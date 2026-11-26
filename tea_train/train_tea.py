from collections import defaultdict
from lightning.pytorch import LightningModule
from lightning.pytorch.cli import LightningCLI
from torch.nn import functional as F
import torch
from tea.model import Tea

def get_residue_triplets(batch, output_1, output_2, keys=("encoded", "quantized", "decoded", "embeddings")):
    valid_triplets = (
        torch.arange(batch["indices_1"].size(1), device=batch["indices_1"].device)[
            None, :
        ]
        < batch["num_triplets"][:, None]
    ).flatten()
    output = {}
    for key in keys:
        if key not in output_1:
            continue
        for suffix, values in (("1", output_1[key]), ("2_positive", output_2[key]),
                               ("2_negative", output_2[key])):
            width = values.shape[-1]
            indices = batch[f"indices_{suffix}"].unsqueeze(-1).expand(-1, -1, width)
            output[f"{key}_{suffix}"] = values.gather(1, indices).reshape(-1, width)[valid_triplets]
    return output

def triplet_logits(batch, left, right):
    if batch.get("packed", False):
        return {
            "logits_1": left[batch["indices_1"]],
            "logits_2_positive": right[batch["indices_2_positive"]],
            "logits_2_negative": right[batch["indices_2_negative"]],
        }
    return get_residue_triplets(batch, {"logits": left}, {"logits": right}, keys=("logits",))


class TeaModule(LightningModule):
    def __init__(
        self,
        model: Tea,
        loss_weights: dict,
        uniform_kl_population: str = "triplets",
        training_diagnostics: bool = True,
    ):
        super().__init__()
        if uniform_kl_population not in ("triplets", "full-sequence"):
            raise ValueError("uniform_kl_population must be 'triplets' or 'full-sequence'")
        self.model = model
        self.loss_weights = loss_weights
        self.uniform_kl_population = uniform_kl_population
        self.training_diagnostics = training_diagnostics
        self.track = defaultdict(int)
        self.running_max = {}
        self.save_hyperparameters()

    def compute_uniform_kl_loss(self, logits, attention_mask=None):
        """Compute normalized KL divergence from uniform distribution."""
        probs = F.softmax(logits, dim=-1)
        
        if attention_mask is not None:
            e_mean = (probs * attention_mask.unsqueeze(-1)).sum((0, 1))
            e_mean = e_mean / attention_mask.sum()
        else:
            e_mean = probs.mean((0))
        
        e_mean = torch.clamp(e_mean, min=1e-8)
        target_distribution = torch.ones_like(e_mean) / self.model.codebook_size
        
        kl_div = torch.sum(e_mean * (torch.log(e_mean + 1e-8) - torch.log(target_distribution + 1e-8)))
        return kl_div / torch.log(torch.tensor(self.model.codebook_size, dtype=e_mean.dtype, device=e_mean.device))

    def compute_sequence_identity_loss(self, anchor_vec, positive_vec,
                                       negative_vec):
        anchor_probs = F.softmax(anchor_vec, dim=-1)
        positive_probs = F.softmax(positive_vec, dim=-1)
        negative_probs = F.softmax(negative_vec, dim=-1)
        pos_similarity = F.cosine_similarity(anchor_probs, positive_probs, dim=-1)
        neg_similarity = F.cosine_similarity(anchor_probs, negative_probs, dim=-1)
        
        losses = {}
        losses["pos_identity"] = pos_similarity.mean()
        losses["neg_identity"] = neg_similarity.mean()
        if self.training_diagnostics:
            sharp_anchor_indices = torch.argmax(anchor_probs, dim=-1)
            sharp_pos_indices = torch.argmax(positive_probs, dim=-1)
            sharp_neg_indices = torch.argmax(negative_probs, dim=-1)
            sharp_anchor_probs = F.one_hot(sharp_anchor_indices, num_classes=self.model.codebook_size).float()
            sharp_pos_probs = F.one_hot(sharp_pos_indices, num_classes=self.model.codebook_size).float()
            sharp_neg_probs = F.one_hot(sharp_neg_indices, num_classes=self.model.codebook_size).float()
            sharp_pos_similarity = F.cosine_similarity(sharp_anchor_probs, sharp_pos_probs, dim=-1)
            sharp_neg_similarity = F.cosine_similarity(sharp_anchor_probs, sharp_neg_probs, dim=-1)
            losses["sharp_pos_identity"] = sharp_pos_similarity.mean()
            losses["sharp_neg_identity"] = sharp_neg_similarity.mean()
        losses["si_loss"] = (self.loss_weights["neg_weight"] * neg_similarity - self.loss_weights["pos_weight"] * pos_similarity).mean()
        return losses


    def calculate_loss(self, batch):
        anchor_output = self.model(batch["embedding_1"].float())
        other_output = self.model(batch["embedding_2"].float())
        triplet_output = triplet_logits(batch, anchor_output, other_output)
        losses = {}
        
        compute_entropy = self.training_diagnostics or self.loss_weights.get("mean_entropy", 0) != 0
        if compute_entropy:
            losses["anchor_entropy"] = self.model.compute_shannon_entropy(triplet_output['logits_1']).mean()
            losses["positive_entropy"] = self.model.compute_shannon_entropy(triplet_output['logits_2_positive']).mean()
            losses["negative_entropy"] = self.model.compute_shannon_entropy(triplet_output['logits_2_negative']).mean()
        if self.uniform_kl_population == "triplets":
            losses["uniform_kl_loss"] = self.compute_uniform_kl_loss(
                torch.cat([
                    triplet_output['logits_1'],
                    triplet_output['logits_2_positive'],
                    triplet_output['logits_2_negative'],
                ])
            )
        else:
            if batch.get("packed", False):
                kl_logits = torch.cat([anchor_output, other_output])
            else:
                masks = []
                for index, output in ((1, anchor_output), (2, other_output)):
                    positions = torch.arange(output.size(1), device=output.device)
                    masks.append(
                        positions[None, :] < batch[f"embedding_{index}_length"][:, None]
                    )
                kl_logits = torch.cat([anchor_output[masks[0]], other_output[masks[1]]])
            losses["uniform_kl_loss"] = self.compute_uniform_kl_loss(kl_logits)
        for k, l in self.compute_sequence_identity_loss(
            triplet_output['logits_1'],
            triplet_output['logits_2_positive'],
            triplet_output['logits_2_negative'],
        ).items():
            losses[k] = l
        if compute_entropy:
            losses["mean_entropy"] = (losses["anchor_entropy"] + losses["positive_entropy"] + losses["negative_entropy"]) / 3
        return losses

    def step(self, batch, batch_idx, split):
        losses = self.calculate_loss(batch)
        
        weighted_losses = [
            losses[k] * self.loss_weights[k]
            for k in self.loss_weights if k in losses
        ]
        losses["loss"] = sum(weighted_losses)
   
        batch_size = batch["batch_triplets"]
        for key, value in losses.items():
            self.log(
                f"{split}/{key}",
                value,
                on_step=(split == "train"), on_epoch=True,
                batch_size=batch_size,
            )
        losses = {
            k: losses[k]
            for k in losses if k in self.loss_weights or k == "loss"
        }
        return losses

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, "train")
        return loss["loss"]

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, "val")
        return loss["loss"]


def main():
    """
    Run with python main.py fit -c config.yaml
    Or in an sbatch script with srun python main.py fit -c config.yaml
    """
    torch.set_float32_matmul_precision("medium")
    LightningCLI(save_config_kwargs={"overwrite": True})


if __name__ == "__main__":
    main()
