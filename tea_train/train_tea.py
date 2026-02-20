from collections import defaultdict
from lightning.pytorch import LightningModule
from lightning.pytorch.cli import LightningCLI
from torch.nn import functional as F
import torch
from tea.model import Tea
import warnings

warnings.filterwarnings("ignore")

def get_residue_triplets(batch, output_1, output_2, keys=["encoded", "quantized", "decoded", "embeddings"]):
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
        output[f"{key}_1"] = (
            output_1[key]
            .gather(
                1,
                batch["indices_1"]
                .unsqueeze(-1)
                .expand(-1, -1, output_1[key].shape[-1]),
            )
            .reshape(-1, output_1[key].shape[-1])[valid_triplets]
        )
        output[f"{key}_2_positive"] = (
            output_2[key]
            .gather(
                1,
                batch["indices_2_positive"]
                .unsqueeze(-1)
                .expand(-1, -1, output_2[key].shape[-1]),
            )
            .reshape(-1, output_2[key].shape[-1])[valid_triplets]
        )
        output[f"{key}_2_negative"] = (
            output_2[key]
            .gather(
                1,
                batch["indices_2_negative"]
                .unsqueeze(-1)
                .expand(-1, -1, output_2[key].shape[-1]),
            )
            .reshape(-1, output_2[key].shape[-1])[valid_triplets]
        )
    return output

class TeaModule(LightningModule):
    def __init__(
        self,
        model: Tea,
        loss_weights: dict,
    ):
        super().__init__()
        self.model = model
        self.loss_weights = loss_weights
        self.track = defaultdict(int)
        self.running_max = {}
        self.save_hyperparameters()

    def compute_uniform_kl_loss(self, logits, attention_mask=None):
        """Compute normalized KL divergence from uniform distribution."""
        # Convert to one-hot or probabilities
        probs = F.softmax(logits, dim=-1)
        
        if attention_mask is not None:
            # Apply mask and normalize
            e_mean = (probs * attention_mask.unsqueeze(-1)).sum((0, 1))
            e_mean = e_mean / attention_mask.sum()
        else:
            # Fallback for pre-filtered data
            e_mean = probs.mean((0))
        
        e_mean = torch.clamp(e_mean, min=1e-8)
        target_distribution = torch.ones_like(e_mean) / self.model.codebook_size
        
        # Normalized KL divergence
        kl_div = torch.sum(e_mean * (torch.log(e_mean + 1e-8) - torch.log(target_distribution + 1e-8)))
        return kl_div / torch.log(torch.tensor(self.model.codebook_size, dtype=e_mean.dtype, device=e_mean.device))

    def compute_sequence_identity_loss(self, anchor_vec, positive_vec, negative_vec):
        anchor_probs = F.softmax(anchor_vec, dim=-1)
        positive_probs = F.softmax(positive_vec, dim=-1)
        negative_probs = F.softmax(negative_vec, dim=-1)
        pos_similarity = F.cosine_similarity(anchor_probs, positive_probs, dim=-1)
        neg_similarity = F.cosine_similarity(anchor_probs, negative_probs, dim=-1)
        
        losses = {}
        losses["pos_identity"] = pos_similarity.mean()
        losses["neg_identity"] = neg_similarity.mean()
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
        anchor_output = self.model(batch["embedding_1"])  # anchor sequence
        other_output = self.model(batch["embedding_2"])  # sequence for positive/negative samples
        triplet_output = get_residue_triplets(batch, {'logits': anchor_output}, {'logits': other_output}, keys=["logits"])
        losses = {}
        
        # Compute entropy for each set of logits
        losses["anchor_entropy"] = self.model.compute_shannon_entropy(triplet_output['logits_1']).mean()
        losses["positive_entropy"] = self.model.compute_shannon_entropy(triplet_output['logits_2_positive']).mean()
        losses["negative_entropy"] = self.model.compute_shannon_entropy(triplet_output['logits_2_negative']).mean()
        losses["uniform_kl_loss"] = self.compute_uniform_kl_loss(torch.cat([triplet_output['logits_1'], triplet_output['logits_2_positive'], triplet_output['logits_2_negative']], dim=0))
        for k, l in self.compute_sequence_identity_loss(
            triplet_output['logits_1'],
            triplet_output['logits_2_positive'],
            triplet_output['logits_2_negative']
        ).items():
            losses[k] = l
        losses["mean_entropy"] = (losses["anchor_entropy"] + losses["positive_entropy"] + losses["negative_entropy"]) / 3
        return losses

    def step(self, batch, batch_idx, split):
        losses = self.calculate_loss(batch)
        
        weighted_losses = [
            losses[k] * self.loss_weights[k]
            for k in self.loss_weights if k in losses
        ]
        losses["loss"] = sum(weighted_losses)
   
        for key, value in losses.items():
            self.log(
                f"{split}/{key}",
                value.item() if isinstance(value, torch.Tensor) else value,
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
