"""Train a TEA head on validated full-chain EBA triplets using Lightning."""

from pathlib import Path

from lightning.pytorch.cli import LightningCLI
import torch

from tea_train.io import read_json, sha256_file, stable_hash, write_json
from tea.model import Tea
from tea_train.full_chain_data import FullChainDataModule
from tea_train.train_tea import TeaModule


class FullChainModule(TeaModule):
    def __init__(self, embeddings: str, recipe: dict, epochs: int = 10, optimization_steps: int | None = None):
        cache = read_json(Path(embeddings) / "manifest.json")
        self.head_spec = {"representation_size": cache["spec"]["dimension"],
                          "hidden_size": recipe["hidden_size"], "codebook_size": recipe["codebook_size"],
                          "dropout_prob": recipe["dropout"]}
        matmul_precision = recipe.get("float32_matmul_precision", "highest")
        if matmul_precision not in ("highest", "high", "medium"):
            raise ValueError("float32_matmul_precision must be highest, high or medium")
        torch.set_float32_matmul_precision(matmul_precision)
        super().__init__(Tea(**self.head_spec), recipe["loss_weights"],
                         recipe.get("uniform_kl_population", "triplets"),
                         recipe.get("training_diagnostics", True))
        compile_mode = recipe.get("compile_mode")
        if compile_mode not in (None, "default", "reduce-overhead",
                                "max-autotune", "max-autotune-no-cudagraphs"):
            raise ValueError("Unsupported torch.compile mode")
        if compile_mode:
            self.model.compile(dynamic=True, mode=compile_mode)
        self.encoder_spec = cache["spec"]
        self.encoder_id = cache["encoder_id"]
        self.embedding_storage_dtype = cache.get("storage_dtype", "unrecorded")
        self.recipe = recipe
        self.epochs = epochs
        if optimization_steps is not None and optimization_steps <= 0:
            raise ValueError("optimization_steps must be positive")
        self.optimization_steps = optimization_steps

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.recipe["learning_rate"],
            weight_decay=self.recipe["weight_decay"], betas=(0.9, 0.999), eps=1e-8,
            fused=self.recipe.get("fused_adamw", False),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.optimization_steps or self.epochs,
                                      eta_min=self.recipe["minimum_learning_rate"])
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler,
                "interval": "step" if self.optimization_steps is not None else "epoch", "frequency": 1}}


class TeaCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.embeddings", "model.embeddings")
        parser.link_arguments("model.epochs", "trainer.max_epochs")

    def before_fit(self):
        if self.model.optimization_steps is not None:
            self.trainer.fit_loop.epoch_loop.max_steps = self.model.optimization_steps
            self.trainer.fit_loop.max_epochs = -1
        if (Path(self.trainer.default_root_dir) / "head.pt").exists():
            raise FileExistsError("A completed TEA head already exists in this run directory")
        if self.datamodule.full_data:
            self.trainer.limit_val_batches = 0
            self.trainer.num_sanity_val_steps = 0

    def after_fit(self):
        output = Path(self.trainer.default_root_dir)
        module, data = self.model, self.datamodule
        if module.optimization_steps is not None and self.trainer.global_step != module.optimization_steps:
            raise RuntimeError("Training ended before the requested optimizer-step budget")
        numerics = {"encoder_precision": module.encoder_spec["precision"],
                    "encoder_frozen": True, "embedding_storage_dtype": module.embedding_storage_dtype,
                    "lightning_precision": str(self.trainer.precision),
                    "float32_matmul_precision": torch.get_float32_matmul_precision(),
                    "head_parameter_dtypes": sorted({str(value.dtype) for value in module.model.parameters()})}
        checkpoint = {"schema_version": 1, "head_spec": module.head_spec, "encoder_spec": module.encoder_spec,
                      "numerics": numerics,
                      "training_partition": {"split_id": data.splits["split_id"], "outer_fold": data.outer_fold,
                          "purpose": "full-dataset-training" if data.full_data else "fold-held-out-training",
                          "dataset_id": data.splits.get("dataset_id"),
                          "mapping_id": data.splits.get("mapping_id"),
                          "embedding_cache_sha256": sha256_file(data.embeddings / "manifest.json"),
                          "fit_chain_ids_sha256": stable_hash(data.fitting_ids)},
                      "state_dict": {key: value.detach().cpu() for key, value in module.model.state_dict().items()}}
        output.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, output / "head.pt")
        write_json(output / "manifest.json", {"schema_version": 1, "recipe": module.recipe, "epochs": module.epochs,
            "optimization_steps": module.optimization_steps, "completed_optimizer_steps": self.trainer.global_step,
            "sampling": data.sampling,
            "sampling_mapping_sha256": sha256_file(data.mapping_path) if data.mapping_path else None,
            "scheduler_interval": "step" if module.optimization_steps is not None else "epoch",
            "scheduler_last_epoch": self.trainer.lr_scheduler_configs[0].scheduler.last_epoch,
            "final_learning_rate": self.trainer.optimizers[0].param_groups[0]["lr"],
            "numerics": numerics,
            "checkpoint_id": sha256_file(output / "head.pt"), "encoder_id": module.encoder_id,
            "outer_fold": data.outer_fold, "split_id": data.splits["split_id"],
            "fit_triplets_sha256": sha256_file(data.data_folder / "fit.parquet"),
            "training_partition": checkpoint["training_partition"],
            "validation_triplets_sha256": None if data.full_data else sha256_file(data.data_folder / "validation.parquet"),
            "epoch_selection": "fixed-budget", "optimizer": "torch.optim.AdamW"})


def main():
    TeaCLI(FullChainModule, FullChainDataModule, seed_everything_default=42,
           auto_configure_optimizers=False, save_config_kwargs={"overwrite": False})


if __name__ == "__main__":
    main()
