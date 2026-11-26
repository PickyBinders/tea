"""Write Lightning configurations for the paper's four-fold training runs."""

import copy
import csv
from pathlib import Path

from lightning.pytorch.cli import LightningArgumentParser
import yaml

from tea.model import DEFAULT_CHARACTERS


REFERENCE_BACKBONE = "esm2-650m"
PAPER_PANELS = {
    "reference": ("selected_recipe",),
    "core": ("kl_0", "window_6_10"),
    "alphabet": tuple(f"k{size:02d}_window_1_5" for size in range(4, 41, 4) if size != 20),
    "backbones": ("esm2-3b", "prott5", "prostt5"),
}


def paper_config(base, panel, condition, fold, work_dir, run_dir):
    config = copy.deepcopy(base)
    backbone = condition if panel == "backbones" else REFERENCE_BACKBONE
    window = (6, 10) if condition == "window_6_10" else (1, 5)
    data_root = work_dir if backbone == REFERENCE_BACKBONE else work_dir / "backbones" / backbone
    triplet_root = data_root if window == (1, 5) else data_root / "far-window"
    config["data"].update(
        outer_fold=fold,
        data_folder=str(triplet_root / f"fold{fold}"),
        embeddings=str(data_root / "embeddings"),
        mapping=str(work_dir / "data/raw/scope40_2_08/mapping.json"),
        splits=str(work_dir / "data/splits/scope40_2_08.json"),
    )
    recipe = config["model"]["recipe"]
    if condition == "kl_0":
        recipe["loss_weights"]["uniform_kl_loss"] = 0.0
    if panel == "alphabet":
        recipe["codebook_size"] = int(condition[1:3])
    config["trainer"]["default_root_dir"] = str(run_dir)
    for logger in config["trainer"]["logger"]:
        logger["init_args"]["save_dir"] = str(run_dir)
        if logger["class_path"].endswith("WandbLogger"):
            logger["init_args"].update(name=f"{condition}-fold{fold}", group=panel)
    return config, backbone, window, data_root / "eba/columns.jsonl"


def main():
    parser = LightningArgumentParser(description=__doc__)
    config_dir = Path(__file__).with_name("configs")
    parser.add_argument("--base", type=Path, default=config_dir / "train_full_chain.yaml")
    parser.add_argument("--panel", choices=("all", *PAPER_PANELS), default="all")
    parser.add_argument("--work-dir", type=Path, default=Path("work"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = yaml.safe_load(args.base.read_text())
    panels = PAPER_PANELS if args.panel == "all" else {args.panel: PAPER_PANELS[args.panel]}
    args.output.mkdir(parents=True)
    rows = []
    for panel, conditions in panels.items():
        for condition in conditions:
            for fold in range(4):
                directory = args.output / condition / f"fold{fold}"
                directory.mkdir(parents=True)
                config, backbone, window, columns = paper_config(
                    base, panel, condition, fold, args.work_dir, directory / "training"
                )
                path = directory / "train.yaml"
                path.write_text(yaml.safe_dump(config, sort_keys=False))
                rows.append({
                    "panel": panel,
                    "condition": condition,
                    "fold": fold,
                    "backbone": backbone,
                    "negative_window": f"{window[0]}-{window[1]}",
                    "alphabet": DEFAULT_CHARACTERS[:config["model"]["recipe"]["codebook_size"]],
                    "embeddings": config["data"]["embeddings"],
                    "eba_columns": columns,
                    "checkpoint": directory / "training/head.pt",
                    "config": path,
                })
    with (args.output / "runs.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0], delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
