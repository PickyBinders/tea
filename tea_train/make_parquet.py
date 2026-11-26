"""Combine original structural triplet tables into training parquet files."""

import json
from pathlib import Path

from lightning.pytorch.cli import LightningArgumentParser
import pandas as pd


def make_parquet(
    triplet_folder,
    lookup_file,
    fold_split,
    output_folder,
    validation_fold=0,
    max_shift=5,
):
    tables = [
        pd.read_csv(path, sep="\t")
        for path in sorted(Path(triplet_folder).glob("*.tsv"))
    ]
    data = pd.concat(tables, ignore_index=True)
    lookup = pd.read_csv(
        lookup_file, sep="\t", header=None, names=["id", "scop_id"]
    )
    id_to_fold = dict(
        zip(
            lookup["id"],
            lookup["scop_id"].map(
                lambda value: ".".join(value.split(".")[:2])
            ),
        )
    )
    with Path(fold_split).open() as handle:
        split = json.load(handle)

    data["fold_split"] = data["protein_1"].map(
        lambda key: split[id_to_fold[key]]
    )
    data = (
        data[data["shift"].abs() <= max_shift]
        .sample(frac=1.0, random_state=42)
        .groupby(["protein_1", "protein_2", "index_1", "index_2_positive"])
        .head(1)
    )
    columns = ["index_1", "index_2_positive", "index_2_negative"]
    data = (
        data.groupby(["protein_1", "protein_2", "fold_split"])[columns]
        .agg(list)
        .reset_index()
    )
    data["triplet_data"] = data[columns].apply(dict, axis=1)
    data = data[["protein_1", "protein_2", "fold_split", "triplet_data"]]

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    data.to_parquet(output_folder / "train.parquet", index=False)
    data[data["fold_split"] == validation_fold].to_parquet(
        output_folder / "val.parquet", index=False
    )


def main():
    parser = LightningArgumentParser(description=__doc__)
    for name in ("triplet_folder", "lookup_file", "fold_split", "output_folder"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--validation-fold", type=int, default=0)
    parser.add_argument("--max-shift", type=int, default=5)
    args = parser.parse_args()
    make_parquet(
        args.triplet_folder,
        args.lookup_file,
        args.fold_split,
        args.output_folder,
        args.validation_fold,
        args.max_shift,
    )


if __name__ == "__main__":
    main()
