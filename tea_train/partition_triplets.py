"""Select a fold-safe partition from shared full-chain EBA triplets."""

from pathlib import Path

from lightning.pytorch.cli import LightningArgumentParser
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from tea_train.io import partition_ids, read_json, sha256_file, write_json


def partition(source, output, mapping, splits, subset, outer_fold=None):
    source = Path(source)
    output = Path(output)
    metadata = read_json(source.with_suffix(".parquet.json"))
    allowed = partition_ids(splits, subset, outer_fold)
    dataset = ds.dataset(source, format="parquet")
    selection = (
        ds.field("protein_1").isin(allowed)
        & ds.field("protein_2").isin(allowed)
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".partial")
    triplets = 0
    with pq.ParquetWriter(temporary, dataset.schema) as writer:
        for batch in dataset.scanner(filter=selection).to_batches():
            writer.write_batch(batch)
            triplets += pc.sum(
                pc.list_value_length(batch.column("index_1"))
            ).as_py() or 0
    temporary.replace(output)
    write_json(
        output.with_suffix(".parquet.json"),
        {
            **metadata,
            "purpose": "partition-preprocessing",
            "split_id": splits["split_id"],
            "subset": subset,
            "outer_fold": outer_fold,
            "triplets": triplets,
            "shared_parquet_sha256": metadata["parquet_sha256"],
            "parquet_sha256": sha256_file(output),
            "selection": "both-chain-endpoints-in-partition",
        },
    )


def main():
    parser = LightningArgumentParser(description=__doc__)
    for name in ("source", "output", "mapping", "splits"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--subset", choices=("fit", "validation"), required=True)
    parser.add_argument("--outer-fold", type=int, choices=range(4))
    args = parser.parse_args()
    partition(
        args.source,
        args.output,
        read_json(args.mapping),
        read_json(args.splits),
        args.subset,
        args.outer_fold,
    )


if __name__ == "__main__":
    main()
