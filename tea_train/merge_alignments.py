"""Merge complete compatible EBA shards and retain rejected-pair records."""

from collections import Counter
import json
from pathlib import Path

from lightning.pytorch.cli import LightningArgumentParser

from tea_train.io import read_json, sha256_file, write_json


def merge(directories, output):
    directories, output = [Path(path) for path in directories], Path(output)
    if output.exists():
        raise FileExistsError(output)
    manifests = [read_json(path / "manifest.json") for path in directories]
    if not manifests:
        raise ValueError("No alignment shards supplied")
    reference = {key: value for key, value in manifests[0].items() if key not in ("counts", "files", "shard")}
    if sorted(info["shard"] for info in manifests) != list(range(reference["shards"])):
        raise ValueError("Missing or duplicate alignment shard indices")
    for path, info in zip(directories, manifests):
        if {key: value for key, value in info.items() if key not in ("counts", "files", "shard")} != reference:
            raise ValueError("Alignment shards have different encoder, split or scoring settings")
        for name in ("pairs.jsonl", "columns.jsonl"):
            if sha256_file(path / name) != info["files"][name]:
                raise ValueError("Alignment shard checksum mismatch")
    output.mkdir(parents=True)
    seen, totals = set(), Counter()
    ordered = sorted(zip(directories, manifests), key=lambda item: item[1]["shard"])
    with open(output / "pairs.jsonl.partial", "w") as pairs, open(output / "columns.jsonl.partial", "w") as columns:
        for path, info in ordered:
            accepted = set()
            counts = Counter({key: 0 for key in ("candidates", "accepted", "rejected", "aligned_columns")})
            with open(path / "pairs.jsonl") as handle:
                for line in handle:
                    row = json.loads(line)
                    key = tuple(sorted((row["chain_a"], row["chain_b"])))
                    if key[0] == key[1] or key in seen:
                        raise ValueError("Self or duplicate alignment pair across shards")
                    seen.add(key)
                    if row["accepted"]:
                        accepted.add(key)
                    counts["candidates"] += 1
                    counts["accepted" if row["accepted"] else "rejected"] += 1
                    pairs.write(line.rstrip("\n") + "\n")
            with open(path / "columns.jsonl") as handle:
                for line in handle:
                    row = json.loads(line)
                    key = tuple(sorted((row["chain_a"], row["chain_b"])))
                    if row["kind"] != "eba_aligned" or key not in accepted:
                        raise ValueError("Aligned column lacks an accepted EBA pair")
                    counts["aligned_columns"] += 1
                    columns.write(line.rstrip("\n") + "\n")
            if dict(counts) != info["counts"]:
                raise ValueError("Alignment shard counts disagree with records")
            totals.update(counts)
    for name in ("pairs.jsonl", "columns.jsonl"):
        (output / f"{name}.partial").replace(output / name)
    result = {**reference, "shard": 0, "shards": 1, "counts": dict(totals),
        "source_shards": [{"shard": info["shard"], "manifest_sha256": sha256_file(path / "manifest.json")}
                          for path, info in ordered],
        "files": {name: sha256_file(output / name) for name in ("pairs.jsonl", "columns.jsonl")}}
    write_json(output / "manifest.json", result)
    return result


def main():
    parser = LightningArgumentParser(description=__doc__)
    parser.add_argument("--shards", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(merge(args.shards, args.output)["counts"])


if __name__ == "__main__":
    main()
