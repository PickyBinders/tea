"""Deterministic residue-coordinate negatives from same-family local EBA."""

from lightning.pytorch.cli import LightningArgumentParser
from collections import defaultdict
import json
import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from tea_train.io import partition_ids, read_json, sha256_file, stable_hash, write_json
from tea_train.eba import aligned_columns, training_positive


MATCHED_WINDOWS = ((1, 2), (1, 5), (6, 10))


def negative_candidates(positive, length, window, forbidden):
    low, high = window
    if not 1 <= low <= high:
        raise ValueError("Negative window must have explicit positive inclusive bounds")
    return [index for shift in range(low, high + 1) for index in (positive - shift, positive + shift)
            if 0 <= index < length and index not in forbidden]


def sample_pair(columns, mapping, window=(1, 5), matched_windows=MATCHED_WINDOWS, seed=42,
                min_residue_score=None, directions="both"):
    """Same eligible anchors across windows; explicit single/both direction policy."""
    if directions not in ("single", "both"):
        raise ValueError("Expected single or both triplet directions")
    positives = defaultdict(set)
    if min_residue_score is not None and not math.isfinite(min_residue_score):
        raise ValueError("Residue cutoff must be finite")
    for column in columns:
        if min_residue_score is not None:
            if "substitution_score" not in column or not math.isfinite(column["substitution_score"]):
                raise ValueError("Residue filtering requires recorded finite EBA substitution scores")
            if column["substitution_score"] < min_residue_score:
                continue
        if not training_positive(column, mapping):
            continue
        a, b, i, j = column["chain_a"], column["chain_b"], column["index_a"], column["index_b"]
        positives[(a, b)].add((i, j))
        positives[(b, a)].add((j, i))
    output = []
    orientations = sorted(positives)
    if directions == "single" and orientations:
        if len(orientations) != 2:
            raise ValueError("Single-direction sampling expects exactly one chain pair")
        choice = int(stable_hash([seed, *orientations[0], "triplet-direction"]), 16) % 2
        orientations = [orientations[choice]]
    for left, right in orientations:
        pairs = positives[(left, right)]
        length = mapping["chains"][right]["length"]
        known = defaultdict(set)
        for i, j in pairs:
            known[i].add(j)
        rows = []
        for i, j in sorted(pairs):
            # Below-cutoff aligned residues remain eligible negatives.
            if not all(negative_candidates(j, length, bounds, known[i]) for bounds in matched_windows):
                continue
            options = negative_candidates(j, length, window, known[i])
            if not options:
                continue
            negative = min(options, key=lambda k: stable_hash([seed, left, right, i, j, k]))
            rows.append((i, j, negative))
        if rows:
            output.append({"protein_1": left, "protein_2": right,
                           "index_1": [r[0] for r in rows],
                           "index_2_positive": [r[1] for r in rows],
                           "index_2_negative": [r[2] for r in rows]})
    return output


def main():
    parser = LightningArgumentParser(description=__doc__)
    for name in ("mapping", "splits", "alignments", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, choices=range(4))
    parser.add_argument("--subset", choices=("fit", "validation", "all"), required=True)
    parser.add_argument("--window", nargs=2, type=int, default=[1, 5])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--directions", choices=("single", "both"), default="both",
                        help="Single stores one seeded direction per chain pair; both preserves legacy sampling")
    parser.add_argument("--min-residue-score", type=float, default=None)
    parser.add_argument("--min-alignment-mean", type=float, default=None,
                        help="Training-only mean substitution cutoff; does not change matrix columns")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if any(value is not None and not math.isfinite(value) for value in (args.min_residue_score, args.min_alignment_mean)):
        raise ValueError("Training score cutoffs must be finite")
    mapping, splits = read_json(args.mapping), read_json(args.splits)
    mapping_id = stable_hash(mapping)
    source = read_json(args.alignments / "manifest.json")
    if source["shards"] != 1:
        raise ValueError("Merge the complete EBA shard set before creating training triplets")
    shared = args.subset == "all"
    if shared and (args.outer_fold is not None or source.get("purpose") != "shared-pairwise-preprocessing"
                   or source.get("mapping_id") != mapping_id):
        raise ValueError("Shared triplets require matching shared pairwise preprocessing")
    if (source["split_id"], source["outer_fold"], source["subset"]) != (None if shared else splits["split_id"], args.outer_fold, args.subset):
        raise ValueError("Alignment source does not match requested split")
    if sha256_file(args.alignments / "pairs.jsonl") != source["files"]["pairs.jsonl"]:
        raise ValueError("Alignment input checksum mismatch")
    allowed = set(mapping["chains"]) if shared else set(partition_ids(splits, args.subset, args.outer_fold))
    schema = pa.schema([("protein_1", pa.string()), ("protein_2", pa.string()),
                        ("index_1", pa.list_(pa.int64())), ("index_2_positive", pa.list_(pa.int64())),
                        ("index_2_negative", pa.list_(pa.int64()))])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".partial")
    rows, triplets, seen = [], 0, set()
    with pq.ParquetWriter(temporary, schema) as writer, open(args.alignments / "pairs.jsonl") as handle:
        for line in handle:
            record = json.loads(line)
            left, right = record["chain_a"], record["chain_b"]
            if left not in allowed or right not in allowed:
                raise ValueError("Triplet endpoint outside declared partition")
            key = tuple(sorted((left, right)))
            if key in seen:
                raise ValueError("Duplicate alignment pair")
            seen.add(key)
            if not record["accepted"]:
                continue
            if args.min_alignment_mean is not None and record["alignment"]["mean_paired_score"] < args.min_alignment_mean:
                continue
            columns = aligned_columns(record["alignment"], left, right, mapping)
            for row in sample_pair(columns, mapping, tuple(args.window), seed=args.seed,
                                   min_residue_score=args.min_residue_score, directions=args.directions):
                rows.append(row)
                triplets += len(row["index_1"])
            if len(rows) >= 1000:
                writer.write_table(pa.Table.from_pylist(rows, schema))
                rows = []
        if rows:
            writer.write_table(pa.Table.from_pylist(rows, schema))
    if not triplets:
        raise ValueError("No eligible triplets; inspect family-span/window retention")
    temporary.replace(args.output)
    write_json(args.output.with_suffix(args.output.suffix + ".json"), {
        "schema_version": 1, "split_id": None if shared else splits["split_id"], "outer_fold": args.outer_fold,
        "purpose": "shared-pairwise-preprocessing" if shared else "partition-preprocessing",
        "mapping_id": mapping_id,
        "subset": args.subset, "seed": args.seed, "window_inclusive": args.window,
        "matched_windows": MATCHED_WINDOWS, "coordinate_space": "full-chain-residue",
        "training_positive_mask": "shared-family-spans", "triplets": triplets,
        "directions": args.directions,
        "direction_selection": "seeded-pair-hash" if args.directions == "single" else "both",
        "minimum_residue_score_inclusive": args.min_residue_score,
        "minimum_alignment_mean_inclusive": args.min_alignment_mean,
        "alignment_scoring": {
            "distance_scale": source.get("distance_scale"),
            "linear_gap_penalty": source.get("linear_gap_penalty"),
            "affine_gap_open": source.get("affine_gap_open"),
            "affine_gap_extend": source.get("affine_gap_extend"),
            "alignment_backend": source.get("alignment_backend"),
        },
        "alignments_sha256": source["files"]["pairs.jsonl"], "parquet_sha256": sha256_file(args.output),
        "teacher_id": source["teacher_id"], "embedding_cache_sha256": source["embedding_cache_sha256"],
        "structure_pretrained": source["structure_pretrained"], "pilot_pair_limit": source["pilot_pair_limit"]})
    print(f"Wrote {triplets} triplets in subset {args.subset}")


if __name__ == "__main__":
    main()
