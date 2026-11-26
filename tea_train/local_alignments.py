"""Generate same-family local EBA alignments from full-chain embedding caches."""

from lightning.pytorch.cli import LightningArgumentParser
import json
import math
from pathlib import Path

from tea_train.encoding import EmbeddingCache
from tea_train.io import partition_ids, read_json, sha256_file, stable_hash, write_json
from tea_train.eba import aligned_columns, family_candidates, local_from_embeddings


def main():
    parser = LightningArgumentParser(description=__doc__)
    for name in ("mapping", "splits", "embeddings", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, choices=range(4))
    parser.add_argument("--subset", choices=("fit", "validation", "all"), required=True)
    parser.add_argument("--reuse", nargs="*", type=Path, default=[], help="Completed alignment directories to reuse")
    parser.add_argument("--distance-scale", type=float, default=1.0)
    parser.add_argument("--gap", type=float, default=0.0)
    parser.add_argument("--gap-open", type=float, default=None)
    parser.add_argument("--gap-extend", type=float, default=None)
    parser.add_argument("--preload-embeddings", action="store_true",
                        help="Load required cache shards once in shard order; request sufficient CPU memory")
    parser.add_argument("--min-local-score", type=float, default=None,
                        help="Optional score-filter ablation; omitted for same-family-only supervision")
    parser.add_argument("--min-paired", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--limit", type=int, help="Explicit pilot pair limit; never report as production data")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Alignment destination exists")
    if not 0 <= args.shard < args.shards or args.min_paired < 3:
        parser.error("Invalid shard or minimum aligned length")
    affine = args.gap_open is not None or args.gap_extend is not None
    if affine and (args.gap != 0 or args.gap_open is None or args.gap_extend is None or
                   not math.isfinite(args.gap_open + args.gap_extend) or args.gap_open <= 0 or args.gap_extend < 0):
        parser.error("Affine gaps require --gap-open >0 and --gap-extend >=0, without --gap")
    scoring = {"linear_gap_penalty": None if affine else args.gap}
    if affine:
        import biotite
        scoring.update({"affine_gap_open": args.gap_open, "affine_gap_extend": args.gap_extend,
                        "alignment_backend": "biotite-local-gotoh", "biotite_version": biotite.__version__,
                        "gap_cost": "open + (length - 1) * extend", "fixed_point_policy": "10000-with-int32-headroom-minimum-1000"})
    cache = EmbeddingCache(args.embeddings)
    if cache.manifest["spec"]["precision"] != "4bit":
        raise ValueError("EBA supervision requires the shared 4-bit encoder cache")
    mapping, splits = read_json(args.mapping), read_json(args.splits)
    if args.subset == "all" and args.outer_fold is not None:
        raise ValueError("Shared pairwise preprocessing has no outer fold")
    shared = args.subset == "all"
    identifiers = sorted(mapping["chains"]) if shared else partition_ids(splits, args.subset, args.outer_fold)
    input_settings = {
        "teacher_id": cache.manifest["encoder_id"],
        "mapping_sha256": sha256_file(args.mapping),
        "embedding_cache_sha256": sha256_file(args.embeddings / "manifest.json"),
        "mode": "eba-local-best",
        "recurse": False,
        "distance_scale": args.distance_scale,
        **scoring,
    }
    reused = {}
    for directory in args.reuse:
        source = read_json(directory / "manifest.json")
        if source.get("pilot_pair_limit") is not None or any(
            source.get(key) != value for key, value in input_settings.items()
        ):
            raise ValueError("Reusable alignment source has incompatible inputs or EBA settings")
        if sha256_file(directory / "pairs.jsonl") != source["files"]["pairs.jsonl"]:
            raise ValueError("Reusable pair checksum mismatch")
        with open(directory / "pairs.jsonl") as handle:
            for line in handle:
                row = json.loads(line)
                key = (row["chain_a"], row["chain_b"])
                if key in reused and reused[key] != row["alignment"]:
                    raise ValueError("Conflicting cached pairwise alignments")
                reused[key] = row["alignment"]
    missing = set(identifiers) - set(cache.manifest["chains"])
    if missing:
        raise ValueError(f"Missing embeddings for {len(missing)} chains")
    for key in identifiers:
        if cache.manifest["chains"][key]["sequence_sha256"] != mapping["chains"][key]["sequence_sha256"]:
            raise ValueError(f"Embedding sequence mismatch: {key}")
    candidates = [(left, right, shared) for number, (left, right, shared) in
                  enumerate(family_candidates(mapping, identifiers)) if number % args.shards == args.shard]
    if args.limit is not None:
        candidates = candidates[:args.limit]
    embeddings = cache
    if args.preload_embeddings:
        needed = {key for left, right, _ in candidates if (left, right) not in reused for key in (left, right)}
        embeddings = {key: cache[key] for key in sorted(needed, key=lambda key: (cache.manifest["chains"][key]["shard"], key))}
    args.output.mkdir(parents=True)
    counts = {"candidates": 0, "accepted": 0, "rejected": 0, "aligned_columns": 0}
    with open(args.output / "pairs.jsonl.partial", "w") as pairs, open(args.output / "columns.jsonl.partial", "w") as columns:
        for left, right, shared in candidates:
            counts["candidates"] += 1
            result = reused[(left, right)] if (left, right) in reused else local_from_embeddings(
                embeddings[left].to(args.device), embeddings[right].to(args.device), args.distance_scale, args.gap,
                gap_open=args.gap_open, gap_extend=args.gap_extend)
            accepted = (result is not None and result["paired_residues"] >= args.min_paired
                        and (args.min_local_score is None or result["raw_score"] > args.min_local_score))
            counts["accepted" if accepted else "rejected"] += 1
            pairs.write(json.dumps({"chain_a": left, "chain_b": right, "shared_families": shared,
                                    "accepted": accepted, "alignment": result}, allow_nan=False) + "\n")
            if accepted:
                for column in aligned_columns(result, left, right, mapping):
                    columns.write(json.dumps(column, allow_nan=False) + "\n")
                    counts["aligned_columns"] += 1
            if counts["candidates"] % 100 == 0:
                print(counts, flush=True)
    for name in ("pairs.jsonl", "columns.jsonl"):
        (args.output / (name + ".partial")).replace(args.output / name)
    write_json(args.output / "manifest.json", {"schema_version": 1,
        **input_settings, "split_id": None if shared else splits["split_id"],
        "purpose": "shared-pairwise-preprocessing" if shared else "partition-preprocessing",
        "mapping_id": stable_hash(mapping),
        "splits_sha256": sha256_file(args.splits),
        "structure_pretrained": cache.manifest["spec"]["structure_pretrained"],
        "outer_fold": args.outer_fold, "subset": args.subset, "counts": counts,
        "minimum_local_raw_score_exclusive": args.min_local_score,
        "minimum_paired_residues": args.min_paired, "posterior_calibrated": False,
        "pilot_pair_limit": args.limit, "shard": args.shard, "shards": args.shards,
        "matrix_column_selection": "all-non-gap-no-structural-filter",
        "files": {name: sha256_file(args.output / name) for name in ("pairs.jsonl", "columns.jsonl")}})
    print(counts)


if __name__ == "__main__":
    main()
