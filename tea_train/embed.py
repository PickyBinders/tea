"""Generate complete full-chain embedding shards through the shared encoder."""

from lightning.pytorch.cli import LightningArgumentParser
import hashlib
from pathlib import Path

import torch
from filelock import FileLock

from tea_train.encoding import EncoderSpec, FrozenEncoder, batch_policy, iter_embedding_batches, normalize_sequence
from tea_train.io import read_fasta, read_json, sha256_file, write_json


def generate(fasta, spec, output, encoder_factory, shard_residues=100000, resume=False,
             residue_budget=0, storage_dtype="float16", require_full_context=False):
    output = Path(output)
    if shard_residues <= 0:
        raise ValueError("Positive shard residue budget required")
    batch_policy(residue_budget)
    if storage_dtype not in ("float16", "int8"):
        raise ValueError("Embedding storage must be float16 or int8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output) + ".lock", timeout=0):
        return _generate(fasta, spec, output, encoder_factory, shard_residues, resume,
                         residue_budget, storage_dtype, require_full_context)


def _generate(fasta, spec, output, encoder_factory, shard_residues, resume,
              residue_budget, storage_dtype, require_full_context):
    sequences = read_fasta(fasta)
    metadata = {"schema_version": 1, "spec": spec.to_dict(), "encoder_id": spec.identifier,
                "fasta_sha256": sha256_file(fasta), "shard_residues": shard_residues,
                "stitching": "uniform-overlap-average-float32", "storage_dtype": storage_dtype}
    if storage_dtype == "int8":
        metadata["storage_quantization"] = "per-residue-symmetric-absmax-fp32-scale-v1"
    if require_full_context:
        metadata["stitching"] = "one-encoder-span-per-full-chain"
    if residue_budget:
        metadata["batch_policy"] = batch_policy(residue_budget)
    complete = output / "manifest.json"
    progress = output / "progress.json"
    if output.exists():
        if not resume:
            raise FileExistsError("Embedding destination exists; use --resume or a new versioned cache")
        manifest = read_json(complete if complete.exists() else progress)
        if any(manifest.get(key) != value for key, value in metadata.items()):
            raise ValueError("Resume requires identical FASTA, encoder and shard settings")
        for name, checksum in manifest["shards"].items():
            if sha256_file(output / name) != checksum:
                raise ValueError(f"Committed embedding shard checksum mismatch: {name}")
        for chain, record in manifest["chains"].items():
            if chain not in sequences or record["length"] != len(sequences[chain]):
                raise ValueError("Committed chain is outside the input FASTA")
            if record["sequence_sha256"] != hashlib.sha256(normalize_sequence(sequences[chain]).encode()).hexdigest():
                raise ValueError("Committed chain sequence changed")
            if record["shard"] not in manifest["shards"]:
                raise ValueError("Committed chain references an unknown shard")
        if complete.exists():
            if set(manifest["chains"]) != set(sequences):
                raise ValueError("Completed embedding cache has incomplete coverage")
            return manifest
    else:
        output.mkdir()
        manifest = {**metadata, "chains": {}, "shards": {}}
        write_json(progress, manifest)
    if set(manifest["chains"]) == set(sequences):
        write_json(complete, manifest)
        return manifest
    encoder = encoder_factory()
    if require_full_context and any(
        encoder.context_spans(length) != [(0, length)] for length in {len(seq) for seq in sequences.values()}
    ):
        raise ValueError("Untiled training requires a single encoder span for every chain length")
    shard, residues, index = {}, 0, len(manifest["shards"])

    def flush():
        nonlocal shard, residues, index
        name = f"embeddings_{index:05d}.pt"
        if name in manifest["shards"]:
            raise ValueError("Refusing to replace a committed embedding shard")
        temporary = output / f"{name}.partial"
        torch.save(shard, temporary)
        temporary.replace(output / name)
        manifest["shards"][name] = sha256_file(output / name)
        for chain, value in shard.items():
            length = value.shape[0] if isinstance(value, torch.Tensor) else value["code"].shape[0]
            manifest["chains"][chain] = {"length": length, "shard": name,
                "sequence_sha256": hashlib.sha256(normalize_sequence(sequences[chain]).encode()).hexdigest()}
        # Record progress after the shard is complete.
        write_json(progress, manifest)
        print(f"Wrote {name}: {len(shard)} chains, {residues} residues", flush=True)
        shard, residues, index = {}, 0, index + 1

    remaining = {chain: sequence for chain, sequence in sequences.items()
                 if chain not in manifest["chains"]}
    if residue_budget:
        batches = iter_embedding_batches(encoder, remaining, residue_budget)
    else:
        batches = ([(chain, encoder.encode(sequence))]
                   for chain, sequence in sorted(remaining.items()))
    for batch in batches:
        for chain, value in batch:
            value = value.half()
            if value.shape != (len(sequences[chain]), spec.dimension) or not torch.isfinite(value).all():
                raise ValueError("Invalid or overflowing stored residue embeddings")
            if storage_dtype == "int8":
                rounded = value.float()
                scale = (rounded.abs().amax(dim=1, keepdim=True) / 127).clamp_min(1e-8)
                code = (rounded / scale).round().clamp(-127, 127).to(torch.int8)
                shard[chain] = {"code": code, "scale": scale}
            else:
                shard[chain] = value
            residues += len(sequences[chain])
            if residues >= shard_residues:
                flush()
    if shard:
        flush()
    if set(manifest["chains"]) != set(sequences):
        raise ValueError("Embedding output does not cover input FASTA")
    write_json(complete, manifest)
    return manifest


def main():
    parser = LightningArgumentParser(description=__doc__)
    for name in ("fasta", "spec", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-residues", type=int, default=100000)
    parser.add_argument("--residue-budget", type=int, default=0,
                        help="Batch encoder forwards within this padded-residue budget")
    parser.add_argument("--storage-dtype", choices=("float16", "int8"), default="float16")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--require-full-context", action="store_true")
    args = parser.parse_args()
    spec = EncoderSpec(**read_json(args.spec))
    generate(args.fasta, spec, args.output,
             lambda: FrozenEncoder(spec, args.device, local_files_only=not args.allow_download),
             args.shard_residues, args.resume, args.residue_budget, args.storage_dtype, args.require_full_context)


if __name__ == "__main__":
    main()
