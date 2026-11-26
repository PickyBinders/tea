"""Download a pinned model snapshot to the TEA cache."""

from pathlib import Path

from huggingface_hub import snapshot_download
from lightning.pytorch.cli import LightningArgumentParser

from tea_train.encoding import EncoderSpec
from tea_train.io import read_json, sha256_file, write_json


def main():
    parser = LightningArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    spec = EncoderSpec(**read_json(args.spec))
    if len(spec.revision) != 40 or any(character not in "0123456789abcdef" for character in spec.revision):
        raise ValueError("Model download requires a pinned commit SHA")
    snapshot = Path(snapshot_download(spec.model_id, revision=spec.revision, cache_dir=args.cache,
        allow_patterns=["*.json", "*.model", "*.txt", "*.safetensors", "*.bin", "README.md"], max_workers=2))
    files = {str(path.relative_to(snapshot)): sha256_file(path) for path in sorted(snapshot.rglob("*")) if path.is_file()}
    if not any(name.endswith((".safetensors", ".bin")) for name in files):
        raise ValueError("Downloaded snapshot contains no model weights")
    write_json(args.manifest, {"schema_version": 1, "model_id": spec.model_id,
        "revision": spec.revision, "files": files, "status": "downloaded-not-model-validated"})
    print(f"Staged {len(files)} files for {spec.model_id} at {spec.revision}")


if __name__ == "__main__":
    main()
