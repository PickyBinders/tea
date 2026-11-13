"""Fast ESM residue tokens and reusable block-compiled FP16 inference."""

import hashlib
import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import torch


class ResidueTokenizer:
    """Match the converter's ESM tokenizer call with a byte lookup table."""

    def __init__(self, tokenizer):
        if tokenizer.padding_side != "right":
            raise ValueError("TEA residue lookup requires right padding")
        self.cls, self.eos, self.pad = (
            tokenizer.cls_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id,
        )
        vocabulary = tokenizer.get_vocab()
        if max(vocabulary.values()) > 255:
            raise ValueError("Expected the small ESM residue vocabulary")
        lookup = bytearray([tokenizer.unk_token_id] * 256)
        for token, index in vocabulary.items():
            if len(token) == 1 and ord(token) < 256:
                lookup[ord(token)] = index
        self.lookup = bytes(lookup)

    def __call__(self, sequences, *, return_tensors="pt", padding=True,
                 add_special_tokens=True, return_special_tokens_mask=True):
        if (return_tensors != "pt" or padding is not True or not add_special_tokens
                or not return_special_tokens_mask):
            raise ValueError("ResidueTokenizer supports the TEA batch call only")
        encoded = [sequence.replace(" ", "").encode("ascii").translate(self.lookup)
                   for sequence in sequences]
        shape = len(encoded), max(map(len, encoded)) + 2
        ids = np.full(shape, self.pad, dtype=np.int64)
        attention = np.zeros(shape, dtype=np.int64)
        special = np.ones(shape, dtype=np.int64)
        for row, sequence in enumerate(encoded):
            length = len(sequence)
            ids[row, 0] = self.cls
            ids[row, 1:length + 1] = np.frombuffer(sequence, dtype=np.uint8)
            ids[row, length + 1] = self.eos
            attention[row, :length + 2] = 1
            special[row, 1:length + 1] = 0
        return dict(input_ids=torch.from_numpy(ids), attention_mask=torch.from_numpy(attention),
                    special_tokens_mask=torch.from_numpy(special))


def compile_encoder(encoder):
    """Compile repeated blocks, preserving eager masks, rotary math and FP16 casts."""
    from torch._inductor import config

    config.emulate_precision_casts = True
    encoder._create_attention_masks = torch.compiler.disable(encoder._create_attention_masks)
    encoder.rotary_embeddings.forward = torch.compiler.disable(encoder.rotary_embeddings.forward)
    # Every block belongs to the same inference step; intermediate outputs remain live.
    encoder.register_forward_pre_hook(lambda *_: torch.compiler.cudagraph_mark_step_begin())
    for block in encoder.encoder.layer:
        block.compile(mode="reduce-overhead", dynamic=True)
    return encoder


class CompilerCache:
    """Share PyTorch compiler artifacts between matching GPU/software stacks."""

    def __init__(self, directory=None):
        import triton

        if scratch := os.environ.get("TMPDIR"):
            os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(Path(scratch) / "tea-inductor"))
            os.environ.setdefault("TRITON_CACHE_DIR", str(Path(scratch) / "tea-triton"))
        versions = (torch.__version__, triton.__version__, torch.version.cuda,
                    torch.cuda.get_device_name(), "tea-fp16-v1", "regional-dynamic-step-v1")
        key = hashlib.sha256(repr(versions).encode()).hexdigest()[:20]
        cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        self.path = Path(directory or cache_home / "tea/compiler") / f"{key}.bin"
        self.loaded = False

    def load(self):
        try:
            if self.path.exists():
                self.loaded = torch.compiler.load_cache_artifacts(self.path.read_bytes()) is not None
        except (OSError, RuntimeError, ValueError) as error:
            logging.getLogger(__name__).warning("Ignoring unavailable TEA compiler cache: %s", error)
        return self.loaded

    def save(self):
        if self.loaded:
            return
        temporary = None
        try:
            artifacts = torch.compiler.save_cache_artifacts()
            if artifacts is None:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=self.path.parent, delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(artifacts[0])
            temporary.replace(self.path)
            self.loaded = True
        except (OSError, RuntimeError, ValueError) as error:
            logging.getLogger(__name__).warning("Could not save TEA compiler cache: %s", error)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
