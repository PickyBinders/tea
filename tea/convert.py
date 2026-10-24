"""Convert FASTA sequences with the current ESM2-650M TEA model."""

import argparse
from functools import lru_cache
from importlib.resources import files
import json
import math
import os
from pathlib import Path
import tempfile
import time

from biotite.sequence.io.fasta import FastaFile
import torch
from huggingface_hub import try_to_load_from_cache
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

from tea.model import Tea


ESM2_MODEL = "facebook/esm2_t33_650M_UR50D"
ESM2_REVISION = "08e4846e537177426273712802403f7ba8261b6c"
TEA_MODEL = "PickyBinders/tea"
TEA_REVISION = "dev"
BATCH_RESIDUES = 8192
UNKNOWN_RESIDUES = str.maketrans({letter: "X" for letter in "UZOBJ"})


@lru_cache(maxsize=1)
def _tea_likelihood_weights():
    """Frozen matrix–Markov log odds, bundled independently of STEAM."""
    table = json.loads(files("tea").joinpath("data/tea_likelihood.json").read_text())
    return table["alphabet"], table["scores"]


def _residue_entropy(logits):
    """Native T=1 Shannon entropy, normalized to the 20-letter range [0, 1]."""
    probabilities = logits.float().softmax(-1)
    return (torch.special.entr(probabilities).sum(-1) / math.log(20)).clamp_(0, 1)


def _enable_fp32_rotary(model):
    """Keep ESM2 rotary positions in FP32 without changing FP16 activations.

    The shared rotary module returns cos/sin in the activation dtype, while
    apply_rotary_pos_emb already rotates q/k in FP32 before casting back.
    """
    from transformers.models.esm.modeling_esm import EsmRotaryEmbedding

    rotary = [module for module in model.modules() if isinstance(module, EsmRotaryEmbedding)]
    if len(rotary) != 1:
        raise ValueError("Expected one shared ESM rotary module")
    for module in rotary:
        module.inv_freq = module.compute_default_rope_parameters(module.config)[0].to(
            device=module.inv_freq.device, dtype=torch.float32,
        )
        module.register_forward_pre_hook(
            lambda _module, inputs: (inputs[0].float(), *inputs[1:])
        )


def _load_models(device):
    if device.type != "cuda":
        raise ValueError("4-bit ESM2 conversion requires a CUDA GPU")
    head = Tea.from_pretrained(TEA_MODEL, revision=TEA_REVISION)
    if head.representation_size != 1280 or head.codebook_size != 20:
        raise ValueError("Expected the current 20-state ESM2-650M TEA head")
    head = head.to(device).eval()

    staged_cache = Path(__file__).resolve().parents[1] / "artifacts/model_cache"
    cache_dir = (
        str(staged_cache)
        if isinstance(try_to_load_from_cache(
            ESM2_MODEL, "config.json", cache_dir=staged_cache,
            revision=ESM2_REVISION,
        ), str) else None
    )
    tokenizer = AutoTokenizer.from_pretrained(
        ESM2_MODEL, revision=ESM2_REVISION, local_files_only=False,
        do_lower_case=False, use_fast=True, cache_dir=cache_dir,
    )
    encoder = AutoModel.from_pretrained(
        ESM2_MODEL, revision=ESM2_REVISION, local_files_only=False,
        dtype=torch.float16, cache_dir=cache_dir,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="fp4",
            bnb_4bit_use_double_quant=False,
            bnb_4bit_compute_dtype=torch.float16,
        ),
        device_map={"": str(device)}, add_pooling_layer=False,
    ).eval()
    _enable_fp32_rotary(encoder)
    encoder.requires_grad_(False)
    return tokenizer, encoder, head


def _read_fasta(path):
    identifiers, sequences, seen = [], [], set()
    for header, sequence in FastaFile.read_iter(path):
        identifier = header.split()[0]
        if not identifier or identifier in seen:
            raise ValueError("FASTA identifiers must be unique and nonempty")
        if not sequence or not sequence.isascii() or not sequence.isalpha():
            raise ValueError("Expected nonempty ungapped amino-acid sequences")
        seen.add(identifier)
        identifiers.append(identifier)
        sequences.append(sequence.upper().translate(UNKNOWN_RESIDUES))
    if not identifiers:
        raise ValueError("Input FASTA is empty")
    return identifiers, sequences


def _batches(sequences, budget=BATCH_RESIDUES):
    """Group similarly sized chains while bounding padded encoder tokens."""
    order = sorted(range(len(sequences)), key=lambda i: (-len(sequences[i]), i))
    batch, longest = [], 0
    for index in order:
        length = len(sequences[index])
        if batch and longest * (len(batch) + 1) > budget:
            yield batch
            batch, longest = [], 0
        batch.append(index)
        longest = max(longest, length)
    if batch:
        yield batch


@torch.inference_mode()
def _convert_batch(sequences, tokenizer, encoder, head, device, headers):
    tokens = tokenizer(
        [" ".join(sequence) for sequence in sequences],
        return_tensors="pt", padding=True, add_special_tokens=True,
        return_special_tokens_mask=True,
    )
    special = tokens.pop("special_tokens_mask").bool()
    residue_mask = tokens["attention_mask"].bool() & ~special
    if residue_mask.sum(-1).tolist() != list(map(len, sequences)):
        raise ValueError("ESM2 tokenizer did not preserve every residue")
    hidden = encoder(**{name: value.to(device) for name, value in tokens.items()}).last_hidden_state
    # Match the checkpoint's FP16 embedding-cache precision before the FP32 head.
    logits = head(torch.cat([
        value[mask.to(device)] for value, mask in zip(hidden, residue_mask)
    ]).half().float())
    if logits.shape != (sum(map(len, sequences)), 20):
        raise ValueError("TEA head returned an unexpected number of residue logits")
    lengths = list(map(len, sequences))
    states = logits.argmax(-1).cpu().split(lengths)
    entropy = _residue_entropy(logits).cpu().split(lengths) if headers else None
    if headers:
        alphabet, weights = _tea_likelihood_weights()
        if "".join(head.characters) != alphabet:
            raise ValueError("tea_likelihood requires the current public TEA letter order")
    converted = []
    for index, state in enumerate(states):
        choices = state.tolist()
        letters = [head.characters[token] for token in choices]
        mean_entropy = float(entropy[index].mean()) if headers else None
        # A single residue has no adjacent pair; use the neutral score zero.
        tea_likelihood = (sum(weights[a][b] for a, b in zip(choices, choices[1:]))
                            / (len(choices)-1)) if headers and len(choices) > 1 else 0.0
        converted.append(("".join(letters).upper(), mean_entropy, tea_likelihood))
    return converted


def convert(fasta, output, *, confidence_headers=True, _models=None):
    """Write TEA FASTA in source order and return conversion timing."""
    fasta, output = Path(fasta), Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    identifiers, sequences = _read_fasta(fasta)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, encoder, head = _models or _load_models(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    offsets = [None] * len(identifiers)
    with tempfile.TemporaryFile(mode="w+t", dir=output.parent) as spool:
        for indices in _batches(sequences):
            batch = _convert_batch(
                [sequences[index] for index in indices], tokenizer, encoder, head,
                device, confidence_headers,
            )
            for index, (sequence, entropy, tea_likelihood) in zip(indices, batch):
                header = identifiers[index]
                if confidence_headers:
                    header += f"|TH={entropy:.3f}|TL={tea_likelihood:.3f}"
                offsets[index] = spool.tell()
                spool.write(f">{header}\n{sequence}\n")
        descriptor, name = tempfile.mkstemp(
            prefix=output.name + ".", suffix=".partial", dir=output.parent,
        )
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "w") as handle:
                for offset in offsets:
                    if offset is None:
                        raise ValueError("Conversion did not cover every input sequence")
                    spool.seek(offset)
                    handle.write(spool.readline())
                    handle.write(spool.readline())
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return {"chains": len(identifiers), "residues": sum(map(len, sequences)),
            "seconds_excluding_model_load": time.perf_counter() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", "-f", type=Path, required=True)
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument("--confidence-headers", action=argparse.BooleanOptionalAction, default=True,
                        help="Append mean entropy (TH) and tea_likelihood (TL); enabled by default")
    args = parser.parse_args()
    result = convert(
        args.fasta, args.output,
        confidence_headers=args.confidence_headers,
    )
    print(
        f"Converted {result['chains']} sequences ({result['residues']} residues) "
        f"in {result['seconds_excluding_model_load']:.1f}s, excluding model load."
    )


if __name__ == "__main__":
    main()
