"""Full-chain residue embeddings and cache format for TEA training."""

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from types import MethodType

import torch

from tea_train.io import read_json, sha256_file, stable_hash


@dataclass(frozen=True)
class EncoderSpec:
    model_id: str
    revision: str
    backend: str
    dimension: int
    precision: str = "float16"
    window: int = 1024
    overlap: int = 128
    structure_pretrained: bool = False
    quantization_type: str = "fp4"
    quantization_double: bool = False
    loading_dtype: str | None = None
    compute_dtype: str | None = None
    rotary_dtype: str | None = None
    t5_quantize_output_projection: bool | None = None
    t5_float32_matmul_precision: str | None = None

    def __post_init__(self):
        for value in (self.loading_dtype, self.compute_dtype):
            if value is not None and value not in ("float16", "float32", "bfloat16"):
                raise ValueError(f"Unsupported encoder dtype: {value}")
        if self.precision != "4bit" and (self.loading_dtype or self.compute_dtype):
            raise ValueError("Separate loading/compute dtypes apply to 4-bit encoders only")
        if self.rotary_dtype is not None and (self.rotary_dtype != "float32" or
                                              self.backend != "esm2" or self.precision != "4bit"):
            raise ValueError("FP32 rotary override is supported only for 4-bit ESM2")
        if self.t5_quantize_output_projection not in (None, True):
            raise ValueError("T5 output-projection quantization must be true or omitted")
        if self.t5_quantize_output_projection and (
                self.precision != "4bit" or self.backend not in ("t5", "prostt5")):
            raise ValueError("Output-projection quantization requires a 4-bit T5 encoder")
        if self.t5_float32_matmul_precision not in (None, "highest", "high"):
            raise ValueError("T5 FP32 matmul precision must be 'highest', 'high', or omitted")
        if self.t5_float32_matmul_precision and (
                self.precision != "4bit" or self.backend not in ("t5", "prostt5")
                or self.t5_quantize_output_projection):
            raise ValueError("T5 FP32 matmul policy requires retained projections in a 4-bit T5")

    def to_dict(self):
        # Missing dtype fields mean the original FP16-load/FP16-compute policy.
        # Preserve old cache/checkpoint identities; explicit policies get new IDs.
        return {key: value for key, value in asdict(self).items() if value is not None}

    @property
    def identifier(self):
        return stable_hash(self.to_dict())


def tile_spans(length, window, overlap):
    if length < 1 or window < 1 or not 0 <= overlap < window:
        raise ValueError("Positive length/window and 0 <= overlap < window required")
    if length <= window:
        return [(0, length)]
    starts = list(range(0, length - window + 1, window - overlap))
    if starts[-1] != length - window:
        starts.append(length - window)
    return [(start, start + window) for start in starts]


def normalize_sequence(sequence):
    if not sequence or not sequence.isascii() or not sequence.isalpha():
        raise ValueError("Ungapped nonempty amino-acid sequence required")
    return sequence.upper().translate(str.maketrans({c: "X" for c in "UZOBJ"}))


def stitch(sequence, encode_chunk, window, overlap, spans=None):
    """Average overlaps in FP32; require every residue exactly in output order."""
    sequence = normalize_sequence(sequence)
    accumulator = None
    coverage = torch.zeros(len(sequence), dtype=torch.int32)
    spans = tile_spans(len(sequence), window, overlap) if spans is None else spans
    for start, stop in spans:
        chunk = encode_chunk(sequence[start:stop]).detach().float().cpu()
        if chunk.ndim != 2 or chunk.shape[0] != stop - start or not torch.isfinite(chunk).all():
            raise ValueError("Encoder returned invalid residue length or values")
        if accumulator is None:
            accumulator = torch.zeros((len(sequence), chunk.shape[1]), dtype=torch.float32)
        if chunk.shape[1] != accumulator.shape[1]:
            raise ValueError("Encoder dimension changed between chunks")
        accumulator[start:stop] += chunk
        coverage[start:stop] += 1
    if (coverage == 0).any():
        raise ValueError("Missing residues after encoder tiling")
    return accumulator / coverage[:, None]


def enable_fp32_rotary(model):
    """Compute the shared ESM2 rotary embedding in FP32."""
    from transformers.models.esm.modeling_esm import EsmRotaryEmbedding

    rotary, = (module for module in model.modules() if isinstance(module, EsmRotaryEmbedding))
    rotary.inv_freq = rotary.compute_default_rope_parameters(rotary.config)[0].to(
        device=rotary.inv_freq.device, dtype=torch.float32)
    rotary.register_forward_pre_hook(lambda module, inputs: (inputs[0].float(), *inputs[1:]))


def _t5_dense_forward_4bit(module, hidden_states):
    """T5 dense forward that recognizes bitsandbytes' uint8-packed weights."""
    hidden_states = module.wi(hidden_states)
    hidden_states = module.act(hidden_states)
    hidden_states = module.dropout(hidden_states)
    if (isinstance(module.wo.weight, torch.Tensor)
            and hidden_states.dtype != module.wo.weight.dtype
            and module.wo.weight.dtype not in (torch.int8, torch.uint8)):
        hidden_states = hidden_states.to(module.wo.weight.dtype)
    return module.wo(hidden_states)


def _t5_gated_forward_4bit(module, hidden_states):
    """T5 gated dense forward that recognizes uint8-packed 4-bit weights."""
    hidden_gelu = module.act(module.wi_0(hidden_states))
    hidden_linear = module.wi_1(hidden_states)
    hidden_states = module.dropout(hidden_gelu * hidden_linear)
    if (isinstance(module.wo.weight, torch.Tensor)
            and hidden_states.dtype != module.wo.weight.dtype
            and module.wo.weight.dtype not in (torch.int8, torch.uint8)):
        hidden_states = hidden_states.to(module.wo.weight.dtype)
    return module.wo(hidden_states)


class FrozenEncoder:
    def __init__(self, spec, device="cuda", local_files_only=True):
        from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig, T5EncoderModel
        from huggingface_hub import try_to_load_from_cache
        if spec.backend not in ("esm2", "t5", "prostt5"):
            raise ValueError(f"Unsupported encoder backend: {spec.backend}")
        if not spec.revision or spec.revision == "main":
            raise ValueError("Pin a model revision before generating scientific embeddings")
        self.spec, self.device = spec, torch.device(device)
        if spec.t5_float32_matmul_precision:
            torch.set_float32_matmul_precision(spec.t5_float32_matmul_precision)
        dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16,
                 "4bit": torch.float16}.get(spec.precision)
        if dtype is None:
            raise ValueError(f"Unsupported precision: {spec.precision}")
        if spec.precision == "4bit":
            dtype = getattr(torch, spec.loading_dtype or "float16")
        kwargs = {"revision": spec.revision, "local_files_only": local_files_only, "dtype": dtype}
        # Prefer a pinned snapshot staged in TEA, while retaining the
        # standard Hugging Face cache for already-staged backbones there.
        staged_cache = Path(__file__).resolve().parents[1] / "artifacts/model_cache"
        if isinstance(try_to_load_from_cache(spec.model_id, "config.json",
                      cache_dir=staged_cache, revision=spec.revision), str):
            kwargs["cache_dir"] = str(staged_cache)
        if spec.precision == "4bit":
            if self.device.type != "cuda":
                raise ValueError("4-bit encoder requires CUDA")
            kwargs.update(quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_quant_type=spec.quantization_type, bnb_4bit_use_double_quant=spec.quantization_double,
                bnb_4bit_compute_dtype=getattr(torch, spec.compute_dtype or "float16")),
                device_map={"": str(self.device)})
        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id, revision=spec.revision,
            local_files_only=local_files_only, do_lower_case=False,
            cache_dir=kwargs.get("cache_dir"),
            use_fast=spec.backend not in ("t5", "prostt5"))
        loader = T5EncoderModel if spec.backend in ("t5", "prostt5") else AutoModel
        if spec.backend == "esm2":
            # ESM masked-LM checkpoints have no pooler. Creating the unused
            # random pooler also breaks missing-key initialization in 4-bit.
            kwargs["add_pooling_layer"] = False
        restore_keep = None
        if spec.t5_quantize_output_projection:
            # Transformers keeps every T5 ``wo`` in FP32 for an 8-bit
            # compatibility workaround.  That also excludes one third of
            # ProstT5 from 4-bit quantization.  Temporarily disable the
            # class-level exclusion while the instance is constructed.
            inherited = "_keep_in_fp32_modules" not in loader.__dict__
            restore_keep = (inherited, loader._keep_in_fp32_modules)
            loader._keep_in_fp32_modules = None
        try:
            self.model = loader.from_pretrained(spec.model_id, **kwargs)
        finally:
            if restore_keep is not None:
                inherited, value = restore_keep
                if inherited:
                    del loader._keep_in_fp32_modules
                else:
                    loader._keep_in_fp32_modules = value
        if spec.t5_quantize_output_projection:
            from transformers.models.t5.modeling_t5 import (
                T5DenseActDense, T5DenseGatedActDense,
            )
            projections = 0
            for module in self.model.modules():
                if isinstance(module, T5DenseActDense):
                    module.forward = MethodType(_t5_dense_forward_4bit, module)
                elif isinstance(module, T5DenseGatedActDense):
                    module.forward = MethodType(_t5_gated_forward_4bit, module)
                else:
                    continue
                if type(module.wo).__name__ != "Linear4bit":
                    raise ValueError("T5 output projection was not quantized to 4-bit")
                projections += 1
            if projections != self.model.config.num_layers:
                raise ValueError("Expected one quantized output projection per T5 layer")
        if spec.precision != "4bit":
            self.model.to(self.device)
        self.model.eval()
        if spec.rotary_dtype == "float32":
            enable_fp32_rotary(self.model)
        self.model.requires_grad_(False)

    @property
    def uses_unbounded_rotary_context(self):
        """Whether this encoder can run a complete chain without windowing.

        ESM2's rotary embedding implementation grows its sine/cosine tables to
        the observed sequence length.  ``max_position_embeddings`` is not an
        absolute-position lookup limit for this model and must not force the
        historical 1,024-residue tiling policy.
        """
        config = getattr(getattr(self, "model", None), "config", None)
        return (
            self.spec.backend == "esm2"
            and getattr(config, "position_embedding_type", None) == "rotary"
        )

    @property
    def context_policy(self):
        if self.uses_unbounded_rotary_context:
            return "single-pass-rotary-full-chain"
        return (
            f"overlap-average-window={self.spec.window};"
            f"overlap={self.spec.overlap}"
        )

    def context_spans(self, length):
        if length < 1:
            raise ValueError("Positive sequence length required")
        if self.uses_unbounded_rotary_context:
            return [(0, length)]
        return tile_spans(length, self.spec.window, self.spec.overlap)

    @torch.inference_mode()
    def encode_chunks(self, sequences):
        if not sequences:
            return []
        text = [" ".join(sequence) for sequence in sequences]
        if self.spec.backend == "prostt5":
            text = ["<AA2fold> " + value for value in text]
        tokens = self.tokenizer(text, return_tensors="pt", add_special_tokens=True,
                                padding=True, return_special_tokens_mask=True)
        special = tokens.pop("special_tokens_mask").bool()
        if self.spec.backend == "prostt5":
            # The task prefix is not a residue, whether or not this tokenizer
            # classifies it as a special token.
            prefix = self.tokenizer.convert_tokens_to_ids("<AA2fold>")
            special |= tokens["input_ids"] == prefix
        residue_mask = tokens["attention_mask"].bool() & ~special
        if residue_mask.sum(-1).tolist() != [len(sequence) for sequence in sequences]:
            raise ValueError("Tokenizer does not yield exactly one token per residue")
        maximum = getattr(self.model.config, "max_position_embeddings", None)
        if (maximum is not None and not self.uses_unbounded_rotary_context
                and tokens["input_ids"].shape[1] > maximum):
            raise ValueError("Encoder tile exceeds the model's positional limit")
        tokens = {key: value.to(self.device) for key, value in tokens.items()}
        output = self.model(**tokens)
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is None:
            hidden = getattr(output, "embeddings", None)
        if hidden is None:
            raise ValueError("Encoder API did not return final residue representations")
        results = [value[mask.to(hidden.device)] for value, mask in zip(hidden, residue_mask)]
        if len(results) != len(sequences) or any(value.shape != (len(sequence), self.spec.dimension)
                                               for value, sequence in zip(results, sequences)):
            raise ValueError("Unexpected encoder batch or representation dimension")
        return results

    def encode_chunk(self, sequence):
        return self.encode_chunks([sequence])[0]

    def encode(self, sequence):
        sequence = normalize_sequence(sequence)
        return stitch(
            sequence, self.encode_chunk, self.spec.window, self.spec.overlap,
            spans=self.context_spans(len(sequence)),
        )

    def encode_batch(self, sequences, residue_budget=4096, output_device="cpu"):
        """Batch library forwards; preserve the same per-sequence tiles/averaging.

        The budget bounds padded residue positions in each forward, excluding
        tokenizer special tokens. It is not a GPU-memory guarantee.
        """
        sequences = [normalize_sequence(sequence) for sequence in sequences]
        tiles = [(i, start, stop) for i, sequence in enumerate(sequences)
                 for start, stop in self.context_spans(len(sequence))]
        if residue_budget <= 0:
            raise ValueError("Positive batch budget required")
        accumulators = [torch.zeros((len(sequence), self.spec.dimension), dtype=torch.float32, device=output_device) for sequence in sequences]
        coverage = [torch.zeros(len(sequence), dtype=torch.int32, device=output_device) for sequence in sequences]
        # Group similar lengths to limit tokenizer padding; restore original
        # sequence/residue positions through the tile indices below.
        ordered = sorted(tiles, key=lambda tile: (-(tile[2] - tile[1]), tile[0], tile[1]))
        cursor = 0
        while cursor < len(ordered):
            longest = ordered[cursor][2] - ordered[cursor][1]
            batch = ordered[cursor:cursor + max(1, residue_budget // longest)]
            chunks = [sequences[i][start:stop] for i, start, stop in batch]
            results = self.encode_chunks(chunks)
            if len(results) != len(batch):
                raise ValueError("Encoder dropped a batch item")
            for (i, start, stop), value in zip(batch, results):
                value = value.detach().to(device=output_device, dtype=torch.float32)
                if value.shape != (stop - start, self.spec.dimension):
                    raise ValueError("Invalid batched residue embeddings")
                accumulators[i][start:stop] += value
                coverage[i][start:stop] += 1
            cursor += len(batch)
        if torch.stack([(counts == 0).any() for counts in coverage]).any():
            raise ValueError("Missing residues after batched encoder tiling")
        if not torch.stack([torch.isfinite(value).all() for value in accumulators]).all():
            raise ValueError("Nonfinite batched residue embeddings")
        return [value / counts[:, None] for value, counts in zip(accumulators, coverage)]


def batch_policy(residue_budget):
    if residue_budget < 0:
        raise ValueError("Batch budget must be nonnegative")
    return f"padded_residue_budget={residue_budget}" if residue_budget else "one-chain-at-a-time"


def iter_embeddings(encoder, sequences, residue_budget=0):
    """Bound the number of full-sequence accumulators retained during conversion."""
    for batch in iter_embedding_batches(encoder, sequences, residue_budget):
        yield from batch


def iter_embedding_batches(encoder, sequences, residue_budget=0, output_device="cpu"):
    """Keep bounded groups together so inference can batch its per-residue head."""
    batch_policy(residue_budget)
    if not residue_budget:
        for key, sequence in sequences.items():
            yield [(key, encoder.encode(sequence).to(output_device))]
        return
    keys = sorted(sequences, key=lambda key: (-len(sequences[key]), key))
    pending, residues = [], 0
    for key in keys:
        if pending and residues + len(sequences[key]) > residue_budget:
            values = encoder.encode_batch([sequences[item] for item in pending], residue_budget, output_device)
            yield list(zip(pending, values))
            pending, residues = [], 0
        pending.append(key)
        residues += len(sequences[key])
    if pending:
        values = encoder.encode_batch([sequences[item] for item in pending], residue_budget, output_device)
        yield list(zip(pending, values))


class EmbeddingCache:
    """Small LRU of tensor-only shards, with strict chain/length validation."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = read_json(self.directory / "manifest.json")
        self._load = lru_cache(maxsize=2)(self._read_shard)

    def _read_shard(self, name):
        if sha256_file(self.directory / name) != self.manifest["shards"][name]:
            raise ValueError(f"Embedding shard checksum mismatch: {name}")
        return torch.load(self.directory / name, map_location="cpu", weights_only=True)

    def __getitem__(self, chain):
        record = self.manifest["chains"][chain]
        stored = self._load(record["shard"])[chain]
        if self.manifest.get("storage_dtype") == "int8":
            if (not isinstance(stored, dict) or set(stored) != {"code", "scale"}
                    or stored["code"].dtype != torch.int8
                    or stored["scale"].dtype != torch.float32
                    or stored["scale"].shape != (record["length"], 1)
                    or not torch.isfinite(stored["scale"]).all()
                    or not (stored["scale"] > 0).all()):
                raise ValueError(f"Invalid quantized cached embedding: {chain}")
            tensor = stored["code"].float() * stored["scale"]
        else:
            tensor = stored
        if tensor.ndim != 2 or tensor.shape != (record["length"], self.manifest["spec"]["dimension"]):
            raise ValueError(f"Invalid cached chain embedding: {chain}")
        return tensor
