"""Small, model-free checks for the public converter."""

import importlib
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import tea.convert as public_convert
from tea.convert import _batches, convert


class FakeTokenizer:
    def __call__(self, sequences, **_):
        lengths = [len(sequence.split()) for sequence in sequences]
        width = max(lengths) + 2
        identifiers = torch.zeros((len(sequences), width), dtype=torch.long)
        attention = torch.zeros_like(identifiers)
        special = torch.ones_like(identifiers)
        for row, (sequence, length) in enumerate(zip(sequences, lengths)):
            identifiers[row, 1:length + 1] = torch.tensor([
                1 if letter == "A" else 2 for letter in sequence.split()
            ])
            attention[row, :length + 2] = 1
            special[row, 1:length + 1] = 0
        return {"input_ids": identifiers, "attention_mask": attention,
                "special_tokens_mask": special}


class FakeEncoder:
    def eval(self):
        return self

    def requires_grad_(self, _):
        return self

    def __call__(self, input_ids, **_):
        values = torch.nn.functional.one_hot(
            (input_ids - 1).clamp_min(0), num_classes=20,
        ).float()
        values *= torch.where(input_ids == 1, 1.0, 20.0).unsqueeze(-1)
        return SimpleNamespace(last_hidden_state=values)


class FakeHead:
    characters = list("ACDEFGHIKLMNPQRSTVWY")
    representation_size = 1280
    codebook_size = 20

    def to(self, _):
        return self

    def eval(self):
        return self

    def __call__(self, embeddings):
        return embeddings


class PublicConvertTests(unittest.TestCase):
    def test_import_does_not_need_benchmark_modules(self):
        with patch.dict(sys.modules, {"tea.io": None, "tea.encoding": None}):
            importlib.reload(public_convert)

    def test_default_model_uses_dev_revision(self):
        with patch.object(public_convert.Tea, "from_pretrained",
                          return_value=FakeHead()) as load_head, \
             patch.object(public_convert.AutoTokenizer, "from_pretrained",
                          return_value=FakeTokenizer()), \
             patch.object(public_convert.AutoModel, "from_pretrained",
                          return_value=FakeEncoder()) as load_encoder, \
             patch.object(public_convert, "BitsAndBytesConfig", return_value=object()), \
             patch.object(public_convert, "_enable_fp32_rotary") as rotary:
            public_convert._load_models(torch.device("cuda"))
        load_head.assert_called_once_with("PickyBinders/tea", revision="dev")
        self.assertEqual(load_encoder.call_args.kwargs["dtype"], torch.float16)
        self.assertNotIn("torch_dtype", load_encoder.call_args.kwargs)
        rotary.assert_called_once()

    def test_shared_rotary_preserves_fp32_positions_and_fp16_attention(self):
        from transformers import EsmConfig
        from transformers.models.esm.modeling_esm import (
            EsmRotaryEmbedding, apply_rotary_pos_emb,
        )

        config = EsmConfig(hidden_size=320, num_attention_heads=5,
                           position_embedding_type="rotary")
        rotary = EsmRotaryEmbedding(config).half()
        model = torch.nn.Sequential(rotary)
        public_convert._enable_fp32_rotary(model)
        expected_freq, _ = rotary.compute_default_rope_parameters(config)
        torch.testing.assert_close(rotary.inv_freq, expected_freq, rtol=0, atol=0)
        self.assertEqual(rotary.inv_freq.dtype, torch.float32)

        positions = torch.tensor([[0, 511, 1023, 2049, 4097]])
        hidden = torch.ones(1, positions.shape[1], config.hidden_size, dtype=torch.float16)
        cos, sin = rotary(hidden, positions)
        angles = positions[..., None].float() * expected_freq
        angles = torch.cat((angles, angles), dim=-1)
        self.assertEqual(cos.dtype, torch.float32)
        self.assertEqual(sin.dtype, torch.float32)
        torch.testing.assert_close(cos, angles.cos(), rtol=0, atol=0)
        torch.testing.assert_close(sin, angles.sin(), rtol=0, atol=0)

        query = torch.ones(1, config.num_attention_heads, positions.shape[1], 64,
                           dtype=torch.float16)
        rotated_query, rotated_key = apply_rotary_pos_emb(query, query, cos, sin)
        self.assertEqual(rotated_query.dtype, torch.float16)
        self.assertEqual(rotated_key.dtype, torch.float16)

    def test_rotary_rejects_missing_shared_module(self):
        with self.assertRaisesRegex(ValueError, "one shared ESM rotary module"):
            public_convert._enable_fp32_rotary(torch.nn.Identity())

    def test_length_batches_bound_padding(self):
        sequences = ["A", "C" * 5, "A" * 9, "C" * 4, "A" * 2]
        batches = list(_batches(sequences, budget=12))
        self.assertEqual(sorted(index for batch in batches for index in batch),
                         list(range(len(sequences))))
        self.assertTrue(all(
            len(batch) == 1 or max(len(sequences[i]) for i in batch) * len(batch) <= 12
            for batch in batches
        ))
        self.assertGreater(len(batches), 1)

    def test_confidence_and_order(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "input.fasta"
            output = Path(directory) / "output.fasta"
            source.write_text(">short description\nA\n>long\nCCC\n")
            result = convert(
                source, output, lowercase_logit_spread_below=3.5,
                confidence_headers=True,
                _models=(FakeTokenizer(), FakeEncoder(), FakeHead()),
            )
            lines = output.read_text().splitlines()
            self.assertEqual(lines[1], "a")
            self.assertEqual(lines[3], "CCC")
            self.assertTrue(lines[0].startswith(">short|TLS=0.22|TCP="))
            self.assertTrue(lines[2].startswith(">long|TLS=4.36|TCP="))
            self.assertEqual(result["chains"], 2)
            self.assertEqual(result["residues"], 4)
            self.assertGreaterEqual(result["seconds_excluding_model_load"], 0)

    def test_duplicate_identifiers_fail_without_output(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "input.fasta"
            output = Path(directory) / "output.fasta"
            source.write_text(">same first\nA\n>same second\nC\n")
            with self.assertRaisesRegex(ValueError, "unique"):
                convert(source, output, _models=(
                    FakeTokenizer(), FakeEncoder(), FakeHead(),
                ))
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
