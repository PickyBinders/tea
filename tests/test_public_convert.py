"""Small, model-free checks for the public converter."""

import importlib
import csv
import json
import math
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import tea.convert as public_convert
from tea.convert import _batches, _convert_batch, _residue_entropy, _tea_likelihood_weights, convert


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
                source, output,
                _models=(FakeTokenizer(), FakeEncoder(), FakeHead()),
            )
            lines = output.read_text().splitlines()
            self.assertEqual(lines[1], "A")
            self.assertEqual(lines[3], "CCC")
            self.assertTrue(lines[0].startswith(">short|TH="))
            self.assertTrue(lines[0].endswith("|TL=0.000"))
            self.assertTrue(lines[2].startswith(">long|TH=0.000|TL="))
            _, weights = _tea_likelihood_weights()
            self.assertTrue(lines[2].endswith(f"|TL={weights[1][1]:.3f}"))
            self.assertEqual(result["chains"], 2)
            self.assertEqual(result["residues"], 4)
            self.assertGreaterEqual(result["seconds_excluding_model_load"], 0)

    def test_entropy_is_normalized_native_shannon(self):
        uniform = torch.zeros(3, 20)
        torch.testing.assert_close(_residue_entropy(uniform), torch.ones(3))
        logits = torch.tensor([[1.0] + [0.0]*19, [20.0] + [0.0]*19])
        probabilities = logits.double().softmax(-1)
        expected = -(probabilities * probabilities.log()).sum(-1) / math.log(20)
        torch.testing.assert_close(_residue_entropy(logits).double(), expected, atol=2e-7, rtol=0)

    def test_high_entropy_residues_stay_uppercase(self):
        models = FakeTokenizer(), FakeEncoder(), FakeHead()
        result = _convert_batch(["AC"], *models, torch.device("cpu"), True)
        self.assertEqual(result[0][0], "AC")
        self.assertGreater(result[0][1], .25)

    def test_confidence_headers_can_be_disabled(self):
        with TemporaryDirectory() as folder:
            source, output = Path(folder)/"aa.fa", Path(folder)/"tea.fa"
            source.write_text(">sp|P12345|NAME\nAC\n")
            convert(source, output, confidence_headers=False,
                    _models=(FakeTokenizer(), FakeEncoder(), FakeHead()))
            self.assertEqual(output.read_text(), ">sp|P12345|NAME\nAC\n")

    def test_removed_lowercase_option_is_rejected(self):
        with self.assertRaisesRegex(TypeError, 'lowercase_entropy_above'):
            convert("unused.fa", "unused-output.fa", lowercase_entropy_above=.25)
        for option in ('--lowercase-entropy-above', '--lowercase-logit-spread-below'):
            with self.subTest(option=option), \
                 patch.object(sys, 'argv', ['tea_convert', '--fasta', 'aa.fa',
                                          '--output', 'tea.fa', option, '.25']), \
                 patch.object(sys, 'stderr'), \
                 patch.object(public_convert, 'convert') as run:
                with self.assertRaises(SystemExit) as error:
                    public_convert.main()
                self.assertEqual(error.exception.code, 2)
                run.assert_not_called()

    def test_bundled_tea_likelihood_name(self):
        from importlib.resources import files
        payload = json.loads(files('tea').joinpath('data/tea_likelihood.json').read_text())
        self.assertEqual(payload['name'], 'tea_likelihood')
        self.assertIn('C is matcha.out', payload['definition'])
        self.assertFalse(files('tea').joinpath('data/structural_confidence.json').is_file())

    def test_packaged_likelihood_scores_match_frozen_simple_matrix(self):
        root = Path(__file__).resolve().parents[1]
        source = root/"revision/data/plot_tables/confidence/simple_two_signals/additional_fit_free"
        if not source.is_dir():
            self.skipTest("Revision source tables are not part of the public package")
        with (source/"substitution_smoothed_markov_initial.tsv").open() as handle:
            pi = {row["letter"]:float(row["initial_probability"]) for row in csv.DictReader(handle,delimiter="\t")}
        with (source/"substitution_smoothed_markov_transition.tsv").open() as handle:
            transition = {row["from_letter"]:row for row in csv.DictReader(handle,delimiter="\t")}
        alphabet, scores = _tea_likelihood_weights()
        for i,a in enumerate(alphabet):
            for j,b in enumerate(alphabet):
                self.assertAlmostEqual(scores[i][j], math.log2(float(transition[a][b])/pi[b]), places=12)

    def test_cli_defaults_and_new_flags(self):
        with patch.object(sys, "argv", ["tea_convert", "--fasta", "aa.fa", "--output", "tea.fa"]), \
             patch.object(public_convert, "convert", return_value={"chains":1,"residues":1,"seconds_excluding_model_load":0}) as run:
            public_convert.main()
        self.assertNotIn('lowercase_entropy_above', run.call_args.kwargs)
        self.assertTrue(run.call_args.kwargs["confidence_headers"])

    def test_tea_likelihood_uses_adjacent_uppercase_pairs(self):
        models = FakeTokenizer(), FakeEncoder(), FakeHead()
        results = _convert_batch(["AC", "CA", "AAA"], *models, torch.device("cpu"), True)
        _, weights = _tea_likelihood_weights()
        for result, expected in zip(results, [weights[0][1],weights[1][0],weights[0][0]]):
            self.assertAlmostEqual(result[2], expected, places=12)
            self.assertTrue(result[0].isupper())

    @unittest.skipUnless(os.environ.get("TEA_TEST_REAL_MODEL") == "1", "Opt-in real 4-bit GPU smoke")
    def test_real_4bit_entropy_likelihood_headers_and_uppercase(self):
        from bitsandbytes.nn import Linear4bit
        self.assertTrue(torch.cuda.is_available())
        tokenizer, encoder, head = public_convert._load_models(torch.device("cuda"))
        quantized = [module for module in encoder.modules() if isinstance(module, Linear4bit)]
        self.assertTrue(quantized)
        self.assertTrue(all(module.weight.quant_state is not None for module in quantized))
        self.assertTrue(all(module.compute_dtype == torch.float16 for module in quantized))
        identifiers = ["short", "second", "one", "long_rotary_smoke"]
        # Artificial repeated chain exercises >2048 rotary positions, not model quality.
        sequences = ["ACDEFGHIKLMNPQRSTVWY", "MKKLAVVAAA", "A", "ACDEFGHIKLMNPQRSTVWY"*110]
        captured = []
        forward = head.forward
        def record(values):
            result = forward(values)
            captured.append(result.detach().cpu())
            return result
        with TemporaryDirectory() as folder:
            source, output = Path(folder)/"aa.fa", Path(folder)/"tea.fa"
            source.write_text("".join(f">{name}\n{sequence}\n" for name,sequence in zip(identifiers,sequences)))
            with patch.object(head, "forward", side_effect=record):
                convert(source, output, _models=(tokenizer, encoder, head))
            records = output.read_text().splitlines()
            expected = {}
            _, weights = _tea_likelihood_weights()
            for batch, logits in zip(_batches(sequences), captured, strict=True):
                offset = 0
                for index in batch:
                    values = logits[offset:offset+len(sequences[index])]
                    offset += len(values)
                    entropy = _residue_entropy(values)
                    probabilities = values.double().softmax(-1)
                    independent = -(probabilities*probabilities.log()).sum(-1)/math.log(20)
                    torch.testing.assert_close(entropy.double(), independent, atol=3e-7, rtol=0)
                    choices = values.argmax(-1).tolist()
                    sequence = "".join(head.characters[c] for c in choices).upper()
                    score = sum(weights[a][b] for a,b in zip(choices,choices[1:]))/(len(choices)-1) if len(choices)>1 else 0
                    expected[identifiers[index]] = (sequence,float(entropy.mean()),score)
            for index,name in enumerate(identifiers):
                sequence, entropy, score = expected[name]
                self.assertEqual(records[2*index], f">{name}|TH={entropy:.3f}|TL={score:.3f}")
                self.assertEqual(records[2*index+1], sequence)
            print(json.dumps(dict(status="passed",gpu=torch.cuda.get_device_name(),
                quantized_layers=len(quantized),chains=len(sequences),residues=sum(map(len,sequences)),
                longest_chain=max(map(len,sequences)),lowercase_residues=sum(c.islower() for row in expected.values() for c in row[0]),
                checks="TH versus native entropy; TL versus frozen adjacent-pair scores; always uppercase; input order; 4-bit FP4/FP16; >2048 rotary positions")),flush=True)

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
