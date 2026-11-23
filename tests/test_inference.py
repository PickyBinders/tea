"""Residue-token equivalence, optional compiler caches and a real GPU control."""

import gc
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from tea._inference import CompilerCache, ResidueTokenizer
from tea.convert import Converter
from transformers import EsmTokenizer


class InferenceTests(unittest.TestCase):
    def test_tokens_match_esm_with_padding_unknowns_and_long_chains(self):
        vocabulary = "<cls> <pad> <eos> <unk> L A G V S E R T I D P K Q N F Y M H W C X B U Z O . - <null_1> <mask>"
        with TemporaryDirectory() as folder:
            vocab = Path(folder) / "vocab.txt"
            vocab.write_text("\n".join(vocabulary.split()) + "\n")
            native = EsmTokenizer(vocab_file=str(vocab))
            lookup = ResidueTokenizer(native)
            for sequences in (["M", "ACDX", "GGGGGGG"],
                              ["ABCDEFGHIJKLMNOPQRSTUVWXYZ", "X", ""],
                              ["ACDEFGHIKLMNPQRSTVWY" * 600, "M"]):
                spaced = [" ".join(sequence) for sequence in sequences]
                expected = native(spaced, padding=True, return_tensors="pt",
                                  add_special_tokens=True, return_special_tokens_mask=True)
                observed = lookup(spaced)
                self.assertEqual(set(expected), set(observed))
                for name in expected:
                    self.assertEqual(observed[name].dtype, expected[name].dtype)
                    self.assertTrue(torch.equal(observed[name], expected[name]), name)

    def test_compiler_cache_roundtrip_and_unavailable_directory(self):
        with TemporaryDirectory() as folder, \
             patch("torch.cuda.get_device_name", return_value="test GPU"):
            cache = CompilerCache(folder)
            self.assertFalse(cache.load())
            with patch("torch.compiler.save_cache_artifacts", return_value=(b"artifacts", None)):
                cache.save()
            self.assertEqual(cache.path.read_bytes(), b"artifacts")
            restored = CompilerCache(folder)
            with patch("torch.compiler.load_cache_artifacts", return_value=object()) as load:
                self.assertTrue(restored.load())
            load.assert_called_once_with(b"artifacts")
            unavailable = CompilerCache(cache.path)
            with patch("torch.compiler.save_cache_artifacts", return_value=(b"artifacts", None)):
                unavailable.save()
            self.assertFalse(unavailable.loaded)

    @unittest.skipUnless(os.environ.get("TEA_TEST_REAL_MODEL") == "1", "Opt-in compiled GPU control")
    def test_compiled_conversion_against_eager_and_repeated_files(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            sequences = ["M", "ACDEFGHIKLMNPQRSTVWY", "MKKLAVVAAA",
                         "ACDEFGHIKLMNPQRSTVWY" * 110]
            source = root / "aa.fa"
            source.write_text("".join(f">{i}\n{sequence}\n" for i, sequence in enumerate(sequences)))
            eager = Converter(compile=False)
            eager.convert(source, root / "eager.fa")
            self.assertEqual(eager._models[1].rotary_embeddings.inv_freq.dtype, torch.float32)
            del eager
            gc.collect()
            torch.cuda.empty_cache()

            compiled = Converter(compiler_cache=root / "compiler")
            compiled.convert(source, root / "compiled.fa")
            compiled.convert(source, root / "repeated.fa")
            self.assertEqual((root / "compiled.fa").read_bytes(), (root / "repeated.fa").read_bytes())
            reference = (root / "eager.fa").read_text().splitlines()
            observed = (root / "compiled.fa").read_text().splitlines()
            matches = sum(a == b for left, right in zip(reference[1::2], observed[1::2])
                          for a, b in zip(left, right))
            self.assertGreaterEqual(matches / sum(map(len, sequences)), .999)
            for index, sequence in enumerate(sequences):
                self.assertTrue(observed[2 * index].startswith(f">{index}|TH="))
                self.assertIn("|TL=", observed[2 * index])
                self.assertEqual(len(observed[2 * index + 1]), len(sequence))
            self.assertTrue(compiled._cache.path.is_file())
            self.assertTrue(compiled._cache.loaded)


if __name__ == "__main__":
    unittest.main()
