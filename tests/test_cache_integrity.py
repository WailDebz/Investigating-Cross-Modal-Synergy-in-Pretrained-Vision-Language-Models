import tempfile
import unittest

import torch

from synergy_bench.evaluation.cache import CacheMetadata, EmbeddingCache
from synergy_bench.evaluation.extraction import extract_to_cache
from synergy_bench.models import adapt_batch
from synergy_bench.models.foundation import FrozenCLIP
from model_fakes import FakeCLIP, FakeProcessor


def metadata(**updates):
    values = {
        "sample_manifest_hash": "manifest",
        "checkpoint": "clip",
        "revision": "commit",
        "preprocessing_hash": "processor",
        "code_hash": "code",
        "config_hash": "config",
        "dtype": "float32",
        "modality": "image",
    }
    values.update(updates)
    return CacheMetadata(**values)


class CacheIntegrityTests(unittest.TestCase):
    def test_realigns_rows_by_sample_id(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = EmbeddingCache(directory)
            cache.save(metadata(), ("b", "a"), torch.tensor([[2.0], [1.0]]))

            loaded = cache.load(metadata(), ("a", "b"))

            torch.testing.assert_close(loaded, torch.tensor([[1.0], [2.0]]))

    def test_metadata_change_is_a_cache_miss(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = EmbeddingCache(directory)
            cache.save(metadata(), ("a",), torch.tensor([[1.0]]))
            with self.assertRaises(FileNotFoundError):
                cache.load(metadata(revision="different"))

    def test_get_or_compute_reuses_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = EmbeddingCache(directory)
            calls = []

            first, first_hit = cache.get_or_compute(
                metadata(), ("a",), lambda: calls.append(1) or torch.tensor([[3.0]])
            )
            second, second_hit = cache.get_or_compute(
                metadata(), ("a",), lambda: calls.append(2) or torch.tensor([[9.0]])
            )

            torch.testing.assert_close(first, second)
            self.assertFalse(first_hit)
            self.assertTrue(second_hit)
            self.assertEqual(calls, [1])

    def test_extracts_both_modalities_once_then_hits_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = EmbeddingCache(directory)
            encoder = FrozenCLIP(FakeCLIP(), FakeProcessor(), "fake", "commit")
            batch = adapt_batch(
                {
                    "sample_id": ["a", "b"],
                    "image": torch.randn(2, 3, 8, 8),
                    "text": ["left", "right"],
                    "target": torch.tensor([0, 1]),
                    "availability": {"image": [True, True], "text": [True, True]},
                }
            )
            cache_metadata = {
                name: metadata(modality=name) for name in ("image", "text")
            }

            _, first, first_hit = extract_to_cache(
                encoder, (batch,), cache, cache_metadata
            )
            _, second, second_hit = extract_to_cache(
                encoder, (batch,), cache, cache_metadata
            )

            self.assertFalse(first_hit)
            self.assertTrue(second_hit)
            torch.testing.assert_close(first["image"], second["image"])


if __name__ == "__main__":
    unittest.main()
