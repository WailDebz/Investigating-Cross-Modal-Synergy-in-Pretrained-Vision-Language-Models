import unittest

import torch

from synergy_bench.models import (
    BilinearFusionHead,
    LateFusionHead,
    LinearFusionHead,
    MLPFusionHead,
)


class FusionShapeTests(unittest.TestCase):
    def setUp(self):
        self.embeddings = {"image": torch.randn(5, 4), "text": torch.randn(5, 3)}
        self.available = {
            "image": torch.tensor([1, 1, 1, 0, 1], dtype=torch.bool),
            "text": torch.tensor([1, 1, 0, 1, 1], dtype=torch.bool),
        }

    def test_linear_and_mlp_shapes(self):
        dims = {"image": 4, "text": 3}
        self.assertEqual(LinearFusionHead(dims)(self.embeddings, self.available).shape, (5,))
        self.assertEqual(MLPFusionHead(dims)(self.embeddings, self.available).shape, (5,))

    def test_bilinear_dimension_and_shape(self):
        head = BilinearFusionHead({"image": 4, "text": 3}, projection_dim=8)

        output = head(self.embeddings, self.available)

        self.assertEqual(head.tensor_dim, 81)
        self.assertEqual(output.shape, (5,))

    def test_late_fusion_averages_only_available_logits(self):
        head = LateFusionHead(("image", "text"))
        logits = {"image": torch.tensor([[2.0], [4.0]]), "text": torch.tensor([[6.0], [8.0]])}
        availability = {
            "image": torch.tensor([True, True]),
            "text": torch.tensor([True, False]),
        }

        output = head(logits, availability)

        torch.testing.assert_close(output, torch.tensor([4.0, 4.0]))

    def test_rejects_samples_with_no_available_modality(self):
        availability = {name: torch.zeros(5, dtype=torch.bool) for name in self.embeddings}
        with self.assertRaises(ValueError):
            LinearFusionHead({"image": 4, "text": 3})(self.embeddings, availability)


if __name__ == "__main__":
    unittest.main()
