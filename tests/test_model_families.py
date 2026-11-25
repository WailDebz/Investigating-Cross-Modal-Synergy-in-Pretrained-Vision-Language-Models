import unittest
from types import SimpleNamespace

import torch
from torch import Tensor, nn

from synergy_bench.models import (
    FusionHead,
    ModalityEncoder,
    adapt_batch,
    inference,
    seeded_model_init,
    summarize_parameters,
)
from synergy_bench.models.adaptation import (
    AdaptiveCLIPClassifier,
    changed_parameter_names,
    configure_final_block_adaptation,
    parameter_snapshot,
)
from synergy_bench.models.foundation import FrozenCLIP, clip_feature_tensor
from synergy_bench.models.fusion import MLPFusionHead
from synergy_bench.models.scratch import ScratchClassifier, TextVocabulary
from model_fakes import FakeCLIP, FakeProcessor


class MockEncoder(ModalityEncoder):
    def __init__(self, scale: float) -> None:
        super().__init__(output_dim=2)
        self.projection = nn.Linear(2, 2, bias=False)
        self.scale = scale

    def forward(self, inputs: Tensor) -> Tensor:
        return self.projection(inputs) * self.scale


class SumHead(FusionHead):
    def forward(self, embeddings, availability=None):
        return torch.stack(tuple(embeddings.values())).sum(0)


class ModelContractTests(unittest.TestCase):
    def test_encoders_are_swappable_without_changing_the_head(self):
        with seeded_model_init(7):
            image_encoder = MockEncoder(1.0)
            text_encoder = MockEncoder(2.0)
        inputs = torch.ones(3, 2)
        head = SumHead()

        first = head({"image": image_encoder(inputs), "text": text_encoder(inputs)})
        second = head({"image": text_encoder(inputs), "text": image_encoder(inputs)})

        torch.testing.assert_close(first, second)

    def test_batch_adapter_excludes_metadata_and_moves_tensors(self):
        batch = {
            "sample_id": ["C/0", "C/1"],
            "image": torch.zeros(2, 3, 4, 4),
            "text": ["left", "right"],
            "target": torch.tensor([0, 1]),
            "availability": {"image": [True, True], "text": [True, False]},
            "metadata": {"target_copy": [0, 1]},
        }

        adapted = adapt_batch(batch)

        self.assertEqual(adapted.sample_ids, ("C/0", "C/1"))
        self.assertEqual(set(adapted.modalities), {"image", "text"})
        self.assertFalse(hasattr(adapted, "metadata"))
        self.assertEqual(adapted.targets.device.type, "cpu")
        self.assertEqual(adapted.availability["text"].dtype, torch.bool)

    def test_batch_adapter_rejects_duplicate_sample_ids(self):
        batch = {
            "sample_id": ["C/0", "C/0"],
            "image": torch.zeros(2, 3, 4, 4),
            "text": ["left", "right"],
            "target": torch.tensor([0, 1]),
            "availability": {"image": [True, True], "text": [True, True]},
        }

        with self.assertRaises(ValueError):
            adapt_batch(batch)

    def test_seeded_forward_and_backward_are_repeatable(self):
        with seeded_model_init(11):
            first = MockEncoder(1.0)
        with seeded_model_init(11):
            second = MockEncoder(1.0)
        inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        first_output = first(inputs).sum()
        second_output = second(inputs).sum()
        first_output.backward()
        second_output.backward()

        for first_parameter, second_parameter in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(first_parameter, second_parameter)
            torch.testing.assert_close(first_parameter.grad, second_parameter.grad)

    def test_inference_uses_module_device_and_restores_training_mode(self):
        module = nn.Linear(2, 1)
        module.train()

        output = inference(module, torch.ones(2, 2))

        self.assertEqual(output.device, next(module.parameters()).device)
        self.assertTrue(module.training)
        self.assertFalse(output.requires_grad)

    def test_parameter_summary_reports_trainable_names(self):
        module = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))
        module[0].weight.requires_grad_(False)

        summary = summarize_parameters(module)

        self.assertEqual(summary.total, 13)
        self.assertEqual(summary.trainable, 7)
        self.assertEqual(summary.trainable_names, ("0.bias", "1.weight", "1.bias"))

    def test_scratch_image_text_classifier_forward(self):
        vocabulary = TextVocabulary.from_texts(["left pattern", "right pattern"])
        head = MLPFusionHead({"image": 8, "text": 8}, hidden_dim=4)
        model = ScratchClassifier(head, vocabulary, embedding_dim=8)
        batch = adapt_batch(
            {
                "sample_id": ["a", "b"],
                "image": torch.randn(2, 3, 224, 224),
                "text": ["left pattern", "right pattern"],
                "target": torch.tensor([0, 1]),
                "availability": {"image": [True, True], "text": [True, True]},
            }
        )

        self.assertEqual(model(batch).shape, (2,))

    def test_frozen_clip_encodes_both_modalities(self):
        backbone = FrozenCLIP(FakeCLIP(), FakeProcessor(), "fake", "commit")
        batch = adapt_batch(
            {
                "sample_id": ["a", "b"],
                "image": torch.randn(2, 3, 8, 8),
                "text": ["left", "right"],
                "target": torch.tensor([0, 1]),
                "availability": {"image": [True, True], "text": [True, True]},
            }
        )

        embeddings = backbone.encode_batch(batch)

        self.assertEqual(embeddings["image"].shape, (2, 4))
        self.assertEqual(embeddings["text"].shape, (2, 4))
        self.assertEqual(summarize_parameters(backbone).trainable, 0)

    def test_clip_feature_output_supports_transformers_five(self):
        expected = torch.ones(2, 4)
        torch.testing.assert_close(
            clip_feature_tensor(SimpleNamespace(pooler_output=expected)), expected
        )

    def test_final_block_adaptation_changes_only_approved_parameters(self):
        model = FakeCLIP()
        trainable = configure_final_block_adaptation(model)
        before = parameter_snapshot(model)
        loss = sum(parameter.sum() for parameter in model.parameters() if parameter.requires_grad)
        loss.backward()
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.add_(parameter.grad)

        changed = changed_parameter_names(model, before)

        self.assertEqual(set(changed), set(trainable))
        self.assertTrue(all(name in trainable for name, p in model.named_parameters() if p.requires_grad))

    def test_adapted_clip_uses_raw_batch_and_updates_encoder(self):
        model = FakeCLIP()
        head = MLPFusionHead({"image": 4, "text": 4}, hidden_dim=4)
        classifier = AdaptiveCLIPClassifier(model, FakeProcessor(), head)
        batch = adapt_batch(
            {
                "sample_id": ["a", "b"],
                "image": torch.randn(2, 3, 8, 8),
                "text": ["left", "right"],
                "target": torch.tensor([0, 1]),
                "availability": {"image": [True, True], "text": [True, True]},
            }
        )

        classifier(batch).sum().backward()

        parameters = dict(model.named_parameters())
        self.assertTrue(
            all(parameters[name].grad is not None for name in classifier.trainable_encoder_names)
        )


if __name__ == "__main__":
    unittest.main()
