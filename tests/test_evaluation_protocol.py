import unittest
import tempfile
from pathlib import Path

import torch
from torch import nn

from synergy_bench.evaluation import RunResult
from synergy_bench.evaluation.core import (
    aggregate_results,
    evaluate_locked_test,
    mark_missing,
    select_on_validation,
    validate_nested_subsets,
)
from synergy_bench.evaluation.reporting import build_report, read_results, write_results
from synergy_bench.evaluation.training import fit_binary, predict_logits
from synergy_bench.models.contracts import ModelBatch


def valid_result(**updates):
    values = {
        "config_hash": "config-123",
        "sample_manifest_hash": "manifest-456",
        "seed": 0,
        "family": "from_scratch",
        "modalities": ("image", "text"),
        "head": "linear",
        "metrics": {"accuracy": 0.75},
        "total_parameters": 100,
        "trainable_parameters": 80,
        "timings_seconds": {"training": 1.25},
        "status": "completed",
    }
    values.update(updates)
    return RunResult(**values)


class RunResultTests(unittest.TestCase):
    def test_flattens_to_a_tidy_row(self):
        row = valid_result().to_row()

        self.assertEqual(row["sample_manifest_hash"], "manifest-456")
        self.assertEqual(row["modalities"], "image,text")
        self.assertEqual(row["metric_accuracy"], 0.75)
        self.assertEqual(row["time_training_seconds"], 1.25)

    def test_rejects_missing_experiment_identity(self):
        with self.assertRaises(ValueError):
            valid_result(config_hash="")

    def test_rejects_invalid_parameter_counts(self):
        with self.assertRaises(ValueError):
            valid_result(total_parameters=10, trainable_parameters=11)

    def test_rejects_negative_timing(self):
        with self.assertRaises(ValueError):
            valid_result(timings_seconds={"training": -1.0})

    def test_completed_result_requires_metrics(self):
        with self.assertRaises(ValueError):
            valid_result(metrics={})

    def test_selection_does_not_touch_test_evaluator(self):
        test_calls = []
        selected = select_on_validation(("small", "large"), lambda value: len(value))
        self.assertEqual(test_calls, [])

        result = evaluate_locked_test(selected, lambda value: test_calls.append(value) or value)

        self.assertEqual(result, "large")
        self.assertEqual(test_calls, ["large"])

    def test_nested_subsets_are_validated(self):
        validate_nested_subsets({"small": ("a",), "large": ("a", "b")})
        with self.assertRaises(ValueError):
            validate_nested_subsets({"small": ("a", "b"), "large": ("b", "c")})

    def test_aggregation_keeps_raw_seed_results(self):
        results = [valid_result(seed=seed, metrics={"accuracy": value}) for seed, value in enumerate((0.6, 0.7, 0.8))]

        rows = aggregate_results(results)

        self.assertEqual(len(results), 3)
        self.assertEqual(rows[0]["n"], 3)
        self.assertAlmostEqual(rows[0]["mean"], 0.7)

    def test_report_is_rebuilt_from_raw_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, table, figure = root / "raw.jsonl", root / "summary.csv", root / "accuracy.svg"
            results = [valid_result(seed=seed) for seed in range(3)]
            write_results(raw, results)

            build_report(raw, table, figure)

            self.assertEqual(len(read_results(raw)), 3)
            self.assertIn("metric,mean,std,n", table.read_text())
            self.assertIn("<svg", figure.read_text())

    def test_missingness_preserves_at_least_one_modality(self):
        batch = ModelBatch(
            sample_ids=("a", "b"),
            modalities={"image": torch.ones(2, 1), "text": torch.ones(2, 1)},
            targets=torch.tensor([0, 1]),
            availability={
                "image": torch.tensor([True, True]),
                "text": torch.tensor([True, True]),
            },
        )

        missing = mark_missing(batch, "image", torch.tensor([True, False]))

        self.assertEqual(missing.availability["image"].tolist(), [False, True])

    def test_training_uses_validation_checkpoint_and_predicts(self):
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(1, 1)

            def forward(self, batch):
                return self.layer(batch.modalities["image"])[:, 0]

        batch = ModelBatch(
            sample_ids=("a", "b", "c", "d"),
            modalities={"image": torch.tensor([[-1.0], [-0.5], [0.5], [1.0]])},
            targets=torch.tensor([0, 0, 1, 1]),
            availability={"image": torch.ones(4, dtype=torch.bool)},
        )
        model = TinyModel()

        summary = fit_binary(model, (batch,), (batch,), epochs=20, patience=5, seed=0)
        logits, targets = predict_logits(model, (batch,))

        self.assertGreater(summary.epochs, 0)
        self.assertEqual(logits.shape, targets.shape)


if __name__ == "__main__":
    unittest.main()
