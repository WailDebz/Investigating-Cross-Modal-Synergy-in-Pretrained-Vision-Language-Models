"""Turn a trained binary model into one complete result row."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

from torch import nn

from synergy_bench.models.contracts import summarize_parameters

from .contracts import RunResult
from .core import binary_metrics, timed_call
from .training import TrainingSummary, predict_logits


@dataclass(frozen=True)
class ExperimentIdentity:
    config_hash: str
    sample_manifest_hash: str
    seed: int
    family: str
    modalities: tuple[str, ...]
    head: str


def evaluate_model(
    model: nn.Module,
    test_batches: Sequence[Any],
    identity: ExperimentIdentity,
    *,
    training: TrainingSummary | None = None,
    extraction_seconds: float | None = None,
    extra_metrics: Mapping[str, Real] | None = None,
) -> RunResult:
    (logits, targets), inference_seconds = timed_call(predict_logits, model, test_batches)
    metrics = binary_metrics(logits, targets)
    metrics.update(extra_metrics or {})
    parameters = summarize_parameters(model)
    timings = {"inference": inference_seconds}
    if training is not None:
        timings["training"] = training.training_seconds
        metrics["best_validation_loss"] = training.best_validation_loss
    if extraction_seconds is not None:
        timings["extraction"] = extraction_seconds
    return RunResult(
        config_hash=identity.config_hash,
        sample_manifest_hash=identity.sample_manifest_hash,
        seed=identity.seed,
        family=identity.family,
        modalities=identity.modalities,
        head=identity.head,
        metrics=metrics,
        total_parameters=parameters.total,
        trainable_parameters=parameters.trainable,
        timings_seconds=timings,
        status="completed",
        epochs=training.epochs if training is not None else None,
        peak_memory_bytes=training.peak_device_bytes if training is not None else None,
    )
