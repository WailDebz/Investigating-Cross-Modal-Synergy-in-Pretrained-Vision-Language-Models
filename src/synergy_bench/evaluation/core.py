"""Metrics, validation-only selection, nesting checks, and aggregation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from math import isfinite
import platform
from statistics import mean, stdev
import sys
import time
from typing import Any, Generic, TypeVar

import torch
from torch import Tensor
from torch.nn import functional as F

from .contracts import RunResult
from synergy_bench.models.contracts import ModelBatch


def binary_metrics(logits: Tensor, targets: Tensor) -> dict[str, float]:
    if logits.shape != targets.shape or logits.ndim != 1 or not len(logits):
        raise ValueError("logits and targets must share non-empty shape [samples]")
    targets = targets.to(torch.long)
    if not set(targets.tolist()).issubset({0, 1}):
        raise ValueError("binary targets must contain only 0 and 1")
    predictions = logits.ge(0).to(torch.long)
    accuracy = float(predictions.eq(targets).float().mean())
    recalls = []
    for label in (0, 1):
        mask = targets.eq(label)
        if mask.any():
            recalls.append(float(predictions[mask].eq(label).float().mean()))
    return {
        "accuracy": accuracy,
        "balanced_accuracy": mean(recalls),
        "nll": float(F.binary_cross_entropy_with_logits(logits, targets.to(logits.dtype))),
    }


T = TypeVar("T")


@dataclass(frozen=True)
class Selection(Generic[T]):
    candidate: T
    validation_score: float


def select_on_validation(candidates: Sequence[T], score: Callable[[T], float]) -> Selection[T]:
    if not candidates:
        raise ValueError("at least one candidate is required")
    scored = [(float(score(candidate)), candidate) for candidate in candidates]
    if any(not isfinite(value) for value, _ in scored):
        raise ValueError("validation scores must be finite")
    best_score, best_candidate = max(scored, key=lambda item: item[0])
    return Selection(best_candidate, best_score)


def evaluate_locked_test(selection: Selection[T], evaluate: Callable[[T], Any]) -> Any:
    return evaluate(selection.candidate)


def validate_nested_subsets(subsets: Mapping[str, Sequence[str]]) -> None:
    previous: set[str] = set()
    for name, sample_ids in subsets.items():
        current = set(sample_ids)
        if len(current) != len(sample_ids):
            raise ValueError(f"subset {name!r} contains duplicate sample IDs")
        if not previous.issubset(current):
            raise ValueError(f"subset {name!r} is not nested")
        previous = current


def evaluate_sample_efficiency(
    subsets: Mapping[str, Sequence[str]], evaluate: Callable[[Sequence[str]], T]
) -> dict[str, T]:
    validate_nested_subsets(subsets)
    return {name: evaluate(sample_ids) for name, sample_ids in subsets.items()}


def mark_missing(batch: ModelBatch, modality: str, missing: Tensor) -> ModelBatch:
    if modality not in batch.availability:
        raise ValueError(f"unknown modality {modality!r}")
    missing = missing.to(device=batch.targets.device, dtype=torch.bool)
    if missing.shape != batch.targets.shape:
        raise ValueError("missing mask must be shaped [batch]")
    availability = dict(batch.availability)
    availability[modality] = availability[modality] & ~missing
    if torch.stack(tuple(availability.values())).sum(0).eq(0).any():
        raise ValueError("missingness cannot remove every modality from a sample")
    return replace(batch, availability=availability)


def timed_call(function: Callable[..., T], *args: Any, **kwargs: Any) -> tuple[T, float]:
    started = time.perf_counter()
    value = function(*args, **kwargs)
    return value, time.perf_counter() - started


def runtime_metadata() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def aggregate_results(results: Sequence[RunResult]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for result in results:
        if result.status != "completed":
            continue
        for metric, value in result.metrics.items():
            key = (result.config_hash, result.family, result.modalities, result.head, metric)
            groups[key].append(float(value))
    rows = []
    for (config_hash, family, modalities, head, metric), values in sorted(groups.items()):
        rows.append(
            {
                "config_hash": config_hash,
                "family": family,
                "modalities": ",".join(modalities),
                "head": head,
                "metric": metric,
                "mean": mean(values),
                "std": stdev(values) if len(values) > 1 else 0.0,
                "n": len(values),
            }
        )
    return rows
