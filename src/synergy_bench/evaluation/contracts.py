"""Machine-readable result records for Phase 1 experiments."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Any, Mapping


_STATUSES = {"completed", "failed", "skipped"}


def _validate_numbers(values: Mapping[str, Real], *, non_negative: bool) -> None:
    for name, value in values.items():
        if (
            not isinstance(name, str)
            or not name
            or isinstance(value, bool)
            or not isinstance(value, Real)
        ):
            raise ValueError("metric and timing names must be non-empty with numeric values")
        if not isfinite(float(value)) or (non_negative and value < 0):
            raise ValueError(f"invalid numeric value for {name!r}: {value!r}")


@dataclass(frozen=True)
class RunResult:
    config_hash: str
    sample_manifest_hash: str
    seed: int
    family: str
    modalities: tuple[str, ...]
    head: str
    metrics: Mapping[str, Real]
    total_parameters: int
    trainable_parameters: int
    timings_seconds: Mapping[str, Real]
    status: str
    epochs: int | None = None
    peak_memory_bytes: int | None = None

    def __post_init__(self) -> None:
        text_fields = {
            "config_hash": self.config_hash,
            "sample_manifest_hash": self.sample_manifest_hash,
            "family": self.family,
            "head": self.head,
        }
        if any(not isinstance(value, str) or not value.strip() for value in text_fields.values()):
            raise ValueError("experiment identity fields must be non-empty strings")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not self.modalities:
            raise ValueError("modalities must be non-empty and unique")
        if any(not isinstance(value, str) or not value for value in self.modalities):
            raise ValueError("modality names must be non-empty strings")
        if len(set(self.modalities)) != len(self.modalities):
            raise ValueError("modalities must be non-empty and unique")
        if not isinstance(self.status, str) or self.status not in _STATUSES:
            raise ValueError(f"status must be one of {sorted(_STATUSES)}")
        counts = (self.total_parameters, self.trainable_parameters)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in counts):
            raise ValueError("parameter counts must be integers")
        if self.total_parameters < 0 or not 0 <= self.trainable_parameters <= self.total_parameters:
            raise ValueError("parameter counts must satisfy 0 <= trainable <= total")
        for name, value in (("epochs", self.epochs), ("peak_memory_bytes", self.peak_memory_bytes)):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer or None")
        if not isinstance(self.metrics, Mapping) or not isinstance(self.timings_seconds, Mapping):
            raise ValueError("metrics and timings_seconds must be mappings")
        if self.status == "completed" and not self.metrics:
            raise ValueError("completed results require at least one metric")
        _validate_numbers(self.metrics, non_negative=False)
        _validate_numbers(self.timings_seconds, non_negative=True)

    def to_row(self) -> dict[str, Any]:
        """Flatten the record into one stable table row."""

        row: dict[str, Any] = {
            "config_hash": self.config_hash,
            "sample_manifest_hash": self.sample_manifest_hash,
            "seed": self.seed,
            "family": self.family,
            "modalities": ",".join(self.modalities),
            "head": self.head,
            "total_parameters": self.total_parameters,
            "trainable_parameters": self.trainable_parameters,
            "status": self.status,
            "epochs": self.epochs,
            "peak_memory_bytes": self.peak_memory_bytes,
        }
        row.update((f"metric_{name}", value) for name, value in sorted(self.metrics.items()))
        row.update(
            (f"time_{name}_seconds", value) for name, value in sorted(self.timings_seconds.items())
        )
        return row
