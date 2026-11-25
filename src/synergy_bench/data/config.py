"""Validated data configuration (plan section 3.5) and its canonical hash."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import get_type_hints

import yaml

from .manifest import BENCHMARK_VARIANTS, MAX_SAMPLES, SCENARIOS
from .splits import validate_split_fractions


class ConfigError(ValueError):
    """Invalid data configuration; the message names the offending field."""


@dataclass(frozen=True)
class ImageConfig:
    size: int
    channels: int
    render_chunk_size: int


@dataclass(frozen=True)
class TextConfig:
    noise: float
    template_set: str


@dataclass(frozen=True)
class SplitConfig:
    train: float
    validation: float
    test: float


@dataclass(frozen=True)
class DataConfig:
    benchmark_variant: str
    scenario: str
    n_samples: int
    data_seed: int
    split_seed: int
    image: ImageConfig
    text: TextConfig
    split: SplitConfig


def _build(cls, data, prefix: str):
    """Strictly build a (nested) dataclass: exact keys, exact scalar types."""
    where = prefix or "config"
    if not isinstance(data, dict):
        raise ConfigError(f"{where} must be a mapping, got {type(data).__name__}")
    hints = get_type_hints(cls)
    missing = [name for name in hints if name not in data]
    if missing:
        raise ConfigError(f"missing field {prefix}{missing[0]}")
    unknown = sorted(str(k) for k in data if k not in hints)
    if unknown:
        raise ConfigError(f"unknown field {prefix}{unknown[0]}")
    values = {}
    for name, typ in hints.items():
        field, value = f"{prefix}{name}", data[name]
        if dataclasses.is_dataclass(typ):
            values[name] = _build(typ, value, f"{field}.")
        elif typ is float:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigError(f"{field} must be a number, got {type(value).__name__}")
            value = float(value)
            if not math.isfinite(value):
                raise ConfigError(f"{field} must be finite, got {value}")
            values[name] = value
        elif isinstance(value, bool) or not isinstance(value, typ):
            raise ConfigError(f"{field} must be {typ.__name__}, got {type(value).__name__}")
        else:
            values[name] = value
    return cls(**values)


def _check(ok: bool, field: str, message: str) -> None:
    if not ok:
        raise ConfigError(f"{field} {message}")


def config_from_dict(data: dict) -> DataConfig:
    """Build and validate a DataConfig; any missing, unknown or invalid field raises ConfigError."""
    cfg = _build(DataConfig, data, "")
    _check(cfg.benchmark_variant in BENCHMARK_VARIANTS, "benchmark_variant",
           f"must be one of {BENCHMARK_VARIANTS}, got {cfg.benchmark_variant!r}")
    _check(cfg.scenario in SCENARIOS, "scenario", f"must be one of {SCENARIOS}, got {cfg.scenario!r}")
    _check(0 < cfg.n_samples < MAX_SAMPLES, "n_samples", f"must be in [1, {MAX_SAMPLES - 1}], got {cfg.n_samples}")
    _check(cfg.data_seed >= 0, "data_seed", f"must be >= 0, got {cfg.data_seed}")
    _check(cfg.split_seed >= 0, "split_seed", f"must be >= 0, got {cfg.split_seed}")
    _check(cfg.image.size > 0, "image.size", f"must be > 0, got {cfg.image.size}")
    _check(cfg.image.channels in (1, 3), "image.channels", f"must be 1 or 3, got {cfg.image.channels}")
    _check(cfg.image.render_chunk_size > 0, "image.render_chunk_size",
           f"must be > 0, got {cfg.image.render_chunk_size}")
    _check(cfg.text.noise >= 0, "text.noise", f"must be >= 0, got {cfg.text.noise}")
    _check(bool(cfg.text.template_set), "text.template_set", "must be non-empty")
    try:
        validate_split_fractions(dataclasses.asdict(cfg.split))
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"split: {exc}") from None
    return cfg


def load_data_config(path: str | Path) -> DataConfig:
    """Load and validate a YAML data config."""
    with open(path, encoding="utf-8") as fh:
        return config_from_dict(yaml.safe_load(fh))


def config_to_canonical_json(config: DataConfig) -> str:
    return json.dumps(dataclasses.asdict(config), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def config_hash(config: DataConfig) -> str:
    """SHA-256 hex digest of the canonical JSON form of the config."""
    return hashlib.sha256(config_to_canonical_json(config).encode("ascii")).hexdigest()
