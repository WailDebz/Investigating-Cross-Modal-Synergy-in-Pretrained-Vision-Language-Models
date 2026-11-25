"""Stable sample identities and persisted split manifests.

A ``sample_id`` identifies one generated sample independently of how it is used:

    <benchmark_variant>/<scenario>/<data_seed>/<sample_index:06d>
    e.g. "vl_matched/C/0/000042"

It never depends on the split, the render chunk size, the model or the training seed.
Phase 1.2 joins embeddings and results on ``sample_id``, never on row position.

A manifest has one row per sample with the contract columns ``MANIFEST_COLUMNS``.
It is written as a canonical CSV (UTF-8, LF line endings, fixed column order, rows
sorted by sample index), so the same configuration always produces byte-identical
files and the same SHA-256 hash on every machine.
"""
from __future__ import annotations

import hashlib
import operator
import re
from pathlib import Path
from typing import Sequence

import pandas as pd

BENCHMARK_VARIANTS: tuple[str, ...] = ("legacy_tp1", "vl_matched")
SCENARIOS: tuple[str, ...] = ("A", "B", "C", "D", "E")
SPLITS: tuple[str, ...] = ("train", "validation", "test")
MANIFEST_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "benchmark_variant",
    "scenario",
    "data_seed",
    "split",
    "target",
)

SAMPLE_INDEX_WIDTH = 6
MAX_SAMPLES = 10**SAMPLE_INDEX_WIDTH

_SAMPLE_ID_RE = re.compile(
    r"^(?P<benchmark_variant>[a-z0-9_]+)/(?P<scenario>[A-Z])/"
    r"(?P<data_seed>0|[1-9][0-9]*)/(?P<sample_index>[0-9]{%d})$" % SAMPLE_INDEX_WIDTH
)


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #
def _check_variant(variant: str) -> str:
    if variant not in BENCHMARK_VARIANTS:
        raise ValueError(f"unknown benchmark_variant {variant!r}; expected one of {BENCHMARK_VARIANTS}")
    return variant


def _check_scenario(scenario: str) -> str:
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; expected one of {SCENARIOS}")
    return scenario


def _check_int(value, name: str, upper: int | None = None) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        value = operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}") from None
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    if upper is not None and value >= upper:
        raise ValueError(f"{name} must be < {upper}, got {value}")
    return value


# --------------------------------------------------------------------------- #
# Sample identity
# --------------------------------------------------------------------------- #
def make_sample_id(benchmark_variant: str, scenario: str, data_seed: int, sample_index: int) -> str:
    """Build the stable identifier of one sample."""
    _check_variant(benchmark_variant)
    _check_scenario(scenario)
    data_seed = _check_int(data_seed, "data_seed")
    sample_index = _check_int(sample_index, "sample_index", MAX_SAMPLES)
    return f"{benchmark_variant}/{scenario}/{data_seed}/{sample_index:0{SAMPLE_INDEX_WIDTH}d}"


def parse_sample_id(sample_id: str) -> dict:
    """Inverse of ``make_sample_id``; raises ValueError on any malformed identifier."""
    match = _SAMPLE_ID_RE.match(sample_id) if isinstance(sample_id, str) else None
    if match is None:
        raise ValueError(f"malformed sample_id {sample_id!r}")
    parts = match.groupdict()
    _check_variant(parts["benchmark_variant"])
    _check_scenario(parts["scenario"])
    return {
        "benchmark_variant": parts["benchmark_variant"],
        "scenario": parts["scenario"],
        "data_seed": int(parts["data_seed"]),
        "sample_index": int(parts["sample_index"]),
    }


def make_sample_ids(benchmark_variant: str, scenario: str, data_seed: int, n_samples: int) -> list[str]:
    """Identifiers for sample indices 0 .. n_samples - 1."""
    n_samples = _check_int(n_samples, "n_samples", MAX_SAMPLES + 1)
    if n_samples == 0:
        raise ValueError("n_samples must be > 0")
    return [make_sample_id(benchmark_variant, scenario, data_seed, i) for i in range(n_samples)]


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #
def build_manifest(
    benchmark_variant: str,
    scenario: str,
    data_seed: int,
    splits: Sequence[str],
    targets: Sequence[int],
) -> pd.DataFrame:
    """Manifest for samples 0 .. len(targets) - 1, with one split name and target per sample."""
    splits = list(splits)
    targets = [_check_int(t, "target") for t in targets]
    if len(splits) != len(targets):
        raise ValueError(f"splits ({len(splits)}) and targets ({len(targets)}) must have the same length")
    sample_ids = make_sample_ids(benchmark_variant, scenario, data_seed, len(targets))
    manifest = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "benchmark_variant": benchmark_variant,
            "scenario": scenario,
            "data_seed": _check_int(data_seed, "data_seed"),
            "split": splits,
            "target": targets,
        },
        columns=list(MANIFEST_COLUMNS),
    )
    return validate_manifest(manifest)


def validate_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    """Check the contract and return a canonical copy (column order, dtypes, row order)."""
    missing = [c for c in MANIFEST_COLUMNS if c not in manifest.columns]
    if missing:
        raise ValueError(f"manifest is missing required columns {missing}")
    extra = [c for c in manifest.columns if c not in MANIFEST_COLUMNS]
    if extra:
        raise ValueError(f"manifest has unexpected columns {extra}; analysis columns belong in a separate file")
    if len(manifest) == 0:
        raise ValueError("manifest is empty")

    out = manifest.loc[:, list(MANIFEST_COLUMNS)].copy()
    for col in ("sample_id", "benchmark_variant", "scenario", "split"):
        out[col] = out[col].astype(str)
    for col in ("data_seed", "target"):
        out[col] = out[col].astype("int64")

    if out["sample_id"].duplicated().any():
        dup = out.loc[out["sample_id"].duplicated(), "sample_id"].iloc[0]
        raise ValueError(f"duplicate sample_id {dup!r}")
    bad_split = sorted(set(out["split"]) - set(SPLITS))
    if bad_split:
        raise ValueError(f"unknown split names {bad_split}; expected {SPLITS}")
    bad_target = sorted(set(out["target"]) - {0, 1})
    if bad_target:
        raise ValueError(f"targets must be 0 or 1, found {bad_target}")

    indices = []
    for row in out.itertuples(index=False):
        parts = parse_sample_id(row.sample_id)
        if (parts["benchmark_variant"], parts["scenario"], parts["data_seed"]) != (
            row.benchmark_variant,
            row.scenario,
            row.data_seed,
        ):
            raise ValueError(f"sample_id {row.sample_id!r} disagrees with its row columns")
        indices.append(parts["sample_index"])

    out["_index"] = indices
    out = out.sort_values("_index", kind="stable").drop(columns="_index").reset_index(drop=True)
    return out


def manifest_to_csv_bytes(manifest: pd.DataFrame) -> bytes:
    """Canonical CSV encoding used for writing and hashing."""
    canonical = validate_manifest(manifest)
    text = canonical.to_csv(index=False, lineterminator="\n")
    return text.encode("utf-8")


def manifest_hash(manifest: pd.DataFrame) -> str:
    """SHA-256 of the canonical CSV encoding."""
    return hashlib.sha256(manifest_to_csv_bytes(manifest)).hexdigest()


def write_manifest(manifest: pd.DataFrame, path: str | Path) -> str:
    """Write the canonical CSV and return its SHA-256."""
    data = manifest_to_csv_bytes(manifest)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def read_manifest(path: str | Path) -> pd.DataFrame:
    """Read and validate a manifest written by ``write_manifest``."""
    manifest = pd.read_csv(
        Path(path),
        dtype={
            "sample_id": str,
            "benchmark_variant": str,
            "scenario": str,
            "data_seed": "int64",
            "split": str,
            "target": "int64",
        },
        keep_default_na=False,
    )
    return validate_manifest(manifest)