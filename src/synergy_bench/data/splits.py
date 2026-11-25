"""Fixed stratified splits and nested, class-balanced training subsets.

Splits
    Each sample index is assigned to train / validation / test once, from
    ``split_seed`` only, stratified by target: every class is shuffled with its
    own stream and cut with the same fractions. Validation and test never change
    when the training budget changes.

Nested training subsets
    For sample-efficiency curves, the training rows of each class are shuffled
    once with the ``subset`` stream; a subset of fraction ``f`` takes the first
    ``round(f * n_class)`` rows of every class (at least one per class). Because
    every subset is a prefix of the same order, smaller subsets are always
    contained in larger ones, and the class balance of each subset matches the
    full training split within one sample per class.

The legacy TP1 regenerated the whole dataset (and therefore a new test set) for
each training size; this module replaces that protocol.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .manifest import SPLITS, build_manifest, parse_sample_id, validate_manifest
from .seeds import make_rng

DEFAULT_SPLIT_FRACTIONS: dict[str, float] = {"train": 0.6, "validation": 0.2, "test": 0.2}
_FRACTION_TOLERANCE = 1e-9


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #
def validate_split_fractions(fractions: Mapping[str, float] | Sequence[float]) -> dict[str, float]:
    """Return {'train', 'validation', 'test'} fractions; each in (0, 1), summing to 1."""
    if isinstance(fractions, Mapping):
        keys = set(fractions)
        if keys != set(SPLITS):
            raise ValueError(f"split fractions must have exactly the keys {SPLITS}, got {sorted(keys)}")
        values = [fractions[name] for name in SPLITS]
    else:
        values = list(fractions)
        if len(values) != len(SPLITS):
            raise ValueError(f"expected {len(SPLITS)} split fractions {SPLITS}, got {len(values)}")
    out = {}
    for name, value in zip(SPLITS, values):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise TypeError(f"fraction for {name!r} must be a number, got {type(value).__name__}")
        value = float(value)
        if not math.isfinite(value) or not 0.0 < value < 1.0:
            raise ValueError(f"fraction for {name!r} must be in (0, 1), got {value}")
        out[name] = value
    total = sum(out.values())
    if abs(total - 1.0) > _FRACTION_TOLERANCE:
        raise ValueError(f"split fractions must sum to 1, got {total}")
    return out


def _round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


def _class_counts(n: int, fractions: dict[str, float]) -> dict[str, int]:
    """Largest-remainder allocation of n items to the splits (exact total, deterministic)."""
    raw = {name: n * fractions[name] for name in SPLITS}
    counts = {name: int(math.floor(raw[name])) for name in SPLITS}
    remainder = n - sum(counts.values())
    order = sorted(SPLITS, key=lambda name: (-(raw[name] - counts[name]), SPLITS.index(name)))
    for name in order[:remainder]:
        counts[name] += 1
    return counts


def _as_binary_targets(targets: Sequence[int]) -> np.ndarray:
    y = np.asarray(targets)
    if y.ndim != 1 or len(y) == 0:
        raise ValueError("targets must be a non-empty 1-D sequence")
    if not np.issubdtype(y.dtype, np.integer) or y.dtype == bool:
        raise TypeError(f"targets must be integers 0/1, got dtype {y.dtype}")
    bad = sorted(set(np.unique(y).tolist()) - {0, 1})
    if bad:
        raise ValueError(f"targets must be 0 or 1, found {bad}")
    return y.astype(np.int64)


# --------------------------------------------------------------------------- #
# Fixed splits
# --------------------------------------------------------------------------- #
def assign_splits(
    targets: Sequence[int],
    fractions: Mapping[str, float] | Sequence[float] = DEFAULT_SPLIT_FRACTIONS,
    split_seed: int = 0,
) -> list[str]:
    """Split name for every sample index, stratified by target and fixed by split_seed."""
    y = _as_binary_targets(targets)
    fractions = validate_split_fractions(fractions)
    labels = np.empty(len(y), dtype=object)
    for cls in (0, 1):
        idx = np.flatnonzero(y == cls)
        if len(idx) == 0:
            continue
        order = idx[make_rng(split_seed, "split", cls).permutation(len(idx))]
        counts = _class_counts(len(idx), fractions)
        start = 0
        for name in SPLITS:
            labels[order[start : start + counts[name]]] = name
            start += counts[name]
    missing = [name for name in SPLITS if not np.any(labels == name)]
    if missing:
        raise ValueError(f"too few samples: splits {missing} would be empty")
    return labels.tolist()


def make_split_manifest(
    benchmark_variant: str,
    scenario: str,
    data_seed: int,
    targets: Sequence[int],
    fractions: Mapping[str, float] | Sequence[float] = DEFAULT_SPLIT_FRACTIONS,
    split_seed: int = 0,
) -> pd.DataFrame:
    """Manifest of all samples with their fixed split."""
    splits = assign_splits(targets, fractions, split_seed)
    return build_manifest(benchmark_variant, scenario, data_seed, splits, list(np.asarray(targets).tolist()))


# --------------------------------------------------------------------------- #
# Nested training subsets
# --------------------------------------------------------------------------- #
def subset_key(fraction: float) -> str:
    """Canonical dictionary key for a subset fraction, e.g. 0.05 -> '0.05', 1 -> '1'."""
    return f"{float(fraction):.6g}"


def make_nested_train_subsets(
    manifest: pd.DataFrame,
    fractions: Sequence[float],
    seed: int,
) -> dict[str, list[str]]:
    """Nested, class-balanced training subsets as lists of sample_ids.

    Keys are ``subset_key(fraction)`` in increasing fraction order; values are
    sample_ids sorted by sample index. Only rows with split == 'train' are used.
    """
    manifest = validate_manifest(manifest)
    fracs = []
    for f in fractions:
        if isinstance(f, bool) or not isinstance(f, (int, float, np.integer, np.floating)):
            raise TypeError(f"subset fraction must be a number, got {type(f).__name__}")
        f = float(f)
        if not math.isfinite(f) or not 0.0 < f <= 1.0:
            raise ValueError(f"subset fraction must be in (0, 1], got {f}")
        fracs.append(f)
    if not fracs:
        raise ValueError("at least one subset fraction is required")
    keys = [subset_key(f) for f in fracs]
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate subset fractions {fracs}")

    train = manifest[manifest["split"] == "train"]
    if train.empty:
        raise ValueError("manifest has no training rows")
    ordered_by_class = {}
    for cls in (0, 1):
        ids = train.loc[train["target"] == cls, "sample_id"].tolist()
        perm = make_rng(seed, "subset", cls).permutation(len(ids))
        ordered_by_class[cls] = [ids[i] for i in perm]

    subsets = {}
    for f in sorted(fracs):
        chosen = []
        for cls, ids in ordered_by_class.items():
            if not ids:
                continue
            k = len(ids) if f == 1.0 else min(len(ids), max(1, _round_half_up(f * len(ids))))
            chosen.extend(ids[:k])
        chosen.sort(key=lambda sid: parse_sample_id(sid)["sample_index"])
        subsets[subset_key(f)] = chosen
    return subsets