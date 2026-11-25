"""Thin adapter around the frozen TP1 generator (``legacy/tp1/multimodal_gen.py``).

The generator is loaded by path, after checking its SHA-256, and is never copied
or edited. This module only attaches the research contract to it: ``legacy_tp1``
sample IDs, a manifest with fixed stratified splits, and the oracle reported
under its precise name ``latent_observation_oracle``.

Hashes are computed on the file content with CRLF normalized to LF. This gives
the Git-stored hashes listed in ``legacy/tp1/SHA256SUMS.txt`` whatever the local
line-ending conversion (``core.autocrlf``) of the checkout.
"""
from __future__ import annotations

import hashlib
import sys
import types
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DataConfig
from .splits import make_split_manifest

LEGACY_DIR = Path(__file__).resolve().parents[3] / "legacy" / "tp1"
LEGACY_GENERATOR = LEGACY_DIR / "multimodal_gen.py"
LEGACY_NOTEBOOK = LEGACY_DIR / "multimodal_fusion_lab_student.ipynb"

LEGACY_SHA256 = {
    "multimodal_gen.py": "e222d35a191a3bed3c29ae1fb7f5b51e310f0055d81a97c4296462bc7d4ebac3",
    "multimodal_fusion_lab_student.ipynb": "b0d5f9560e45090bea20d5934ea57148c6116504c31b9be9e8644624bb8f19e1",
}

# Data-level values printed by the legacy generator / stored in the notebook outputs.
# They depend only on the generator, so they must reproduce to the printed precision.
LEGACY_ORACLE_N5000 = {"A": 0.869, "B": 0.852, "C": 0.915, "D": 0.885, "E": 0.869}

# Stored single-seed notebook outputs (seed 0, n=6000 split 0.6/0.2/0.2 unless noted).
# Model metrics are historical observations, kept for the later end-to-end regression
# of Phase 1.2; they are compared with MODEL_METRIC_TOLERANCE, data-level values with
# DATA_METRIC_TOLERANCE (rounding of the printed value).
DATA_METRIC_TOLERANCE = 5e-4
MODEL_METRIC_TOLERANCE = 0.03
NOTEBOOK_REFERENCE_METRICS = {
    "data": {
        # make_dataset(scn, n=4000).summary(): oracle accuracy and P(y=1)
        "summary_n4000_oracle": {"A": 0.872, "B": 0.850, "C": 0.917, "D": 0.883, "E": 0.872},
        "summary_n4000_p_y1": {"A": 0.49, "B": 0.51, "C": 0.51, "D": 0.51, "E": 0.49},
        # unimodal_oracle(make_dataset(scn, n=5000)): accuracy of sign(obs[:, m]), m = img, sig, tab
        "unimodal_oracle_n5000": {
            "A": [0.786, 0.785, 0.799],
            "B": [0.692, 0.696, 0.633],
            "C": [0.5, 0.493, 0.502],
            "D": [0.751, 0.708, 0.513],
            "E": [0.786, 0.785, 0.799],
        },
        # te.oracle_accuracy() on the notebook test split of load(scn) (n=6000)
        "test_oracle_n6000": {"A": 0.866, "B": 0.850, "C": 0.928, "D": 0.858},
    },
    "models": {
        # sk_grid(scn): test accuracy
        "sklearn_grid": {
            "A": {"uni-img": 0.736, "uni-sig": 0.785, "uni-tab": 0.792, "early-linear": 0.843,
                  "early-MLP": 0.847, "early-GBM": 0.850, "late-avg": 0.853, "late-stack": 0.823},
            "B": {"uni-img": 0.682, "uni-sig": 0.686, "uni-tab": 0.640, "early-linear": 0.828,
                  "early-MLP": 0.819, "early-GBM": 0.832, "late-avg": 0.802, "late-stack": 0.788},
            "C": {"uni-img": 0.509, "uni-sig": 0.481, "uni-tab": 0.500, "early-linear": 0.507,
                  "early-MLP": 0.898, "early-GBM": 0.881, "late-avg": 0.501, "late-stack": 0.502},
        },
        # torch_suite(scn): test accuracy
        "torch_suite": {
            "A": {"uni-img": 0.756, "uni-sig": 0.777, "uni-tab": 0.793, "late-avg": 0.858,
                  "concat+linear": 0.862, "concat+MLP": 0.849, "gated+MLP": 0.790, "bilinear+linear": 0.848},
            "B": {"uni-img": 0.685, "uni-sig": 0.673, "uni-tab": 0.634, "late-avg": 0.811,
                  "concat+linear": 0.840, "concat+MLP": 0.839, "gated+MLP": 0.825, "bilinear+linear": 0.832},
            "C": {"uni-img": 0.509, "uni-sig": 0.483, "uni-tab": 0.480, "late-avg": 0.502,
                  "concat+linear": 0.496, "concat+MLP": 0.908, "gated+MLP": 0.903, "bilinear+linear": 0.914},
        },
        # noise_sweep("A"): test accuracy by global noise multiplier
        "noise_sweep_A": {
            "0.5": {"early-linear": 0.917, "late-avg": 0.922, "oracle": 0.932},
            "1.0": {"early-linear": 0.843, "late-avg": 0.853, "oracle": 0.866},
            "1.5": {"early-linear": 0.778, "late-avg": 0.780, "oracle": 0.808},
            "2.0": {"early-linear": 0.720, "late-avg": 0.716, "oracle": 0.760},
            "3.0": {"early-linear": 0.642, "late-avg": 0.645, "oracle": 0.690},
        },
    },
}


def normalized_sha256(path: str | Path) -> str:
    """SHA-256 of a text file with CRLF normalized to LF (equals the Git-stored blob hash)."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def check_legacy_hashes() -> None:
    """Raise if a frozen legacy file differs from its recorded hash."""
    for name, expected in LEGACY_SHA256.items():
        actual = normalized_sha256(LEGACY_DIR / name)
        if actual != expected:
            raise RuntimeError(f"legacy/tp1/{name} was modified: sha256 {actual} != {expected}")


@lru_cache(maxsize=1)
def load_legacy_module() -> types.ModuleType:
    """Import the frozen generator by path, without writing bytecode into legacy/."""
    check_legacy_hashes()
    name = "synergy_bench_legacy_multimodal_gen"
    module = types.ModuleType(name)
    module.__file__ = str(LEGACY_GENERATOR)
    sys.modules[name] = module  # dataclasses resolve annotations through sys.modules
    try:
        exec(compile(LEGACY_GENERATOR.read_text(encoding="utf-8"), str(LEGACY_GENERATOR), "exec"), module.__dict__)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def make_legacy_dataset(scenario: str, n_samples: int, data_seed: int, img_size: int = 16):
    """Legacy ``MultimodalData`` for (scenario, n, seed), exactly as the TP1 generator makes it."""
    return load_legacy_module().make_dataset(scenario, n=n_samples, seed=data_seed, img_size=img_size)


def latent_observation_oracle(data) -> float:
    """Legacy oracle accuracy under its precise name: Bayes rule on noisy latents, before
    pixel/sample rendering noise, so an optimistic ceiling rather than an observable-data one."""
    return float(np.mean(data.oracle_pred == data.y))


def unimodal_latent_oracle(data) -> list[float]:
    """Accuracy of sign(obs[:, m]) for m = img, sig, tab (the notebook's unimodal oracle)."""
    return [float(np.mean((data.obs[:, m] > 0) == data.y)) for m in range(data.obs.shape[1])]


def legacy_manifest(config: DataConfig) -> pd.DataFrame:
    """Contract manifest for a ``legacy_tp1`` config (splits from split_seed, not the legacy split)."""
    if config.benchmark_variant != "legacy_tp1":
        raise ValueError(f"expected benchmark_variant 'legacy_tp1', got {config.benchmark_variant!r}")
    data = make_legacy_dataset(config.scenario, config.n_samples, config.data_seed, config.image.size)
    fractions = {"train": config.split.train, "validation": config.split.validation, "test": config.split.test}
    return make_split_manifest("legacy_tp1", config.scenario, config.data_seed, data.y, fractions, config.split_seed)
