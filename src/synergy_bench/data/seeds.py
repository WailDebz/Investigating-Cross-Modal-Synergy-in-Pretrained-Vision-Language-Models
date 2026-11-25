"""Named, independent random streams for every stochastic step of the benchmark.

The legacy TP1 generator draws everything from one RNG in sequence, so adding or
reordering a step (for example a text channel) shifts every later draw. Here each
purpose gets its own stream, derived from a base seed with NumPy's SeedSequence:
consuming or changing one stream never affects another.

Which base seed feeds which purpose:

    data_seed   -> latents, image_render, text_template, corruption
    split_seed  -> split, subset
    model seed  -> model_init, train_order   (used by Phase 1.2)

Extra non-negative integer keys (for example a sample index) give one independent
sub-stream per item, so results do not depend on processing order or chunk size.
"""
from __future__ import annotations

import hashlib
import operator

import numpy as np

SEED_PURPOSES: tuple[str, ...] = (
    "latents",
    "split",
    "image_render",
    "text_template",
    "corruption",
    "subset",
    "model_init",
    "train_order",
)


def _as_non_negative_int(value, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        value = operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}") from None
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value


def purpose_key(purpose: str) -> int:
    """Stable 32-bit integer for a purpose name (identical on every machine and run)."""
    if purpose not in SEED_PURPOSES:
        raise ValueError(f"unknown seed purpose {purpose!r}; expected one of {SEED_PURPOSES}")
    digest = hashlib.sha256(purpose.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def seed_sequence(base_seed: int, purpose: str, *keys: int) -> np.random.SeedSequence:
    """SeedSequence for (base_seed, purpose, *keys)."""
    base_seed = _as_non_negative_int(base_seed, "base_seed")
    keys = tuple(_as_non_negative_int(k, "key") for k in keys)
    return np.random.SeedSequence(entropy=base_seed, spawn_key=(purpose_key(purpose), *keys))


def make_rng(base_seed: int, purpose: str, *keys: int) -> np.random.Generator:
    """Independent NumPy Generator for (base_seed, purpose, *keys)."""
    return np.random.Generator(np.random.PCG64(seed_sequence(base_seed, purpose, *keys)))


def derive_seed(base_seed: int, purpose: str, *keys: int) -> int:
    """Plain integer seed in [0, 2**32) for libraries that need one (e.g. torch.manual_seed)."""
    state = seed_sequence(base_seed, purpose, *keys).generate_state(1, dtype=np.uint32)
    return int(state[0])