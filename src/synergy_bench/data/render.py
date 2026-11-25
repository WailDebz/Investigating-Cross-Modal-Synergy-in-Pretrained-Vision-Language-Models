"""Deterministic, chunked grayscale rendering for ``vl_matched`` (version ``render_v1``).

Scene semantics are those of the legacy TP1 image, on a normalized [-1, 1] grid,
so they do not depend on resolution:

    blob     exp(-((X - cx)^2 + Y^2) / (2 * 0.25^2)),  cx = clip(0.6 * obs_img, -0.85, 0.85)
    grating  0.3 * cos(2*pi*2 * (X cos(theta) + Y sin(theta)) + phase),
             theta = (pi/2) / (1 + exp(-nuisance))
    noise    pix_std * N(0, 1) per pixel

Each sample uses its own stream ``make_rng(data_seed, "image_render", index)``:
the phase is drawn first, then the pixel noise. Content therefore never depends
on the chunk size, the processing order or the other samples.

Float values are mapped to uint8 with one fixed global transform
(clip to ``UINT8_RANGE`` then scale to 0..255), never with per-image statistics.
Images are grayscale; RGB is obtained by replicating the channel at load time.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np

from .seeds import make_rng

RENDER_VERSION = "render_v1"
UINT8_RANGE = (-1.5, 2.5)  # covers blob + grating + > 4.5 std of the largest pixel noise (0.26)
BLOB_WIDTH = 0.25
GRATING_AMPLITUDE = 0.3
GRATING_CYCLES = 2


@lru_cache(maxsize=8)
def _grid(size: int) -> tuple[np.ndarray, np.ndarray]:
    g = np.linspace(-1.0, 1.0, size)
    X, Y = np.meshgrid(g, g)
    X.flags.writeable = Y.flags.writeable = False
    return X, Y


def blob_center(obs_img: float) -> float:
    """Normalized x-position of the blob."""
    return float(np.clip(0.6 * obs_img, -0.85, 0.85))


def grating_orientation(nuisance: float) -> float:
    """Grating orientation in (0, pi/2)."""
    return float((np.pi / 2) / (1 + np.exp(-nuisance)))


def render_float(obs_img: float, nuisance: float, pix_std: float, size: int, data_seed: int, index: int) -> np.ndarray:
    """Float64 grayscale image (size, size) of one sample."""
    rng = make_rng(data_seed, "image_render", index)
    phase = rng.uniform(0, 2 * np.pi)
    X, Y = _grid(size)
    cx, theta = blob_center(obs_img), grating_orientation(nuisance)
    img = np.exp(-((X - cx) ** 2 + Y**2) / (2 * BLOB_WIDTH**2))
    img += GRATING_AMPLITUDE * np.cos(2 * np.pi * GRATING_CYCLES * (X * np.cos(theta) + Y * np.sin(theta)) + phase)
    img += pix_std * rng.normal(size=(size, size))
    return img


def to_uint8(img: np.ndarray) -> np.ndarray:
    """Fixed global mapping UINT8_RANGE -> 0..255."""
    lo, hi = UINT8_RANGE
    return np.rint((np.clip(img, lo, hi) - lo) * (255.0 / (hi - lo))).astype(np.uint8)


def from_uint8(img: np.ndarray) -> np.ndarray:
    """Inverse mapping back to scene units (up to quantization), for analysis."""
    lo, hi = UINT8_RANGE
    return lo + np.asarray(img, dtype=np.float64) * ((hi - lo) / 255.0)


def render_uint8(obs_img: float, nuisance: float, pix_std: float, size: int, data_seed: int, index: int) -> np.ndarray:
    return to_uint8(render_float(obs_img, nuisance, pix_std, size, data_seed, index))


def render_chunked(
    obs_img: Sequence[float],
    nuisance: Sequence[float],
    pix_std: float,
    size: int,
    data_seed: int,
    indices: Sequence[int],
    chunk_size: int,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Render samples ``indices`` into ``out`` (uint8, (n, size, size); may be a memmap).

    At most ``chunk_size`` rendered images are buffered at a time, so peak memory does
    not grow with the number of samples.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    n = len(indices)
    if out is None:
        out = np.empty((n, size, size), dtype=np.uint8)
    if out.shape != (n, size, size) or out.dtype != np.uint8:
        raise ValueError(f"out must be uint8 {(n, size, size)}, got {out.dtype} {out.shape}")
    buffer = np.empty((min(chunk_size, n), size, size), dtype=np.uint8)
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        for k in range(start, stop):
            buffer[k - start] = render_uint8(obs_img[k], nuisance[k], pix_std, size, data_seed, int(indices[k]))
        out[start:stop] = buffer[: stop - start]
    return out
