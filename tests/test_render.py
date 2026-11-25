"""Tests for deterministic chunked rendering and its resolution-independent semantics."""
import tracemalloc

import numpy as np
import pytest

from synergy_bench.data.render import (
    BLOB_WIDTH,
    GRATING_CYCLES,
    UINT8_RANGE,
    _grid,
    blob_center,
    from_uint8,
    grating_orientation,
    render_chunked,
    render_uint8,
    to_uint8,
)

RENDER_PEAK_LIMIT_BYTES = 64 * 2**20  # declared budget at 224 px, chunk 128 (docs/decisions.md)
PIX_STD = 0.1 + 0.2 * 0.8  # largest legacy pixel noise (scenarios A/E)

rng = np.random.default_rng(123)
OBS = rng.normal(scale=1.2, size=40)
NUIS = rng.normal(size=40)
IDX = np.arange(1000, 1040)


def test_same_sample_identical_across_runs_chunks_and_order():
    ref = render_chunked(OBS, NUIS, PIX_STD, 64, 0, IDX, chunk_size=40)
    for chunk in (1, 3, 7, 128):
        assert np.array_equal(render_chunked(OBS, NUIS, PIX_STD, 64, 0, IDX, chunk_size=chunk), ref)
    rev = render_chunked(OBS[::-1], NUIS[::-1], PIX_STD, 64, 0, IDX[::-1], chunk_size=5)
    assert np.array_equal(rev[::-1], ref)
    assert np.array_equal(render_uint8(OBS[9], NUIS[9], PIX_STD, 64, 0, int(IDX[9])), ref[9])


def test_different_index_or_seed_changes_noise():
    a = render_uint8(0.3, 0.1, PIX_STD, 32, 0, 1)
    assert not np.array_equal(a, render_uint8(0.3, 0.1, PIX_STD, 32, 0, 2))
    assert not np.array_equal(a, render_uint8(0.3, 0.1, PIX_STD, 32, 1, 1))


def test_fixed_global_uint8_mapping():
    lo, hi = UINT8_RANGE
    assert to_uint8(np.array([lo - 1, lo, (lo + hi) / 2, hi, hi + 1])).tolist() == [0, 0, 128, 255, 255]
    flat = np.full((4, 4), 0.5)
    assert np.all(to_uint8(flat) == to_uint8(np.array(0.5)))  # no per-image normalization
    x = np.linspace(lo, hi, 101)
    assert np.max(np.abs(from_uint8(to_uint8(x)) - x)) <= (hi - lo) / 255 / 2 + 1e-12


def test_clipping_is_negligible():
    imgs = render_chunked(OBS, NUIS, PIX_STD, 64, 0, IDX, chunk_size=16)
    assert np.mean((imgs == 0) | (imgs == 255)) < 1e-3


def _estimate_center(img, size):
    X, Y = _grid(size)
    img = img - img.mean()
    cands = np.linspace(-0.9, 0.9, 361)
    scores = []
    for c in cands:
        t = np.exp(-((X - c) ** 2 + Y**2) / (2 * BLOB_WIDTH**2))
        t -= t.mean()
        scores.append(np.sum(img * t) / np.linalg.norm(t))
    return cands[int(np.argmax(scores))]


def _estimate_orientation(img, size):
    X, Y = _grid(size)
    img = img - img.mean()
    cands = np.linspace(0, np.pi / 2, 361)
    scores = [abs(np.sum(img * np.exp(-2j * np.pi * GRATING_CYCLES * (X * np.cos(t) + Y * np.sin(t)))))
              for t in cands]
    return cands[int(np.argmax(scores))]


@pytest.mark.parametrize("k", range(6))
def test_resolution_preserves_blob_position_and_orientation(k):
    obs, nuis = float(OBS[k]), float(NUIS[k])
    est = {}
    for size in (16, 224):
        img = from_uint8(render_uint8(obs, nuis, PIX_STD, size, 0, k))
        est[size] = (_estimate_center(img, size), _estimate_orientation(img, size))
    cx, theta = blob_center(obs), grating_orientation(nuis)
    for size, (c, t) in est.items():
        assert abs(c - cx) < 0.06, (size, c, cx)
        assert abs(t - theta) < 0.08, (size, t, theta)
    assert abs(est[16][0] - est[224][0]) < 0.08
    assert abs(est[16][1] - est[224][1]) < 0.1


def test_peak_memory_is_bounded_by_chunk_not_dataset(tmp_path):
    def peak(n):
        out = np.lib.format.open_memmap(tmp_path / f"cache_{n}.npy", mode="w+", dtype=np.uint8, shape=(n, 224, 224))
        obs, nuis = np.zeros(n), np.zeros(n)
        tracemalloc.start()
        render_chunked(obs, nuis, PIX_STD, 224, 0, np.arange(n), chunk_size=32, out=out)
        _, p = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        out.flush()
        del out
        return p

    small, large = peak(64), peak(256)
    assert large < RENDER_PEAK_LIMIT_BYTES
    assert large < 1.5 * small  # does not grow with the number of samples
    assert large < 256 * 224 * 224  # far below even a uint8 copy of the whole set


def test_out_validation():
    with pytest.raises(ValueError, match="chunk_size"):
        render_chunked(OBS, NUIS, PIX_STD, 8, 0, IDX, chunk_size=0)
    with pytest.raises(ValueError, match="uint8"):
        render_chunked(OBS, NUIS, PIX_STD, 8, 0, IDX, chunk_size=4, out=np.empty((40, 8, 8)))
