"""Reproducibility tests for Phase 1.1 (named seeds)."""
import numpy as np
import pytest

from synergy_bench.data.seeds import (
    SEED_PURPOSES,
    derive_seed,
    make_rng,
    purpose_key,
    seed_sequence,
)


def draws(rng, n=8):
    return rng.integers(0, 2**31, size=n)


def test_same_inputs_give_identical_streams():
    for purpose in SEED_PURPOSES:
        np.testing.assert_array_equal(draws(make_rng(0, purpose)), draws(make_rng(0, purpose)))


def test_purposes_give_different_streams():
    streams = {p: tuple(draws(make_rng(0, p))) for p in SEED_PURPOSES}
    assert len(set(streams.values())) == len(SEED_PURPOSES)


def test_purpose_keys_are_unique_and_32_bit():
    keys = [purpose_key(p) for p in SEED_PURPOSES]
    assert len(set(keys)) == len(keys)
    assert all(0 <= k < 2**32 for k in keys)


def test_base_seed_changes_the_stream():
    assert not np.array_equal(draws(make_rng(0, "latents")), draws(make_rng(1, "latents")))


def test_consuming_one_stream_does_not_affect_another():
    reference = draws(make_rng(0, "split"))
    model_rng = make_rng(0, "model_init")
    model_rng.normal(size=10_000)
    np.testing.assert_array_equal(draws(make_rng(0, "split")), reference)


def test_model_seed_cannot_change_split_stream():
    split_seed = 0
    reference = draws(make_rng(split_seed, "split"))
    for model_seed in (0, 1, 2, 123):
        _ = derive_seed(model_seed, "model_init")
        np.testing.assert_array_equal(draws(make_rng(split_seed, "split")), reference)


def test_per_item_keys_are_independent_of_order():
    forward = {i: tuple(draws(make_rng(0, "image_render", i))) for i in range(20)}
    backward = {i: tuple(draws(make_rng(0, "image_render", i))) for i in reversed(range(20))}
    assert forward == backward
    assert len(set(forward.values())) == 20


def test_derive_seed_range_and_type():
    seed = derive_seed(0, "model_init")
    assert isinstance(seed, int)
    assert 0 <= seed < 2**32


def test_derive_seed_golden_values():
    # SeedSequence output is stable across NumPy versions and platforms.
    assert derive_seed(0, "latents") == 2187148373
    assert derive_seed(0, "split") == 393882064
    assert derive_seed(1, "latents") == 1013340154
    assert derive_seed(0, "image_render", 42) == 810857716


def test_seed_sequence_matches_make_rng():
    a = np.random.Generator(np.random.PCG64(seed_sequence(3, "subset", 7)))
    np.testing.assert_array_equal(draws(a), draws(make_rng(3, "subset", 7)))


@pytest.mark.parametrize("purpose", ["", "Latents", "model", "text"])
def test_unknown_purpose_is_rejected(purpose):
    with pytest.raises(ValueError, match="unknown seed purpose"):
        make_rng(0, purpose)


@pytest.mark.parametrize("bad", [-1, -100])
def test_negative_values_are_rejected(bad):
    with pytest.raises(ValueError):
        make_rng(bad, "latents")
    with pytest.raises(ValueError):
        make_rng(0, "image_render", bad)


@pytest.mark.parametrize("bad", [True, 1.5, "0", None])
def test_non_integer_values_are_rejected(bad):
    with pytest.raises(TypeError):
        make_rng(bad, "latents")


def test_numpy_integers_are_accepted():
    np.testing.assert_array_equal(
        draws(make_rng(np.int64(5), "corruption", np.uint32(2))),
        draws(make_rng(5, "corruption", 2)),
    )