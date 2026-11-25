"""Reproducibility tests for Phase 1.1 (seeds, sample identity, manifests, splits, subsets)."""
import numpy as np
import pandas as pd
import pytest

from synergy_bench.data.manifest import (
    MANIFEST_COLUMNS,
    build_manifest,
    make_sample_id,
    make_sample_ids,
    manifest_hash,
    manifest_to_csv_bytes,
    parse_sample_id,
    read_manifest,
    validate_manifest,
    write_manifest,
)
from synergy_bench.data.seeds import (
    SEED_PURPOSES,
    derive_seed,
    make_rng,
    purpose_key,
    seed_sequence,
)
from synergy_bench.data.splits import (
    assign_splits,
    make_nested_train_subsets,
    make_split_manifest,
    validate_split_fractions,
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


# --------------------------------------------------------------------------- #
# Sample identity and manifests
# --------------------------------------------------------------------------- #
GOLDEN_TOY_MANIFEST_HASH = "f4b74afc1a9d29edec34ac6e13f1be1884e0e5aa78e3a2231f179dd91468dcc7"
TOY_SPLITS = ["train", "train", "validation", "test", "train", "test"]
TOY_TARGETS = [0, 1, 1, 0, 1, 0]


def toy_manifest(data_seed=0):
    return build_manifest("vl_matched", "C", data_seed, TOY_SPLITS, TOY_TARGETS)


def test_sample_id_format():
    assert make_sample_id("vl_matched", "C", 0, 42) == "vl_matched/C/0/000042"
    assert make_sample_id("legacy_tp1", "A", 12, 0) == "legacy_tp1/A/12/000000"


def test_sample_id_round_trip():
    for args in [("vl_matched", "C", 0, 42), ("legacy_tp1", "E", 7, 999_999)]:
        parsed = parse_sample_id(make_sample_id(*args))
        assert (parsed["benchmark_variant"], parsed["scenario"], parsed["data_seed"], parsed["sample_index"]) == args


def test_sample_ids_sort_in_index_order():
    ids = make_sample_ids("vl_matched", "C", 0, 1500)
    assert ids == sorted(ids)


@pytest.mark.parametrize(
    "args",
    [
        ("other_variant", "C", 0, 1),
        ("vl_matched", "F", 0, 1),
        ("vl_matched", "c", 0, 1),
        ("vl_matched", "C", -1, 1),
        ("vl_matched", "C", 0, -1),
        ("vl_matched", "C", 0, 1_000_000),
    ],
)
def test_invalid_sample_id_inputs_are_rejected(args):
    with pytest.raises(ValueError):
        make_sample_id(*args)


@pytest.mark.parametrize(
    "bad",
    ["vl_matched/C/0/42", "vl_matched/C/00/000042", "vl_matched/G/0/000042", "x/C/0/000042", "", None],
)
def test_malformed_sample_ids_are_rejected(bad):
    with pytest.raises(ValueError):
        parse_sample_id(bad)


def test_manifest_has_contract_columns_and_dtypes():
    m = toy_manifest()
    assert tuple(m.columns) == MANIFEST_COLUMNS
    assert m["data_seed"].dtype == "int64"
    assert m["target"].dtype == "int64"
    assert m["sample_id"].is_unique


def test_manifest_is_byte_identical_across_builds(tmp_path):
    h1 = write_manifest(toy_manifest(), tmp_path / "a.csv")
    h2 = write_manifest(toy_manifest(), tmp_path / "b.csv")
    assert (tmp_path / "a.csv").read_bytes() == (tmp_path / "b.csv").read_bytes()
    assert h1 == h2 == manifest_hash(toy_manifest())


def test_manifest_bytes_use_lf_and_utf8():
    data = manifest_to_csv_bytes(toy_manifest())
    assert b"\r" not in data
    assert data.decode("utf-8").splitlines()[0] == ",".join(MANIFEST_COLUMNS)


def test_manifest_hash_golden_value():
    # Same bytes on every OS and pandas version: proves cross-machine reproducibility.
    assert manifest_hash(toy_manifest()) == GOLDEN_TOY_MANIFEST_HASH


def test_manifest_round_trip(tmp_path):
    path = tmp_path / "nested" / "manifest.csv"
    written_hash = write_manifest(toy_manifest(), path)
    loaded = read_manifest(path)
    pd.testing.assert_frame_equal(loaded, toy_manifest())
    assert manifest_hash(loaded) == written_hash


def test_row_order_does_not_change_the_hash():
    shuffled = toy_manifest().sample(frac=1.0, random_state=3)
    assert manifest_hash(shuffled) == manifest_hash(toy_manifest())


def test_data_seed_changes_ids_and_hash():
    assert manifest_hash(toy_manifest(0)) != manifest_hash(toy_manifest(1))


def test_manifest_rejects_bad_content():
    with pytest.raises(ValueError, match="same length"):
        build_manifest("vl_matched", "C", 0, ["train"], [0, 1])
    with pytest.raises(ValueError, match="unknown split"):
        build_manifest("vl_matched", "C", 0, ["train", "dev"], [0, 1])
    with pytest.raises(ValueError, match="0 or 1"):
        build_manifest("vl_matched", "C", 0, ["train", "test"], [0, 2])

    duplicated = pd.concat([toy_manifest(), toy_manifest().iloc[:1]])
    with pytest.raises(ValueError, match="duplicate"):
        validate_manifest(duplicated)

    extra = toy_manifest().assign(latent_b_img=0)
    with pytest.raises(ValueError, match="unexpected columns"):
        validate_manifest(extra)

    inconsistent = toy_manifest()
    inconsistent.loc[0, "scenario"] = "A"
    with pytest.raises(ValueError, match="disagrees"):
        validate_manifest(inconsistent)


# --------------------------------------------------------------------------- #
# Fixed splits and nested training subsets
# --------------------------------------------------------------------------- #
GOLDEN_SPLIT_MANIFEST_HASH = "a8bb570031360744fe405db190d92a6d00ec7ea91a74c4a118892100d111c900"
SUBSET_FRACTIONS = [0.05, 0.1, 0.25, 0.5, 1.0]


def balanced_targets(n=5000, seed=0):
    return make_rng(seed, "latents").integers(0, 2, n)


def split_manifest(split_seed=0, n=5000):
    return make_split_manifest("vl_matched", "C", 0, balanced_targets(n), split_seed=split_seed)


def test_splits_are_deterministic():
    assert manifest_hash(split_manifest()) == manifest_hash(split_manifest())


def test_split_seed_changes_assignment_but_not_ids():
    a, b = split_manifest(0), split_manifest(1)
    assert list(a["sample_id"]) == list(b["sample_id"])
    assert list(a["split"]) != list(b["split"])


def test_split_sizes_and_stratification():
    m = split_manifest()
    sizes = m["split"].value_counts()
    assert abs(sizes["train"] - 3000) <= 2
    assert abs(sizes["validation"] - 1000) <= 2
    assert abs(sizes["test"] - 1000) <= 2
    overall = m["target"].mean()
    for name in ("train", "validation", "test"):
        assert abs(m.loc[m["split"] == name, "target"].mean() - overall) < 0.005


def test_split_manifest_golden_value():
    targets = [(i * 7) % 3 % 2 for i in range(200)]
    m = make_split_manifest("vl_matched", "C", 0, targets)
    assert manifest_hash(m) == GOLDEN_SPLIT_MANIFEST_HASH


@pytest.mark.parametrize(
    "bad",
    [
        (0.6, 0.2),
        (0.6, 0.3, 0.2),
        (0.8, 0.2, 0.0),
        (1.0, 0.0, 0.0),
        (0.7, 0.4, -0.1),
        {"train": 0.6, "val": 0.2, "test": 0.2},
        (0.6, float("nan"), 0.2),
    ],
)
def test_invalid_split_fractions_fail_clearly(bad):
    with pytest.raises(ValueError):
        validate_split_fractions(bad)


def test_split_fractions_accept_mapping_and_sequence():
    expected = {"train": 0.6, "validation": 0.2, "test": 0.2}
    assert validate_split_fractions(expected) == expected
    assert validate_split_fractions((0.6, 0.2, 0.2)) == expected


def test_assign_splits_rejects_bad_targets():
    with pytest.raises(ValueError):
        assign_splits([0, 1, 2, 1, 0, 1])
    with pytest.raises(ValueError):
        assign_splits([])
    with pytest.raises(ValueError, match="empty"):
        assign_splits([0, 1])


def test_subsets_are_nested_and_complete():
    m = split_manifest()
    subsets = make_nested_train_subsets(m, SUBSET_FRACTIONS, seed=0)
    assert list(subsets) == ["0.05", "0.1", "0.25", "0.5", "1"]
    ordered = list(subsets.values())
    for small, large in zip(ordered, ordered[1:]):
        assert set(small) < set(large)
    train_ids = set(m.loc[m["split"] == "train", "sample_id"])
    assert set(subsets["1"]) == train_ids


def test_subsets_are_class_balanced():
    m = split_manifest()
    train = m[m["split"] == "train"].set_index("sample_id")
    train_rate = train["target"].mean()
    subsets = make_nested_train_subsets(m, SUBSET_FRACTIONS, seed=0)
    for ids in subsets.values():
        n_pos = int(train.loc[ids, "target"].sum())
        # Within one sample per class of the exact proportional allocation.
        assert abs(n_pos - train_rate * len(ids)) <= 1.0


def test_subsets_never_touch_validation_or_test():
    m = split_manifest()
    held_out = set(m.loc[m["split"] != "train", "sample_id"])
    for ids in make_nested_train_subsets(m, SUBSET_FRACTIONS, seed=0).values():
        assert held_out.isdisjoint(ids)


def test_subsets_are_deterministic_and_seeded():
    m = split_manifest()
    a = make_nested_train_subsets(m, SUBSET_FRACTIONS, seed=0)
    b = make_nested_train_subsets(m, SUBSET_FRACTIONS, seed=0)
    c = make_nested_train_subsets(m, SUBSET_FRACTIONS, seed=1)
    assert a == b
    assert a["0.1"] != c["0.1"]
    assert a["1"] == c["1"]


def test_subsets_do_not_depend_on_requested_fractions():
    m = split_manifest()
    alone = make_nested_train_subsets(m, [0.1], seed=0)
    together = make_nested_train_subsets(m, SUBSET_FRACTIONS, seed=0)
    assert alone["0.1"] == together["0.1"]


def test_subsets_are_sorted_by_sample_index():
    m = split_manifest()
    for ids in make_nested_train_subsets(m, SUBSET_FRACTIONS, seed=0).values():
        assert ids == sorted(ids)


@pytest.mark.parametrize("bad", [[0.0], [1.5], [-0.1], [], [0.1, 0.1], [float("nan")]])
def test_invalid_subset_fractions_fail_clearly(bad):
    with pytest.raises(ValueError):
        make_nested_train_subsets(split_manifest(n=200), bad, seed=0)