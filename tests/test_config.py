"""Tests for the validated data config loader and its canonical hash."""
import copy
import dataclasses
from pathlib import Path

import pytest

from synergy_bench.data.config import (
    ConfigError,
    DataConfig,
    config_from_dict,
    config_hash,
    config_to_canonical_json,
    load_data_config,
)

REPO = Path(__file__).resolve().parents[1]
VL_MATCHED_C = REPO / "configs" / "phase1" / "data" / "vl_matched_C.yaml"
VL_MATCHED_C_HASH = "41e59e9e319dd0f45c09108735b0fbe0c9b7010606611fc3555d494573b4049b"

BASE = {
    "benchmark_variant": "vl_matched",
    "scenario": "C",
    "n_samples": 5000,
    "data_seed": 0,
    "split_seed": 0,
    "image": {"size": 224, "channels": 3, "render_chunk_size": 128},
    "text": {"noise": 1.0, "template_set": "v1"},
    "split": {"train": 0.6, "validation": 0.2, "test": 0.2},
}


def _with(path: str, value):
    data = copy.deepcopy(BASE)
    *parents, leaf = path.split(".")
    node = data
    for key in parents:
        node = node[key]
    node[leaf] = value
    return data


def _without(path: str):
    data = copy.deepcopy(BASE)
    *parents, leaf = path.split(".")
    node = data
    for key in parents:
        node = node[key]
    del node[leaf]
    return data


def test_reference_yaml_hash():
    assert config_hash(load_data_config(VL_MATCHED_C)) == VL_MATCHED_C_HASH


def test_reference_yaml_matches_plan_values():
    assert dataclasses.asdict(load_data_config(VL_MATCHED_C)) == BASE


def test_canonical_json_is_sorted_compact_and_float_typed():
    text = config_to_canonical_json(config_from_dict(BASE))
    assert " " not in text
    assert text.startswith('{"benchmark_variant":"vl_matched","data_seed":0,')
    assert '"noise":1.0' in text and '"train":0.6' in text


def test_int_noise_and_fractions_are_coerced_to_float():
    data = _with("text.noise", 1)
    cfg = config_from_dict(data)
    assert isinstance(cfg.text.noise, float)
    assert config_hash(cfg) == VL_MATCHED_C_HASH


def test_roundtrip_and_key_order_do_not_change_hash():
    cfg = config_from_dict(BASE)
    again = config_from_dict(dataclasses.asdict(cfg))
    reordered = dict(reversed(list(BASE.items())))
    assert again == cfg
    assert config_hash(again) == config_hash(config_from_dict(reordered)) == config_hash(cfg)


def test_any_value_change_changes_hash():
    assert config_hash(config_from_dict(_with("data_seed", 1))) != VL_MATCHED_C_HASH
    assert config_hash(config_from_dict(_with("image.render_chunk_size", 64))) != VL_MATCHED_C_HASH


def test_config_is_frozen():
    cfg = config_from_dict(BASE)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.scenario = "A"
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.image.size = 16
    assert isinstance(cfg, DataConfig)


@pytest.mark.parametrize("path", ["scenario", "image", "image.size", "text.noise", "split.test"])
def test_missing_field_is_named(path):
    with pytest.raises(ConfigError, match=rf"missing field {path}"):
        config_from_dict(_without(path))


@pytest.mark.parametrize("path", ["model_seed", "image.color", "text.extra", "split.holdout"])
def test_unknown_field_is_named(path):
    with pytest.raises(ConfigError, match=rf"unknown field {path}"):
        config_from_dict(_with(path, 1))


@pytest.mark.parametrize(
    "path, value",
    [
        ("n_samples", "5000"),
        ("n_samples", 5000.0),
        ("data_seed", True),
        ("image.size", 224.0),
        ("text.noise", "1.0"),
        ("text.noise", True),
        ("text.template_set", 1),
        ("image", [224, 3, 128]),
    ],
)
def test_wrong_type_is_named(path, value):
    with pytest.raises(ConfigError, match=path):
        config_from_dict(_with(path, value))


@pytest.mark.parametrize(
    "path, value",
    [
        ("benchmark_variant", "extended_four_modal"),
        ("scenario", "F"),
        ("n_samples", 0),
        ("n_samples", 10**6),
        ("data_seed", -1),
        ("split_seed", -1),
        ("image.size", 0),
        ("image.channels", 2),
        ("image.render_chunk_size", 0),
        ("text.noise", -0.1),
        ("text.noise", float("nan")),
        ("text.template_set", ""),
    ],
)
def test_invalid_value_is_named(path, value):
    with pytest.raises(ConfigError, match=path):
        config_from_dict(_with(path, value))


@pytest.mark.parametrize("path, value", [("split.train", 0.7), ("split.test", 0.0), ("split.test", 1.0)])
def test_invalid_split_fractions(path, value):
    with pytest.raises(ConfigError, match="split"):
        config_from_dict(_with(path, value))


def test_non_mapping_config_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="config must be a mapping"):
        load_data_config(path)
