"""The committed integration fixture matches the generator and the documented contract."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from synergy_bench.data import build_dataset, load_split
from synergy_bench.data.config import config_from_dict, config_hash
from synergy_bench.data.fixture import FIXTURE_SCENARIOS, ITEM_KEYS, PER_SPLIT, _config_path, load_fixture_items
from synergy_bench.data.manifest import SPLITS
from synergy_bench.data.render import RENDER_VERSION

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "data_contract"


@pytest.fixture(scope="module")
def contract():
    return json.loads((FIXTURE / "contract.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def npz():
    with np.load(FIXTURE / "images.npz") as f:
        return {"images": f["images"], "sample_ids": f["sample_ids"]}


def test_fixture_is_small_and_complete(contract, npz):
    assert sum(p.stat().st_size for p in FIXTURE.iterdir()) < 2 * 2**20
    assert set(contract["scenarios"]) == set(FIXTURE_SCENARIOS)
    n = len(FIXTURE_SCENARIOS) * len(SPLITS) * PER_SPLIT
    assert npz["images"].shape == (n, 224, 224) and npz["images"].dtype == np.uint8
    for sc in contract["scenarios"].values():
        assert [it["split"] for it in sc["items"]] == [s for s in SPLITS for _ in range(PER_SPLIT)]


def test_contract_fields_match_documentation(contract):
    assert contract["item_keys"] == ITEM_KEYS
    assert contract["image"]["shape"] == [3, 224, 224] and contract["image"]["dtype"] == "float32"
    assert contract["render_version"] == RENDER_VERSION
    bundle = build_dataset(_config_path("C"))
    assert list(load_split(bundle, "test")[0]) == ITEM_KEYS


@pytest.mark.parametrize("scenario", FIXTURE_SCENARIOS)
def test_fixture_items_follow_the_contract_without_the_generator(scenario):
    items = load_fixture_items(FIXTURE, scenario)
    for item in items:
        assert list(item) == ITEM_KEYS
        assert item["image"].dtype == torch.float32 and tuple(item["image"].shape) == (3, 224, 224)
        assert item["target"].dtype == torch.int64 and item["target"].dim() == 0
        assert item["availability"] == {"image": True, "text": True}
        assert item["metadata"] == {}
    assert load_fixture_items(FIXTURE, scenario, with_metadata=True)[0]["metadata"]["render_version"] == RENDER_VERSION


@pytest.mark.parametrize("scenario", FIXTURE_SCENARIOS)
def test_fixture_matches_regenerated_dataset(contract, npz, scenario):
    rec = contract["scenarios"][scenario]
    cfg = config_from_dict(rec["config"])
    assert config_hash(cfg) == rec["config_hash"]
    bundle = build_dataset(_config_path(scenario))
    assert bundle.config == cfg
    assert bundle.metadata["manifest_hash"] == rec["manifest_hash"]
    assert bundle.metadata["split_counts"] == rec["split_counts"]
    for split in SPLITS:
        ds = load_split(bundle, split, with_metadata=True)
        expected = [it for it in rec["items"] if it["split"] == split]
        for k, it in enumerate(expected):
            item = ds[k]
            assert item["sample_id"] == it["sample_id"] == npz["sample_ids"][it["image_index"]]
            assert int(item["target"]) == it["target"]
            assert item["text"] == it["text"]
            assert item["availability"] == it["availability"]
            assert item["metadata"] == it["metadata"]
            fresh = bundle.gray_image(int(ds.indices[k])).astype(int)
            stored = npz["images"][it["image_index"]].astype(int)
            # exp/cos may differ in the last bit across CPUs; allow one uint8 level
            assert np.max(np.abs(fresh - stored)) <= contract["image_tolerance_uint8"]


def test_e_fixture_mirrors_a(contract):
    a, e = contract["scenarios"]["A"], contract["scenarios"]["E"]
    assert [it["text"] for it in a["items"]] == [it["text"] for it in e["items"]]
    assert [it["target"] for it in a["items"]] == [it["target"] for it in e["items"]]
    assert all(it["sample_id"].startswith("vl_matched/E/") for it in e["items"])
