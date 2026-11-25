"""Semantics and contract tests for the matched image-text benchmark (vl_matched)."""
import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

import synergy_bench.data.vl_matched as vlm
from synergy_bench.data import build_dataset, load_manifest, load_split, make_nested_train_subsets
from synergy_bench.data.config import config_from_dict
from synergy_bench.data.manifest import MANIFEST_COLUMNS, manifest_to_csv_bytes, parse_sample_id
from synergy_bench.data.render import RENDER_VERSION
from synergy_bench.data.text import text_category

REPO = Path(__file__).resolve().parents[1]


def make_config(scenario="C", n=120, size=32, chunk=16, noise=1.0, data_seed=0, split_seed=0, channels=3):
    return config_from_dict({
        "benchmark_variant": "vl_matched", "scenario": scenario, "n_samples": n,
        "data_seed": data_seed, "split_seed": split_seed,
        "image": {"size": size, "channels": channels, "render_chunk_size": chunk},
        "text": {"noise": noise, "template_set": "v1"},
        "split": {"train": 0.6, "validation": 0.2, "test": 0.2},
    })


def all_images(bundle):
    return np.stack([bundle.gray_image(i) for i in range(len(bundle.manifest))])


# --------------------------------------------------------------------------- #
# Item contract (plan section 3.2)
# --------------------------------------------------------------------------- #
def test_item_contract_without_metadata():
    bundle = build_dataset(make_config())
    ds = load_split(bundle, "train")
    item = ds[0]
    assert set(item) == {"sample_id", "image", "text", "target", "availability", "metadata"}
    assert isinstance(item["sample_id"], str) and parse_sample_id(item["sample_id"])["scenario"] == "C"
    assert item["image"].dtype == torch.float32 and tuple(item["image"].shape) == (3, 32, 32)
    assert 0.0 <= float(item["image"].min()) and float(item["image"].max()) <= 1.0
    assert isinstance(item["text"], str) and item["text"]
    assert item["target"].dtype == torch.int64 and item["target"].dim() == 0 and int(item["target"]) in (0, 1)
    assert item["availability"] == {"image": True, "text": True}
    assert item["metadata"] == {}


def test_metadata_only_with_flag():
    bundle = build_dataset(make_config())
    item = load_split(bundle, "validation", with_metadata=True)[0]
    meta = item["metadata"]
    for key in ("z_img", "z_txt", "obs_img", "obs_txt", "nuisance", "text_category", "template_id", "oracle_pred"):
        assert key in meta
    assert meta["render_version"] == RENDER_VERSION and meta["split"] == "validation"
    assert meta["text_category"] == int(text_category(meta["obs_txt"]))


def test_rgb_is_replicated_grayscale_and_matches_uint8_render():
    bundle = build_dataset(make_config())
    ds = load_split(bundle, "test")
    img = ds[1]["image"]
    assert torch.equal(img[0], img[1]) and torch.equal(img[1], img[2])
    gray = bundle.gray_image(int(ds.indices[1]))
    assert torch.equal(img[0], torch.from_numpy(gray.astype(np.float32) / 255.0))


def test_single_channel_config():
    bundle = build_dataset(make_config(channels=1))
    assert tuple(load_split(bundle, "train")[0]["image"].shape) == (1, 32, 32)


def test_splits_partition_and_ids_match_manifest():
    bundle = build_dataset(make_config())
    ids = []
    for split in ("train", "validation", "test"):
        ds = load_split(bundle, split)
        assert ds.sample_ids == [ds[k]["sample_id"] for k in range(len(ds))]
        assert (bundle.manifest.set_index("sample_id").loc[ds.sample_ids, "split"] == split).all()
        ids += ds.sample_ids
    assert sorted(ids) == sorted(bundle.manifest.sample_id)
    with pytest.raises(ValueError, match="split"):
        load_split(bundle, "dev")


def test_dataloader_collates_batches():
    bundle = build_dataset(make_config())
    batch = next(iter(DataLoader(load_split(bundle, "train"), batch_size=4)))
    assert tuple(batch["image"].shape) == (4, 3, 32, 32)
    assert batch["target"].dtype == torch.int64 and len(batch["text"]) == 4
    assert batch["availability"]["image"].tolist() == [True] * 4


def test_manifest_has_contract_columns_only_and_latents_are_separate():
    bundle = build_dataset(make_config())
    assert tuple(bundle.manifest.columns) == MANIFEST_COLUMNS
    assert list(bundle.latents.sample_id) == list(bundle.manifest.sample_id)
    assert manifest_to_csv_bytes(load_manifest(make_config())) == manifest_to_csv_bytes(bundle.manifest)


def test_nested_subsets_from_public_api():
    bundle = build_dataset(make_config(n=400))
    subsets = make_nested_train_subsets(bundle.manifest, [0.1, 0.5, 1.0], seed=0)
    assert set(subsets["0.1"]) <= set(subsets["0.5"]) <= set(subsets["1"])


def test_bundle_metadata_declares_render_text_and_oracle():
    meta = build_dataset(make_config()).metadata
    assert meta["render"]["version"] == RENDER_VERSION
    assert meta["render"]["uint8_range"] == [-1.5, 2.5]
    assert meta["oracle"]["name"] == "text_aware_oracle"
    assert meta["oracle"]["inputs"] == ["obs_img", "text_category"]
    assert meta["oracle"]["limitations"]
    assert meta["sigma_img"] == 0.6 and meta["sigma_txt"] == 0.6


# --------------------------------------------------------------------------- #
# Determinism and independence of streams
# --------------------------------------------------------------------------- #
def test_repeated_build_is_identical():
    a, b = build_dataset(make_config()), build_dataset(make_config())
    assert manifest_to_csv_bytes(a.manifest) == manifest_to_csv_bytes(b.manifest)
    assert a.latents.equals(b.latents)
    assert np.array_equal(all_images(a), all_images(b))


def test_same_sample_id_same_image_across_chunk_sizes_and_cache(tmp_path):
    on_demand = all_images(build_dataset(make_config(chunk=16)))
    for chunk in (1, 7, 128):
        cached = build_dataset(make_config(chunk=chunk), cache_dir=tmp_path)
        assert cached.image_cache is not None
        assert np.array_equal(np.asarray(cached.image_cache), on_demand)


def test_sample_content_does_not_depend_on_n_samples():
    small, large = build_dataset(make_config(n=60)), build_dataset(make_config(n=200))
    cols = ["sample_id", "obs_img", "obs_txt", "text", "oracle_pred"]
    assert small.latents[cols].equals(large.latents[cols].iloc[:60])
    assert np.array_equal(all_images(small), all_images(large)[:60])


def test_text_noise_changes_only_text():
    base, noisy = build_dataset(make_config(noise=1.0)), build_dataset(make_config(noise=3.0))
    assert np.array_equal(all_images(base), all_images(noisy))
    for col in ("z_img", "z_txt", "obs_img", "nuisance", "template_id"):
        assert np.array_equal(base.latents[col], noisy.latents[col])
    assert np.array_equal(base.manifest.target, noisy.manifest.target)
    assert not np.array_equal(base.latents.text_category, noisy.latents.text_category)


def test_template_selection_does_not_change_images_or_other_modalities(monkeypatch):
    base = build_dataset(make_config())
    original = vlm.draw_template_ids
    monkeypatch.setattr(vlm, "draw_template_ids", lambda seed, idx, name: original(seed + 99, idx, name))
    other = build_dataset(make_config())
    assert not np.array_equal(base.latents.template_id, other.latents.template_id)
    assert np.array_equal(all_images(base), all_images(other))
    assert base.manifest.equals(other.manifest)
    for col in ("obs_img", "obs_txt", "text_category", "oracle_pred"):
        assert np.array_equal(base.latents[col], other.latents[col])


def test_resolution_changes_only_pixels_not_latents():
    lo, hi = build_dataset(make_config(size=16)), build_dataset(make_config(size=224, n=30))
    cols = ["obs_img", "nuisance", "text", "oracle_pred"]
    assert lo.latents[cols].iloc[:30].equals(hi.latents[cols])
    assert hi.gray_image(0).shape == (224, 224) and lo.gray_image(0).shape == (16, 16)


def test_split_seed_changes_splits_not_samples():
    a, b = build_dataset(make_config(split_seed=0)), build_dataset(make_config(split_seed=1))
    assert list(a.manifest.sample_id) == list(b.manifest.sample_id)
    assert np.array_equal(a.manifest.target, b.manifest.target)
    assert not np.array_equal(a.manifest.split, b.manifest.split)


def test_cache_is_reused(tmp_path):
    cfg = make_config()
    build_dataset(cfg, cache_dir=tmp_path)
    files = list(tmp_path.rglob("images_*.npy"))
    assert len(files) == 1
    mtime = files[0].stat().st_mtime_ns
    build_dataset(cfg, cache_dir=tmp_path)
    assert files[0].stat().st_mtime_ns == mtime
    folder = files[0].parent
    assert {p.name for p in folder.iterdir()} >= {"manifest.csv", "latents.csv", "metadata.json", "config.json"}


# --------------------------------------------------------------------------- #
# Scenario semantics
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scenario", "ABCDE")
def test_reference_configs_balance_and_channel_roles(scenario):
    bundle = build_dataset(REPO / "configs" / "phase1" / "data" / f"vl_matched_{scenario}.yaml")
    y = bundle.manifest.target.to_numpy()
    lat = bundle.latents
    assert len(y) == 5000 and abs(y.mean() - 0.5) < 0.03
    img_acc = np.mean((lat.obs_img > 0) == y)
    txt_acc = np.mean(lat.text_category == y)
    if scenario == "C":
        assert abs(img_acc - 0.5) < 0.03 and abs(txt_acc - 0.5) < 0.03
        assert set(lat.z_img) == set(lat.z_txt) == {-1.0, 1.0}
    else:
        assert img_acc > 0.65 and txt_acc > 0.65
    if scenario in "AE":
        assert np.array_equal(lat.z_img, lat.z_txt)


def test_scenario_e_matches_a_except_ids():
    a, e = build_dataset(make_config("A")), build_dataset(make_config("E"))
    assert np.array_equal(all_images(a), all_images(e))
    assert list(a.latents.text) == list(e.latents.text)
    assert a.manifest.sample_id[0].startswith("vl_matched/A/") and e.manifest.sample_id[0].startswith("vl_matched/E/")


def test_scenario_d_image_cleaner_than_text():
    bundle = build_dataset(make_config("D", n=2000))
    assert bundle.metadata["sigma_img"] < bundle.metadata["sigma_txt"]


# --------------------------------------------------------------------------- #
# Variant handling
# --------------------------------------------------------------------------- #
def test_legacy_variant_manifest_and_build_rejection():
    cfg = dataclasses.replace(make_config(), benchmark_variant="legacy_tp1")
    manifest = load_manifest(cfg)
    assert manifest.sample_id[0].startswith("legacy_tp1/C/0/")
    with pytest.raises(ValueError, match="legacy"):
        build_dataset(cfg)


def test_unknown_template_set_rejected():
    cfg = dataclasses.replace(make_config(), text=dataclasses.replace(make_config().text, template_set="v9"))
    with pytest.raises(ValueError, match="template_set"):
        build_dataset(cfg)
