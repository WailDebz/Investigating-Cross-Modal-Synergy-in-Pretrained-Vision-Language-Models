"""Regression coverage of the frozen TP1 generator through the thin legacy adapter."""
import dataclasses
import hashlib

import numpy as np
import pytest

from synergy_bench.data import legacy as L
from synergy_bench.data.config import config_from_dict
from synergy_bench.data.manifest import manifest_to_csv_bytes, parse_sample_id

SCENARIOS = "ABCDE"
LATENT_KEYS = {"A": {"s"}, "B": {"u_img", "u_sig", "u_tab"}, "C": {"b_img", "b_sig"},
               "D": {"u_img", "u_sig", "u_tab"}, "E": {"s"}}

# Golden SHA-256 of RNG-only arrays (n=200, seed=0). Rendered float arrays use
# exp/cos whose last bits may vary across CPUs, so they are checked by sums.
GOLDEN_SHA256 = {
    "A": {"y": "363313a25993488acf07f4672f0769d29157d48e07ab92c2664b3b413a3e71b2",
          "obs": "804b593c9d44efc0ae3429655f2cddc68926b64a604b86de5e1721db471f7742",
          "oracle": "9f591cd41c5c9cbf842fa257d8a5e3cd5a675bcb10bab14ab03ab5d1669afeae",
          "s": "399e37bff38237be63c88ca2cdbdfb4442891f1c0568b7bfca25a5def8fdd7a5"},
    "B": {"y": "094c72985794625e4de310690ad70b529adffdcf2537f05bc7f8ef2fc7f89f62",
          "obs": "c731242101d7ad645e9d5b39db83d5a79c529c02a9461246b6b14f7cfcff72e6",
          "oracle": "a539627415ef160d2ef7e684dea0ffdcc5c15210220becbe68e6b2c2c101e38f",
          "u_img": "8ca99f97e1eb892008b494375305897f8b2dfd171218c140a74219095200e0cc",
          "u_sig": "2b30b73b0f9de539b2665c63da5405b4731b83477e462ffb28a96397de92927b",
          "u_tab": "d0b4db67e7ca63dca583de874074484c19c703d0dff7bca47c5ce388baa1710b"},
    "C": {"y": "a40933631096789c30dd795a1485d1e7b2e4d0ae1780a95847211b123e2d9a4a",
          "obs": "53354514015f1ecbac2b065809889ec85d92e3ff0737333c6b8bf78ab91cd20e",
          "oracle": "be7581d2dcd49eb20076ef7af71a276ea530a0113640da1307db7bd0371f192a",
          "b_img": "45bc3e1190ffcfb9cc8fde9444ad10e06410a5545e34d636d19cce0ad7dac8f8",
          "b_sig": "a2a049529d5e7fb4e1cfcbbc465a2b87e94ced197089fbc7a6b3560289c8f610"},
    "D": {"y": "d5d37aed0ab9824a8be472f3b7372169fc418a317e4b06e8d8764a55cceeab92",
          "obs": "1ea7d1f032a9aa7a651674507801264c43cbae2ac54de20a0faa747555ac9fce",
          "oracle": "69b0f23c735961a8e41c79761128ae1a62c7bd6560ec126e80f6b37ce35e2f40",
          "u_img": "8ca99f97e1eb892008b494375305897f8b2dfd171218c140a74219095200e0cc",
          "u_sig": "2b30b73b0f9de539b2665c63da5405b4731b83477e462ffb28a96397de92927b",
          "u_tab": "d0b4db67e7ca63dca583de874074484c19c703d0dff7bca47c5ce388baa1710b"},
}
GOLDEN_SHA256["E"] = GOLDEN_SHA256["A"]
GOLDEN_SUMS = {  # float64 sums of img, sig, tab (n=200, seed=0)
    "A": (4128.95451281256, 18.80117677258204, 35.80479691363871),
    "B": (4176.71965147521, -27.564347426722634, 117.72194754716475),
    "C": (4112.675157207956, -99.04150834553002, 250.80131727200933),
    "D": (4214.7128390292455, 25.175601585498953, 109.55574578023516),
}
GOLDEN_SUMS["E"] = GOLDEN_SUMS["A"]
C_UNIMODAL_N5000 = [0.4998, 0.4934, 0.502]


def _sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


@pytest.fixture(scope="module")
def n5000():
    return {sc: L.make_legacy_dataset(sc, 5000, 0) for sc in SCENARIOS}


# --------------------------------------------------------------------------- #
# Frozen files
# --------------------------------------------------------------------------- #
def test_legacy_hashes_match_constants():
    for name, expected in L.LEGACY_SHA256.items():
        assert L.normalized_sha256(L.LEGACY_DIR / name) == expected


def test_legacy_hashes_match_sha256sums_file():
    recorded = {}
    for line in (L.LEGACY_DIR / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split()
            recorded[name.lstrip("*")] = digest
    assert recorded == L.LEGACY_SHA256


def test_crlf_normalization_gives_lf_hash(tmp_path):
    lf, crlf = tmp_path / "lf.txt", tmp_path / "crlf.txt"
    lf.write_bytes(b"a\nb\n")
    crlf.write_bytes(b"a\r\nb\r\n")
    assert L.normalized_sha256(lf) == L.normalized_sha256(crlf) == hashlib.sha256(b"a\nb\n").hexdigest()


def test_loading_does_not_write_into_legacy_folder():
    L.load_legacy_module()
    assert not (L.LEGACY_DIR / "__pycache__").exists()


# --------------------------------------------------------------------------- #
# Shapes, balance, latents
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_shapes_dtypes_latent_keys_and_balance(n5000, scenario):
    d = n5000[scenario]
    assert d.img.shape == (5000, 1, 16, 16) and d.img.dtype == np.float32
    assert d.sig.shape == (5000, 1, 128) and d.sig.dtype == np.float32
    assert d.tab.shape == (5000, 10) and d.tab.dtype == np.float32
    assert d.y.dtype == np.int64 and set(np.unique(d.y)) == {0, 1}
    assert d.obs.shape == (5000, 3)
    assert set(d.latents) == LATENT_KEYS[scenario]
    assert abs(d.y.mean() - 0.5) < 0.02


# --------------------------------------------------------------------------- #
# Oracles and stored notebook values
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_latent_observation_oracle_reference(n5000, scenario):
    assert L.latent_observation_oracle(n5000[scenario]) == pytest.approx(
        L.LEGACY_ORACLE_N5000[scenario], abs=L.DATA_METRIC_TOLERANCE)


def test_scenario_c_each_channel_near_chance(n5000):
    uni = L.unimodal_latent_oracle(n5000["C"])
    assert uni == pytest.approx(C_UNIMODAL_N5000, abs=1e-4)
    assert all(abs(a - 0.5) < 0.02 for a in uni)
    b = n5000["C"].latents
    assert abs(b["b_img"].mean() - 0.5) < 0.02 and abs(b["b_sig"].mean() - 0.5) < 0.02


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_notebook_unimodal_oracle(n5000, scenario):
    ref = L.NOTEBOOK_REFERENCE_METRICS["data"]["unimodal_oracle_n5000"][scenario]
    assert L.unimodal_latent_oracle(n5000[scenario]) == pytest.approx(ref, abs=L.DATA_METRIC_TOLERANCE)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_notebook_summary_n4000(scenario):
    d = L.make_legacy_dataset(scenario, 4000, 0)
    data = L.NOTEBOOK_REFERENCE_METRICS["data"]
    assert L.latent_observation_oracle(d) == pytest.approx(data["summary_n4000_oracle"][scenario],
                                                           abs=L.DATA_METRIC_TOLERANCE)
    assert d.y.mean() == pytest.approx(data["summary_n4000_p_y1"][scenario], abs=0.005)


@pytest.mark.parametrize("scenario", "ABCD")
def test_notebook_test_split_oracle(scenario):
    gen = L.load_legacy_module()
    _, _, te = gen.split_dataset(L.make_legacy_dataset(scenario, 6000, 0), (0.6, 0.2, 0.2), seed=0)
    ref = L.NOTEBOOK_REFERENCE_METRICS["data"]["test_oracle_n6000"][scenario]
    assert L.latent_observation_oracle(te) == pytest.approx(ref, abs=L.DATA_METRIC_TOLERANCE)


def test_model_reference_metrics_are_well_formed():
    models = L.NOTEBOOK_REFERENCE_METRICS["models"]
    assert set(models["sklearn_grid"]) == set(models["torch_suite"]) == {"A", "B", "C"}
    for table in models.values():
        for row in table.values():
            assert all(0.0 <= v <= 1.0 for v in row.values())
    assert 0 < L.MODEL_METRIC_TOLERANCE < 0.1


# --------------------------------------------------------------------------- #
# Reproducible arrays
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_golden_arrays(scenario):
    d = L.make_legacy_dataset(scenario, 200, 0)
    gold = GOLDEN_SHA256[scenario]
    assert _sha(d.y) == gold["y"]
    assert _sha(d.obs) == gold["obs"]
    assert _sha(d.oracle_pred) == gold["oracle"]
    for key, value in d.latents.items():
        assert _sha(value) == gold[key]
    sums = [float(np.float64(getattr(d, k)).sum()) for k in ("img", "sig", "tab")]
    assert sums == pytest.approx(GOLDEN_SUMS[scenario], rel=1e-6)


def test_same_call_gives_identical_arrays():
    a, b = L.make_legacy_dataset("C", 300, 7), L.make_legacy_dataset("C", 300, 7)
    for key in ("img", "sig", "tab", "y", "obs"):
        assert np.array_equal(getattr(a, key), getattr(b, key))


# --------------------------------------------------------------------------- #
# Contract attachment
# --------------------------------------------------------------------------- #
def _legacy_config(scenario="C", n=600):
    return config_from_dict({
        "benchmark_variant": "legacy_tp1", "scenario": scenario, "n_samples": n,
        "data_seed": 0, "split_seed": 0,
        "image": {"size": 16, "channels": 1, "render_chunk_size": 128},
        "text": {"noise": 1.0, "template_set": "v1"},
        "split": {"train": 0.6, "validation": 0.2, "test": 0.2},
    })


def test_legacy_manifest_contract():
    cfg = _legacy_config()
    manifest = L.legacy_manifest(cfg)
    data = L.make_legacy_dataset("C", 600, 0)
    assert len(manifest) == 600
    assert parse_sample_id(manifest.sample_id[42]) == {
        "benchmark_variant": "legacy_tp1", "scenario": "C", "data_seed": 0, "sample_index": 42}
    assert np.array_equal(manifest.target.to_numpy(), data.y)
    assert manifest_to_csv_bytes(manifest) == manifest_to_csv_bytes(L.legacy_manifest(cfg))


def test_legacy_manifest_rejects_other_variant():
    cfg = dataclasses.replace(_legacy_config(), benchmark_variant="vl_matched")
    with pytest.raises(ValueError, match="legacy_tp1"):
        L.legacy_manifest(cfg)
