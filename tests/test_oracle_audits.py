"""Text-aware oracle behaviour and leakage audits."""
from pathlib import Path

import numpy as np
import pytest

from synergy_bench.data import build_dataset
from synergy_bench.data.audits import CHANCE_TOLERANCE, run_audits
from synergy_bench.data.oracle import (
    c_expected_accuracy,
    decision_threshold,
    posterior_y1,
    text_aware_oracle,
)
from synergy_bench.data.vl_matched import BASE_SIGMA, generate_latents

REPO = Path(__file__).resolve().parents[1]
NOISE_LEVELS = (0.25, 0.5, 1.0, 2.0, 4.0)


def simulate(scenario, n, sigma_img, sigma_txt, seed=0):
    """Vectorized draw of the vl_matched generative model (common random numbers across sigmas)."""
    r = np.random.default_rng(seed)
    u, bits, e = r.normal(size=(n, 2)), r.integers(0, 2, size=(n, 2)), r.normal(size=(n, 2))
    if scenario in "AE":
        z_img = z_txt = u[:, 0]
        y = u[:, 0] > 0
    elif scenario in "BD":
        z_img, z_txt = u[:, 0], u[:, 1]
        y = u.sum(1) > 0
    else:
        z_img, z_txt = 2.0 * bits[:, 0] - 1, 2.0 * bits[:, 1] - 1
        y = bits[:, 0] != bits[:, 1]
    obs_img = z_img + sigma_img * e[:, 0]
    cat = (z_txt + sigma_txt * e[:, 1] > 0).astype(int)
    return obs_img, cat, y.astype(int)


def oracle_acc(scenario, sigma_img, sigma_txt, n=40000):
    obs_img, cat, y = simulate(scenario, n, sigma_img, sigma_txt)
    return float(np.mean(text_aware_oracle(scenario, obs_img, cat, sigma_img, sigma_txt) == y))


# --------------------------------------------------------------------------- #
# Oracle
# --------------------------------------------------------------------------- #
def test_c_expected_accuracy_closed_form_and_monotone():
    assert c_expected_accuracy(0.6, 0.6) == pytest.approx(0.9090, abs=1e-4)
    for fixed in (0.3, 0.6, 1.2):
        txt = [c_expected_accuracy(fixed, s) for s in NOISE_LEVELS]
        img = [c_expected_accuracy(s, fixed) for s in NOISE_LEVELS]
        assert all(a > b for a, b in zip(txt, txt[1:]))
        assert all(a > b for a, b in zip(img, img[1:]))
    assert c_expected_accuracy(1e6, 0.6) == pytest.approx(0.5, abs=1e-5)


def test_c_oracle_matches_closed_form():
    assert oracle_acc("C", 0.6, 0.6) == pytest.approx(c_expected_accuracy(0.6, 0.6), abs=0.01)


@pytest.mark.parametrize("scenario", "ABCDE")
def test_oracle_decreases_monotonically_with_each_channel_noise(scenario):
    s_img, s_txt = BASE_SIGMA[scenario]
    by_txt = [oracle_acc(scenario, s_img, s_txt * k) for k in NOISE_LEVELS]
    by_img = [oracle_acc(scenario, s_img * k, s_txt) for k in NOISE_LEVELS]
    assert all(a >= b - 1e-3 for a, b in zip(by_txt, by_txt[1:])), by_txt
    assert all(a >= b - 1e-3 for a, b in zip(by_img, by_img[1:])), by_img
    assert by_txt[0] - by_txt[-1] > 0.02 and by_img[0] - by_img[-1] > 0.02


@pytest.mark.parametrize("scenario", "ABDE")
def test_threshold_rule_is_bayes_optimal_locally(scenario):
    s_img, s_txt = BASE_SIGMA[scenario]
    obs_img, cat, y = simulate(scenario, 200000, s_img, s_txt, seed=1)
    tau = decision_threshold(scenario, s_img, s_txt)
    assert posterior_y1(scenario, tau, 1, s_img, s_txt) == pytest.approx(0.5, abs=1e-6)
    assert posterior_y1(scenario, -tau, 0, s_img, s_txt) == pytest.approx(0.5, abs=1e-6)

    def acc(t):
        return np.mean((obs_img > np.where(cat == 1, t, -t)) == y)

    best = acc(tau)
    assert best >= acc(tau - 0.3) - 1e-3 and best >= acc(tau + 0.3) - 1e-3


@pytest.mark.parametrize("scenario", "ABCDE")
def test_oracle_beats_single_channels(scenario):
    s_img, s_txt = BASE_SIGMA[scenario]
    obs_img, cat, y = simulate(scenario, 40000, s_img, s_txt)
    oracle = np.mean(text_aware_oracle(scenario, obs_img, cat, s_img, s_txt) == y)
    assert oracle > max(np.mean((obs_img > 0) == y), np.mean(cat == y))


def test_generator_oracle_follows_text_noise():
    accs = []
    for noise in (0.5, 1.0, 3.0):
        latents, y = generate_latents("C", 6000, 0, text_noise=noise)
        accs.append(np.mean(latents.oracle_pred.to_numpy() == y))
    assert accs[0] > accs[1] > accs[2] > 0.5


def test_noiseless_text_is_handled():
    latents, y = generate_latents("A", 2000, 0, text_noise=0.0)
    assert np.array_equal(latents.text_category.to_numpy(), y)
    assert np.mean(latents.oracle_pred.to_numpy() == y) == 1.0


# --------------------------------------------------------------------------- #
# Audits on the reference configs
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def audits():
    return {s: run_audits(build_dataset(REPO / "configs" / "phase1" / "data" / f"vl_matched_{s}.yaml"))
            for s in "ABCDE"}


def test_scenario_c_passes_all_leakage_checks(audits):
    rep = audits["C"]
    assert rep["passed"]
    for acc in (*rep["unimodal_predictability"].values(), *rep["shortcuts"].values()):
        assert abs(acc - 0.5) <= CHANCE_TOLERANCE
    assert rep["oracle_accuracy"]["test"] > 0.85


@pytest.mark.parametrize("scenario", "ABCDE")
def test_balance_templates_lengths_duplicates(audits, scenario):
    rep = audits[scenario]
    assert all(abs(p - 0.5) < 0.03 for p in rep["class_balance"].values())
    assert rep["template_frequencies"]["max_deviation_from_uniform"] < 0.02
    by_cat = np.array(rep["template_frequencies"]["by_category"])
    shares = by_cat / by_cat.sum(0, keepdims=True)  # template distribution within each category
    assert np.max(np.abs(shares[:, 0] - shares[:, 1])) < 0.04
    lengths = rep["caption_lengths"]
    assert lengths["mean_chars_by_category"][0] == pytest.approx(lengths["mean_chars_by_category"][1], abs=0.3)
    assert abs(rep["shortcuts"]["template_id"] - 0.5) <= CHANCE_TOLERANCE
    assert abs(rep["shortcuts"]["caption_tokens"] - 0.5) <= CHANCE_TOLERANCE
    assert rep["duplicates"]["duplicate_sample_ids"] == 0
    assert rep["duplicates"]["duplicate_latent_rows"] == 0
    assert rep["duplicates"]["unique_captions"] == 8


@pytest.mark.parametrize("scenario", "ABDE")
def test_bow_diagnostic_detects_real_text_signal(audits, scenario):
    # outside C the caption is informative, so the diagnostic must have power
    assert audits[scenario]["shortcuts"]["bag_of_words"] > 0.65
