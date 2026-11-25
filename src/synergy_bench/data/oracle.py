"""Text-aware Bayes oracle for ``vl_matched``.

Inputs per sample: the noisy image observation ``obs_img`` and the binary
``text_category`` that the caption actually states. The continuous text
observation is never read. Generative model (see ``vl_matched``):

    obs_img = z_img + sigma_img * e1        text_category = 1[z_txt + sigma_txt * e2 > 0]

A/E  z_img = z_txt = s ~ N(0,1),           y = 1[s > 0]
B/D  z_img, z_txt ~ N(0,1) independent,   y = 1[z_img + z_txt > 0]
C    z = 2b - 1, b ~ Bernoulli(1/2),       y = b_img XOR b_txt

For C the posterior is closed form. For A/B/D/E P(y=1 | obs_img, c) increases
with obs_img, so the Bayes rule is ``obs_img > tau_c``; tau is found once by
bisection on a 1-D quadrature of the posterior (tau_0 = -tau_1 by symmetry).
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

ORACLE_INFO = {
    "name": "text_aware_oracle",
    "inputs": ["obs_img", "text_category"],
    "rule": "Bayes decision for y given obs_img and the caption category under the declared generative model",
    "limitations": [
        "obs_img is the latent image observation before pixel noise and rendering, so the oracle is an "
        "optimistic ceiling for image models (same convention as the legacy latent_observation_oracle)",
        "it assumes the generative model and noise levels are known exactly",
        "the text side is exact: it reads only the category stated by the caption, not obs_txt",
    ],
}

_GRID = np.linspace(-8.0, 8.0, 4001)
_erf = np.vectorize(math.erf, otypes=[float])


def _phi(x):
    return 0.5 * (1.0 + _erf(np.asarray(x, dtype=float) / math.sqrt(2.0)))


def _text_likelihood(z: np.ndarray, category: int, sigma_txt: float) -> np.ndarray:
    """P(text_category | z_txt = z)."""
    if sigma_txt == 0:
        p1 = (z > 0).astype(float)
    else:
        p1 = _phi(z / sigma_txt)
    return p1 if category == 1 else 1.0 - p1


def posterior_y1(scenario: str, obs_img: float, category: int, sigma_img: float, sigma_txt: float) -> float:
    """P(y = 1 | obs_img, text_category) for one sample."""
    if sigma_img <= 0:
        raise ValueError("sigma_img must be > 0")
    if scenario == "C":
        p_img = 1.0 / (1.0 + math.exp(-2.0 * obs_img / sigma_img**2))
        q = 1.0 if sigma_txt == 0 else float(_phi(1.0 / sigma_txt))
        p_txt = q if category == 1 else 1.0 - q
        return p_img * (1 - p_txt) + (1 - p_img) * p_txt
    g = _GRID
    with np.errstate(divide="ignore"):
        log_lik_txt = np.log(_text_likelihood(g, category, sigma_txt))
    if scenario in ("A", "E"):
        logw = -0.5 * g**2 - (obs_img - g) ** 2 / (2 * sigma_img**2) + log_lik_txt
        w = np.exp(logw - logw.max())
        return float(np.sum(w * np.heaviside(g, 0.5)) / w.sum())  # half weight at s = 0 keeps the symmetry
    if scenario in ("B", "D"):
        logw = -0.5 * g**2 + log_lik_txt  # posterior of z_txt given the caption
        w = np.exp(logw - logw.max())
        mean, var = obs_img / (1 + sigma_img**2), sigma_img**2 / (1 + sigma_img**2)
        return float(np.sum(w * _phi((mean + g) / math.sqrt(var))) / w.sum())
    raise ValueError(f"unknown scenario {scenario!r}")


@lru_cache(maxsize=64)
def decision_threshold(scenario: str, sigma_img: float, sigma_txt: float) -> float:
    """tau_1 such that the oracle predicts 1 iff obs_img > tau_1 when text_category = 1."""
    lo, hi = -20.0, 20.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if posterior_y1(scenario, mid, 1, sigma_img, sigma_txt) > 0.5:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def text_aware_oracle(scenario: str, obs_img, text_category, sigma_img: float, sigma_txt: float) -> np.ndarray:
    """Bayes prediction (0/1, int64) from obs_img and the caption category."""
    obs_img = np.asarray(obs_img, dtype=float)
    cat = np.asarray(text_category).astype(bool)
    if scenario == "C":
        return ((obs_img > 0) != cat).astype(np.int64)  # equals posterior > 1/2 since flip prob < 1/2
    tau = decision_threshold(scenario, float(sigma_img), float(sigma_txt))
    return (obs_img > np.where(cat, tau, -tau)).astype(np.int64)


def c_expected_accuracy(sigma_img: float, sigma_txt: float) -> float:
    """Exact accuracy of the C oracle: a_i a_t + (1 - a_i)(1 - a_t), a = Phi(1 / sigma)."""
    a_i = float(_phi(1.0 / sigma_img)) if sigma_img > 0 else 1.0
    a_t = float(_phi(1.0 / sigma_txt)) if sigma_txt > 0 else 1.0
    return a_i * a_t + (1 - a_i) * (1 - a_t)
