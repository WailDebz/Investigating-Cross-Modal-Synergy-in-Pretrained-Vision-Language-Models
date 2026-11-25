"""Matched image-text variants A-E (``vl_matched``): latents, captions and oracle.

The text channel replaces the legacy signal channel; signal and tabular are not
generated. Each channel sees its own noisy observation of its latent:

    obs_img = z_img + sigma_img * e_img        (rendered as the blob position)
    obs_txt = z_txt + sigma_txt * e_txt        (caption states 1[obs_txt > 0])

sigma_img is the legacy image sigma; sigma_txt is the legacy signal sigma times
``text.noise``.

    A, E  z_img = z_txt = s ~ N(0,1)      y = 1[s > 0]
    B     z_img, z_txt ~ N(0,1)           y = 1[z_img + z_txt > 0]   (legacy weights 1, 1)
    C     z = 2b - 1, b ~ Bernoulli(1/2)  y = b_img XOR b_txt
    D     as B, clean image / noisier text

Every sample draws a fixed set of values from its own stream
``make_rng(data_seed, "latents", index)``, so a sample's latents never depend on
n_samples, on the scenario's draw count or on text settings.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .manifest import make_sample_ids
from .oracle import text_aware_oracle
from .seeds import make_rng
from .text import draw_template_ids, render_captions, text_category

# base (image, text) latent noise per scenario, from the legacy (img, sig) sigmas
BASE_SIGMA = {"A": (0.8, 0.8), "B": (0.5, 0.5), "C": (0.6, 0.6), "D": (0.2, 0.6), "E": (0.8, 0.8)}

LATENT_COLUMNS = (
    "sample_id", "z_img", "z_txt", "obs_img", "obs_txt", "nuisance",
    "text_category", "template_id", "text", "oracle_pred",
)


def channel_sigmas(scenario: str, text_noise: float) -> tuple[float, float]:
    s_img, s_txt = BASE_SIGMA[scenario]
    return s_img, s_txt * text_noise


def pixel_noise_std(scenario: str) -> float:
    """Legacy pixel noise rule 0.1 + 0.2 * sigma_img."""
    return 0.1 + 0.2 * BASE_SIGMA[scenario][0]


def _draw(data_seed: int, n: int) -> np.ndarray:
    """(n, 7): u0, u1, bit0, bit1, e_img, e_txt, nuisance."""
    out = np.empty((n, 7))
    for i in range(n):
        rng = make_rng(data_seed, "latents", i)
        out[i, 0:2] = rng.normal(size=2)
        out[i, 2:4] = rng.integers(0, 2, size=2)
        out[i, 4:7] = rng.normal(size=3)
    return out


def generate_latents(
    scenario: str,
    n_samples: int,
    data_seed: int,
    text_noise: float = 1.0,
    template_set: str = "v1",
    sigma_img: float | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Latent manifest (columns LATENT_COLUMNS) and targets for one scenario.

    ``sigma_img`` overrides the scenario's image sigma (analysis only, e.g. oracle sweeps).
    """
    if scenario not in BASE_SIGMA:
        raise ValueError(f"unknown scenario {scenario!r}")
    s_img, s_txt = channel_sigmas(scenario, text_noise)
    if sigma_img is not None:
        s_img = sigma_img
    d = _draw(data_seed, n_samples)
    u0, u1, b0, b1, e_img, e_txt, nuis = d.T
    if scenario in ("A", "E"):
        z_img = z_txt = u0
        y = u0 > 0
    elif scenario in ("B", "D"):
        z_img, z_txt = u0, u1
        y = u0 + u1 > 0
    else:
        z_img, z_txt = 2 * b0 - 1, 2 * b1 - 1
        y = b0.astype(bool) ^ b1.astype(bool)
    obs_img = z_img + s_img * e_img
    obs_txt = z_txt + s_txt * e_txt
    category = text_category(obs_txt)
    template_ids = draw_template_ids(data_seed, range(n_samples), template_set)
    latents = pd.DataFrame({
        "sample_id": make_sample_ids("vl_matched", scenario, data_seed, n_samples),
        "z_img": z_img, "z_txt": z_txt, "obs_img": obs_img, "obs_txt": obs_txt, "nuisance": nuis,
        "text_category": category, "template_id": template_ids,
        "text": render_captions(category, template_ids, template_set),
        "oracle_pred": text_aware_oracle(scenario, obs_img, category, s_img, s_txt),
    }, columns=list(LATENT_COLUMNS))
    return latents, y.astype(np.int64)
