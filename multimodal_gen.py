"""
multimodal_gen.py -- controlled synthetic multimodal data for teaching fusion.

Three modalities are rendered from latent variables:
    img : (N, 1, H, W)  a Gaussian blob (x-position <- latent slot 1) on top of an
                        oriented grating (orientation <- latent slot 2, random phase)
    sig : (N, 1, T)     a sinusoid whose FREQUENCY <- slot 1 (random phase!) plus a
                        second sinusoid whose amplitude <- slot 2
    tab : (N, d)        noisy linear / tanh features of (slot 1, slot 2)

Slot 1 of each modality is the "informative" slot, slot 2 is a nuisance.
The scenario decides how the label y relates to the slot-1 latents:

    A  Redundancy  : y = 1[s > 0], all three modalities observe the same s
    B  Uniqueness  : y = 1[w . u > 0], each modality observes its own u_m
    C  Synergy     : y = XOR(b_img, b_sig); tab is pure distractor.
                     Each unimodal view is at chance, the pair determines y.
    D  Imbalance   : y = 1[u_img + u_sig > 0]; image is very clean, signal is noisy,
                     tab is a distractor
    E  Corruption  : same generator as A. Use `corrupt()` at test time.

Everything is deterministic given `seed`. The "world" (tabular mixing matrix) is
fixed, so train/val/test generated with different seeds share the same mapping.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

MODALITIES = ("img", "sig", "tab")

SCENARIOS = {
    "A": "Redundancy  - every modality sees the same latent s",
    "B": "Uniqueness  - each modality carries its own part of the label",
    "C": "Synergy     - XOR of a bit in the image and a bit in the signal",
    "D": "Imbalance   - easy/clean image vs. harder/noisier signal",
    "E": "Corruption  - redundant data, corrupt a modality at test time",
}

# per-modality latent noise (img, sig, tab); multiplied by the `noise` argument
_DEFAULT_SIGMA = {
    "A": (0.8, 0.8, 0.8),
    "B": (0.5, 0.5, 0.5),
    "C": (0.6, 0.6, 0.6),
    "D": (0.2, 0.6, 0.6),
    "E": (0.8, 0.8, 0.8),
}
_B_WEIGHTS = np.array([1.0, 1.0, 0.7])
_D_WEIGHTS = np.array([1.0, 1.0, 0.0])
_WORLD_SEED = 12345


# --------------------------------------------------------------------------- #
# Container
# --------------------------------------------------------------------------- #
@dataclass
class MultimodalData:
    img: np.ndarray
    sig: np.ndarray
    tab: np.ndarray
    y: np.ndarray
    scenario: str = ""
    noise: float = 1.0
    sigma: tuple = ()
    obs: np.ndarray = None            # noisy slot-1 observations (N, 3), for analysis
    oracle_pred: np.ndarray = None    # Bayes-optimal prediction from `obs`
    latents: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.y)

    def subset(self, idx):
        idx = np.asarray(idx)
        return MultimodalData(
            self.img[idx], self.sig[idx], self.tab[idx], self.y[idx],
            self.scenario, self.noise, self.sigma,
            None if self.obs is None else self.obs[idx],
            None if self.oracle_pred is None else self.oracle_pred[idx],
            {k: v[idx] for k, v in self.latents.items()},
        )

    def copy(self):
        return self.subset(np.arange(len(self)))

    def oracle_accuracy(self):
        """Accuracy of the Bayes-optimal rule that sees the noisy *latents*.
        Pixel/sample noise on top makes this a (slightly optimistic) ceiling."""
        return float((self.oracle_pred == self.y).mean())

    def summary(self):
        return (f"scenario={self.scenario} n={len(self)} noise={self.noise} "
                f"sigma(img,sig,tab)={tuple(round(float(s) * self.noise, 2) for s in self.sigma)} "
                f"P(y=1)={self.y.mean():.2f} oracle_acc={self.oracle_accuracy():.3f}")


# --------------------------------------------------------------------------- #
# Renderers
# --------------------------------------------------------------------------- #
def _render_images(s1, s2, size, pix_std, rng):
    n = len(s1)
    g = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(g, g)                       # (H, W)
    cx = np.clip(0.6 * s1, -0.85, 0.85)[:, None, None]
    blob = np.exp(-((X - cx) ** 2 + Y ** 2) / (2 * 0.25 ** 2))
    theta = (np.pi / 2) / (1 + np.exp(-s2))        # orientation in (0, pi/2)
    theta = theta[:, None, None]
    phase = rng.uniform(0, 2 * np.pi, size=(n, 1, 1))
    grating = 0.3 * np.cos(2 * np.pi * 2 * (X * np.cos(theta) + Y * np.sin(theta)) + phase)
    img = blob + grating + pix_std * rng.normal(size=(n, size, size))
    return img[:, None].astype(np.float32)


def _render_signals(s1, s2, length, noise_std, rng):
    n = len(s1)
    t = np.linspace(0, 1, length, endpoint=False)[None, :]
    f = np.clip(6 + 2 * s1, 2, 14)[:, None]        # cycles per window
    main = np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi, size=(n, 1)))
    amp2 = (0.5 + 0.3 * np.tanh(s2))[:, None]
    second = amp2 * np.sin(2 * np.pi * 20 * t + rng.uniform(0, 2 * np.pi, size=(n, 1)))
    sig = main + second + noise_std * rng.normal(size=(n, length))
    return sig[:, None].astype(np.float32)


def _render_tabular(s1, s2, n_tab, noise_std, rng):
    rng_w = np.random.default_rng(_WORLD_SEED)    # fixed mixing across splits
    A = rng_w.normal(size=(n_tab, 2))
    A = 1.5 * A / np.linalg.norm(A, axis=1, keepdims=True)
    bias = rng_w.normal(scale=0.3, size=n_tab)
    nonlin = np.arange(n_tab) >= n_tab // 2
    lin = np.stack([s1, s2], 1) @ A.T + bias
    feats = np.where(nonlin[None, :], np.tanh(lin), lin)
    return (feats + noise_std * rng.normal(size=feats.shape)).astype(np.float32)


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def make_dataset(scenario="A", n=5000, noise=1.0, seed=0, img_size=16,
                 sig_len=128, n_tab=10, sigma=None) -> MultimodalData:
    """Generate a dataset.

    scenario : 'A'..'E' (see module docstring)
    noise    : global multiplier on every modality's latent noise
    sigma    : optional (img, sig, tab) base noise levels overriding the scenario
               default, e.g. sigma=(0.2, 1.5, 1.0). Still multiplied by `noise`.
    """
    scenario = scenario.upper()
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {list(SCENARIOS)}")
    rng = np.random.default_rng(seed)
    base = np.array(sigma if sigma is not None else _DEFAULT_SIGMA[scenario], float)
    sig_m = base * noise

    # ---- true latents (slot 1 per modality) and labels
    if scenario in ("A", "E"):
        s = rng.normal(size=n)
        z = np.stack([s, s, s], 1)
        y = s > 0
        latents = {"s": s}
    elif scenario in ("B", "D"):
        z = rng.normal(size=(n, 3))
        w = _B_WEIGHTS if scenario == "B" else _D_WEIGHTS
        y = z @ w > 0
        latents = {"u_img": z[:, 0], "u_sig": z[:, 1], "u_tab": z[:, 2]}
    else:  # C
        b = rng.integers(0, 2, size=(n, 2))
        z = np.stack([2.0 * b[:, 0] - 1, 2.0 * b[:, 1] - 1, rng.normal(size=n)], 1)
        y = b[:, 0] ^ b[:, 1]
        latents = {"b_img": b[:, 0], "b_sig": b[:, 1]}

    obs = z + sig_m[None, :] * rng.normal(size=(n, 3))   # what each modality "sees"
    nuis = rng.normal(size=(n, 3))                       # slot 2: nuisance

    # ---- Bayes-optimal decision from the noisy latent observations
    if scenario in ("A", "E"):
        oracle = (obs / np.maximum(sig_m, 1e-6) ** 2).sum(1) > 0
    elif scenario in ("B", "D"):
        w = _B_WEIGHTS if scenario == "B" else _D_WEIGHTS
        shrink = 1.0 / (1.0 + sig_m ** 2)               # posterior mean of Gaussian u
        oracle = (obs * shrink * w).sum(1) > 0
    else:
        oracle = (obs[:, 0] > 0) != (obs[:, 1] > 0)

    # ---- render
    img = _render_images(obs[:, 0], nuis[:, 0], img_size, 0.1 + 0.2 * sig_m[0], rng)
    sig = _render_signals(obs[:, 1], nuis[:, 1], sig_len, 0.3 + 0.3 * sig_m[1], rng)
    tab = _render_tabular(obs[:, 2], nuis[:, 2], n_tab, 0.1 + 0.2 * sig_m[2], rng)

    return MultimodalData(img, sig, tab, y.astype(np.int64), scenario, noise,
                          tuple(float(b) for b in base), obs.astype(np.float32), oracle.astype(np.int64),
                          latents)


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def split_dataset(data: MultimodalData, fracs=(0.6, 0.2, 0.2), seed=0):
    """Random train/val/test split -> tuple of MultimodalData."""
    idx = np.random.default_rng(seed).permutation(len(data))
    cuts = np.cumsum([int(f * len(data)) for f in fracs])[:-1]
    return tuple(data.subset(p) for p in np.split(idx, cuts))


def corrupt(data: MultimodalData, modality: str, mode="noise", level=1.0, seed=0):
    """Return a corrupted copy (use on TEST data only).

    mode='noise': add Gaussian noise with std = level * std(modality)
    mode='drop' : zero the modality for a fraction `level` of the samples
    """
    assert modality in MODALITIES
    out = data.copy()
    arr = getattr(out, modality)
    rng = np.random.default_rng(seed)
    if mode == "noise":
        arr += (level * arr.std() * rng.normal(size=arr.shape)).astype(arr.dtype)
    elif mode == "drop":
        mask = rng.random(len(arr)) < level
        arr[mask] = 0
    else:
        raise ValueError("mode must be 'noise' or 'drop'")
    return out


def handcrafted_features(data: MultimodalData, n_fft_bins=40):
    """Fixed features for the sklearn track: flattened pixels, FFT magnitudes
    of the signal (phase is random, so raw samples are useless for a linear model),
    raw tabular features."""
    spec = np.abs(np.fft.rfft(data.sig[:, 0, :], axis=1))[:, :n_fft_bins]
    return {"img": data.img.reshape(len(data), -1), "sig": spec, "tab": data.tab}


def plot_samples(data: MultimodalData, n=4, seed=0):
    import matplotlib.pyplot as plt
    idx = np.random.default_rng(seed).choice(len(data), n, replace=False)
    fig, ax = plt.subplots(n, 3, figsize=(10, 2.0 * n), squeeze=False)
    for r, i in enumerate(idx):
        ax[r, 0].imshow(data.img[i, 0], cmap="gray"); ax[r, 0].axis("off")
        ax[r, 1].plot(data.sig[i, 0], lw=0.8); ax[r, 1].set_xticks([])
        ax[r, 2].bar(range(data.tab.shape[1]), data.tab[i]); ax[r, 2].set_xticks([])
        ax[r, 0].set_title(f"image  (y={data.y[i]})", fontsize=9)
        ax[r, 1].set_title("signal", fontsize=9)
        ax[r, 2].set_title("tabular", fontsize=9)
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    for sc in SCENARIOS:
        print(sc, make_dataset(sc, n=5000).summary())
