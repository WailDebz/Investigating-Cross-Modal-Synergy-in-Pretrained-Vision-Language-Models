"""Leakage and sanity audits for ``vl_matched`` (NumPy only).

Shortcut diagnostics are fitted on the train split and scored on the test split:

    template_id -> target     majority class per template_id
    caption length -> target  majority class per token count
    bag of words -> target    L2-regularized logistic regression on binary word features

In Scenario C none of them, nor image-only or text-only latent rules, may exceed
chance by more than ``CHANCE_TOLERANCE``.
"""
from __future__ import annotations

import numpy as np

from .manifest import SPLITS

CHANCE_TOLERANCE = 0.05  # |accuracy - 0.5|; about 3 standard errors on a 1000-sample test split


def class_balance(manifest) -> dict[str, float]:
    """P(target = 1) per split and overall."""
    y = manifest.target.to_numpy()
    out = {s: float(y[(manifest.split == s).to_numpy()].mean()) for s in SPLITS}
    out["all"] = float(y.mean())
    return out


def unimodal_predictability(latents, y: np.ndarray) -> dict[str, float]:
    """Accuracy of the latent sign rules of each channel alone."""
    return {
        "image_sign": float(np.mean((latents.obs_img.to_numpy() > 0) == y)),
        "text_category": float(np.mean(latents.text_category.to_numpy() == y)),
    }


def template_frequencies(latents, y: np.ndarray) -> dict:
    """Counts per (template_id, text_category) and per (template_id, target), and max deviation from uniform."""
    tid = latents.template_id.to_numpy()
    n_t = int(tid.max()) + 1
    by_cat = np.zeros((n_t, 2), dtype=int)
    by_y = np.zeros((n_t, 2), dtype=int)
    np.add.at(by_cat, (tid, latents.text_category.to_numpy()), 1)
    np.add.at(by_y, (tid, y), 1)
    freq = np.bincount(tid, minlength=n_t) / len(tid)
    return {
        "by_category": by_cat.tolist(),
        "by_target": by_y.tolist(),
        "max_deviation_from_uniform": float(np.max(np.abs(freq - 1.0 / n_t))),
    }


def caption_lengths(latents, y: np.ndarray) -> dict:
    """Token and character lengths overall and by target / category."""
    tokens = np.array([len(t.split()) for t in latents.text])
    chars = np.array([len(t) for t in latents.text])
    cat = latents.text_category.to_numpy()
    return {
        "token_min": int(tokens.min()), "token_max": int(tokens.max()),
        "mean_tokens_by_target": [float(tokens[y == c].mean()) for c in (0, 1)],
        "mean_chars_by_target": [float(chars[y == c].mean()) for c in (0, 1)],
        "mean_chars_by_category": [float(chars[cat == c].mean()) for c in (0, 1)],
    }


def duplicates(manifest, latents) -> dict:
    """Duplicate identifiers and caption reuse (captions are drawn from a small closed set by design)."""
    return {
        "duplicate_sample_ids": int(manifest.sample_id.duplicated().sum()),
        "unique_captions": int(latents.text.nunique()),
        "duplicate_latent_rows": int(latents.drop(columns=["sample_id"]).duplicated().sum()),
    }


def lookup_shortcut_accuracy(feature: np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray) -> float:
    """Majority-class-per-value predictor fitted on train, accuracy on test."""
    default = int(y[train].mean() >= 0.5)
    table = {v: int(y[train][feature[train] == v].mean() >= 0.5) for v in np.unique(feature[train])}
    pred = np.array([table.get(v, default) for v in feature[test]])
    return float(np.mean(pred == y[test]))


def bag_of_words(texts) -> tuple[np.ndarray, list[str]]:
    vocab = sorted({w for t in texts for w in t.split()})
    index = {w: j for j, w in enumerate(vocab)}
    X = np.zeros((len(texts), len(vocab)))
    for i, t in enumerate(texts):
        X[i, [index[w] for w in t.split()]] = 1.0
    return X, vocab


def bow_shortcut_accuracy(texts, y: np.ndarray, train: np.ndarray, test: np.ndarray,
                          l2: float = 1e-2, steps: int = 500, lr: float = 0.5) -> float:
    """Logistic regression on binary bag-of-words features (full-batch gradient descent)."""
    X, _ = bag_of_words(list(texts))
    X = np.hstack([X, np.ones((len(X), 1))])
    Xtr, ytr = X[train], y[train].astype(float)
    w = np.zeros(X.shape[1])
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-Xtr @ w))
        w -= lr * (Xtr.T @ (p - ytr) / len(ytr) + l2 * w)
    return float(np.mean(((X[test] @ w) > 0).astype(int) == y[test]))


def run_audits(bundle) -> dict:
    """All audits for a bundle; ``passed`` is set only for Scenario C shortcut checks."""
    manifest, latents = bundle.manifest, bundle.latents
    y = manifest.target.to_numpy()
    split = manifest.split.to_numpy()
    train, test = split == "train", split == "test"
    n_tokens = np.array([len(t.split()) for t in latents.text])
    report = {
        "class_balance": class_balance(manifest),
        "unimodal_predictability": unimodal_predictability(latents, y),
        "oracle_accuracy": {s: float(np.mean(latents.oracle_pred.to_numpy()[split == s] == y[split == s]))
                            for s in SPLITS},
        "template_frequencies": template_frequencies(latents, y),
        "caption_lengths": caption_lengths(latents, y),
        "duplicates": duplicates(manifest, latents),
        "shortcuts": {
            "template_id": lookup_shortcut_accuracy(latents.template_id.to_numpy(), y, train, test),
            "caption_tokens": lookup_shortcut_accuracy(n_tokens, y, train, test),
            "bag_of_words": bow_shortcut_accuracy(latents.text, y, train, test),
        },
        "chance_tolerance": CHANCE_TOLERANCE,
    }
    if bundle.config.scenario == "C":
        near_chance = [*report["unimodal_predictability"].values(), *report["shortcuts"].values()]
        report["passed"] = bool(all(abs(a - 0.5) <= CHANCE_TOLERANCE for a in near_chance)
                                and report["oracle_accuracy"]["test"] > 0.5 + CHANCE_TOLERANCE)
    return report
