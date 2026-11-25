"""Controlled template-based text channel.

The text carries one binary category, the sign of its own noisy observation
(``obs_txt > 0``). It never states the image value, the target or the XOR relation.

Template set ``v1``: four sentence frames shared by both categories, filled with
one of two neutral code words. Both words have the same length in characters and
whitespace tokens, so for a fixed frame the two categories give captions of
identical length. No word refers to position, orientation, colour or shape.
The frame (``template_id``) is drawn uniformly from its own per-sample stream
``make_rng(data_seed, "text_template", index)``, independently of the category.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .seeds import make_rng

TEMPLATE_SETS: dict[str, dict] = {
    "v1": {
        "words": ("maple", "cedar"),  # category 0, category 1
        "frames": (
            "the code word is {word}",
            "this record is tagged {word}",
            "the note here simply says {word}",
            "the keyword for this item is {word}",
        ),
    }
}


def template_set(name: str) -> dict:
    if name not in TEMPLATE_SETS:
        raise ValueError(f"unknown template_set {name!r}; expected one of {tuple(TEMPLATE_SETS)}")
    return TEMPLATE_SETS[name]


def text_category(obs_txt: np.ndarray) -> np.ndarray:
    """Binary category stated by the caption: 1 if the text observation is positive."""
    return (np.asarray(obs_txt) > 0).astype(np.int64)


def draw_template_ids(data_seed: int, indices: Sequence[int], name: str = "v1") -> np.ndarray:
    """One uniform template_id per sample, from its own stream (independent of category)."""
    n_frames = len(template_set(name)["frames"])
    return np.array([make_rng(data_seed, "text_template", int(i)).integers(n_frames) for i in indices],
                    dtype=np.int64)


def render_captions(categories: Sequence[int], template_ids: Sequence[int], name: str = "v1") -> list[str]:
    """Captions for (category, template_id) pairs."""
    ts = template_set(name)
    return [ts["frames"][int(t)].format(word=ts["words"][int(c)]) for c, t in zip(categories, template_ids)]
