"""Tests for the template-based text channel."""
import re

import numpy as np
import pytest

from synergy_bench.data.text import (
    TEMPLATE_SETS,
    draw_template_ids,
    render_captions,
    template_set,
    text_category,
)

FORBIDDEN = re.compile(
    r"\b(left|right|top|bottom|up|down|center|centre|middle|position|horizontal|vertical|diagonal|"
    r"orientation|angle|tilt|stripe|stripes|grating|blob|circle|round|square|shape|line|lines|"
    r"red|green|blue|black|white|gray|grey|dark|bright|light|colou?r|image|picture|class|label|target|"
    r"xor|same|different|yes|no|true|false|positive|negative|high|low)\b"
)


def test_category_is_sign_of_observation():
    assert text_category([-0.3, 0.0, 0.2, 5.0]).tolist() == [0, 0, 1, 1]


def test_v1_has_four_frames_and_matched_lengths():
    ts = template_set("v1")
    w0, w1 = ts["words"]
    assert len(ts["frames"]) == 4
    assert len(w0) == len(w1) and len(w0.split()) == len(w1.split()) == 1
    for t in range(4):
        c0, c1 = render_captions([0, 1], [t, t])
        assert c0 != c1
        assert len(c0) == len(c1) and len(c0.split()) == len(c1.split())


def test_v1_vocabulary_is_neutral():
    ts = TEMPLATE_SETS["v1"]
    for text in (*ts["frames"], *ts["words"]):
        assert not FORBIDDEN.search(text.lower()), text


def test_caption_states_only_its_category_word():
    w0, w1 = template_set("v1")["words"]
    caps = render_captions([0, 1, 1, 0], [0, 1, 2, 3])
    assert [w1 in c for c in caps] == [False, True, True, False]
    assert [w0 in c for c in caps] == [True, False, False, True]


def test_template_ids_deterministic_uniform_and_per_sample():
    ids = draw_template_ids(0, range(8000))
    assert np.array_equal(ids, draw_template_ids(0, range(8000)))
    counts = np.bincount(ids, minlength=4) / len(ids)
    assert np.all(np.abs(counts - 0.25) < 0.02)
    # per-sample stream: the id of a sample does not depend on which other samples are drawn
    assert np.array_equal(draw_template_ids(0, [5, 4000, 7999]), ids[[5, 4000, 7999]])
    assert not np.array_equal(ids[:200], draw_template_ids(1, range(200)))


def test_unknown_template_set():
    with pytest.raises(ValueError, match="template_set"):
        template_set("v2")
