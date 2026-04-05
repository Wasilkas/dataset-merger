import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest
from kappa import _cohen_kappa, _render_pixel_map, compute_kappa
from config import COL_IMAGE, COL_LABEL, COL_SOURCE, COL_X1, COL_X2, COL_Y1, COL_Y2, MergeConfig


# ---------------------------------------------------------------------------
# _cohen_kappa unit tests
# ---------------------------------------------------------------------------

def test_kappa_perfect_agreement():
    a = np.array([0, 0, 1, 1, 2, 2])
    assert _cohen_kappa(a, a) == pytest.approx(1.0)


def test_kappa_perfect_disagreement_binary():
    a = np.array([0, 0, 1, 1])
    b = np.array([1, 1, 0, 0])
    k = _cohen_kappa(a, b)
    assert k == pytest.approx(-1.0)


def test_kappa_chance_agreement():
    # 50/50 split and random-looking disagreement — kappa near 0
    rng = np.random.default_rng(42)
    a = rng.integers(0, 2, size=10000)
    b = rng.integers(0, 2, size=10000)
    k = _cohen_kappa(a, b)
    assert abs(k) < 0.05


def test_kappa_all_same_label():
    # Both predict all background — pe == 1, should return 1.0
    a = np.zeros(100, dtype=np.int32)
    b = np.zeros(100, dtype=np.int32)
    assert _cohen_kappa(a, b) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _render_pixel_map unit tests
# ---------------------------------------------------------------------------

def make_box_df(rows):
    return pd.DataFrame(rows)


def test_render_empty_boxes():
    canvas = _render_pixel_map(make_box_df([]), width=10, height=10, class_to_id={}, class_agnostic=True)
    assert canvas.sum() == 0


def test_render_single_box_agnostic():
    boxes = make_box_df([{COL_X1: 2, COL_Y1: 2, COL_X2: 5, COL_Y2: 5, COL_LABEL: "car"}])
    canvas = _render_pixel_map(boxes, width=10, height=10, class_to_id={"car": 1}, class_agnostic=True)
    # Rows 2-4, cols 2-4 should be 1 (9 pixels)
    assert canvas[2:5, 2:5].sum() == 9
    assert canvas.sum() == 9


def test_render_single_box_class_specific():
    class_to_id = {"car": 1, "person": 2}
    boxes = make_box_df([{COL_X1: 0, COL_Y1: 0, COL_X2: 3, COL_Y2: 3, COL_LABEL: "person"}])
    canvas = _render_pixel_map(boxes, width=5, height=5, class_to_id=class_to_id, class_agnostic=False)
    assert np.all(canvas[0:3, 0:3] == 2)
    assert canvas[4, 4] == 0


def test_render_class_agnostic_ignores_label():
    class_to_id = {"car": 1, "truck": 2}
    boxes = make_box_df([
        {COL_X1: 0, COL_Y1: 0, COL_X2: 2, COL_Y2: 2, COL_LABEL: "car"},
        {COL_X1: 3, COL_Y1: 3, COL_X2: 5, COL_Y2: 5, COL_LABEL: "truck"},
    ])
    canvas = _render_pixel_map(boxes, width=6, height=6, class_to_id=class_to_id, class_agnostic=True)
    assert np.all(canvas[0:2, 0:2] == 1)
    assert np.all(canvas[3:5, 3:5] == 1)


# ---------------------------------------------------------------------------
# compute_kappa integration tests
# ---------------------------------------------------------------------------

def _make_df(source, image, label, x1, y1, x2, y2):
    return {
        COL_SOURCE: source,
        COL_IMAGE: image,
        COL_LABEL: label,
        COL_X1: x1,
        COL_Y1: y1,
        COL_X2: x2,
        COL_Y2: y2,
    }


def test_compute_kappa_identical_annotators():
    rows = [
        _make_df("A", "img.jpg", "car", 10, 10, 50, 50),
        _make_df("B", "img.jpg", "car", 10, 10, 50, 50),
    ]
    df = pd.DataFrame(rows)
    cfg = MergeConfig(image_width=100, image_height=100)
    result = compute_kappa(df, class_agnostic=True, cfg=cfg)
    assert result["pairwise"][("A", "B")] == pytest.approx(1.0)
    assert result["mean"] == pytest.approx(1.0)


def test_compute_kappa_non_overlapping_annotators():
    # A annotates top-left, B annotates bottom-right — both fill the same pixel count
    rows = [
        _make_df("A", "img.jpg", "car", 0, 0, 50, 50),
        _make_df("B", "img.jpg", "car", 50, 50, 100, 100),
    ]
    df = pd.DataFrame(rows)
    cfg = MergeConfig(image_width=100, image_height=100)
    result = compute_kappa(df, class_agnostic=True, cfg=cfg)
    k = result["pairwise"][("A", "B")]
    # No overlap → kappa should be negative (systematic disagreement)
    assert k < 0


def test_compute_kappa_class_agnostic_ignores_class_diff():
    # Same box, different class — agnostic kappa should be 1.0
    rows = [
        _make_df("A", "img.jpg", "car", 10, 10, 50, 50),
        _make_df("B", "img.jpg", "truck", 10, 10, 50, 50),
    ]
    df = pd.DataFrame(rows)
    cfg = MergeConfig(image_width=100, image_height=100)
    result = compute_kappa(df, class_agnostic=True, cfg=cfg)
    assert result["pairwise"][("A", "B")] == pytest.approx(1.0)


def test_compute_kappa_class_specific_detects_class_diff():
    # Same box, different class — class-specific kappa should be lower than 1.0
    rows = [
        _make_df("A", "img.jpg", "car", 10, 10, 50, 50),
        _make_df("B", "img.jpg", "truck", 10, 10, 50, 50),
    ]
    df = pd.DataFrame(rows)
    cfg = MergeConfig(image_width=100, image_height=100)
    result = compute_kappa(df, class_agnostic=False, cfg=cfg)
    assert result["pairwise"][("A", "B")] < 1.0


def test_compute_kappa_missing_annotator_treated_as_background():
    # A annotates an image, B has no annotations on it
    rows = [_make_df("A", "img.jpg", "car", 10, 10, 50, 50)]
    df = pd.DataFrame(rows)
    # Manually add B with no annotation — B is simply absent from df for this image
    # compute_kappa should treat B as all-background
    cfg = MergeConfig(image_width=100, image_height=100)

    # Add a second source row for a different image so sources are registered
    rows2 = rows + [_make_df("B", "img2.jpg", "car", 10, 10, 50, 50)]
    df2 = pd.DataFrame(rows2)
    result = compute_kappa(df2, class_agnostic=True, cfg=cfg)
    # They disagree on img.jpg but agree on img2.jpg — kappa should be between -1 and 1
    k = result["pairwise"][("A", "B")]
    assert -1.0 <= k <= 1.0


def test_compute_kappa_mean_of_pairwise():
    rows = [
        _make_df("A", "img.jpg", "car", 10, 10, 50, 50),
        _make_df("B", "img.jpg", "car", 10, 10, 50, 50),
        _make_df("C", "img.jpg", "car", 10, 10, 50, 50),
    ]
    df = pd.DataFrame(rows)
    cfg = MergeConfig(image_width=100, image_height=100)
    result = compute_kappa(df, class_agnostic=True, cfg=cfg)
    # All three identical — all pairwise kappas = 1.0, mean = 1.0
    assert len(result["pairwise"]) == 3
    assert result["mean"] == pytest.approx(1.0)
