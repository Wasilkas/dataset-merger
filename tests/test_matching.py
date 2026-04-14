import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import pytest
from matching import compute_iou, compute_center_distance, build_clusters, boxes_are_same
from config import COL_SOURCE, COL_X1, COL_X2, COL_Y1, COL_Y2, MergeConfig


def box(x1, y1, x2, y2, source="A"):
    return pd.Series({COL_X1: x1, COL_Y1: y1, COL_X2: x2, COL_Y2: y2, COL_SOURCE: source})


def make_group(*rows):
    return pd.DataFrame(list(rows))


# --- IoU tests ---

def test_iou_identical():
    b = box(0, 0, 100, 100)
    assert compute_iou(b, b) == pytest.approx(1.0)


def test_iou_no_overlap():
    a = box(0, 0, 10, 10)
    b = box(20, 20, 30, 30)
    assert compute_iou(a, b) == pytest.approx(0.0)


def test_iou_partial():
    a = box(0, 0, 10, 10)   # area 100
    b = box(5, 0, 15, 10)   # area 100, overlap 50
    assert compute_iou(a, b) == pytest.approx(50 / 150)


def test_iou_contained():
    outer = box(0, 0, 100, 100)
    inner = box(25, 25, 75, 75)
    # inner area = 2500, outer area = 10000, intersection = 2500, union = 10000
    assert compute_iou(outer, inner) == pytest.approx(2500 / 10000)


# --- Center distance tests ---

def test_center_distance_same():
    b = box(0, 0, 100, 100)
    assert compute_center_distance(b, b) == pytest.approx(0.0)


def test_center_distance_known():
    a = box(0, 0, 10, 10)   # center (5, 5)
    b = box(10, 0, 20, 10)  # center (15, 5)
    assert compute_center_distance(a, b) == pytest.approx(10.0)


# --- boxes_are_same ---

def test_same_by_iou():
    cfg = MergeConfig(iou_threshold=0.5, dist_threshold=5.0)
    a = box(0, 0, 100, 100)
    b = box(5, 5, 105, 105)
    assert boxes_are_same(a, b, cfg)


def test_same_by_dist_small_boxes():
    # Small boxes may have low IoU but close centers
    cfg = MergeConfig(iou_threshold=0.5, dist_threshold=20.0)
    a = box(0, 0, 5, 5)   # center (2.5, 2.5)
    b = box(3, 3, 8, 8)   # center (5.5, 5.5) — distance ~4.2, IoU very low
    assert boxes_are_same(a, b, cfg)


def test_not_same():
    cfg = MergeConfig(iou_threshold=0.5, dist_threshold=5.0)
    a = box(0, 0, 10, 10)
    b = box(500, 500, 510, 510)
    assert not boxes_are_same(a, b, cfg)


# --- Clustering ---

def test_single_box_one_cluster():
    cfg = MergeConfig()
    group = make_group({COL_X1: 0, COL_Y1: 0, COL_X2: 100, COL_Y2: 100, COL_SOURCE: "A"})
    clusters = build_clusters(group, cfg)
    assert len(clusters) == 1
    assert len(clusters[0]) == 1


def test_two_matching_boxes_one_cluster():
    cfg = MergeConfig(iou_threshold=0.5)
    group = make_group(
        {COL_X1: 0, COL_Y1: 0, COL_X2: 100, COL_Y2: 100, COL_SOURCE: "A"},
        {COL_X1: 5, COL_Y1: 5, COL_X2: 105, COL_Y2: 105, COL_SOURCE: "B"},
    )
    clusters = build_clusters(group, cfg)
    assert len(clusters) == 1
    assert len(clusters[0]) == 2


def test_two_distant_boxes_two_clusters():
    cfg = MergeConfig(iou_threshold=0.5, dist_threshold=10.0)
    group = make_group(
        {COL_X1: 0, COL_Y1: 0, COL_X2: 50, COL_Y2: 50, COL_SOURCE: "A"},
        {COL_X1: 400, COL_Y1: 400, COL_X2: 450, COL_Y2: 450, COL_SOURCE: "B"},
    )
    clusters = build_clusters(group, cfg)
    assert len(clusters) == 2


def test_transitive_clustering():
    """A-B match, B-C match, so A-B-C should be one cluster even if A-C don't directly match."""
    cfg = MergeConfig(iou_threshold=0.4, dist_threshold=5.0)
    # A and C have low IoU but both overlap with B
    group = make_group(
        {COL_X1: 0, COL_Y1: 0, COL_X2: 60, COL_Y2: 60, COL_SOURCE: "A"},    # center (30,30)
        {COL_X1: 20, COL_Y1: 20, COL_X2: 80, COL_Y2: 80, COL_SOURCE: "B"},  # center (50,50)
        {COL_X1: 40, COL_Y1: 40, COL_X2: 100, COL_Y2: 100, COL_SOURCE: "C"}, # center (70,70)
    )
    clusters = build_clusters(group, cfg)
    assert len(clusters) == 1
    assert len(clusters[0]) == 3
