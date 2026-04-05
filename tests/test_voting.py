import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import pytest
from voting import vote, process_all
from config import COL_LABEL, COL_SOURCE, COL_X1, COL_X2, COL_Y1, COL_Y2, QUESTIONABLE_SUFFIX, MergeConfig


def cluster(*rows):
    return pd.DataFrame(list(rows))


def make_box(x1, y1, x2, y2, source):
    return {COL_X1: x1, COL_Y1: y1, COL_X2: x2, COL_Y2: y2, COL_SOURCE: source}


# --- N=2 annotators ---

def test_n2_both_agree_keeps():
    cfg = MergeConfig()
    c = cluster(
        make_box(0, 0, 100, 100, "A"),
        make_box(5, 5, 105, 105, "B"),
    )
    results = vote([c], n_annotators=2, image="img.jpg", cls="car", cfg=cfg)
    assert len(results) == 1
    assert results[0][COL_LABEL] == "car"


def test_n2_only_one_questionable():
    cfg = MergeConfig()
    c = cluster(make_box(0, 0, 100, 100, "A"))
    results = vote([c], n_annotators=2, image="img.jpg", cls="car", cfg=cfg)
    assert len(results) == 1
    assert results[0][COL_LABEL] == "car{QUESTIONABLE_SUFFIX}"


def test_n2_no_questionable_flag_drops():
    cfg = MergeConfig(no_questionable=True)
    c = cluster(make_box(0, 0, 100, 100, "A"))
    results = vote([c], n_annotators=2, image="img.jpg", cls="car", cfg=cfg)
    assert len(results) == 0


# --- N=3 annotators ---

def test_n3_two_of_three_keeps():
    cfg = MergeConfig()
    c = cluster(
        make_box(0, 0, 100, 100, "A"),
        make_box(5, 5, 105, 105, "B"),
    )
    results = vote([c], n_annotators=3, image="img.jpg", cls="dog", cfg=cfg)
    assert len(results) == 1
    assert results[0][COL_LABEL] == "dog"


def test_n3_one_of_three_questionable():
    cfg = MergeConfig()
    c = cluster(make_box(0, 0, 100, 100, "A"))
    results = vote([c], n_annotators=3, image="img.jpg", cls="dog", cfg=cfg)
    assert len(results) == 1
    assert results[0][COL_LABEL] == "dog{QUESTIONABLE_SUFFIX}"


def test_n3_all_three_keeps():
    cfg = MergeConfig()
    c = cluster(
        make_box(0, 0, 100, 100, "A"),
        make_box(5, 5, 105, 105, "B"),
        make_box(2, 2, 102, 102, "C"),
    )
    results = vote([c], n_annotators=3, image="img.jpg", cls="dog", cfg=cfg)
    assert len(results) == 1
    assert results[0][COL_LABEL] == "dog"


# --- N=4 annotators ---

def test_n4_two_of_four_questionable():
    cfg = MergeConfig()
    c = cluster(
        make_box(0, 0, 100, 100, "A"),
        make_box(5, 5, 105, 105, "B"),
    )
    # 2/4 = 50%, not > 50%
    results = vote([c], n_annotators=4, image="img.jpg", cls="car", cfg=cfg)
    assert len(results) == 1
    assert results[0][COL_LABEL] == "car{QUESTIONABLE_SUFFIX}"


def test_n4_three_of_four_keeps():
    cfg = MergeConfig()
    c = cluster(
        make_box(0, 0, 100, 100, "A"),
        make_box(5, 5, 105, 105, "B"),
        make_box(3, 3, 103, 103, "C"),
    )
    results = vote([c], n_annotators=4, image="img.jpg", cls="car", cfg=cfg)
    assert len(results) == 1
    assert results[0][COL_LABEL] == "car"


# --- Box averaging ---

def test_box_coordinates_averaged():
    cfg = MergeConfig()
    c = cluster(
        make_box(0, 0, 100, 100, "A"),
        make_box(10, 10, 110, 110, "B"),
    )
    results = vote([c], n_annotators=2, image="img.jpg", cls="car", cfg=cfg)
    assert results[0][COL_X1] == pytest.approx(5.0)
    assert results[0][COL_Y1] == pytest.approx(5.0)
    assert results[0][COL_X2] == pytest.approx(105.0)
    assert results[0][COL_Y2] == pytest.approx(105.0)


# --- Dirty cluster (same annotator, 2 boxes) ---

def test_dirty_cluster_is_questionable():
    cfg = MergeConfig()
    c = cluster(
        make_box(0, 0, 100, 100, "A"),
        make_box(5, 5, 105, 105, "A"),  # same annotator — dirty
        make_box(2, 2, 102, 102, "B"),
    )
    results = vote([c], n_annotators=2, image="img.jpg", cls="car", cfg=cfg)
    assert len(results) == 1
    assert results[0][COL_LABEL] == "car{QUESTIONABLE_SUFFIX}"


# --- N=1 passthrough ---

def test_n1_passthrough():
    cfg = MergeConfig()
    c = cluster(make_box(10, 20, 50, 80, "A"))
    results = vote([c], n_annotators=1, image="img.jpg", cls="bike", cfg=cfg)
    assert len(results) == 1
    assert results[0][COL_LABEL] == "bike"
    assert results[0][COL_X1] == 10
