import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import pytest
from voting import vote, process_all
from config import COL_LABEL, COL_SOURCE, COL_X1, COL_X2, COL_Y1, COL_Y2, QUESTIONABLE_SUFFIX, MergeConfig


def cluster(*rows):
    return pd.DataFrame(list(rows))


def make_box(x1, y1, x2, y2, source, cls="car"):
    return {COL_X1: x1, COL_Y1: y1, COL_X2: x2, COL_Y2: y2, COL_SOURCE: source, COL_LABEL: cls}


# ---------------------------------------------------------------------------
# N=2 annotators — same class
# ---------------------------------------------------------------------------

def test_n2_both_agree_keeps():
    cfg = MergeConfig()
    c = cluster(
        make_box(0, 0, 100, 100, "A"),
        make_box(5, 5, 105, 105, "B"),
    )
    results = vote([c], n_annotators=2, image="img.jpg", cfg=cfg)
    assert len(results) == 1
    assert results[0][COL_LABEL] == "car"


def test_n2_only_one_questionable():
    cfg = MergeConfig()
    c = cluster(make_box(0, 0, 100, 100, "A"))
    results = vote([c], n_annotators=2, image="img.jpg", cfg=cfg)
    assert len(results) == 1
    assert results[0][COL_LABEL] == f"car{QUESTIONABLE_SUFFIX}"


def test_n2_no_questionable_flag_drops():
    cfg = MergeConfig(no_questionable=True)
    c = cluster(make_box(0, 0, 100, 100, "A"))
    results = vote([c], n_annotators=2, image="img.jpg", cfg=cfg)
    assert len(results) == 0


# ---------------------------------------------------------------------------
# N=3 annotators — same class
# ---------------------------------------------------------------------------

def test_n3_two_of_three_keeps():
    cfg = MergeConfig()
    c = cluster(
        make_box(0, 0, 100, 100, "A", cls="dog"),
        make_box(5, 5, 105, 105, "B", cls="dog"),
    )
    results = vote([c], n_annotators=3, image="img.jpg", cfg=cfg)
    assert len(results) == 1
    assert results[0][COL_LABEL] == "dog"


def test_n3_one_of_three_questionable():
    cfg = MergeConfig()
    c = cluster(make_box(0, 0, 100, 100, "A", cls="dog"))
    results = vote([c], n_annotators=3, image="img.jpg", cfg=cfg)
    assert len(results) == 1
    assert results[0][COL_LABEL] == f"dog{QUESTIONABLE_SUFFIX}"


def test_n3_all_three_keeps():
    cfg = MergeConfig()
    c = cluster(
        make_box(0, 0, 100, 100, "A", cls="dog"),
        make_box(5, 5, 105, 105, "B", cls="dog"),
        make_box(2, 2, 102, 102, "C", cls="dog"),
    )
    results = vote([c], n_annotators=3, image="img.jpg", cfg=cfg)
    assert len(results) == 1
    assert results[0][COL_LABEL] == "dog"


# ---------------------------------------------------------------------------
# N=4 annotators — same class
# ---------------------------------------------------------------------------

def test_n4_two_of_four_questionable():
    cfg = MergeConfig()
    c = cluster(
        make_box(0, 0, 100, 100, "A"),
        make_box(5, 5, 105, 105, "B"),
    )
    # 2/4 = 50%, not > 50%
    results = vote([c], n_annotators=4, image="img.jpg", cfg=cfg)
    assert len(results) == 1
    assert results[0][COL_LABEL] == f"car{QUESTIONABLE_SUFFIX}"


def test_n4_three_of_four_keeps():
    cfg = MergeConfig()
    c = cluster(
        make_box(0, 0, 100, 100, "A"),
        make_box(5, 5, 105, 105, "B"),
        make_box(3, 3, 103, 103, "C"),
    )
    results = vote([c], n_annotators=4, image="img.jpg", cfg=cfg)
    assert len(results) == 1
    assert results[0][COL_LABEL] == "car"


# ---------------------------------------------------------------------------
# Box coordinate averaging
# ---------------------------------------------------------------------------

def test_box_coordinates_averaged():
    cfg = MergeConfig()
    c = cluster(
        make_box(0, 0, 100, 100, "A"),
        make_box(10, 10, 110, 110, "B"),
    )
    results = vote([c], n_annotators=2, image="img.jpg", cfg=cfg)
    assert results[0][COL_X1] == pytest.approx(5.0)
    assert results[0][COL_Y1] == pytest.approx(5.0)
    assert results[0][COL_X2] == pytest.approx(105.0)
    assert results[0][COL_Y2] == pytest.approx(105.0)


# ---------------------------------------------------------------------------
# Dirty cluster (same annotator, 2 boxes)
# ---------------------------------------------------------------------------

def test_dirty_cluster_is_questionable():
    cfg = MergeConfig()
    c = cluster(
        make_box(0, 0, 100, 100, "A"),
        make_box(5, 5, 105, 105, "A"),  # same annotator — dirty
        make_box(2, 2, 102, 102, "B"),
    )
    results = vote([c], n_annotators=2, image="img.jpg", cfg=cfg)
    assert len(results) == 1
    assert results[0][COL_LABEL] == f"car{QUESTIONABLE_SUFFIX}"


# ---------------------------------------------------------------------------
# N=1 passthrough
# ---------------------------------------------------------------------------

def test_n1_passthrough():
    cfg = MergeConfig()
    c = cluster(make_box(10, 20, 50, 80, "A", cls="bike"))
    results = vote([c], n_annotators=1, image="img.jpg", cfg=cfg)
    assert len(results) == 1
    assert results[0][COL_LABEL] == "bike"
    assert results[0][COL_X1] == 10


# ---------------------------------------------------------------------------
# Rule 4: Mixed classes — individual questionable boxes, no coord merging
# ---------------------------------------------------------------------------

def test_rule4_two_boxes_different_classes_both_questionable():
    """Two annotators draw the same location — one calls it 'cat', one 'dog'.
    Both boxes must be emitted individually as questionable; coords not averaged."""
    cfg = MergeConfig()
    c = cluster(
        make_box(0, 0, 100, 100, "A", cls="cat"),
        make_box(5, 5, 105, 105, "B", cls="dog"),
    )
    results = vote([c], n_annotators=2, image="img.jpg", cfg=cfg)
    assert len(results) == 2
    labels = {r[COL_LABEL] for r in results}
    assert labels == {f"cat{QUESTIONABLE_SUFFIX}", f"dog{QUESTIONABLE_SUFFIX}"}
    # Original coords preserved — not averaged
    cat_box = next(r for r in results if r[COL_LABEL].startswith("cat"))
    dog_box = next(r for r in results if r[COL_LABEL].startswith("dog"))
    assert cat_box[COL_X1] == pytest.approx(0.0)
    assert dog_box[COL_X1] == pytest.approx(5.0)


def test_rule4_three_boxes_majority_class_still_all_questionable():
    """Group of 3: 2×cat (A, B) + 1×dog (C).
    Majority class is cat, but classes disagree → Rule 4: all 3 questionable individually.
    This uses the strict interpretation: non-unanimous class → all questionable."""
    cfg = MergeConfig()
    c = cluster(
        make_box(0, 0, 100, 100, "A", cls="cat"),
        make_box(5, 5, 105, 105, "B", cls="cat"),
        make_box(2, 2, 102, 102, "C", cls="dog"),
    )
    results = vote([c], n_annotators=3, image="img.jpg", cfg=cfg)
    assert len(results) == 3
    assert all(r[COL_LABEL].endswith(QUESTIONABLE_SUFFIX) for r in results)
    labels = [r[COL_LABEL] for r in results]
    assert labels.count(f"cat{QUESTIONABLE_SUFFIX}") == 2
    assert labels.count(f"dog{QUESTIONABLE_SUFFIX}") == 1


def test_rule4_50_50_class_split_both_questionable():
    """Exact 50/50 class split (1 cat, 1 dog) — tie-break is questionable per Rule 4."""
    cfg = MergeConfig()
    c = cluster(
        make_box(10, 10, 90, 90, "A", cls="cat"),
        make_box(10, 10, 90, 90, "B", cls="dog"),
    )
    results = vote([c], n_annotators=2, image="img.jpg", cfg=cfg)
    assert len(results) == 2
    assert all(r[COL_LABEL].endswith(QUESTIONABLE_SUFFIX) for r in results)


def test_rule4_no_questionable_flag_drops_mixed_class():
    """With --no-questionable, mixed-class groups are entirely suppressed."""
    cfg = MergeConfig(no_questionable=True)
    c = cluster(
        make_box(0, 0, 100, 100, "A", cls="cat"),
        make_box(5, 5, 105, 105, "B", cls="dog"),
    )
    results = vote([c], n_annotators=2, image="img.jpg", cfg=cfg)
    assert len(results) == 0


# ---------------------------------------------------------------------------
# Rule 3: Singleton from an annotator who only labeled one image
# ---------------------------------------------------------------------------

def test_rule3_singleton_one_annotator_one_image():
    """Annotator A drew exactly one box on one image; annotator B drew nothing there.
    The single cluster is a singleton → questionable."""
    cfg = MergeConfig()
    c = cluster(make_box(50, 50, 200, 200, "A", cls="person"))
    results = vote([c], n_annotators=2, image="single_image.jpg", cfg=cfg)
    assert len(results) == 1
    assert results[0][COL_LABEL] == f"person{QUESTIONABLE_SUFFIX}"
    # Coordinates unchanged
    assert results[0][COL_X1] == pytest.approx(50.0)
    assert results[0][COL_Y2] == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# Two boxes same class, below IoU threshold → two singletons, both questionable
# ---------------------------------------------------------------------------

def test_two_same_class_below_threshold_two_singletons():
    """Two boxes of the same class that did NOT match (separate singleton clusters).
    Each singleton → questionable. Validates Rule 3 applied twice."""
    cfg = MergeConfig()
    # Two separate singleton clusters — caller (merge.py) would have split these
    c1 = cluster(make_box(0, 0, 50, 50, "A", cls="car"))
    c2 = cluster(make_box(400, 400, 450, 450, "B", cls="car"))
    results = vote([c1, c2], n_annotators=2, image="img.jpg", cfg=cfg)
    assert len(results) == 2
    assert all(r[COL_LABEL] == f"car{QUESTIONABLE_SUFFIX}" for r in results)
