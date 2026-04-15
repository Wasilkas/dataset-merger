"""
Majority voting on box clusters.

For each cluster (connected component from matching.py):

Rule 1 — Class voting:
  The group's class is determined by majority vote across boxes. Tie-break: mark as
  questionable (falls under Rule 4 since a 50/50 split implies class disagreement).

Rule 2 — Same-class, majority annotators → one merged normal box.
  When all boxes in the cluster share the same class AND the number of distinct
  annotator sources exceeds 50% of total annotators, collapse into a single output
  box with averaged corner coordinates. No individual source boxes are emitted.

Rule 3 — Singleton → questionable.
  A cluster with exactly one box (only one annotator drew it) is emitted with a
  questionable label using the original box coordinates.

Rule 4 — Mixed classes → all boxes individually questionable.
  If the cluster contains boxes of more than one class (i.e. unanimous class
  agreement is absent), emit every source box individually with its original
  class label suffixed by QUESTIONABLE_SUFFIX. Coordinates are NOT averaged.
  This also covers the 50/50 tie-break for class voting.

Additional cases preserved from prior behaviour:
- Dirty cluster (one annotator contributed 2+ boxes): always questionable,
  single averaged box, majority class label. Annotator intent is ambiguous.
- N == 1: no voting is performed; return all annotations unchanged.
- Same class, not majority annotators: questionable averaged box.
"""

from __future__ import annotations

import pandas as pd

from config import (
    COL_IMAGE,
    COL_LABEL,
    COL_SOURCE,
    COL_X1,
    COL_X2,
    COL_Y1,
    COL_Y2,
    COORD_COLS,
    OUTPUT_COLS,
    QUESTIONABLE_SUFFIX,
    MergeConfig,
)


def _average_boxes(df: pd.DataFrame) -> dict:
    """Return averaged coordinates over all rows in df."""
    return {col: df[col].mean() for col in COORD_COLS}


def _is_dirty(cluster: pd.DataFrame) -> bool:
    """True if any single annotator contributed more than one box to this cluster."""
    return cluster[COL_SOURCE].duplicated().any()


def vote(
    clusters: list[pd.DataFrame],
    n_annotators: int,
    image: str,
    cfg: MergeConfig,
) -> list[dict]:
    """
    Apply voting rules to a list of clusters for one image.

    Each cluster DataFrame must contain COL_LABEL so that class agreement can be
    assessed across annotators. Returns a list of output row dicts with keys
    matching OUTPUT_COLS.
    """
    if n_annotators == 1:
        # Nothing to vote on — return all boxes as-is.
        results = []
        for cluster in clusters:
            for _, row in cluster.iterrows():
                results.append({
                    COL_IMAGE: image,
                    COL_LABEL: row[COL_LABEL],
                    COL_X1: row[COL_X1],
                    COL_Y1: row[COL_Y1],
                    COL_X2: row[COL_X2],
                    COL_Y2: row[COL_Y2],
                })
        return results

    results = []
    for cluster in clusters:
        # Rule 3: Singleton group → questionable with original class.
        if len(cluster) == 1:
            if not cfg.no_questionable:
                row = cluster.iloc[0]
                results.append({
                    COL_IMAGE: image,
                    COL_LABEL: f"{row[COL_LABEL]}{QUESTIONABLE_SUFFIX}",
                    COL_X1: row[COL_X1],
                    COL_Y1: row[COL_Y1],
                    COL_X2: row[COL_X2],
                    COL_Y2: row[COL_Y2],
                })
            continue

        # Dirty cluster: one annotator contributed 2+ boxes — intent is ambiguous.
        if _is_dirty(cluster):
            if not cfg.no_questionable:
                majority_cls = cluster[COL_LABEL].mode()[0]
                avg = _average_boxes(cluster)
                results.append({
                    COL_IMAGE: image,
                    COL_LABEL: f"{majority_cls}{QUESTIONABLE_SUFFIX}",
                    **avg,
                })
            continue

        # Rule 4: Mixed classes → emit every box individually as questionable.
        # This also handles the 50/50 tie-break for class voting (2 different classes
        # in a group of 2 — no class has a strict majority, so all are questionable).
        if cluster[COL_LABEL].nunique() > 1:
            if not cfg.no_questionable:
                for _, row in cluster.iterrows():
                    results.append({
                        COL_IMAGE: image,
                        COL_LABEL: f"{row[COL_LABEL]}{QUESTIONABLE_SUFFIX}",
                        COL_X1: row[COL_X1],
                        COL_Y1: row[COL_Y1],
                        COL_X2: row[COL_X2],
                        COL_Y2: row[COL_Y2],
                    })
            continue

        # All boxes share the same class — apply majority-of-annotators voting.
        cls = cluster[COL_LABEL].iloc[0]
        support = cluster[COL_SOURCE].nunique()
        majority = support > n_annotators / 2.0

        if majority:
            # Rule 2: Same class, majority → one normal merged box with averaged coords.
            avg = _average_boxes(cluster)
            results.append({COL_IMAGE: image, COL_LABEL: cls, **avg})
        else:
            # Same class but not enough annotators agree → questionable averaged box.
            if not cfg.no_questionable:
                avg = _average_boxes(cluster)
                results.append({COL_IMAGE: image, COL_LABEL: f"{cls}{QUESTIONABLE_SUFFIX}", **avg})

    return results


def process_all(
    df: pd.DataFrame,
    n_annotators: int,
    clusters_by_image: dict[str, list[pd.DataFrame]],
    cfg: MergeConfig,
) -> pd.DataFrame:
    """
    Run voting for all images and return the merged output DataFrame.

    clusters_by_image: {image_name: [cluster_df, ...]}
    Each cluster_df must contain COL_LABEL.
    """
    rows = []
    for image, clusters in clusters_by_image.items():
        rows.extend(vote(clusters, n_annotators, image, cfg))

    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLS)

    result = pd.DataFrame(rows)
    # Round coordinates to 2 decimal places for cleaner output
    for col in COORD_COLS:
        result[col] = result[col].round(2)
    return result.sort_values([COL_IMAGE, COL_LABEL]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Controversial cases report
# ---------------------------------------------------------------------------

def _is_controversial(cluster: pd.DataFrame, n_annotators: int) -> bool:
    """Return True if this cluster should appear in the controversial cases report.

    Matches the same conditions that cause a cluster to produce questionable output:
    singleton, dirty, mixed classes, or same class without majority support.
    """
    if len(cluster) == 1:
        return True  # Rule 3: only one annotator drew this object
    if _is_dirty(cluster):
        return True  # one annotator contributed 2+ boxes — ambiguous
    if cluster[COL_LABEL].nunique() > 1:
        return True  # Rule 4: annotators disagree on class
    # Same class — controversial if not a strict majority of annotators agree
    return cluster[COL_SOURCE].nunique() <= n_annotators / 2.0


def _make_controversy_row(
    cluster: pd.DataFrame,
    image: str,
    source_names: list[str],
) -> dict:
    """Build one controversy report row from a cluster.

    Each cell is the class label that annotator assigned, or None if they
    contributed no box to this cluster. For dirty annotators (2+ boxes in one
    cluster), the modal class is used.
    """
    source_to_class: dict[str, str] = {}
    for source, grp in cluster.groupby(COL_SOURCE):
        source_to_class[source] = grp[COL_LABEL].mode()[0]

    row: dict = {"image_name": image}
    for i, name in enumerate(source_names, 1):
        row[f"annotator_{i}"] = source_to_class.get(name)  # None if absent
    return row


def collect_controversy_records(
    clusters_by_image: dict[str, list[pd.DataFrame]],
    n_annotators: int,
    source_names: list[str],
) -> list[dict]:
    """Return one row per controversial cluster across all images, in image order.

    A record has keys: image_name, annotator_1, annotator_2, ..., annotator_N.
    Cells are class label strings or None when an annotator did not label that object.
    Returns an empty list when n_annotators == 1 (nothing to compare).
    """
    if n_annotators == 1:
        return []
    records = []
    for image, clusters in clusters_by_image.items():
        for cluster in clusters:
            if _is_controversial(cluster, n_annotators):
                records.append(_make_controversy_row(cluster, image, source_names))
    return records
