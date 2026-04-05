"""
Majority voting on box clusters.

For each cluster (connected component from matching.py):
- If the fraction of distinct agreeing sources exceeds 50% of total annotators → keep, merge boxes.
- Otherwise → questionable (emit with QUESTIONABLE_SUFFIX appended to the class label).

Special case:
- If a single annotator contributed 2+ boxes to the cluster ("dirty cluster"),
  the annotator's intent is ambiguous. Mark the entire cluster as questionable.
- If N == 1, no voting is performed; return all annotations unchanged.
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
    cls: str,
    cfg: MergeConfig,
) -> list[dict]:
    """
    Apply majority voting to a list of clusters for one (image, class) group.

    Returns a list of output row dicts with keys matching OUTPUT_COLS.
    """
    if n_annotators == 1:
        # Nothing to vote on — return all boxes as-is.
        results = []
        for cluster in clusters:
            for _, row in cluster.iterrows():
                results.append({
                    COL_IMAGE: image,
                    COL_LABEL: cls,
                    COL_X1: row[COL_X1],
                    COL_Y1: row[COL_Y1],
                    COL_X2: row[COL_X2],
                    COL_Y2: row[COL_Y2],
                })
        return results

    results = []
    for cluster in clusters:
        questionable = _is_dirty(cluster)
        support = cluster[COL_SOURCE].nunique()
        majority = support > n_annotators / 2.0

        if not questionable and majority:
            # Confirmed detection — average all boxes in the cluster.
            avg = _average_boxes(cluster)
            results.append({COL_IMAGE: image, COL_LABEL: cls, **avg})
        else:
            if cfg.no_questionable:
                continue  # drop uncertain detections
            avg = _average_boxes(cluster)
            results.append({COL_IMAGE: image, COL_LABEL: f"{cls}{QUESTIONABLE_SUFFIX}", **avg})

    return results


def process_all(
    df: pd.DataFrame,
    n_annotators: int,
    clusters_by_group: dict[tuple[str, str], list[pd.DataFrame]],
    cfg: MergeConfig,
) -> pd.DataFrame:
    """
    Run voting for all (image, class) groups and return the merged output DataFrame.

    clusters_by_group: {(image, class): [cluster_df, ...]}
    """
    rows = []
    for (image, cls), clusters in clusters_by_group.items():
        rows.extend(vote(clusters, n_annotators, image, cls, cfg))

    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLS)

    result = pd.DataFrame(rows)
    # Round coordinates to 2 decimal places for cleaner output
    for col in COORD_COLS:
        result[col] = result[col].round(2)
    return result.sort_values([COL_IMAGE, COL_LABEL]).reset_index(drop=True)
