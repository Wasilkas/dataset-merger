"""
Box matching and clustering.

Given a group of boxes (same image, same class) from multiple annotators,
build a similarity graph and find connected components via union-find.
Each component is one "object candidate".
"""

from __future__ import annotations

import math

import pandas as pd

from config import COL_SOURCE, COL_X1, COL_X2, COL_Y1, COL_Y2, MergeConfig


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def compute_iou(a: pd.Series, b: pd.Series) -> float:
    """Compute Intersection-over-Union for two boxes."""
    inter_x1 = max(a[COL_X1], b[COL_X1])
    inter_y1 = max(a[COL_Y1], b[COL_Y1])
    inter_x2 = min(a[COL_X2], b[COL_X2])
    inter_y2 = min(a[COL_Y2], b[COL_Y2])

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = (a[COL_X2] - a[COL_X1]) * (a[COL_Y2] - a[COL_Y1])
    area_b = (b[COL_X2] - b[COL_X1]) * (b[COL_Y2] - b[COL_Y1])
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def compute_center_distance(a: pd.Series, b: pd.Series) -> float:
    """Euclidean distance between box centers."""
    cx_a = (a[COL_X1] + a[COL_X2]) / 2
    cy_a = (a[COL_Y1] + a[COL_Y2]) / 2
    cx_b = (b[COL_X1] + b[COL_X2]) / 2
    cy_b = (b[COL_Y1] + b[COL_Y2]) / 2
    return math.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2)


def boxes_are_same(a: pd.Series, b: pd.Series, cfg: MergeConfig) -> bool:
    """Return True if two boxes should be considered the same object."""
    return (
        compute_iou(a, b) > cfg.iou_threshold
        or compute_center_distance(a, b) < cfg.dist_threshold
    )


# ---------------------------------------------------------------------------
# Union-Find
# ---------------------------------------------------------------------------

class UnionFind:
    def __init__(self, n: int) -> None:
        self._parent = list(range(n))
        self._rank = [0] * n

    def find(self, x: int) -> int:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]  # path compression
            x = self._parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1

    def components(self) -> dict[int, list[int]]:
        """Return {root: [member indices]} mapping."""
        groups: dict[int, list[int]] = {}
        for i in range(len(self._parent)):
            root = self.find(i)
            groups.setdefault(root, []).append(i)
        return groups


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_clusters(group: pd.DataFrame, cfg: MergeConfig) -> list[pd.DataFrame]:
    """
    Given a DataFrame of boxes from the same (image, class) group,
    return a list of clusters (DataFrames), one per object candidate.

    The input DataFrame must have columns: COL_X1, COL_Y1, COL_X2, COL_Y2, COL_SOURCE.
    """
    rows = [row for _, row in group.iterrows()]
    n = len(rows)

    if n == 0:
        return []
    if n == 1:
        return [group]

    uf = UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if boxes_are_same(rows[i], rows[j], cfg):
                uf.union(i, j)

    components = uf.components()
    clusters = []
    for indices in components.values():
        cluster_df = group.iloc[indices].copy()
        clusters.append(cluster_df)

    return clusters
