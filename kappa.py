"""
Pixel-wise Cohen's kappa for inter-annotator agreement.

Each annotator's bounding boxes are rasterized onto a pixel canvas.
Kappa is then computed by comparing these pixel-level label maps.

Two variants:
- Class-agnostic: pixels are binary (0 = background, 1 = any object).
- Class-specific: pixels carry the class label id (0 = background, N = class N).

Annotators with no annotations for an image are treated as all-background.
Canvas size is inferred per image from the union of all bounding box extents,
unless overridden via image_width / image_height in MergeConfig.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from config import COL_IMAGE, COL_LABEL, COL_SOURCE, COL_X1, COL_X2, COL_Y1, COL_Y2, MergeConfig


def _infer_canvas_size(img_df: pd.DataFrame) -> tuple[int, int]:
    """Return (width, height) inferred from max bounding box coordinates."""
    width = int(img_df[COL_X2].max()) + 1
    height = int(img_df[COL_Y2].max()) + 1
    return width, height


def _render_pixel_map(
    boxes: pd.DataFrame,
    width: int,
    height: int,
    class_to_id: dict[str, int],
    class_agnostic: bool,
) -> np.ndarray:
    """
    Rasterize bounding boxes onto a (height x width) int32 pixel map.
    Background = 0. Later boxes overwrite earlier ones in overlapping areas.
    """
    canvas = np.zeros((height, width), dtype=np.int32)
    for _, row in boxes.iterrows():
        x1 = max(0, int(row[COL_X1]))
        y1 = max(0, int(row[COL_Y1]))
        x2 = min(width, int(row[COL_X2]))
        y2 = min(height, int(row[COL_Y2]))
        label = 1 if class_agnostic else class_to_id[row[COL_LABEL]]
        canvas[y1:y2, x1:x2] = label
    return canvas


def _cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's kappa between two flat integer label arrays via sklearn."""
    try:
        return float(cohen_kappa_score(a, b))
    except ValueError:
        # Degenerate case: arrays are identical with a single unique label.
        return 1.0 if np.array_equal(a, b) else float("nan")


def compute_kappa(df: pd.DataFrame, class_agnostic: bool, cfg: MergeConfig) -> dict:
    """
    Compute pairwise pixel-wise Cohen's kappa for all annotator pairs.

    Kappa is computed over all images concatenated into a single pixel sequence,
    so images with more pixels contribute more to the overall score.

    Returns:
        {
            "pairwise": {(source_a, source_b): kappa},
            "mean": float,
        }
    """
    sources = sorted(df[COL_SOURCE].unique())
    class_to_id = {cls: i + 1 for i, cls in enumerate(sorted(df[COL_LABEL].unique()))}
    images = sorted(df[COL_IMAGE].unique())

    source_pixels: dict[str, list[np.ndarray]] = {s: [] for s in sources}

    for image in images:
        img_df = df[df[COL_IMAGE] == image]

        if cfg.image_width and cfg.image_height:
            w, h = cfg.image_width, cfg.image_height
        else:
            w, h = _infer_canvas_size(img_df)

        for source in sources:
            src_df = img_df[img_df[COL_SOURCE] == source]
            pixel_map = _render_pixel_map(src_df, w, h, class_to_id, class_agnostic)
            source_pixels[source].append(pixel_map.ravel())

    flat_pixels = {s: np.concatenate(source_pixels[s]) for s in sources}

    pairwise: dict[tuple[str, str], float] = {}
    kappa_values: list[float] = []
    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            s1, s2 = sources[i], sources[j]
            k = _cohen_kappa(flat_pixels[s1], flat_pixels[s2])
            pairwise[(s1, s2)] = k
            kappa_values.append(k)

    mean = float(np.mean(kappa_values)) if kappa_values else float("nan")
    return {"pairwise": pairwise, "mean": mean}


def print_kappa_report(df: pd.DataFrame, cfg: MergeConfig) -> None:
    """Compute and print both kappa variants to stdout."""
    agnostic = compute_kappa(df, class_agnostic=True, cfg=cfg)
    specific = compute_kappa(df, class_agnostic=False, cfg=cfg)

    print("\n--- Cohen's Kappa (class-agnostic) ---")
    for (s1, s2), k in agnostic["pairwise"].items():
        print(f"  {s1} vs {s2}: {k:.4f}")
    print(f"  Mean: {agnostic['mean']:.4f}")

    print("\n--- Cohen's Kappa (class-specific) ---")
    for (s1, s2), k in specific["pairwise"].items():
        print(f"  {s1} vs {s2}: {k:.4f}")
    print(f"  Mean: {specific['mean']:.4f}")
