# dataset-merger

A CLI tool for merging object detection annotations from multiple annotators into a single authoritative dataset.

## Features

- **Spatial matching** — boxes of the same class on the same image are matched via IoU and/or center distance, so slightly different draws of the same object are treated as one
- **Majority voting** — an object is kept only if more than 50% of annotators agree it exists; below that it is flagged as questionable
- **Questionable flagging** — uncertain detections are kept with a `_questionable` suffix on their class label instead of being silently dropped
- **Cohen's kappa** — pixel-wise inter-annotator agreement measured in two modes: class-agnostic (object vs. background) and class-specific

## CSV Schema

All input files and the output share the same schema:

```
image_name, instance_label, bbox_x_tl, bbox_y_tl, bbox_x_br, bbox_y_br
```

| Column | Description |
|---|---|
| `image_name` | Filename of the image (e.g. `frame_001.jpg`) |
| `instance_label` | Object class name (e.g. `car`, `person`) |
| `bbox_x_tl` | Bounding box left edge (pixels) |
| `bbox_y_tl` | Bounding box top edge (pixels) |
| `bbox_x_br` | Bounding box right edge (pixels) |
| `bbox_y_br` | Bounding box bottom edge (pixels) |

## Installation

Requires Python 3.10+. Uses [uv](https://github.com/astral-sh/uv) for environment management.

```bash
git clone <repo>
cd dataset-merger
uv venv && source .venv/bin/activate
uv sync
```

## Usage

```bash
python merge.py annotator_a.csv annotator_b.csv [annotator_c.csv ...] [OPTIONS]
```

### Options

| Option | Default | Description |
|---|---|---|
| `-o / --output PATH` | `merged.csv` | Output file path |
| `--iou-threshold FLOAT` | `0.5` | IoU threshold for matching two boxes as the same object |
| `--dist-threshold FLOAT` | `20.0` | Center distance threshold in pixels (fallback for small boxes) |
| `--no-questionable` | off | Drop uncertain detections instead of flagging them |
| `--kappa` | off | Compute and print pixel-wise Cohen's kappa |
| `--image-width INT` | inferred | Canvas width in pixels for kappa computation |
| `--image-height INT` | inferred | Canvas height in pixels for kappa computation |
| `--config PATH` | — | YAML config file (CLI flags override file values) |

### Examples

```bash
# Basic merge of two annotators
python merge.py alice.csv bob.csv -o merged.csv

# Three annotators with custom matching thresholds
python merge.py a.csv b.csv c.csv --iou-threshold 0.4 --dist-threshold 15

# Drop uncertain detections entirely
python merge.py a.csv b.csv --no-questionable

# Compute inter-annotator agreement before merging
python merge.py a.csv b.csv c.csv --kappa --image-width 1920 --image-height 1080

# Use a config file
python merge.py a.csv b.csv --config config.yaml
```

### Config file (YAML)

```yaml
iou_threshold: 0.5
dist_threshold: 20.0
output: merged.csv
no_questionable: false
compute_kappa: true
image_width: 1920
image_height: 1080
```

## Voting Logic

For each group of boxes with the same `(image_name, instance_label)`:

1. Boxes are linked into clusters via IoU > threshold **or** center distance < threshold (OR condition handles tiny boxes where IoU is unreliable)
2. Clusters are found using connected components (union-find), so matching is transitive
3. A cluster with support from **more than 50%** of annotators → merged into one box (coordinate average)
4. A cluster below majority → emitted with `_questionable` class suffix
5. A "dirty cluster" (one annotator drew 2+ boxes that merged) → always questionable

| Annotators (N) | Keep if support ≥ | Questionable if support ≤ |
|---|---|---|
| 2 | 2 | 1 |
| 3 | 2 | 1 |
| 4 | 3 | 2 |

## Inter-annotator Agreement (Cohen's Kappa)

When `--kappa` is used, each annotator's boxes are rasterised onto a pixel canvas and kappa is computed pairwise over all pixels across all images.

- **Class-agnostic**: each pixel is binary — covered by any box (`1`) or background (`0`)
- **Class-specific**: each pixel carries the class ID — measures whether annotators agree on both presence and class

Canvas size is inferred from annotation extents by default; use `--image-width` / `--image-height` for more accurate background estimation.

## Development

```bash
# Lint
uvx ruff check .
uvx ruff format .

# Tests
uv run pytest tests/

# Single test
uv run pytest tests/test_voting.py::test_n3_two_of_three_keeps
```
