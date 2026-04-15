# dataset-merger

A CLI tool for merging object detection annotations from multiple annotators into a single authoritative dataset.

## Features

- **Cross-class spatial matching** — boxes across annotators are matched by IoU and/or center distance regardless of class, so disagreements on class label are detected rather than silently split
- **Majority voting** — an object is kept only if more than 50% of annotators agree it exists; below that it is flagged as questionable
- **Questionable flagging** — uncertain detections are kept with a `_questionable` suffix on their class label instead of being silently dropped
- **Controversial cases report** — optional CSV listing every questionable cluster with the per-annotator class label (or blank if that annotator did not label the object)

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
python src/merge.py annotator_a.csv annotator_b.csv [annotator_c.csv ...] [OPTIONS]
```

### Options

| Option | Default | Description |
|---|---|---|
| `-o / --output PATH` | `merged.csv` | Output file path |
| `--iou-threshold FLOAT` | `0.5` | IoU threshold for matching two boxes as the same object |
| `--dist-threshold FLOAT` | `20.0` | Center distance threshold in pixels (fallback for small boxes) |
| `--no-questionable` | off | Drop uncertain detections instead of flagging them |
| `--controversial-report` | off | Write `controversial_report.csv` alongside the merged output |
| `--config PATH` | — | YAML config file (CLI flags override file values) |

### Examples

```bash
# Basic merge of two annotators
python src/merge.py alice.csv bob.csv -o merged.csv

# Three annotators with custom matching thresholds
python src/merge.py a.csv b.csv c.csv --iou-threshold 0.4 --dist-threshold 15

# Drop uncertain detections entirely
python src/merge.py a.csv b.csv --no-questionable

# Merge and write a report of every disputed object
python src/merge.py a.csv b.csv c.csv -o out/merged.csv --controversial-report

# Use a config file
python src/merge.py a.csv b.csv --config config.yaml
```

### Config file (YAML)

```yaml
iou_threshold: 0.5
dist_threshold: 20.0
output: merged.csv
no_questionable: false
```

## Voting Logic

For each image, all boxes across all annotators are clustered together (regardless of class label):

1. Boxes are linked into clusters via IoU > threshold **or** center distance < threshold (OR condition handles tiny boxes where IoU is unreliable)
2. Clusters are found using connected components (union-find), so matching is transitive
3. Per cluster, four rules apply:

| Situation | Output |
|---|---|
| All annotators agree on class, strict majority present | One merged box (coordinate average), normal label |
| Only one annotator drew this object (singleton) | Original box, `_questionable` label |
| Annotators drew boxes here but disagree on class | Every source box emitted individually, each `_questionable` |
| Same class but below majority threshold | One merged box, `_questionable` label |
| One annotator drew 2+ overlapping boxes (dirty cluster) | One averaged box, `_questionable` label |

Class tie-breaks (50/50 split) fall under the "disagree on class" rule — all boxes are emitted as questionable.

| Annotators (N) | Keep if support ≥ | Questionable if support ≤ |
|---|---|---|
| 2 | 2 | 1 |
| 3 | 2 | 1 |
| 4 | 3 | 2 |

## Controversial Cases Report

When `--controversial-report` is passed, a second CSV (`controversial_report.csv`) is written to the same directory as the output. It contains one row per questionable cluster:

```
image_name,annotator_1,annotator_2,annotator_3
frame_042.jpg,car,car,
frame_042.jpg,truck,,
frame_101.jpg,car,bus,car
```

- `annotator_N` is the class label that annotator N assigned to that object
- Blank cell means that annotator did not place a box on this object
- Annotator column order matches the input file order

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
