# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Use `uv` — never `pip`, `conda`, or the system Python.

```bash
uv venv && source .venv/bin/activate   # create and activate venv
uv sync                                # install dependencies from pyproject.toml / uv.lock
uv add <package>                       # add a dependency (updates pyproject.toml)
```

## Commands

```bash
# Run the merger
python src/merge.py annotator1.csv annotator2.csv [annotator3.csv ...] -o merged.csv

# Lint (ruff)
uvx ruff check .
uvx ruff format .

# Run all tests
uv run pytest tests/

# Run a single test file
uv run pytest tests/test_matching.py

# Run a single test by name
uv run pytest tests/test_voting.py::test_n3_two_of_three_keeps
```

## Architecture

Linear pipeline: **load → cluster → vote → write**

| File | Responsibility |
|---|---|
| [src/config.py](src/config.py) | `MergeConfig` dataclass — all thresholds and flags |
| [src/io.py](src/io.py) | CSV load/validate (adds `source` column), write output |
| [src/matching.py](src/matching.py) | IoU, center distance, union-find clustering |
| [src/voting.py](src/voting.py) | Majority vote, box averaging, questionable flagging |
| [src/kappa.py](src/kappa.py) | Pixel-wise Cohen's kappa (class-agnostic and class-specific) |
| [src/merge.py](src/merge.py) | `click` CLI, orchestrates the pipeline |

## Key Algorithm Details

**Box matching** (`src/matching.py`): Two boxes (same image, same class) are the same object if `IoU > iou_threshold` **OR** `center_distance < dist_threshold`. Uses union-find to find connected components — matching is transitive (A↔B, B↔C → A/B/C are one cluster even if A↔C don't directly match).

**Voting** (`src/voting.py`): A cluster is kept if `distinct_sources > N / 2.0` (strict majority). For N=2 this requires both annotators to agree. "Dirty clusters" — where one annotator contributed 2+ boxes — are always questionable.

**Questionable objects**: Emitted with `{class}_questionable` suffix. Suppressed with `--no-questionable`.

## CSV Schema

Input and output share the same schema:
```
image_name, instance_label, bbox_x_tl, bbox_y_tl, bbox_x_br, bbox_y_br
```

## CLI Options

```
python src/merge.py a.csv b.csv [c.csv ...]
  -o / --output PATH          Output file (default: merged.csv)
  --iou-threshold FLOAT       Default: 0.5
  --dist-threshold FLOAT      Center distance in pixels, default: 20.0
  --no-questionable           Drop uncertain detections
  --kappa                     Compute and print pixel-wise Cohen's kappa (both variants)
  --image-width INT           Canvas width for kappa (default: inferred from annotations)
  --image-height INT          Canvas height for kappa (default: inferred from annotations)
  --config PATH               YAML config file (CLI flags override)
```

Run `python src/merge.py --help` for full usage. Click handles exit codes automatically.

## Code Style

- PEP8 enforced via ruff (configured in `pyproject.toml`)
- Prefer readability over cleverness
- Do not duplicate code, variables that repeat more than 1 times at code should be moved to config file
