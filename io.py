import sys
import pandas as pd
from pathlib import Path

from config import COL_IMAGE, COL_LABEL, COL_SOURCE, COL_X1, COL_X2, COL_Y1, COL_Y2
from config import COORD_COLS, OUTPUT_COLS, REQUIRED_COLUMNS


def load_annotator_csv(path: str, source_name: str | None = None) -> pd.DataFrame:
    """Load and validate a single annotator CSV. Returns a DataFrame with a 'source' column."""
    path = Path(path)
    if not path.exists():
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"ERROR: Cannot read {path}: {e}", file=sys.stderr)
        sys.exit(1)

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        print(f"ERROR: {path} is missing columns: {missing}", file=sys.stderr)
        sys.exit(1)

    # Coerce coordinate columns to numeric, drop unconvertible rows
    before = len(df)
    df[COORD_COLS] = df[COORD_COLS].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=COORD_COLS)

    # Validate coordinate sanity: tl < br
    valid_mask = (df[COL_X1] < df[COL_X2]) & (df[COL_Y1] < df[COL_Y2])
    bad = (~valid_mask).sum()
    if bad > 0:
        print(f"WARNING: {path}: skipping {bad} row(s) with invalid coordinates (tl >= br)")
    df = df[valid_mask].copy()

    dropped = before - len(df)
    if dropped > 0:
        print(f"WARNING: {path}: dropped {dropped} invalid row(s)")

    df[COL_SOURCE] = source_name or str(path)
    df = df.reset_index(drop=True)
    return df[OUTPUT_COLS + [COL_SOURCE]]


def load_all(paths: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Load multiple annotator CSVs. Returns concatenated DataFrame and list of source names."""
    source_names = [Path(p).stem for p in paths]
    frames = [load_annotator_csv(p, name) for p, name in zip(paths, source_names)]
    combined = pd.concat(frames, ignore_index=True)
    return combined, source_names


def write_output(df: pd.DataFrame, path: str) -> None:
    """Write the merged output CSV, dropping the 'source' column if present."""
    df[OUTPUT_COLS].to_csv(path, index=False)
    print(f"Wrote {len(df)} annotations to {path}")
