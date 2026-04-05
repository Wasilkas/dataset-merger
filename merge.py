#!/usr/bin/env python3
"""Merge object detection CSV annotations from multiple annotators."""

import click
import yaml

from config import COL_IMAGE, COL_LABEL, QUESTIONABLE_SUFFIX, MergeConfig
from io import load_all, write_output
from kappa import print_kappa_report
from matching import build_clusters
from voting import process_all


def _load_config_file(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _build_config(
    config_path: str | None,
    output: str | None,
    iou_threshold: float | None,
    dist_threshold: float | None,
    no_questionable: bool,
    kappa: bool,
    image_width: int | None,
    image_height: int | None,
) -> MergeConfig:
    cfg = MergeConfig()

    if config_path:
        file_cfg = _load_config_file(config_path)
        if "iou_threshold" in file_cfg:
            cfg.iou_threshold = float(file_cfg["iou_threshold"])
        if "dist_threshold" in file_cfg:
            cfg.dist_threshold = float(file_cfg["dist_threshold"])
        if "output" in file_cfg:
            cfg.output = file_cfg["output"]
        if "no_questionable" in file_cfg:
            cfg.no_questionable = bool(file_cfg["no_questionable"])
        if "compute_kappa" in file_cfg:
            cfg.compute_kappa = bool(file_cfg["compute_kappa"])
        if "image_width" in file_cfg:
            cfg.image_width = int(file_cfg["image_width"])
        if "image_height" in file_cfg:
            cfg.image_height = int(file_cfg["image_height"])

    if output is not None:
        cfg.output = output
    if iou_threshold is not None:
        cfg.iou_threshold = iou_threshold
    if dist_threshold is not None:
        cfg.dist_threshold = dist_threshold
    if no_questionable:
        cfg.no_questionable = True
    if kappa:
        cfg.compute_kappa = True
    if image_width is not None:
        cfg.image_width = image_width
    if image_height is not None:
        cfg.image_height = image_height

    return cfg


@click.command()
@click.argument("files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("-o", "--output", default=None, help="Output CSV path (default: merged.csv)")
@click.option("--iou-threshold", default=None, type=float, help="IoU threshold for box matching (default: 0.5)")
@click.option("--dist-threshold", default=None, type=float, help="Center distance threshold in pixels (default: 20.0)")
@click.option("--no-questionable", is_flag=True, help="Drop questionable objects instead of flagging them")
@click.option("--kappa", is_flag=True, help="Compute pixel-wise Cohen's kappa (class-agnostic and class-specific)")
@click.option("--image-width", default=None, type=int, help="Canvas width for kappa in pixels (default: inferred from annotations)")
@click.option("--image-height", default=None, type=int, help="Canvas height for kappa in pixels (default: inferred from annotations)")
@click.option("--config", "config_path", default=None, type=click.Path(exists=True), help="YAML config file (CLI flags override)")
def main(files, output, iou_threshold, dist_threshold, no_questionable, kappa, image_width, image_height, config_path):
    if len(files) < 2:
        raise click.UsageError("At least 2 annotator CSV files are required.")

    cfg = _build_config(config_path, output, iou_threshold, dist_threshold, no_questionable, kappa, image_width, image_height)

    try:
        cfg.validate()
    except ValueError as e:
        raise click.BadParameter(str(e))

    click.echo(f"Loading {len(files)} annotator file(s)...")
    df, source_names = load_all(list(files))
    n_annotators = len(source_names)
    click.echo(f"Sources: {', '.join(source_names)}")
    click.echo(f"Total annotations loaded: {len(df)}")
    click.echo(
        f"Config: iou_threshold={cfg.iou_threshold}, dist_threshold={cfg.dist_threshold}, "
        f"no_questionable={cfg.no_questionable}"
    )

    if cfg.compute_kappa:
        print_kappa_report(df, cfg)

    clusters_by_group: dict[tuple[str, str], list] = {}
    for (image, cls), group in df.groupby([COL_IMAGE, COL_LABEL], sort=False):
        clusters_by_group[(image, cls)] = build_clusters(group, cfg)

    total_clusters = sum(len(v) for v in clusters_by_group.values())
    click.echo(f"Found {total_clusters} object candidate(s) across {len(clusters_by_group)} ({COL_IMAGE}, {COL_LABEL}) group(s)")

    result = process_all(df, n_annotators, clusters_by_group, cfg)

    confirmed = result[~result[COL_LABEL].str.endswith(QUESTIONABLE_SUFFIX)]
    questionable = result[result[COL_LABEL].str.endswith(QUESTIONABLE_SUFFIX)]
    click.echo(f"Confirmed: {len(confirmed)}  |  Questionable: {len(questionable)}")

    write_output(result, cfg.output)


if __name__ == "__main__":
    main()
