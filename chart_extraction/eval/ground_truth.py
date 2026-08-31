"""Loader for the competition's ``train/annotations`` JSON.

Schema (hyphenated keys, which is easy to get wrong)::

    {
      "source": "generated" | "extracted",
      "chart-type": "line" | "scatter" | "dot" | "vertical_bar" | "horizontal_bar",
      "plot-bb": {...},
      "text": [...],
      "axes": {"x-axis": {"ticks": [...]}, "y-axis": {...}},
      "data-series": [{"x": <str|num>, "y": <str|num>}, ...],
      "visual-elements": {...}
    }

Only ``source``, ``chart-type`` and ``data-series`` are needed for scoring. The
axes and text blocks are the ground truth a future OCR axis-label source would
be measured against (Phase 3), and are deliberately not read here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

VALID_SOURCES = ("generated", "extracted")
VALID_CHART_TYPES = (
    "line", "scatter", "dot", "vertical_bar", "horizontal_bar",
)


@dataclass(frozen=True)
class Annotation:
    """Ground truth for one image."""

    image_id: str
    source: str
    chart_type: str
    x_series: tuple
    y_series: tuple

    @property
    def length(self) -> int:
        return len(self.x_series)


def parse_annotation(image_id: str, payload: dict) -> Annotation:
    """Parse one annotation dict.

    Raises on a missing required key rather than defaulting, because a silently
    defaulted chart type or source would corrupt the split and every per-type
    breakdown built on it.
    """
    for key in ("source", "chart-type", "data-series"):
        if key not in payload:
            raise KeyError(f"{image_id}: annotation missing required key {key!r}")

    source = payload["source"]
    chart_type = payload["chart-type"]

    if source not in VALID_SOURCES:
        raise ValueError(f"{image_id}: unexpected source {source!r}")
    if chart_type not in VALID_CHART_TYPES:
        raise ValueError(f"{image_id}: unexpected chart-type {chart_type!r}")

    series = payload["data-series"]
    x_series, y_series = [], []
    for point in series:
        if "x" not in point or "y" not in point:
            raise KeyError(f"{image_id}: data-series point missing x or y: {point!r}")
        x_series.append(point["x"])
        y_series.append(point["y"])

    return Annotation(
        image_id=image_id,
        source=source,
        chart_type=chart_type,
        x_series=tuple(x_series),
        y_series=tuple(y_series),
    )


def load_annotation(path: Path | str) -> Annotation:
    path = Path(path)
    with open(path) as handle:
        payload = json.load(handle)
    return parse_annotation(path.stem, payload)


def load_annotations(
    annotation_dir: Path | str,
    image_ids: Sequence[str] | None = None,
    strict: bool = True,
) -> dict[str, Annotation]:
    """Load annotations, keyed on image id.

    ``image_ids`` restricts the load to a split, so an evaluation pass never
    touches the 60k-file directory when it only needs a few thousand.
    """
    annotation_dir = Path(annotation_dir)
    if not annotation_dir.is_dir():
        raise NotADirectoryError(f"annotation_dir does not exist: {annotation_dir}")

    if image_ids is None:
        paths = sorted(annotation_dir.glob("*.json"))
    else:
        paths = [annotation_dir / f"{image_id}.json" for image_id in image_ids]

    annotations: dict[str, Annotation] = {}
    for path in paths:
        if not path.exists():
            if strict:
                raise FileNotFoundError(f"missing annotation: {path}")
            continue
        annotations[path.stem] = load_annotation(path)
    return annotations


def annotations_to_frame(annotations: Iterable[Annotation]) -> pd.DataFrame:
    """Build the scorer's ground-truth frame.

    One row per (image, axis) instance, indexed ``<image_id>_x`` /
    ``<image_id>_y`` -- the competition's instance granularity.
    """
    rows = []
    for annotation in annotations:
        rows.append(
            {
                "id": f"{annotation.image_id}_x",
                "data_series": list(annotation.x_series),
                "chart_type": annotation.chart_type,
            }
        )
        rows.append(
            {
                "id": f"{annotation.image_id}_y",
                "data_series": list(annotation.y_series),
                "chart_type": annotation.chart_type,
            }
        )
    frame = pd.DataFrame(rows, columns=["id", "data_series", "chart_type"])
    return frame.set_index("id")


def summarise(annotations: Iterable[Annotation]) -> pd.DataFrame:
    """Counts per (source, chart_type)."""
    rows = [
        {"source": a.source, "chart_type": a.chart_type} for a in annotations
    ]
    if not rows:
        return pd.DataFrame(columns=["source", "chart_type", "count"])
    return (
        pd.DataFrame(rows)
        .value_counts(["source", "chart_type"])
        .reset_index(name="count")
        .sort_values(["source", "count"], ascending=[True, False])
        .reset_index(drop=True)
    )
