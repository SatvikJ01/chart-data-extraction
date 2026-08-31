"""Single-pass evaluation harness.

One pass over a split produces every Phase 1 and Phase 2 number together:
overall score, per-chart-type, per-source (extracted is the headline), latency,
model sizes, and the error taxonomy. Splitting these across separate passes
would burn a GPU cycle per number for no benefit.
"""

from __future__ import annotations

import logging
import platform
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from chart_extraction.config import PipelineConfig
from chart_extraction.data.images import ImageRef
from chart_extraction.eval.ground_truth import Annotation, annotations_to_frame
from chart_extraction.eval.metric import score_instance
from chart_extraction.eval.splits import Split
from chart_extraction.eval.taxonomy import taxonomy_by_chart_type, taxonomy_counts
from chart_extraction.pipeline import ImageOutcome, decode_all

logger = logging.getLogger(__name__)

#: Carried into every result file. See splits.py for the full statement.
LEAKAGE_CAVEAT = (
    "Validation ids are held out with respect to future training in this repo "
    "only. The checkpoints under evaluation were fine-tuned elsewhere on "
    "train/ with no recorded partition, so these images may have been in their "
    "training data. Scores are optimistic for these checkpoints. Separately, "
    "the generated slice is a far easier distribution than the competition's "
    "test set; the extracted slice is the headline number."
)


@dataclass
class StageTimings:
    """Wall-clock seconds per stage, and the image count they covered."""

    n_images: int = 0
    donut_s: float = 0.0
    axis_s: float = 0.0
    markers_s: float = 0.0
    decode_s: float = 0.0

    @property
    def total_s(self) -> float:
        return self.donut_s + self.axis_s + self.markers_s + self.decode_s

    def per_image_ms(self) -> dict[str, float]:
        n = max(self.n_images, 1)
        return {
            "donut_ms": 1000.0 * self.donut_s / n,
            "axis_ms": 1000.0 * self.axis_s / n,
            "markers_ms": 1000.0 * self.markers_s / n,
            "decode_ms": 1000.0 * self.decode_s / n,
            "total_ms": 1000.0 * self.total_s / n,
        }


@contextmanager
def _timed(timings: StageTimings, field_name: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        setattr(timings, field_name, getattr(timings, field_name) + time.perf_counter() - start)


def model_size(model, name: str) -> dict:
    """Parameter count and dense float32-equivalent size in MB."""
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    n_bytes += sum(b.numel() * b.element_size() for b in model.buffers())
    return {
        "name": name,
        "parameters": int(n_params),
        "trainable_parameters": int(n_trainable),
        "size_mb": round(n_bytes / (1024 * 1024), 2),
    }


def outcomes_to_prediction_frame(
    outcomes: Mapping[str, ImageOutcome], image_ids: Sequence[str]
) -> pd.DataFrame:
    """Build the scorer's prediction frame at (image, axis) granularity."""
    rows = []
    for image_id in image_ids:
        outcome = outcomes[image_id]
        rows.append(
            {
                "id": f"{image_id}_x",
                "data_series": list(outcome.x_series),
                "chart_type": outcome.chart_type,
            }
        )
        rows.append(
            {
                "id": f"{image_id}_y",
                "data_series": list(outcome.y_series),
                "chart_type": outcome.chart_type,
            }
        )
    return pd.DataFrame(rows, columns=["id", "data_series", "chart_type"]).set_index("id")


def score_breakdown(
    outcomes: Mapping[str, ImageOutcome],
    annotations: Mapping[str, Annotation],
    image_ids: Sequence[str],
) -> dict:
    """Score every instance once, then aggregate every way at once.

    Per-instance scores are computed a single time and grouped afterwards, so
    the overall, per-chart-type and per-source numbers are guaranteed to be
    consistent aggregations of the same values.
    """
    truth = annotations_to_frame(annotations[i] for i in image_ids)
    predictions = outcomes_to_prediction_frame(outcomes, image_ids)
    predictions = predictions.reindex(truth.index)

    records = []
    for instance_id in truth.index:
        gt_row = truth.loc[instance_id]
        pred_row = predictions.loc[instance_id]
        image_id, axis = instance_id.rsplit("_", 1)
        annotation = annotations[image_id]

        pred_series = pred_row["data_series"]
        pred_type = pred_row["chart_type"]
        if not isinstance(pred_series, list):  # reindex produced NaN
            pred_series, pred_type = [], ""

        records.append(
            {
                "instance_id": instance_id,
                "image_id": image_id,
                "axis": axis,
                "source": annotation.source,
                "chart_type": annotation.chart_type,
                "predicted_chart_type": pred_type,
                "score": score_instance(
                    gt_row["data_series"], gt_row["chart_type"], pred_series, pred_type
                ),
            }
        )

    frame = pd.DataFrame(records)
    if frame.empty:
        return {"overall": 0.0, "n_instances": 0, "by_chart_type": {}, "by_source": {}}

    def _group(column: str) -> dict:
        grouped = frame.groupby(column)["score"].agg(["mean", "count"])
        return {
            str(key): {"score": round(float(row["mean"]), 6), "n_instances": int(row["count"])}
            for key, row in grouped.iterrows()
        }

    by_source_and_type: dict[str, dict] = {}
    for (source, chart_type), group in frame.groupby(["source", "chart_type"]):
        by_source_and_type.setdefault(str(source), {})[str(chart_type)] = {
            "score": round(float(group["score"].mean()), 6),
            "n_instances": int(len(group)),
        }

    chart_type_accuracy = float(
        (frame["chart_type"] == frame["predicted_chart_type"]).mean()
    )

    return {
        "overall": round(float(frame["score"].mean()), 6),
        "n_instances": int(len(frame)),
        "by_chart_type": _group("chart_type"),
        "by_source": _group("source"),
        "by_axis": _group("axis"),
        "by_source_and_chart_type": by_source_and_type,
        "chart_type_accuracy": round(chart_type_accuracy, 6),
        "per_instance": frame,
    }


@dataclass
class EvaluationResult:
    """Everything one pass produced."""

    run_id: str
    config: dict
    split: dict
    scores: dict
    taxonomy: dict
    latency: dict
    models: list
    populations: dict
    caveats: list = field(default_factory=lambda: [LEAKAGE_CAVEAT])
    per_instance: pd.DataFrame | None = None

    @property
    def headline(self) -> float:
        """The extracted-slice score -- the number that goes on the CV."""
        return self.scores.get("by_source", {}).get("extracted", {}).get("score", 0.0)

    @property
    def generated_score(self) -> float:
        return self.scores.get("by_source", {}).get("generated", {}).get("score", 0.0)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "config": self.config,
            "split": self.split,
            "scores": {k: v for k, v in self.scores.items() if k != "per_instance"},
            "taxonomy": self.taxonomy,
            "latency": self.latency,
            "models": self.models,
            "populations": self.populations,
            "caveats": self.caveats,
        }


def evaluate(
    refs: Sequence[ImageRef],
    annotations: Mapping[str, Annotation],
    donut_predictions: Mapping,
    axis_ticks: Mapping,
    marker_detections: Mapping,
    timings: StageTimings,
    models: list,
    config: PipelineConfig,
    split: Split,
    run_id: str | None = None,
) -> EvaluationResult:
    """Aggregate one pass. Pure -- no models are run here."""
    image_ids = [r.image_id for r in refs]

    with _timed(timings, "decode_s"):
        outcomes = decode_all(refs, donut_predictions, axis_ticks, marker_detections, config)

    scores = score_breakdown(outcomes, annotations, image_ids)
    per_instance = scores.pop("per_instance", None)

    populations = {
        # Finding F: horizontal_bar had no decode branch in the notebook and
        # still has no x-axis calibration. This is the size of the affected
        # population, which is what decides whether Phase 3 should build one.
        "horizontal_bar_ground_truth": sum(
            1 for i in image_ids if annotations[i].chart_type == "horizontal_bar"
        ),
        "horizontal_bar_predicted": sum(
            1 for o in outcomes.values() if o.chart_type == "horizontal_bar"
        ),
        "by_ground_truth_chart_type": {
            chart_type: sum(1 for i in image_ids if annotations[i].chart_type == chart_type)
            for chart_type in sorted({annotations[i].chart_type for i in image_ids})
        },
    }

    return EvaluationResult(
        run_id=run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        config={
            "generation": _generation_label(config),
            "num_beams": config.generation.num_beams,
            "do_sample": config.generation.do_sample,
            "axis_label_source": config.axis_label_source,
            "marker_score_threshold": config.marker_score_threshold,
            "donut_random_padding": config.donut_random_padding,
            "python": platform.python_version(),
        },
        split={
            "name": split.name,
            "salt": split.salt,
            "fraction": split.fraction,
            "n_images": len(split),
            "composition": split.composition,
        },
        scores=scores,
        taxonomy={
            "counts": taxonomy_counts(outcomes, annotations),
            "by_ground_truth_chart_type": taxonomy_by_chart_type(outcomes, annotations),
        },
        latency=timings.per_image_ms() | {"total_wall_s": round(timings.total_s, 3)},
        models=models,
        populations=populations,
        per_instance=per_instance,
    )


def _generation_label(config: PipelineConfig) -> str:
    """Human-readable decoding label.

    Never describes a run as temperature or nucleus tuning. Neither notebook set
    do_sample, so those parameters were inert; the only real axis of variation
    is beam width (Phase 0 finding E).
    """
    generation = config.generation
    if generation.do_sample:
        parts = [f"sampling(top_k={generation.top_k}, top_p={generation.top_p}"]
        parts.append(f", temperature={generation.temperature})")
        return "".join(parts)
    if generation.num_beams == 1:
        return "greedy"
    return f"beam_search(num_beams={generation.num_beams})"
