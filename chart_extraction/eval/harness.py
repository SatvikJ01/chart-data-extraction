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
from chart_extraction.eval.sanity import check_against_reference, composition_from_scores
from chart_extraction.eval.taxonomy import taxonomy_report
from chart_extraction.stages import MODE_DONUT_ONLY, MODE_FULL
from chart_extraction.pipeline import ImageOutcome, decode_all

logger = logging.getLogger(__name__)

#: Carried into every result file. See splits.py for the full statement. This
#: half is true of every run regardless of what it scored.
LEAKAGE_CAVEAT = (
    "Validation ids are held out with respect to future training in this repo "
    "only. The checkpoints under evaluation were fine-tuned elsewhere on "
    "train/ with no recorded partition, so these images may have been in their "
    "training data. Scores are optimistic for these checkpoints."
)


def holdout_caveat(provenance: dict | None) -> str | None:
    """Replace the generic leakage caveat when a run scores a real held-out set.

    A set held out from the specialization fine-tune was genuinely never
    optimised against -- but it was not held out from the **base** checkpoint
    that fine-tune started from, whose own partition is unrecorded. So the
    absolute score is still optimistic; what is defensible is the difference
    between two models scored on the same set, since both inherit the same base
    history. Saying otherwise would overclaim exactly the thing this experiment
    exists to establish.
    """
    if not provenance or not provenance.get("held_out_from_finetune"):
        return None

    seed = provenance.get("seed")
    n_holdout = provenance.get("n_holdout")
    if provenance.get("held_out_from_base_checkpoint"):
        return (
            f"Scored on {n_holdout} images held out from all training "
            f"(seed {seed}). This number is leakage-free."
        )
    return (
        f"PARTIALLY LEAKAGE-FREE. These {n_holdout} images were held out from "
        f"the specialization fine-tune (seed {seed}) and were never optimised "
        "against by it, nor used for checkpoint selection. They were NOT held "
        "out from the base checkpoint the fine-tune started from, whose own "
        "train/validation partition is unrecorded, so the absolute score is "
        "still optimistic. The defensible quantity is the DIFFERENCE between "
        "models scored on this same set, since both inherit the same base "
        "checkpoint history."
    )


def distribution_caveat(composition) -> str | None:
    """The distribution half of the caveat, or None when it does not apply.

    Built from what the run actually scored. An extracted-only run contains no
    synthetic data, so asserting "the generated slice is easier" there would be
    describing data the run never touched.
    """
    if not composition.total:
        return None
    if not composition.distribution_applies:
        return (
            "This run scored extracted (real textbook) charts only -- the same "
            "kind the competition test set was weighted toward -- so the "
            "synthetic-is-easier caveat does not apply to this number. Leakage "
            "above is the only known upward pressure."
        )
    if composition.extracted_fraction == 0:
        return (
            "This run scored generated (synthetic) charts only. That is a far "
            "easier distribution than the competition test set, so this number "
            "is not indicative of real-world performance. Score the extracted "
            "slice for the headline number."
        )
    return (
        f"{composition.generated_fraction:.0%} of the instances scored were "
        "generated (synthetic), a far easier distribution than the competition "
        "test set. The extracted slice is the headline number."
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

    # Standard error clustered by image. The x and y instances of one image
    # share a chart type, a generation and a failure mode, so they are strongly
    # correlated; treating 2N instances as 2N independent draws would understate
    # the error by up to sqrt(2). Averaging within an image first and taking the
    # error across images avoids that.
    per_image = frame.groupby("image_id")["score"].mean()
    n_images = len(per_image)
    stderr = (
        float(per_image.std(ddof=1) / np.sqrt(n_images)) if n_images > 1 else 0.0
    )

    return {
        "overall": round(float(frame["score"].mean()), 6),
        "n_instances": int(len(frame)),
        "n_images": int(n_images),
        "stderr": round(stderr, 6),
        "ci95": [
            round(float(frame["score"].mean()) - 1.96 * stderr, 6),
            round(float(frame["score"].mean()) + 1.96 * stderr, 6),
        ],
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
    runtime: dict = field(default_factory=dict)
    oom: dict = field(default_factory=dict)
    stages: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
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
            "runtime": self.runtime,
            "oom": self.oom,
            "stages": self.stages,
            "warnings": self.warnings,
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
    runtime: dict | None = None,
    oom_policy=None,
    stages: dict | None = None,
    sampling: dict | None = None,
    provenance: dict | None = None,
    model_tag: str | None = None,
) -> EvaluationResult:
    """Aggregate one pass. Pure -- no models are run here."""
    image_ids = [r.image_id for r in refs]

    with _timed(timings, "decode_s"):
        outcomes = decode_all(refs, donut_predictions, axis_ticks, marker_detections, config)

    scores = score_breakdown(outcomes, annotations, image_ids)
    per_instance = scores.pop("per_instance", None)

    oom_summary = oom_policy.summary() if oom_policy is not None else {}
    composition = composition_from_scores(scores.get("by_source"))

    # A recorded holdout replaces the generic leakage caveat with a precise
    # statement of what was and was not held out.
    holdout = holdout_caveat(provenance)
    caveats = [holdout] if holdout else [LEAKAGE_CAVEAT]

    distribution = distribution_caveat(composition)
    if distribution:
        caveats.append(distribution)

    if sampling and sampling.get("sampled"):
        stderr = scores.get("stderr", 0.0)
        caveats.append(
            f"SUBSAMPLED RUN: {sampling['n_selected']} of "
            f"{sampling['n_population']} images "
            f"({sampling.get('fraction', 0):.1%}), stratified on "
            f"(source, chart_type), seed {sampling['seed']}. The score carries "
            f"sampling error of roughly +/-{1.96 * stderr:.4f} at 95% "
            "confidence (clustered by image). Differences smaller than that "
            "between this row and another are not measurable at this sample "
            "size. Reproduce exactly with --sample "
            f"{sampling['n_requested']} --seed {sampling['seed']}."
        )

    if config.mode == MODE_DONUT_ONLY:
        caveats.append(
            "DONUT-ONLY RUN. The axis CNN and marker detector did not run; "
            "Donut's generated series was used directly. This is a different "
            "system from the full pipeline and its score must never be "
            "compared with, or reported as, a full-pipeline score."
        )
    if oom_summary.get("n_recovered"):
        # An OOM retry lowers that image's Donut input resolution, which changes
        # its prediction. A run containing degraded images is not directly
        # comparable with one that has none, so the caveat travels with the row
        # rather than sitting only in the log.
        caveats.append(
            f"{oom_summary['n_recovered']} image(s) were processed at reduced "
            "Donut input resolution after an out-of-memory retry and are not "
            "directly comparable with the rest of this run. See oom.events."
        )
    if oom_summary.get("n_unrecovered"):
        caveats.append(
            f"{oom_summary['n_unrecovered']} image(s) could not be processed at "
            "any resolution and scored as failures."
        )

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
            "mode": config.mode,
            "model_tag": model_tag or "base",
            "axis_label_source": config.axis_label_source,
            "marker_score_threshold": config.marker_score_threshold,
            "donut_random_padding": config.donut_random_padding,
            # Latency is uninterpretable without the batch sizes it was
            # measured at, so they are part of the recorded config.
            "donut_batch_size": config.donut_batch_size,
            "axis_batch_size": config.axis_batch_size,
            "marker_batch_size": config.marker_batch_size,
            "python": platform.python_version(),
        },
        split={
            "name": split.name,
            "salt": split.salt,
            "fraction": split.fraction,
            # What was actually scored. A --subset run evaluates a fraction of
            # its split, so reporting the split's own totals here would
            # overstate the run.
            "n_images": len(image_ids),
            "evaluated_composition": composition.as_dict(),
            # Present and sampled=True only for a --sample run. Its absence
            # means every image of the selected subset was evaluated.
            "sampling": sampling or {"sampled": False},
            "provenance": provenance or {},
            # The split this run was drawn from, for reproducibility.
            "source_split_n_images": len(split),
            "source_split_composition": split.composition,
        },
        scores=scores,
        taxonomy=taxonomy_report(outcomes, annotations, config.mode),
        latency=timings.per_image_ms() | {"total_wall_s": round(timings.total_s, 3)},
        models=models,
        populations=populations,
        runtime=runtime or {},
        oom=oom_summary,
        stages=stages or {"mode": config.mode},
        warnings=[
            w.as_dict()
            for w in check_against_reference(
                scores.get("overall", 0.0),
                config.mode,
                scores.get("n_instances", 0),
                composition=composition,
            )
        ],
        caveats=caveats,
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
