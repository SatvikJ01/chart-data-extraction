"""End-to-end orchestration.

Stage order matches the notebook: Donut -> axis ticks -> markers -> per-type
decoding -> submission. The structural difference is that every stage returns a
dict keyed on image id and every join is a dict lookup (bug 3).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import pandas as pd

from chart_extraction.axis.calibration import AxisCalibration
from chart_extraction.axis.inference import AxisTicks
from chart_extraction.axis.labels import build_axis_label_source
from chart_extraction.config import PipelineConfig
from chart_extraction.data.images import ImageRef, discover_images, require_all_ids
from chart_extraction.data.submission import build_submission
from chart_extraction.decoding import DecodeContext, build_decoder
from chart_extraction.donut.parsing import DonutPrediction
from chart_extraction.markers.inference import MarkerDetections
from chart_extraction.stages import MODE_DONUT_ONLY, MODE_FULL

logger = logging.getLogger(__name__)


@dataclass
class ImageOutcome:
    """Per-image record, including why a prediction is empty.

    The notebook could not distinguish "Donut produced nothing", "the chart type
    has no decoder", "no markers passed the threshold" and "the axis had no
    usable scale" -- all four collapsed to the same 0;0 placeholder. Phase 2's
    error taxonomy needs them separated, so they are recorded now.
    """

    image_id: str
    chart_type: str
    x_series: list = field(default_factory=list)
    y_series: list = field(default_factory=list)
    failure_mode: str | None = None
    #: Which pipeline mode produced this outcome. A donut_only outcome took its
    #: y series straight from Donut; a full outcome re-derived it from the
    #: detection stages. The two are not comparable.
    mode: str = MODE_FULL


def build_calibration(
    image_id: str,
    ticks: AxisTicks | None,
    prediction: DonutPrediction,
    label_source,
) -> AxisCalibration:
    """Build the y-axis calibration for one image via the configured seam.

    AUDIT NOTE: the notebook indexed ``df2['y_points'][i]`` with no guard, so an
    image whose axis model returned fewer than two ticks raised IndexError
    inside ``extend_y_axis``. Guarded here; a degenerate calibration is returned
    and the image is recorded as an axis failure instead of killing the run.
    """
    if ticks is None or len(ticks.y_points) < 2:
        return AxisCalibration(pixels=(), values=())
    return label_source.build_calibration(
        image_id, ticks.y_pixels, prediction.y
    )


def decode_all(
    refs: Sequence[ImageRef],
    donut_predictions: Mapping[str, DonutPrediction],
    axis_ticks: Mapping[str, AxisTicks] | None = None,
    marker_detections: Mapping[str, MarkerDetections] | None = None,
    config: PipelineConfig | None = None,
) -> dict[str, ImageOutcome]:
    """Join the available stages by image id and decode each image.

    In donut_only mode the detection mappings are unused and may be None.
    """
    config = config or PipelineConfig()
    axis_ticks = axis_ticks or {}
    marker_detections = marker_detections or {}
    label_source = build_axis_label_source(config.axis_label_source)

    image_ids = [r.image_id for r in refs]
    require_all_ids("donut", donut_predictions, image_ids)

    if config.mode not in (MODE_FULL, MODE_DONUT_ONLY):
        raise ValueError(f"unknown pipeline mode {config.mode!r}")

    outcomes: dict[str, ImageOutcome] = {}

    for image_id in image_ids:
        prediction = donut_predictions[image_id]
        chart_type = prediction.chart_type or config.placeholder_chart_type

        if not prediction.is_well_formed:
            outcomes[image_id] = ImageOutcome(
                image_id=image_id,
                chart_type=chart_type,
                failure_mode=prediction.failure_mode,
            )
            continue

        # x series comes straight from Donut for every chart type, exactly as in
        # the notebook. Only the y series is re-derived from the CV stages.
        outcome = ImageOutcome(
            image_id=image_id,
            chart_type=chart_type,
            x_series=list(prediction.x),
            mode=config.mode,
        )

        if config.mode == MODE_DONUT_ONLY:
            # Donut's generated y series is used as-is -- this is exactly what
            # tuned-donut did, and it is the configuration the published
            # leaderboard score for this checkpoint refers to. No axis
            # calibration, no marker detection, no per-chart-type decoder.
            outcome.y_series = list(prediction.y)
            if not outcome.y_series:
                outcome.failure_mode = "empty_series"
            outcomes[image_id] = outcome
            continue

        decoder = build_decoder(chart_type)
        if decoder is None:
            outcome.failure_mode = f"no_decoder_for_{chart_type}"
            outcomes[image_id] = outcome
            continue

        ticks = axis_ticks.get(image_id)
        detections = marker_detections.get(image_id)
        if detections is None:
            outcome.failure_mode = "missing_marker_detections"
            outcomes[image_id] = outcome
            continue

        calibration = build_calibration(image_id, ticks, prediction, label_source)
        if not calibration.is_usable:
            outcome.failure_mode = "unusable_axis_calibration"
            outcomes[image_id] = outcome
            continue

        ctx = DecodeContext(
            image_id=image_id,
            chart_type=chart_type,
            boxes=detections.boxes,
            labels=detections.labels,
            scores=detections.scores,
            calibration=calibration,
            x_tick_pixels=ticks.x_pixels if ticks is not None else [],
            donut_x=prediction.x,
            donut_y=prediction.y,
            marker_label_id=config.marker_label_id,
            score_threshold=config.marker_score_threshold,
        )

        try:
            outcome.y_series = decoder.decode(ctx)
        except Exception:
            logger.exception("decode failed for %s (%s)", image_id, chart_type)
            outcome.failure_mode = "decode_error"

        if not outcome.y_series and outcome.failure_mode is None:
            outcome.failure_mode = "no_markers_decoded"

        outcomes[image_id] = outcome

    return outcomes


def outcomes_to_submission(
    refs: Sequence[ImageRef],
    outcomes: Mapping[str, ImageOutcome],
    config: PipelineConfig | None = None,
) -> pd.DataFrame:
    config = config or PipelineConfig()
    image_ids = [r.image_id for r in refs]
    return build_submission(
        image_ids=image_ids,
        chart_types={i: o.chart_type for i, o in outcomes.items()},
        x_series={i: o.x_series for i, o in outcomes.items()},
        y_series={i: o.y_series for i, o in outcomes.items()},
        config=config,
    )


def failure_summary(outcomes: Mapping[str, ImageOutcome]) -> pd.DataFrame:
    """Counts per (chart_type, failure_mode). Groundwork for Phase 2."""
    rows = [
        {
            "chart_type": o.chart_type,
            "failure_mode": o.failure_mode or "ok",
        }
        for o in outcomes.values()
    ]
    if not rows:
        return pd.DataFrame(columns=["chart_type", "failure_mode", "count"])
    return (
        pd.DataFrame(rows)
        .value_counts(["chart_type", "failure_mode"])
        .reset_index(name="count")
        .sort_values(["chart_type", "count"], ascending=[True, False])
        .reset_index(drop=True)
    )


def run_pipeline(config: PipelineConfig, device: str = "cuda:0") -> pd.DataFrame:
    """Full run. Imports the heavy model deps lazily so the package stays
    importable (and unit-testable) without transformers/cv2 installed."""
    from transformers import DonutProcessor, VisionEncoderDecoderModel

    from chart_extraction.axis.inference import detect_axis_ticks
    from chart_extraction.axis.model import load_axis_model
    from chart_extraction.donut.inference import run_donut
    from chart_extraction.markers.inference import detect_markers
    from chart_extraction.markers.model import load_marker_model

    config.require_paths(
        "image_dir", "donut_model_dir", "x_axis_model_path",
        "y_axis_model_path", "marker_model_path",
    )
    refs = discover_images(config.image_dir, config.image_glob)
    logger.info("discovered %d images", len(refs))

    donut_model = VisionEncoderDecoderModel.from_pretrained(config.donut_model_dir).to(device)
    processor = DonutProcessor.from_pretrained(config.donut_model_dir)
    donut_predictions = run_donut(
        refs, donut_model, processor, config.generation, device=device,
        batch_size=config.donut_batch_size, num_workers=config.num_workers,
        random_padding=config.donut_random_padding,
    )

    model_x = load_axis_model(config.x_axis_model_path, config.axis_max_num_points, device)
    model_y = load_axis_model(config.y_axis_model_path, config.axis_max_num_points, device)
    axis_ticks = detect_axis_ticks(
        refs, model_x, model_y, device=device,
        batch_size=config.axis_batch_size, num_workers=config.num_workers,
    )

    marker_model = load_marker_model(
        config.marker_model_path, config.marker_num_classes, device
    )
    marker_detections = detect_markers(
        refs, marker_model, device=device,
        batch_size=config.marker_batch_size, num_workers=config.num_workers,
    )

    outcomes = decode_all(refs, donut_predictions, axis_ticks, marker_detections, config)
    logger.info("failure summary:\n%s", failure_summary(outcomes).to_string(index=False))
    return outcomes_to_submission(refs, outcomes, config)
