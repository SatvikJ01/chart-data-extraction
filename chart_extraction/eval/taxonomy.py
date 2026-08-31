"""Phase 2 error taxonomy.

Maps the pipeline's fine-grained ``failure_mode`` strings onto the four
categories the project brief asks for, and adds the one category that cannot be
derived without ground truth.

    malformed_sequence  - Donut emitted something unparseable
    wrong_chart_type    - parsed fine, but classified the chart wrongly
    axis_misestimation  - no usable axis calibration
    marker_miss         - axis fine, but no markers survived detection/threshold

``wrong_chart_type`` is deliberately not a pipeline failure_mode: the pipeline
cannot know it is wrong. It is assigned here by comparing against the
annotation, which is why the taxonomy lives in the eval package rather than
alongside failure_summary().
"""

from __future__ import annotations

from typing import Mapping

from chart_extraction.eval.ground_truth import Annotation
from chart_extraction.pipeline import ImageOutcome
from chart_extraction.stages import MODE_DONUT_ONLY

MALFORMED_SEQUENCE = "malformed_sequence"
WRONG_CHART_TYPE = "wrong_chart_type"
AXIS_MISESTIMATION = "axis_misestimation"
MARKER_MISS = "marker_miss"
UNSUPPORTED_CHART_TYPE = "unsupported_chart_type"
DECODE_ERROR = "decode_error"
OK = "ok"

CATEGORIES = (
    OK,
    MALFORMED_SEQUENCE,
    WRONG_CHART_TYPE,
    AXIS_MISESTIMATION,
    MARKER_MISS,
    UNSUPPORTED_CHART_TYPE,
    DECODE_ERROR,
)

#: pipeline failure_mode -> taxonomy category
_FAILURE_MODE_MAP = {
    "no_chart_type_token": MALFORMED_SEQUENCE,
    "missing_series_delimiters": MALFORMED_SEQUENCE,
    "empty_series": MALFORMED_SEQUENCE,
    "generation_error": MALFORMED_SEQUENCE,
    "unusable_axis_calibration": AXIS_MISESTIMATION,
    "missing_marker_detections": MARKER_MISS,
    "no_markers_decoded": MARKER_MISS,
    "decode_error": DECODE_ERROR,
}


def categorise(outcome: ImageOutcome, annotation: Annotation | None) -> str:
    """Assign one outcome to a taxonomy category.

    Order matters. A malformed sequence is reported as such even when the chart
    type also happens to be wrong, because the malformed sequence is the
    upstream cause -- reporting the downstream symptom would misattribute it.
    """
    mode = outcome.failure_mode

    if mode and mode.startswith("no_decoder_for_"):
        return UNSUPPORTED_CHART_TYPE

    if mode in _FAILURE_MODE_MAP:
        category = _FAILURE_MODE_MAP[mode]
        if category is MALFORMED_SEQUENCE:
            return MALFORMED_SEQUENCE
        # An axis or marker failure on an image whose chart type is already
        # wrong is not really an axis/marker problem.
        if annotation and outcome.chart_type != annotation.chart_type:
            return WRONG_CHART_TYPE
        return category

    if annotation and outcome.chart_type != annotation.chart_type:
        return WRONG_CHART_TYPE

    if mode:
        return mode

    return OK


#: Categories that only exist when the detection stages run. In donut_only mode
#: they are structurally impossible, and reporting them as 0 would read as
#: "the detector missed nothing" rather than "there was no detector".
DETECTION_ONLY_CATEGORIES = (AXIS_MISESTIMATION, MARKER_MISS, UNSUPPORTED_CHART_TYPE)


def applicable_categories(mode: str) -> tuple[str, ...]:
    """Categories that can occur in a given pipeline mode."""
    if mode == MODE_DONUT_ONLY:
        return tuple(c for c in CATEGORIES if c not in DETECTION_ONLY_CATEGORIES)
    return CATEGORIES


def taxonomy_counts(
    outcomes: Mapping[str, ImageOutcome],
    annotations: Mapping[str, Annotation],
    mode: str = "full",
) -> dict[str, int]:
    """Category counts over a whole split.

    Only categories that are *possible* in this mode are reported. A donut_only
    run omits marker_miss and axis_misestimation entirely rather than reporting
    them as zero, because a zero there would be read as a clean detection pass
    instead of an absent one. ``not_applicable`` lists what was omitted.
    """
    applicable = applicable_categories(mode)
    counts = {category: 0 for category in applicable}
    for image_id, outcome in outcomes.items():
        category = categorise(outcome, annotations.get(image_id))
        counts[category] = counts.get(category, 0) + 1
    return counts


def taxonomy_report(
    outcomes: Mapping[str, ImageOutcome],
    annotations: Mapping[str, Annotation],
    mode: str = "full",
) -> dict:
    """Counts plus an explicit record of what this mode could not produce."""
    omitted = [c for c in CATEGORIES if c not in applicable_categories(mode)]
    return {
        "mode": mode,
        "counts": taxonomy_counts(outcomes, annotations, mode),
        "by_ground_truth_chart_type": taxonomy_by_chart_type(outcomes, annotations),
        "not_applicable": omitted,
    }


def taxonomy_by_chart_type(
    outcomes: Mapping[str, ImageOutcome],
    annotations: Mapping[str, Annotation],
) -> dict[str, dict[str, int]]:
    """Category counts broken down by *ground-truth* chart type.

    Grouping on ground truth rather than prediction is deliberate: grouping on
    the prediction would hide wrong-chart-type errors inside whichever type the
    model wrongly guessed.
    """
    out: dict[str, dict[str, int]] = {}
    for image_id, outcome in outcomes.items():
        annotation = annotations.get(image_id)
        chart_type = annotation.chart_type if annotation else "unknown"
        bucket = out.setdefault(chart_type, {c: 0 for c in CATEGORIES})
        category = categorise(outcome, annotation)
        bucket[category] = bucket.get(category, 0) + 1
    return out
