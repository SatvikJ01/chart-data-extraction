"""End-to-end decode/join tests using fabricated stage outputs.

No models are loaded: these exercise the joining and failure-recording logic,
which is where bug 3 and the placeholder-collapse problems lived.
"""

from __future__ import annotations

import numpy as np
import pytest

from chart_extraction.axis.inference import AxisTicks
from chart_extraction.data.images import ImageRef
from chart_extraction.donut.parsing import DonutPrediction
from chart_extraction.markers.inference import MarkerDetections
from chart_extraction.pipeline import (
    decode_all, failure_summary, outcomes_to_submission,
)


def _refs(*ids):
    return [ImageRef(image_id=i, path=f"/tmp/{i}.jpg") for i in ids]


def _ticks(image_id):
    return AxisTicks(
        image_id=image_id,
        x_points=np.array([[15.0, 210.0], [45.0, 210.0]]),
        y_points=np.array([[40.0, 100.0], [40.0, 120.0], [40.0, 140.0]]),
    )


def _detections(image_id, n=2):
    boxes = np.array([[10 + 30 * i, 95, 20 + 30 * i, 105] for i in range(n)], dtype=float)
    return MarkerDetections(
        image_id=image_id,
        boxes=boxes,
        labels=np.full(n, 3),
        scores=np.full(n, 0.9),
    )


def _prediction(image_id, chart_type="vertical_bar"):
    return DonutPrediction(
        image_id=image_id, chart_type=chart_type, x=["a", "b"], y=[10.0, 20.0, 30.0]
    )


def test_happy_path_produces_a_series_per_image():
    refs = _refs("a", "b")
    outcomes = decode_all(
        refs,
        {i.image_id: _prediction(i.image_id) for i in refs},
        {i.image_id: _ticks(i.image_id) for i in refs},
        {i.image_id: _detections(i.image_id) for i in refs},
    )
    assert set(outcomes) == {"a", "b"}
    for outcome in outcomes.values():
        assert outcome.failure_mode is None
        assert outcome.y_series
        assert outcome.x_series == ["a", "b"]


def test_bug3_shuffled_stage_dicts_give_identical_results():
    """The join is keyed on image id, so mapping order is irrelevant."""
    refs = _refs("a", "b", "c")
    donut = {i.image_id: _prediction(i.image_id) for i in refs}
    ticks = {i.image_id: _ticks(i.image_id) for i in refs}
    marks = {i.image_id: _detections(i.image_id) for i in refs}

    forward = decode_all(refs, donut, ticks, marks)
    backward = decode_all(
        refs,
        dict(reversed(list(donut.items()))),
        dict(reversed(list(ticks.items()))),
        dict(reversed(list(marks.items()))),
    )
    assert {k: v.y_series for k, v in forward.items()} == {
        k: v.y_series for k, v in backward.items()
    }


def test_missing_image_from_a_stage_raises():
    refs = _refs("a", "b")
    with pytest.raises(ValueError, match="missing"):
        decode_all(refs, {"a": _prediction("a")}, {}, {})


@pytest.mark.parametrize(
    "setup,expected_mode",
    [
        ("malformed_donut", "missing_series_delimiters"),
        ("no_markers", "missing_marker_detections"),
        ("bad_axis", "unusable_axis_calibration"),
    ],
)
def test_failure_modes_are_distinguished(setup, expected_mode):
    """The notebook collapsed every one of these to the same 0;0 placeholder,
    which is why malformed-sequence rate was unmeasurable. Phase 2 needs them
    separated."""
    refs = _refs("a")
    donut = {"a": _prediction("a")}
    ticks = {"a": _ticks("a")}
    marks = {"a": _detections("a")}

    if setup == "malformed_donut":
        donut = {"a": DonutPrediction("a", "line", failure_mode="missing_series_delimiters")}
    elif setup == "no_markers":
        marks = {}
    elif setup == "bad_axis":
        ticks = {"a": AxisTicks("a", np.empty((0, 2)), np.empty((0, 2)))}

    outcomes = decode_all(refs, donut, ticks, marks)
    assert outcomes["a"].failure_mode == expected_mode


def test_submission_shape_and_placeholders():
    refs = _refs("a", "b")
    outcomes = decode_all(
        refs,
        {"a": _prediction("a"), "b": DonutPrediction("b", "line", failure_mode="empty_series")},
        {i.image_id: _ticks(i.image_id) for i in refs},
        {i.image_id: _detections(i.image_id) for i in refs},
    )
    sub = outcomes_to_submission(refs, outcomes)

    assert list(sub.columns) == ["id", "data_series", "chart_type"]
    assert len(sub) == 4
    assert set(sub["id"]) == {"a_x", "a_y", "b_x", "b_y"}
    # The failed image falls back to the placeholder rather than an empty cell.
    assert sub.set_index("id").loc["b_y", "data_series"] == "0;0"
    assert (sub["data_series"].str.len() > 0).all()


def test_failure_summary_groups_by_type_and_mode():
    refs = _refs("a", "b")
    outcomes = decode_all(
        refs,
        {"a": _prediction("a"), "b": DonutPrediction("b", "line", failure_mode="empty_series")},
        {i.image_id: _ticks(i.image_id) for i in refs},
        {i.image_id: _detections(i.image_id) for i in refs},
    )
    summary = failure_summary(outcomes)
    assert set(summary.columns) == {"chart_type", "failure_mode", "count"}
    assert summary["count"].sum() == 2
