"""Tests for the decoders, including bug 2 (line charts stubbed)."""

from __future__ import annotations

import numpy as np
import pytest

from chart_extraction.axis.calibration import AxisCalibration
from chart_extraction.decoding import DecodeContext, build_decoder, available_decoders


@pytest.fixture
def calibration():
    # 50px -> 30.0 down to 200px -> 0.0, i.e. -0.2 units per pixel.
    return AxisCalibration.from_ticks([50.0, 100.0, 150.0, 200.0], [30.0, 20.0, 10.0, 0.0])


def make_ctx(calibration, boxes, scores=None, x_ticks=(), chart_type="vertical_bar"):
    boxes = np.asarray(boxes, dtype=float)
    n = len(boxes)
    return DecodeContext(
        image_id="img",
        chart_type=chart_type,
        boxes=boxes,
        labels=np.full(n, 3),
        scores=np.full(n, 0.9) if scores is None else np.asarray(scores, dtype=float),
        calibration=calibration,
        x_tick_pixels=list(x_ticks),
        donut_x=[],
        donut_y=[],
    )


def test_bug2_line_decoder_is_registered_and_produces_real_values(calibration):
    """Every line chart got a hardcoded [0.0, 0.0] in the notebook -- the
    PredLinePlot call was commented out. It must now decode."""
    ctx = make_ctx(
        calibration,
        boxes=[[10, 95, 20, 105], [40, 145, 50, 155], [70, 45, 80, 55]],
        x_ticks=[15.0, 45.0, 75.0],
        chart_type="line",
    )
    result = build_decoder("line").decode(ctx)
    assert result != [0.0, 0.0], "line decoding must not return the old stub"
    assert result == pytest.approx([20.0, 10.0, 30.0])


def test_bug2_line_decoder_samples_one_value_per_x_tick(calibration):
    ctx = make_ctx(
        calibration,
        boxes=[[10, 95, 20, 105], [40, 145, 50, 155]],
        x_ticks=[15.0, 45.0, 100.0, 200.0],
        chart_type="line",
    )
    assert len(build_decoder("line").decode(ctx)) == 4


def test_line_decoder_handles_no_markers(calibration):
    ctx = make_ctx(calibration, boxes=np.empty((0, 4)), x_ticks=[10.0], chart_type="line")
    assert build_decoder("line").decode(ctx) == []


def test_bar_decoder_deduplicates_overlaps(calibration):
    """PredBarPlot removed any overlapping box (IoU > 0.0); two bars cannot
    occupy the same pixels."""
    overlapping = [[10, 95, 20, 105], [11, 96, 21, 106]]
    ctx = make_ctx(calibration, boxes=overlapping)
    assert len(build_decoder("vertical_bar").decode(ctx)) == 1


def test_scatter_decoder_keeps_overlaps(calibration):
    """PredScatterPlot did NOT deduplicate -- scatter points legitimately
    overlap, and removing them would discard real data."""
    overlapping = [[10, 95, 20, 105], [11, 96, 21, 106]]
    ctx = make_ctx(calibration, boxes=overlapping, chart_type="scatter")
    assert len(build_decoder("scatter").decode(ctx)) == 2


def test_score_threshold_is_applied(calibration):
    ctx = make_ctx(
        calibration,
        boxes=[[10, 95, 20, 105], [40, 145, 50, 155]],
        scores=[0.9, 0.2],
        chart_type="scatter",
    )
    assert len(build_decoder("scatter").decode(ctx)) == 1


def test_dot_decoder_counts_stacked_dots(calibration):
    """Three dots stacked in one column, one in another."""
    boxes = [
        [10, 190, 20, 200], [10, 180, 20, 190], [10, 170, 20, 180],
        [60, 190, 70, 200],
    ]
    ctx = make_ctx(calibration, boxes=boxes, x_ticks=[15.0, 65.0], chart_type="dot")
    result = build_decoder("dot").decode(ctx)
    assert len(result) == 2
    assert result[0] > result[1], "the taller column must decode to a larger value"


def test_unknown_chart_type_returns_none():
    assert build_decoder("pie") is None


def test_all_registered_decoders_handle_empty_input(calibration):
    """No decoder may raise on an image where nothing was detected."""
    for chart_type in available_decoders():
        ctx = make_ctx(calibration, boxes=np.empty((0, 4)), chart_type=chart_type)
        assert build_decoder(chart_type).decode(ctx) == []
