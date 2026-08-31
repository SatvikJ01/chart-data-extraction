"""Tests for the AxisLabelSource seam (finding B).

Finding B is a DESIGN ERROR that is deliberately PRESERVED in Phase 0: the
default source feeds Donut's predicted y data series in as if it were the y-axis
tick labels. Phase 0 is a faithful refactor, not a better model. These tests pin
the seam's shape so Phase 3 can swap in an OCR implementation as a
single-variable change.
"""

from __future__ import annotations

import pytest

from chart_extraction.axis.labels import (
    AxisLabelSource,
    DonutSeriesAxisLabelSource,
    available_axis_label_sources,
    build_axis_label_source,
    register_axis_label_source,
)
from chart_extraction.config import PipelineConfig


def test_default_source_is_the_faithful_flawed_one():
    assert PipelineConfig().axis_label_source == "donut_series"
    source = build_axis_label_source(PipelineConfig().axis_label_source)
    assert isinstance(source, DonutSeriesAxisLabelSource)


def test_findingB_default_source_consumes_the_donut_data_series():
    """Documents the flaw explicitly: the y *data series* is what reaches the
    label slot. This test asserts the wrong-but-preserved behaviour, and should
    be updated -- not deleted -- when Phase 3 adds a real tick-label source."""
    source = DonutSeriesAxisLabelSource()
    donut_y_series = [10.0, 20.0, 30.0]
    labels = source.y_tick_labels("img", tick_pixels=[100.0, 120.0], donut_y_series=donut_y_series)
    assert labels == [10.0, 20.0, 30.0]


def test_ladder_reproduces_extend_y_axis():
    """The original extrapolated a uniform ladder from the first two pairs
    rather than using every detected tick. Preserved."""
    source = DonutSeriesAxisLabelSource(ladder_length=25)
    cal = source.build_calibration("img", [100.0, 120.0, 140.0], ["10", "20", "30"])
    assert len(cal.pixels) == 26
    assert cal.pixels[:3] == (100.0, 120.0, 140.0)
    # step_val = labels[0] - labels[1] = -10, applied as val[i+1] = val[i] - step
    assert cal.values[:3] == (10.0, 20.0, 30.0)


def test_fewer_than_two_labels_gives_zero_scale():
    """Preserved baseline behaviour: a zero scale is a visible failure rather
    than a silent plausible-looking one."""
    source = DonutSeriesAxisLabelSource()
    assert source.y_tick_labels("img", [100.0], ["5"]) == [0.0, 0.0]
    assert source.y_tick_labels("img", [100.0], []) == [0.0, 0.0]


def test_non_numeric_labels_are_dropped_not_zeroed():
    """Bug 4 is fixed *inside* the preserved-flawed source. Bug 4 (parsing) and
    finding B (wrong input) are independent."""
    source = DonutSeriesAxisLabelSource()
    labels = source.y_tick_labels("img", [1.0, 2.0], ["-5", "abc", "3.14"])
    assert labels == [-5.0, 3.14]


def test_registry_accepts_a_phase3_implementation():
    """The seam must be a real swap point, not a hardcoded default."""

    class FakeOcrSource:
        name = "fake_ocr"

        def y_tick_labels(self, image_id, tick_pixels, donut_y_series):
            return [100.0, 200.0]

        def build_calibration(self, image_id, tick_pixels, donut_y_series):
            from chart_extraction.axis.calibration import AxisCalibration
            return AxisCalibration.from_ticks(list(tick_pixels)[:2], [100.0, 200.0])

    register_axis_label_source("fake_ocr", FakeOcrSource)
    try:
        assert "fake_ocr" in available_axis_label_sources()
        source = build_axis_label_source("fake_ocr")
        assert isinstance(source, AxisLabelSource)
        cal = source.build_calibration("img", [10.0, 20.0], ["ignored"])
        assert cal.values == (100.0, 200.0)
    finally:
        from chart_extraction.axis import labels as labels_module
        labels_module._SOURCES.pop("fake_ocr", None)


def test_unknown_source_name_is_rejected():
    with pytest.raises(ValueError, match="unknown axis_label_source"):
        build_axis_label_source("does_not_exist")
