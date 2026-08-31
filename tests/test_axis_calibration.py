"""Tests for findings A and C -- both ACTIVE bugs, both fixed."""

from __future__ import annotations

import pytest

from chart_extraction.axis.calibration import AxisCalibration
from chart_extraction.axis.labels import parse_tick_label


# --- Finding A: find_element_above sorted the caller's list in place. -------

def test_findingA_calibration_pairs_cannot_desync():
    """Points and labels are bound together and sorted together.

    The notebook sorted y_points in place while y_labels kept its original
    order, silently mispairing every tick from the first call onward.
    """
    pixels = [200.0, 150.0, 100.0, 50.0]
    values = [0.0, 10.0, 20.0, 30.0]
    cal = AxisCalibration.from_ticks(pixels, values)

    assert list(zip(cal.pixels, cal.values)) == [
        (50.0, 30.0), (100.0, 20.0), (150.0, 10.0), (200.0, 0.0)
    ]
    # The inputs must not have been mutated.
    assert pixels == [200.0, 150.0, 100.0, 50.0]
    assert values == [0.0, 10.0, 20.0, 30.0]


def test_findingA_repeated_calls_are_stable():
    """Calibration is immutable, so repeated queries cannot degrade it."""
    cal = AxisCalibration.from_ticks([200.0, 150.0, 100.0, 50.0], [0.0, 10.0, 20.0, 30.0])
    before = (cal.pixels, cal.values)
    for _ in range(5):
        cal.value_at(175.0)
        cal.value_at(10.0)
        cal.value_at(9999.0)
    assert (cal.pixels, cal.values) == before


# --- Finding C: negative-index wraparound on out-of-range markers. ----------

def test_findingC_out_of_range_extrapolates_not_wraps():
    """A marker above the topmost tick must extrapolate from the nearest
    interval, not wrap to the opposite end of the axis."""
    cal = AxisCalibration.from_ticks([50.0, 100.0, 150.0, 200.0], [30.0, 20.0, 10.0, 0.0])
    # Slope is -0.2 units/px. 50px above the top tick -> 30 + 10 = 40.
    assert cal.value_at(0.0) == pytest.approx(40.0)
    # 50px below the bottom tick -> 0 - 10 = -10.
    assert cal.value_at(250.0) == pytest.approx(-10.0)


def test_findingC_interpolation_is_exact_at_ticks():
    cal = AxisCalibration.from_ticks([50.0, 100.0, 150.0], [30.0, 20.0, 10.0])
    for px, val in [(50.0, 30.0), (100.0, 20.0), (150.0, 10.0), (75.0, 25.0)]:
        assert cal.value_at(px) == pytest.approx(val)


def test_degenerate_calibrations_do_not_raise():
    assert not AxisCalibration.from_ticks([], []).is_usable
    assert AxisCalibration.from_ticks([], []).value_at(10.0) == 0.0
    single = AxisCalibration.from_ticks([100.0], [5.0])
    assert not single.is_usable
    assert single.value_at(999.0) == 5.0
    # Duplicate pixels would divide by zero; they are collapsed.
    dup = AxisCalibration.from_ticks([10.0, 10.0, 20.0], [1.0, 2.0, 3.0])
    assert dup.pixels == (10.0, 20.0)


def test_length_mismatch_truncates_to_shorter():
    cal = AxisCalibration.from_ticks([1.0, 2.0, 3.0], [10.0, 20.0])
    assert len(cal.pixels) == 2


# --- Bug 4: label.isdigit() dropped negative and decimal ticks. -------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("-5", -5.0), ("3.14", 3.14), ("1e5", 100000.0), ("12", 12.0),
        ("1,200", 1200.0), ("  7 ", 7.0), (-2, -2.0),
    ],
)
def test_bug4_numeric_labels_parse(raw, expected):
    assert parse_tick_label(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["abc", "", "   ", None, "Monday"])
def test_bug4_non_numeric_labels_return_none_not_zero(raw):
    """Returning None keeps 'not a number' distinguishable from the number 0 --
    the distinction `float(x) if x.isdigit() else 0.0` destroyed."""
    assert parse_tick_label(raw) is None


def test_bug4_isdigit_would_have_dropped_these():
    """Documents the original failure directly."""
    for raw in ["-5", "3.14", "1e5"]:
        assert not raw.isdigit()          # the old guard rejected it
        assert parse_tick_label(raw) is not None   # the new one does not
