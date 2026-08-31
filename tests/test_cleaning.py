"""Tests for bug 5 -- the gutted clean_preds."""

from __future__ import annotations

import pytest

from chart_extraction.donut.cleaning import (
    clean_numeric_series, clean_preds, _numeric_fraction,
)
from chart_extraction.donut.parsing import string2preds


def test_bug5_salvageable_values_are_repaired_not_zeroed():
    """inference-3 had the numeric strip commented out, so anything failing the
    first cast fell through to `temp = 0`. Measured at audit time:

        input : ['11', '1E', '3.14', '-5', '1e5']
        output: [11,    0,   3.14,   -5,   0   ]

    '1E' and '1e5' must no longer collapse to zero.
    """
    result = clean_numeric_series(["11", "1E", "3.14", "-5", "1e5"])
    assert result[0] == 11
    assert result[1] != 0, "'1E' must be repaired, not zeroed"
    assert result[2] == pytest.approx(3.14)
    assert result[3] == -5
    assert result[4] == pytest.approx(1e5), "'1e5' must be repaired, not zeroed"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1 0 0", 100),      # whitespace inside a number
        ("--3", -3),         # doubled sign
        ("1.2.3", 1.23),     # doubled decimal point
        ("", 0),             # nothing to salvage
        ("abc", 0),          # nothing numeric
        ("12abc", 12),       # junk suffix
    ],
)
def test_repair_cases(raw, expected):
    assert clean_numeric_series([raw])[0] == pytest.approx(expected)


def test_categorical_series_is_not_numerically_repaired():
    """Pushing a categorical axis through numeric repair would turn 'Monday'
    into 0 -- which is why the digit-density check exists."""
    x, y = clean_preds(["Mon", "Tue", "Wed"], ["1", "2", "3"])
    assert x == ["Mon", "Tue", "Wed"]
    assert y == [1, 2, 3]


def test_empty_series_does_not_raise_zero_division():
    """The notebook divided by len("".join(values)) with no guard; an all-empty
    series raised ZeroDivisionError, which the bare except then converted into a
    placeholder row for the whole image."""
    assert _numeric_fraction([""]) == 0.0
    assert _numeric_fraction([]) == 0.0
    x, y = clean_preds([""], [""])       # must not raise
    assert x == [""] and y == [""]


def test_bug5_cleaning_is_actually_invoked_by_the_parser():
    """tuned-donut defined clean_preds and never called it."""
    raw = "<vertical_bar><x_start>a;b<x_end><y_start>1E;2<y_end>"
    pred = string2preds(raw, "img", apply_cleaning=True)
    assert pred.y[0] != "1E", "parser must clean, not pass the raw token through"
    assert all(isinstance(v, (int, float)) for v in pred.y)

    uncleaned = string2preds(raw, "img", apply_cleaning=False)
    assert uncleaned.y == ["1E", "2"]
