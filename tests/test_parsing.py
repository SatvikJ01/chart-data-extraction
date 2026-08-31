"""Tests for Donut token-string parsing and failure-mode capture."""

from __future__ import annotations

import pytest

from chart_extraction.donut.parsing import detect_chart_type, string2preds


def test_well_formed_sequence():
    pred = string2preds(
        "<|BOS|><scatter><x_start>1;2;3<x_end><y_start>4;5;6<y_end>", "img"
    )
    assert pred.chart_type == "scatter"
    assert pred.x == [1, 2, 3] and pred.y == [4, 5, 6]
    assert pred.is_well_formed


@pytest.mark.parametrize(
    "raw,mode",
    [
        ("no tokens at all", "no_chart_type_token"),
        ("<line> partial <x_start>1<x_end>", "missing_series_delimiters"),
        ("<line><x_start><x_end><y_start><y_end>", "empty_series"),
    ],
)
def test_failure_modes_are_recorded_not_swallowed(raw, mode):
    """The notebooks wrapped this in a bare except and emitted a placeholder,
    making malformed-sequence rate unmeasurable. Phase 2 needs these counts."""
    pred = string2preds(raw, "img")
    assert pred.failure_mode == mode
    assert not pred.is_well_formed


def test_empty_series_guard_actually_fires():
    """`"".split(";")` returns `['']`, never `[]`, so the notebook's
    `if len(x) == 0` guard was unreachable."""
    assert "".split(";") == [""]
    assert len("".split(";")) != 0
    assert string2preds(
        "<line><x_start><x_end><y_start><y_end>", "i"
    ).failure_mode == "empty_series"


def test_one_token_is_expanded():
    pred = string2preds(
        "<line><x_start><one>;2<x_end><y_start><one>;3<y_end>", "i"
    )
    assert pred.x == [1, 2]


def test_chart_type_detection_order_is_preserved():
    """The original returned on first match in a fixed order."""
    assert detect_chart_type("<dot><line>") == ("dot", True)
    assert detect_chart_type("nothing") == ("vertical_bar", False)
