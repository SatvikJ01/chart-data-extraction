"""Regression tests for the two LATENT contract violations (bugs 1 and 3).

Both were real defects that were NOT corrupting output at audit time. These
tests exist to prove they were real and to fail if either guarantee is ever
relied upon again.

Neither bug may be credited with any part of a Phase 0 -> Phase 1 score delta.
By definition they cannot have changed output. See docs/PHASE0_AUDIT.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from chart_extraction.axis.calibration import AxisCalibration
from chart_extraction.data.images import discover_images, require_all_ids
from chart_extraction.data.submission import build_submission
from chart_extraction.decoding import DecodeContext, build_decoder


def _ctx(boxes, labels, scores, calibration, chart_type="vertical_bar"):
    return DecodeContext(
        image_id="img",
        chart_type=chart_type,
        boxes=boxes,
        labels=labels,
        scores=scores,
        calibration=calibration,
        x_tick_pixels=[],
        donut_x=[],
        donut_y=[],
    )


# --- Bug 1: LATENT. Global `scores` instead of `self.scores`. --------------

@pytest.mark.parametrize("chart_type", ["vertical_bar", "scatter"])
def test_bug1_no_cross_talk_between_decoder_instances(chart_type):
    """Two decoders built from different detections must not influence one
    another.

    The notebook's PredBarPlot/PredScatterPlot filtered on a module-level
    `scores`, so behaviour depended on whatever was last bound to that name.
    Here each decode reads only its own DecodeContext.
    """
    calibration = AxisCalibration.from_ticks([50.0, 150.0], [10.0, 0.0])
    boxes = np.array([[0, 40, 10, 60], [20, 90, 30, 110]], dtype=float)
    labels = np.array([3, 3])

    # Image A: both markers confident. Image B: only the first.
    ctx_a = _ctx(boxes, labels, np.array([0.9, 0.9]), calibration, chart_type)
    ctx_b = _ctx(boxes, labels, np.array([0.9, 0.1]), calibration, chart_type)

    decoder = build_decoder(chart_type)

    a_first = decoder.decode(ctx_a)
    b_after_a = decoder.decode(ctx_b)
    b_first = build_decoder(chart_type).decode(ctx_b)
    a_after_b = build_decoder(chart_type).decode(ctx_a)

    assert len(a_first) == 2, "image A should keep both markers"
    assert len(b_first) == 1, "image B should keep only the confident marker"
    # Order of evaluation must not matter -- this is the actual guarantee.
    assert b_after_a == b_first
    assert a_after_b == a_first


def test_bug1_mismatched_detection_arrays_are_rejected():
    """Scores from a different image than the boxes must raise, not silently
    broadcast or truncate."""
    from chart_extraction.markers.geometry import filter_by_label_and_score

    boxes = np.zeros((3, 4))
    with pytest.raises(ValueError, match="length mismatch"):
        filter_by_label_and_score(boxes, np.array([3, 3, 3]), np.array([0.9, 0.9]), 3, 0.5)


# --- Bug 3: LATENT. Positional joins across independently-built id lists. ---

def test_bug3_submission_joins_are_id_keyed_not_positional():
    """Shuffling the per-stage mappings must not change the output.

    A positional join would produce different rows; an id-keyed join cannot.
    """
    image_ids = ["c", "a", "b"]
    chart_types = {"a": "line", "b": "dot", "c": "scatter"}
    x = {"a": [1, 2], "b": [3], "c": [4, 5, 6]}
    y = {"a": [7.0], "b": [8.0], "c": [9.0]}

    forward = build_submission(image_ids, chart_types, x, y)
    reversed_maps = build_submission(
        image_ids,
        dict(reversed(list(chart_types.items()))),
        dict(reversed(list(x.items()))),
        dict(reversed(list(y.items()))),
    )
    assert forward.equals(reversed_maps)

    # And each row carries its own image's data.
    row = forward.set_index("id").loc["c_x"]
    assert row["data_series"] == "4;5;6"
    assert row["chart_type"] == "scatter"


def test_bug3_discovery_is_deterministic_and_unique(tmp_path):
    for name in ["z9", "a1", "m5"]:
        (tmp_path / f"{name}.jpg").touch()
    first = [r.image_id for r in discover_images(tmp_path)]
    second = [r.image_id for r in discover_images(tmp_path)]
    assert first == second == ["a1", "m5", "z9"]


def test_bug3_stage_dropping_an_image_is_detected():
    """A stage that silently loses an image shifted every later positional
    join in the notebook. It is now an explicit error."""
    with pytest.raises(ValueError, match="missing"):
        require_all_ids("donut", {"a", "b"}, ["a", "b", "c"])
