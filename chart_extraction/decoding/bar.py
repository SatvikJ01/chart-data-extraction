"""Vertical / horizontal bar decoding."""

from __future__ import annotations

from chart_extraction.decoding.base import ChartDecoder, DecodeContext


class BarDecoder(ChartDecoder):
    """Each retained marker's centre row maps to one bar value.

    Preserves the notebook's PredBarPlot behaviour: deduplicate at IoU 0.0 (any
    overlap drops the later box), sort left-to-right, then map each centre's y
    to a value.

    Fixes relative to PredBarPlot:
      * bug 1  -- confidences now come from this image (base.marker_centres)
      * A      -- calibration pairs cannot desync (AxisCalibration)
      * C      -- no negative-index wraparound on out-of-range markers
    """

    chart_type = "vertical_bar"
    dedupe_iou = 0.0

    def decode(self, ctx: DecodeContext) -> list[float]:
        centres = self.marker_centres(ctx)
        return [ctx.calibration.value_at(cy) for _, cy in centres]


class HorizontalBarDecoder(BarDecoder):
    """Horizontal bars.

    AUDIT NOTE: the notebook's final loop had no ``horizontal_bar`` branch at
    all -- Donut could predict that chart type (the token is in the schema and
    ``string2preds`` returns it), but the decode loop tested only line,
    vertical_bar, dot and scatter. Any horizontal_bar image therefore fell
    through every branch and emitted ``result = []``, producing an empty y
    series and the '0;0' placeholder.

    That gap is preserved in *behaviour* here only insofar as the geometry is
    genuinely different: for horizontal bars the value axis is x, not y. Bar
    length is measured along x, so this decoder is registered but deliberately
    NOT given a y-axis calibration path that would silently produce nonsense.
    Phase 2's error taxonomy should quantify how many test images this affects
    before Phase 3 decides whether to build a real x-axis calibration.
    """

    chart_type = "horizontal_bar"

    def decode(self, ctx: DecodeContext) -> list[float]:
        # Deliberately unimplemented rather than wrong: decoding these needs an
        # x-axis calibration, which no current stage produces. Returning an
        # empty series reproduces the notebook's effective output for this type
        # (placeholder) without pretending the geometry was handled.
        return []
