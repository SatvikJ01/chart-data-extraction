"""Scatter decoding."""

from __future__ import annotations

from chart_extraction.decoding.base import ChartDecoder, DecodeContext


class ScatterDecoder(ChartDecoder):
    """One value per detected point, ordered left-to-right.

    Matches PredScatterPlot, which -- unlike PredBarPlot -- did NOT deduplicate
    overlapping boxes. That difference is preserved: scatter points legitimately
    overlap, so IoU-0 deduplication would discard real data.

    Fixes bug 1, A and C as for BarDecoder.
    """

    chart_type = "scatter"
    dedupe_iou = None

    def decode(self, ctx: DecodeContext) -> list[float]:
        centres = self.marker_centres(ctx)
        return [ctx.calibration.value_at(cy) for _, cy in centres]
