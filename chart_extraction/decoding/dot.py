"""Dot-plot decoding."""

from __future__ import annotations

from chart_extraction.decoding.base import ChartDecoder, DecodeContext

#: Pixel tolerance for treating a dot as belonging to an x-tick column.
#: Preserved from the notebook's hardcoded `count_points_with_same_x(..., 10)`.
COLUMN_TOLERANCE_PX = 10.0


class DotDecoder(ChartDecoder):
    """Count stacked dots per x-tick column and scale by the tick step.

    Preserves PredDotPlot: deduplicate at IoU 0.0, then for each x tick count
    the markers within COLUMN_TOLERANCE_PX in x, and report
    ``count * value_per_tick``. Results are integers, as in the original.

    PredDotPlot already used ``self.scores`` correctly, so bug 1 never applied
    here. It did share findings A and C through the common axis helpers; both
    are fixed by AxisCalibration.
    """

    chart_type = "dot"
    dedupe_iou = 0.0

    def decode(self, ctx: DecodeContext) -> list[float]:
        centres = self.marker_centres(ctx)
        x_ticks = sorted(float(x) for x in ctx.x_tick_pixels)
        if not x_ticks:
            return []

        # The notebook took `scale = abs(y_labels[0] - y_labels[1])`, i.e. the
        # value step between adjacent rungs of the calibration ladder.
        values = ctx.calibration.values
        scale = abs(values[1] - values[0]) if len(values) >= 2 else 0.0

        counts = []
        for tick_x in x_ticks:
            count = sum(1 for cx, _ in centres if abs(cx - tick_x) <= COLUMN_TOLERANCE_PX)
            counts.append(int(count * scale))
        return counts
