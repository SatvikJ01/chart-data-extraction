"""Line-chart decoding.

AUDIT NOTE (Phase 0, bug 2) -- ACTIVE, every line chart was a placeholder
=========================================================================
In the notebook's final loop the entire line branch was commented out and
replaced with a hardcoded stub::

    if df['chart_types'][i] == 'line':
        '''
        result = [0.0,0.0]
        ...
        pred_line_plot = PredLinePlot(...)
        result = pred_line_plot.generate_output()'''
        result = [0.0,0.0]

So ``PredLinePlot`` was fully defined but never once invoked, and every line
chart in the submission received ``0.0;0.0``. Note also that the module-level
placeholder chart type is ``"line"``, so any image whose generation failed was
*also* labelled line and given the same stub -- two independent paths to the
same silent placeholder.

The logic is restored here. Two bugs in the dead code are fixed rather than
carried across (finding D):

  * ``el = y_points[0]`` read a module-level global instead of
    ``self.y_points[0]`` -- the same class of defect as bug 1, but in
    PredLinePlot it was never latent because the surrounding loop never bound a
    global named ``y_points``. It would have raised NameError on the
    out-of-range branch had the branch ever run.
  * ``self.scores >= 0.0`` applied no confidence filtering whatsoever, admitting
    every proposal the detector emitted. This decoder uses the configured
    threshold like the others.

Because line decoding produced a constant before this change, there is no
prior behaviour to preserve -- any non-stub output is new.
"""

from __future__ import annotations

from chart_extraction.decoding.base import ChartDecoder, DecodeContext


class LineDecoder(ChartDecoder):
    """Sample the detected line at each x-axis tick.

    For every x tick, take the marker nearest in x and read its y through the
    calibration. Yields one value per x tick, which is what the competition
    format expects for a line series.
    """

    chart_type = "line"
    dedupe_iou = None

    def decode(self, ctx: DecodeContext) -> list[float]:
        centres = self.marker_centres(ctx)
        if not centres:
            return []

        x_ticks = sorted(float(x) for x in ctx.x_tick_pixels)
        if not x_ticks:
            # No axis ticks to sample at: fall back to the markers themselves,
            # left-to-right. Better than the notebook's constant, and it keeps
            # the series length tied to something observed.
            return [ctx.calibration.value_at(cy) for _, cy in centres]

        values: list[float] = []
        for tick_x in x_ticks:
            # argmin rather than the notebook's `list.index(closest_value)`,
            # which returned the first equal x when markers shared a coordinate.
            _, nearest_y = min(centres, key=lambda c: abs(c[0] - tick_x))
            values.append(ctx.calibration.value_at(nearest_y))
        return values
