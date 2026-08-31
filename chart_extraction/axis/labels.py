"""The AxisLabelSource seam.

WHAT THIS SEAM IS FOR
=====================
Converting a marker's pixel position into a data value requires knowing what
values the axis *ticks* carry. This module is the single place that question is
answered, so that answering it differently is a one-line config change rather
than a rewrite.

AUDIT NOTE (Phase 0, finding B) -- DESIGN ERROR, deliberately preserved
=======================================================================
The original pipeline answered it incorrectly. In ``inference-3.ipynb``::

    labels = [float(label) if label.isdigit() else 0.0 for label in df['y_val'][i]]
    extended_y.append(extend_y_axis(y, labels))

``df['y_val'][i]`` is **Donut's predicted y data series** -- the chart's data
points. It is being passed as ``y_labels``, the **y-axis tick labels**. Those
are different quantities and coincide only by accident. Downstream, PredBarPlot
then converts detected marker pixels into values by interpolating against
Donut's own output, which makes the numeric branch of the pipeline circular.

Nothing in the notebook pipeline ever reads the axis tick *text*. That is the
gap Phase 3's OCR component is meant to close, and it is why this seam exists.

Per the Phase 0 decision, the flawed behaviour is **preserved bit-for-bit** as
``DonutSeriesAxisLabelSource`` and left as the default. Phase 0 is a faithful
refactor, not a better model: if the baseline already had correct axis labels,
the Phase 3 OCR row in the ablation table would have nothing to demonstrate.
Phase 3 registers a second implementation and flips
``PipelineConfig.axis_label_source`` -- a single-variable change.

AUDIT NOTE (Phase 0, bug 4) -- ACTIVE BUG, fixed
================================================
The label parse used ``str.isdigit()``, which is False for "-5", "3.14" and
"1e5". Every such tick collapsed to 0.0, zeroing the axis scale for any chart
with negative or decimal ticks. ``parse_tick_label`` below parses actual floats.

Note this fix is *inside* the preserved-flawed source. Bug 4 and finding B are
independent: 4 is a parsing bug and is fixed, B is a wrong-input design error
and is preserved.
"""

from __future__ import annotations

from typing import Callable, Protocol, Sequence, runtime_checkable

from chart_extraction.axis.calibration import AxisCalibration


def parse_tick_label(label: object) -> float | None:
    """Parse one tick label to a float, or None if it isn't numeric.

    Replaces ``float(label) if label.isdigit() else 0.0`` (bug 4). Returning
    None rather than 0.0 lets the caller distinguish "this tick is not a number"
    from "this tick is the number zero" -- a distinction the original destroyed.
    """
    if isinstance(label, (int, float)):
        value = float(label)
        return None if value != value else value  # reject NaN
    if not isinstance(label, str):
        return None
    text = label.strip().replace(",", "")
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return None if value != value else value


@runtime_checkable
class AxisLabelSource(Protocol):
    """Supplies numeric y-axis tick labels for one image.

    Implementations return values positionally aligned with ``tick_pixels``
    (ascending pixel order). Returning fewer labels than pixels is allowed;
    AxisCalibration truncates to the shorter of the two.
    """

    name: str

    def y_tick_labels(
        self,
        image_id: str,
        tick_pixels: Sequence[float],
        donut_y_series: Sequence[object],
    ) -> list[float]:
        ...


class DonutSeriesAxisLabelSource:
    """Faithful reproduction of the notebook's (incorrect) label source.

    Preserves two behaviours of the original on purpose:

    1. It uses Donut's predicted y **data series** as if it were the y-axis tick
       labels (finding B). This is wrong. It is kept so the Phase 0 baseline is
       genuinely the old pipeline.
    2. It extrapolates a uniform tick ladder from the first two (pixel, label)
       pairs rather than using every detected tick -- the ``extend_y_axis``
       behaviour -- so the assumed-uniform-spacing characteristic is preserved
       too.

    It does NOT preserve the ``isdigit`` parse bug (bug 4), which is fixed.
    """

    name = "donut_series"

    def __init__(self, ladder_length: int = 25) -> None:
        self.ladder_length = ladder_length

    def y_tick_labels(
        self,
        image_id: str,
        tick_pixels: Sequence[float],
        donut_y_series: Sequence[object],
    ) -> list[float]:
        # Bug 4 fix: parse real floats, keep only genuinely numeric labels.
        parsed = [parse_tick_label(v) for v in donut_y_series]
        labels = [v for v in parsed if v is not None]

        # The notebook substituted [0.0, 0.0] when it had fewer than two labels,
        # which yields a zero-scale axis. Preserved: it is the honest baseline
        # behaviour, and a zero scale is a visible failure rather than a silent
        # plausible-looking one.
        if len(labels) < 2:
            labels = [0.0, 0.0]

        return labels

    def build_calibration(
        self,
        image_id: str,
        tick_pixels: Sequence[float],
        donut_y_series: Sequence[object],
    ) -> AxisCalibration:
        """Build the calibration, reproducing extend_y_axis's ladder.

        The original walked outward from the first tick using a constant step in
        both pixel and label space:

            diff_points = y_points[1] - y_points[0]
            diff_labels = y_labels[0] - y_labels[1]

        and appended ``ladder_length`` further rungs. That ladder is what the
        decoders interpolated against, so it is reproduced here rather than
        replaced by the true detected ticks.
        """
        pixels = sorted(float(p) for p in tick_pixels)
        labels = self.y_tick_labels(image_id, pixels, donut_y_series)

        if len(pixels) < 2 or len(labels) < 2:
            return AxisCalibration.from_ticks(pixels, labels)

        step_px = pixels[1] - pixels[0]
        step_val = labels[0] - labels[1]

        ladder_px = [pixels[0]]
        ladder_val = [labels[0]]
        for i in range(self.ladder_length):
            ladder_px.append(ladder_px[i] + step_px)
            ladder_val.append(ladder_val[i] - step_val)

        return AxisCalibration.from_ticks(ladder_px, ladder_val)


# --- Registry -------------------------------------------------------------
# Phase 3 registers its OCR implementation here and selects it by setting
# PipelineConfig.axis_label_source. No other code changes.

_SOURCES: dict[str, Callable[[], AxisLabelSource]] = {
    "donut_series": DonutSeriesAxisLabelSource,
}


def register_axis_label_source(
    name: str, factory: Callable[[], AxisLabelSource]
) -> None:
    """Register an implementation under a config-selectable name."""
    if name in _SOURCES:
        raise ValueError(f"axis label source {name!r} is already registered")
    _SOURCES[name] = factory


def available_axis_label_sources() -> list[str]:
    return sorted(_SOURCES)


def build_axis_label_source(name: str) -> AxisLabelSource:
    """Instantiate the configured implementation."""
    try:
        factory = _SOURCES[name]
    except KeyError:
        raise ValueError(
            f"unknown axis_label_source {name!r}; "
            f"available: {available_axis_label_sources()}"
        ) from None
    return factory()
