"""Shared decoder scaffolding.

The notebook's four Pred*Plot classes duplicated ``calculate_center``,
``calculate_iou``, ``remove_high_iou_boxes``, ``find_element_above`` and the
``least_count`` interpolation between them, with divergences that were the
source of several bugs (findings A, C, D and bug 1). All of that shared
machinery now lives here or in markers.geometry, so a fix lands once.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from chart_extraction.axis.calibration import AxisCalibration
from chart_extraction.markers.geometry import (
    box_center,
    deduplicate_boxes,
    filter_by_label_and_score,
)


@dataclass
class DecodeContext:
    """Everything one decoder needs for one image.

    Bundling these means a decoder cannot reach a value belonging to a different
    image. The notebook's decoders read module-level globals (``scores`` in
    PredBarPlot/PredScatterPlot, ``y_points`` in PredLinePlot); there is no
    global to reach for here.
    """

    image_id: str
    chart_type: str
    boxes: np.ndarray
    labels: np.ndarray
    scores: np.ndarray
    calibration: AxisCalibration
    x_tick_pixels: Sequence[float]
    donut_x: Sequence[object]
    donut_y: Sequence[object]
    marker_label_id: int = 3
    score_threshold: float = 0.5


class ChartDecoder(ABC):
    """Converts detected markers into a y data series for one chart type."""

    chart_type: str = ""
    #: Threshold override; None means use the context's threshold.
    score_threshold_override: float | None = None
    #: IoU above which an overlapping later box is dropped. None disables.
    dedupe_iou: float | None = None

    def marker_centres(self, ctx: DecodeContext) -> list[tuple[float, float]]:
        """Filtered, optionally deduplicated marker centres, sorted by x."""
        threshold = (
            self.score_threshold_override
            if self.score_threshold_override is not None
            else ctx.score_threshold
        )
        boxes = filter_by_label_and_score(
            ctx.boxes, ctx.labels, ctx.scores, ctx.marker_label_id, threshold
        )
        if self.dedupe_iou is not None:
            boxes = deduplicate_boxes(boxes, self.dedupe_iou)
        centres = [box_center(b) for b in boxes]
        centres.sort(key=lambda c: c[0])
        return centres

    @abstractmethod
    def decode(self, ctx: DecodeContext) -> list[float]:
        """Return the predicted y data series."""
        raise NotImplementedError
