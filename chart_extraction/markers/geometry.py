"""Box geometry helpers, deduplicated from the four decoder classes.

``calculate_center``, ``calculate_iou`` and ``remove_high_iou_boxes`` were
copy-pasted across PredBarPlot, PredScatterPlot, PredDotPlot and PredLinePlot in
the notebook, with small divergences between copies. Single implementation here.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def box_center(box: Sequence[float]) -> tuple[float, float]:
    """Centre of an (x1, y1, x2, y2) box."""
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def iou(box1: Sequence[float], box2: Sequence[float]) -> float:
    """Intersection over union.

    The +1 terms match the notebook's convention (pixel-inclusive extents).
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0.0, x2 - x1 + 1) * max(0.0, y2 - y1 + 1)
    area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    union = float(area1 + area2 - intersection)
    return intersection / union if union > 0 else 0.0


def filter_by_label_and_score(
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    label_id: int,
    score_threshold: float,
) -> np.ndarray:
    """Select boxes of one class above a confidence threshold.

    AUDIT NOTE (Phase 0, bug 1) -- LATENT, NOT ACTIVE
    --------------------------------------------------
    PredBarPlot.generate_output and PredScatterPlot.generate_output filtered
    using the bare global ``scores`` instead of ``self.scores``::

        marker = self.marker[np.logical_and(self.labels == 3, scores >= ...)]

    This was reported as causing every bar/scatter prediction after the first to
    filter against the previous image's confidences. It did not. The final
    prediction loop rebound the module-level name ``scores = df3['scores'][i]``
    inside each branch *before* constructing the decoder, so at call time the
    global happened to equal ``self.scores``. Verified at audit time: identity
    held on every iteration.

    It was a real contract violation waiting to fire -- it would have begun
    corrupting output the moment the code moved into modules (i.e. this
    refactor) or the branch order changed. Fixed by making the scores an
    explicit argument with no global fallback reachable.

    Because this bug was latent, it MUST NOT be credited with any part of a
    Phase 0 -> Phase 1 score delta. See docs/PHASE0_AUDIT.md.
    """
    boxes = np.asarray(boxes)
    labels = np.asarray(labels)
    scores = np.asarray(scores)

    if boxes.size == 0:
        return np.empty((0, 4), dtype=float)
    if not (len(boxes) == len(labels) == len(scores)):
        raise ValueError(
            "boxes/labels/scores length mismatch: "
            f"{len(boxes)}/{len(labels)}/{len(scores)} -- these must come from "
            "the same detection result"
        )

    keep = np.logical_and(labels == label_id, scores >= score_threshold)
    return boxes[keep]


def deduplicate_boxes(boxes, iou_threshold: float = 0.0) -> list:
    """Drop boxes overlapping an earlier-kept box above the threshold.

    Preserves the notebook's greedy semantics, including that it compares
    against all earlier boxes in input order rather than by descending score.
    Note the pipeline calls this with threshold 0.0, i.e. any overlap at all
    removes the later box.
    """
    boxes = list(boxes)
    redundant: set[int] = set()
    for i in range(len(boxes)):
        if i in redundant:
            continue
        for j in range(i + 1, len(boxes)):
            if j in redundant:
                continue
            if iou(boxes[i], boxes[j]) > iou_threshold:
                redundant.add(j)
    return [b for i, b in enumerate(boxes) if i not in redundant]
