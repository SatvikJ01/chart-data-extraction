"""The official Benetech competition metric, reimplemented.

The metric is asymmetric across data types, which is the point of it:

  * **Categorical** series are scored by summed Levenshtein distance, normalised
    by total ground-truth string length. Getting a label nearly right earns
    partial credit; the penalty scales with how many characters are wrong.
  * **Numeric** series are scored by RMSE normalised against the RMSE of
    predicting the ground-truth mean. So a prediction is measured against the
    trivial baseline of "guess the average" -- beating that baseline is what
    earns score, and a series with no variance cannot earn partial credit.

Both are squashed through ``2 - 2 / (1 + exp(-x))``, which maps an error ratio
of 0 to a score of 1 and decays monotonically to 0.

Two hard gates precede all of that, and both award exactly 0:
  * predicted chart type != ground-truth chart type
  * predicted series length != ground-truth series length

The length gate is why the notebooks' habit of emitting a 2-element ``0;0``
placeholder scores zero rather than "a little": length alone disqualifies it.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from chart_extraction.eval.levenshtein import distance as levenshtein


def sigmoid(x: float) -> float:
    """Official squashing function: 0 -> 1, +inf -> 0, monotone decreasing."""
    # exp overflows for large negative -x; the limit is 0 either way.
    try:
        return 2.0 - 2.0 / (1.0 + math.exp(-x))
    except OverflowError:  # pragma: no cover
        return 0.0


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(np.square(y_true - y_pred))))


def normalized_rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """RMSE normalised by the RMSE of predicting the ground-truth mean.

    When the ground truth has no variance the denominator is zero; the official
    resolution is 1.0 for an exact match and 0.0 otherwise, since there is no
    meaningful scale against which to award partial credit.
    """
    numerator = rmse(y_true, y_pred)
    denominator = rmse(y_true, np.full(len(y_true), np.mean(y_true)))

    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0
    return sigmoid(numerator / denominator)


def normalized_levenshtein_score(
    y_true: Sequence[str], y_pred: Sequence[str]
) -> float:
    """Summed edit distance normalised by total ground-truth length."""
    total_distance = sum(
        levenshtein(str(yt), str(yp)) for yt, yp in zip(y_true, y_pred)
    )
    length_sum = sum(len(str(yt)) for yt in y_true)

    if length_sum == 0:
        return 1.0 if total_distance == 0 else 0.0
    return sigmoid(total_distance / length_sum)


def _is_numeric_series(values: Sequence) -> bool:
    """Ground truth decides which branch of the metric applies.

    The official scorer tested ``isinstance(y_true[0], str)`` -- the first
    element alone determines the whole series' type.
    """
    return not isinstance(values[0], str)


def _coerce_numeric(values: Sequence) -> list[float] | None:
    """Coerce a predicted series to floats, or None if it cannot be.

    A numeric ground truth paired with an uncoercible prediction scores 0: the
    official scorer would raise inside ``rmse``, and a prediction that is not
    numbers cannot be numerically close to numbers.
    """
    out: list[float] = []
    for v in values:
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            f = float(v)
        elif isinstance(v, str):
            try:
                f = float(v.strip().replace(",", ""))
            except ValueError:
                return None
        else:
            return None
        if math.isnan(f) or math.isinf(f):
            return None
        out.append(f)
    return out


def score_series(y_true: Sequence, y_pred: Sequence) -> float:
    """Score one predicted series against one ground-truth series.

    DEVIATION FROM THE OFFICIAL SCORER, deliberate and tested:
    the official implementation indexes ``y_true[0]`` without a guard, so it
    raises IndexError on an empty ground-truth series. Two empty series are
    treated as a perfect match (1.0) here rather than crashing the run. Real
    ground-truth series are never empty, so this affects no real score -- it
    only stops a degenerate case from taking down a whole evaluation pass.
    """
    if len(y_true) == 0 and len(y_pred) == 0:
        return 1.0
    if len(y_true) != len(y_pred):
        return 0.0
    if len(y_true) == 0:  # pragma: no cover - unreachable given the checks above
        return 1.0

    if _is_numeric_series(y_true):
        true_numeric = _coerce_numeric(y_true)
        pred_numeric = _coerce_numeric(y_pred)
        if true_numeric is None:
            return 0.0
        if pred_numeric is None:
            return 0.0
        return normalized_rmse(true_numeric, pred_numeric)

    return normalized_levenshtein_score(
        [str(v) for v in y_true], [str(v) for v in y_pred]
    )


def score_instance(
    gt_series: Sequence,
    gt_chart_type: str,
    pred_series: Sequence,
    pred_chart_type: str,
) -> float:
    """Score one (id, axis) instance, applying the chart-type gate first."""
    if gt_chart_type != pred_chart_type:
        return 0.0
    return score_series(gt_series, pred_series)


def benetech_score(ground_truth, predictions) -> float:
    """Mean score over all instances.

    Both arguments are DataFrames indexed by instance id (``<image_id>_x`` /
    ``<image_id>_y``) with columns ``data_series`` and ``chart_type``. Every
    ground-truth instance must have exactly one prediction.
    """
    if not ground_truth.index.equals(predictions.index):
        raise ValueError(
            "must have exactly one prediction per ground-truth instance "
            f"({len(ground_truth)} truth vs {len(predictions)} predicted)"
        )

    scores = [
        score_instance(
            gt.data_series, gt.chart_type, pred.data_series, pred.chart_type
        )
        for gt, pred in zip(
            ground_truth.itertuples(index=False), predictions.itertuples(index=False)
        )
    ]
    return float(np.mean(scores)) if scores else 0.0
