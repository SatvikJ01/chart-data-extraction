"""Parse Donut's generated token string into a chart type and x/y series."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from chart_extraction.donut.cleaning import clean_preds
from chart_extraction.donut.tokens import (
    CHART_TYPE_TOKENS,
    DEFAULT_CHART_TYPE,
    X_END,
    X_START,
    Y_END,
    Y_START,
)


@dataclass
class DonutPrediction:
    """One parsed generation.

    ``failure_mode`` is recorded rather than discarded. The notebooks wrapped
    this whole path in a bare ``except:`` and emitted a placeholder, which made
    malformed-sequence rate unmeasurable. Phase 2's error taxonomy needs it, so
    it is captured now even though nothing consumes it yet.
    """

    image_id: str
    chart_type: str
    x: List = field(default_factory=list)
    y: List = field(default_factory=list)
    raw: str = ""
    failure_mode: str | None = None

    @property
    def is_well_formed(self) -> bool:
        return self.failure_mode is None


def detect_chart_type(pred_string: str) -> tuple[str, bool]:
    """Return (chart_type, found). Order-sensitive, matching the original."""
    for token, name in CHART_TYPE_TOKENS:
        if token in pred_string:
            return name, True
    return DEFAULT_CHART_TYPE, False


def string2preds(
    pred_string: str,
    image_id: str = "",
    apply_cleaning: bool = True,
) -> DonutPrediction:
    """Convert a generated string to a structured prediction.

    AUDIT NOTE (Phase 0, bug 5): ``clean_preds`` is now actually invoked, and
    the function it calls has its numeric-strip restored. See cleaning.py.
    """
    chart_type, found = detect_chart_type(pred_string)
    if not found:
        return DonutPrediction(
            image_id=image_id,
            chart_type=chart_type,
            raw=pred_string,
            failure_mode="no_chart_type_token",
        )

    if not all(tok in pred_string for tok in (X_START, X_END, Y_START, Y_END)):
        return DonutPrediction(
            image_id=image_id,
            chart_type=chart_type,
            raw=pred_string,
            failure_mode="missing_series_delimiters",
        )

    # Donut emits "<one>" for the literal digit 1 in some checkpoints.
    normalised = re.sub(r"<one>", "1", pred_string)

    x = normalised.split(X_START)[1].split(X_END)[0].split(";")
    y = normalised.split(Y_START)[1].split(Y_END)[0].split(";")

    # split(";") on "" yields [""], never [] -- so the notebook's
    # `if len(x) == 0` guard could never fire. Check for empty content instead.
    if not any(s.strip() for s in x) or not any(s.strip() for s in y):
        return DonutPrediction(
            image_id=image_id,
            chart_type=chart_type,
            raw=pred_string,
            failure_mode="empty_series",
        )

    if apply_cleaning:
        x, y = clean_preds(x, y)

    return DonutPrediction(
        image_id=image_id, chart_type=chart_type, x=list(x), y=list(y), raw=pred_string
    )
