"""Turn a ground-truth annotation into the token string Donut is trained to emit.

This is the single most fragile part of fine-tuning a generative model into
structured output: the target string must be **exactly** what
``chart_extraction.donut.parsing.string2preds`` knows how to read back. If the
two drift apart, training will converge happily and every prediction will parse
to nothing. The round trip is unit-tested for that reason.

Format, matching what the parser expects::

    <|BOS|><vertical_bar><x_start>Mon;Tue;Wed<x_end><y_start>1;2;3<y_end></s>
"""

from __future__ import annotations

import math
from typing import Sequence

from chart_extraction.donut.tokens import (
    BOS_TOKEN, CHART_TYPE_TOKENS, X_END, X_START, Y_END, Y_START,
)

#: chart type name -> its special token
CHART_TYPE_TO_TOKEN = {name: token for token, name in CHART_TYPE_TOKENS}

#: The separator between series values. A categorical label containing this
#: would split into two values on the way back, so it is stripped.
SEPARATOR = ";"

#: Significant digits kept for floats. Donut has to emit every digit as a token,
#: so full repr precision costs sequence length for accuracy the metric cannot
#: reward -- the numeric branch scores on RMSE relative to the series spread.
FLOAT_PRECISION = 6


def format_value(value: object) -> str:
    """Render one series value as the model should emit it."""
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return "0"
        if value == int(value) and abs(value) < 1e15:
            return str(int(value))
        text = f"{value:.{FLOAT_PRECISION}g}"
        return text
    text = str(value).strip()
    # A separator inside a categorical label would create a phantom value on
    # parse-back, changing the series length and scoring zero.
    return text.replace(SEPARATOR, ",")


def _join(values: Sequence[object]) -> str:
    return SEPARATOR.join(format_value(v) for v in values)


def serialize_annotation(
    annotation,
    include_bos: bool = True,
    eos_token: str = "</s>",
) -> str:
    """Build the training target for one annotation.

    Raises on an unknown chart type rather than defaulting: a mislabelled
    target teaches the model the wrong token, and silently training on it is
    worse than failing loudly at dataset construction.
    """
    token = CHART_TYPE_TO_TOKEN.get(annotation.chart_type)
    if token is None:
        raise ValueError(
            f"{annotation.image_id}: no chart-type token for "
            f"{annotation.chart_type!r}; known: {sorted(CHART_TYPE_TO_TOKEN)}"
        )

    if len(annotation.x_series) != len(annotation.y_series):
        raise ValueError(
            f"{annotation.image_id}: x and y series differ in length "
            f"({len(annotation.x_series)} vs {len(annotation.y_series)}); the "
            "competition metric scores unequal lengths as zero, so this cannot "
            "be a training target"
        )

    parts = []
    if include_bos:
        parts.append(BOS_TOKEN)
    parts.append(token)
    parts.append(X_START)
    parts.append(_join(annotation.x_series))
    parts.append(X_END)
    parts.append(Y_START)
    parts.append(_join(annotation.y_series))
    parts.append(Y_END)
    if eos_token:
        parts.append(eos_token)
    return "".join(parts)


def roundtrip_ok(annotation, apply_cleaning: bool = True) -> bool:
    """Whether serialising then parsing recovers the chart type and lengths.

    Used by the dataset builder to drop targets the parser could not read back.
    Values are not compared exactly -- ``clean_preds`` legitimately coerces
    numeric strings -- but chart type and series length must survive, because
    the metric gates on both.
    """
    from chart_extraction.donut.parsing import string2preds

    try:
        text = serialize_annotation(annotation)
    except ValueError:
        return False

    parsed = string2preds(text, annotation.image_id, apply_cleaning=apply_cleaning)
    return (
        parsed.is_well_formed
        and parsed.chart_type == annotation.chart_type
        and len(parsed.x) == len(annotation.x_series)
        and len(parsed.y) == len(annotation.y_series)
    )
