"""Donut's structured-output token schema.

Donut is trained OCR-free to emit a token sequence, and structure is imposed by
special tokens rather than by a grammar. Nothing constrains decoding to produce
well-formed output, which is why every consumer of these tokens has to treat a
malformed sequence as an expected case rather than an error.
"""

from __future__ import annotations

BOS_TOKEN = "<|BOS|>"
X_START = "<x_start>"
X_END = "<x_end>"
Y_START = "<y_start>"
Y_END = "<y_end>"

# Ordered: the original code checked these in this order and returned on the
# first match, so a sequence containing two chart-type tokens resolves to
# whichever appears earliest in this list. Order preserved deliberately.
CHART_TYPE_TOKENS: tuple[tuple[str, str], ...] = (
    ("<dot>", "dot"),
    ("<horizontal_bar>", "horizontal_bar"),
    ("<vertical_bar>", "vertical_bar"),
    ("<scatter>", "scatter"),
    ("<line>", "line"),
)

CHART_TYPES = tuple(name for _, name in CHART_TYPE_TOKENS)

# Fallback when no chart-type token is present at all.
DEFAULT_CHART_TYPE = "vertical_bar"
