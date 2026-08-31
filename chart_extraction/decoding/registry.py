"""Chart-type -> decoder lookup."""

from __future__ import annotations

from chart_extraction.decoding.bar import BarDecoder, HorizontalBarDecoder
from chart_extraction.decoding.base import ChartDecoder
from chart_extraction.decoding.dot import DotDecoder
from chart_extraction.decoding.line import LineDecoder
from chart_extraction.decoding.scatter import ScatterDecoder

_DECODERS: dict[str, type[ChartDecoder]] = {
    "vertical_bar": BarDecoder,
    "horizontal_bar": HorizontalBarDecoder,
    "scatter": ScatterDecoder,
    "line": LineDecoder,
    "dot": DotDecoder,
}


def available_decoders() -> list[str]:
    return sorted(_DECODERS)


def build_decoder(chart_type: str) -> ChartDecoder | None:
    """Return a decoder for the chart type, or None if unsupported.

    Returning None rather than raising is deliberate: an unrecognised chart type
    is an expected Donut failure mode, not a programming error. The pipeline
    records it as such so Phase 2 can count it.
    """
    cls = _DECODERS.get(chart_type)
    return cls() if cls else None
