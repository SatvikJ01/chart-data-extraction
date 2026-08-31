"""Pixel-to-value calibration for a chart axis.

This module replaces the notebook's ``extend_y_axis`` helper plus the
``find_element_above`` / ``least_count`` arithmetic that was duplicated
verbatim across all four decoder classes.

AUDIT NOTE (Phase 0, finding A) -- ACTIVE BUG, silent
-----------------------------------------------------
``find_element_above`` began with ``lst.sort()``, sorting the **caller's** list
in place. It was called with ``self.y_points`` while ``self.y_labels`` was left
untouched, so the index correspondence between points and labels was destroyed
on the first call. Measured at audit time::

    before: [(200.0, 0.0), (150.0, 10.0), (100.0, 20.0), (50.0, 30.0)]
    after : [(50.0,  0.0), (100.0, 10.0), (150.0, 20.0), (200.0, 30.0)]

Every value decoded after that point was calibrated against mispaired labels.
Fixed structurally: points and labels are stored as bound (pixel, value) pairs
that are sorted together and never re-sorted, so they cannot desync.

AUDIT NOTE (Phase 0, finding C) -- ACTIVE BUG, silent
-----------------------------------------------------
When ``find_element_above`` returned None the decoders set ``index = 0`` and
then evaluated ``y_labels[index - 1]`` / ``y_points[index - 1]``, i.e.
``[-1]`` -- wrapping to the opposite end of the axis. This produced a
plausible-looking wrong scale (measured: least_count = -0.2) rather than
raising. Fixed by explicit linear extrapolation from the nearest tick interval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class AxisCalibration:
    """A monotone pixel -> data-value mapping built from detected ticks.

    ``pixels`` and ``values`` are parallel and sorted by pixel, ascending. They
    are stored as immutable tuples specifically so that no downstream helper can
    reorder one without the other (finding A).
    """

    pixels: tuple[float, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.pixels) != len(self.values):
            raise ValueError(
                f"pixels/values length mismatch: {len(self.pixels)} vs {len(self.values)}"
            )

    @classmethod
    def from_ticks(
        cls, pixels: Sequence[float], values: Sequence[float]
    ) -> "AxisCalibration":
        """Bind tick pixel positions to tick values and sort them together."""
        if len(pixels) != len(values):
            n = min(len(pixels), len(values))
            pixels, values = pixels[:n], values[:n]

        paired = sorted(zip(map(float, pixels), map(float, values)), key=lambda p: p[0])
        # Collapse duplicate pixel positions -- a repeated pixel would make the
        # interval width zero and divide by zero during interpolation.
        deduped: list[tuple[float, float]] = []
        for px, val in paired:
            if deduped and abs(px - deduped[-1][0]) < 1e-9:
                continue
            deduped.append((px, val))

        if not deduped:
            return cls(pixels=(), values=())
        return cls(
            pixels=tuple(p for p, _ in deduped),
            values=tuple(v for _, v in deduped),
        )

    @property
    def is_usable(self) -> bool:
        """At least two distinct ticks are needed to define a scale."""
        return len(self.pixels) >= 2

    def value_at(self, pixel: float) -> float:
        """Map a pixel coordinate to a data value.

        Piecewise-linear between ticks, linearly extrapolated from the nearest
        interval outside them. Replaces the original find_element_above +
        least_count arithmetic, which wrapped around the array on the
        out-of-range branch (finding C).
        """
        if not self.is_usable:
            # A single tick gives an offset but no scale; no ticks gives nothing.
            return float(self.values[0]) if self.pixels else 0.0

        pixels, values = self.pixels, self.values

        if pixel <= pixels[0]:
            lo, hi = 0, 1
        elif pixel >= pixels[-1]:
            lo, hi = len(pixels) - 2, len(pixels) - 1
        else:
            hi = next(i for i, p in enumerate(pixels) if p >= pixel)
            lo = hi - 1

        span = pixels[hi] - pixels[lo]
        if span == 0:
            return float(values[lo])
        slope = (values[hi] - values[lo]) / span
        return float(values[lo] + (pixel - pixels[lo]) * slope)

    @property
    def unit_per_pixel(self) -> float:
        """Average data units per pixel across the calibrated span."""
        if not self.is_usable:
            return 0.0
        span = self.pixels[-1] - self.pixels[0]
        if span == 0:
            return 0.0
        return (self.values[-1] - self.values[0]) / span
