"""Periodic progress reporting for long inference passes.

Deliberately not tqdm. These runs are driven from a plain terminal, a Kaggle
notebook and a redirected log file, and tqdm's carriage-return rendering turns
into thousands of unreadable lines in the latter two. This logs a complete line
at a fixed time interval instead, so a captured log stays legible and a live
terminal still updates often enough to tell a slow run from a hung one.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)


def _format_duration(seconds: float) -> str:
    if seconds < 0 or seconds != seconds:  # negative or NaN
        return "?"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


class ProgressReporter:
    """Logs 'label 320/1118 (28.6%) 2.1 img/s eta 6m20s' periodically.

    The first and last updates always emit, so even a run shorter than one
    interval leaves a record of having started and finished.
    """

    def __init__(
        self,
        total: int,
        label: str = "progress",
        interval_s: float = 15.0,
        log: logging.Logger | None = None,
    ) -> None:
        self.total = max(int(total), 0)
        self.label = label
        self.interval_s = float(interval_s)
        self.log = log or logger
        self.done = 0
        self._start = time.monotonic()
        self._last_emit = 0.0
        self._emitted = False

    def start(self) -> "ProgressReporter":
        self._start = time.monotonic()
        self._last_emit = self._start
        self.log.info("%s: starting, %d images", self.label, self.total)
        return self

    def update(self, n: int = 1) -> None:
        self.done += int(n)
        now = time.monotonic()
        if now - self._last_emit >= self.interval_s:
            self._emit(now)
            self._last_emit = now

    def _emit(self, now: float) -> None:
        elapsed = max(now - self._start, 1e-9)
        rate = self.done / elapsed
        pct = (100.0 * self.done / self.total) if self.total else 0.0
        remaining = (self.total - self.done) / rate if rate > 0 else float("nan")
        self.log.info(
            "%s: %d/%d (%.1f%%) %.2f img/s elapsed %s eta %s",
            self.label, self.done, self.total, pct, rate,
            _format_duration(elapsed), _format_duration(remaining),
        )
        self._emitted = True

    def finish(self) -> None:
        now = time.monotonic()
        elapsed = max(now - self._start, 1e-9)
        rate = self.done / elapsed
        self.log.info(
            "%s: done %d/%d in %s (%.2f img/s)",
            self.label, self.done, self.total, _format_duration(elapsed), rate,
        )

    def __enter__(self) -> "ProgressReporter":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        # Report where it got to even when the pass failed -- knowing a run died
        # at image 900 of 1118 is exactly what is wanted at that moment.
        self.finish()
