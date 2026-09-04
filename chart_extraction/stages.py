"""Which pipeline stages can actually run, given the checkpoints present.

The multi-stage pipeline needs three separate checkpoints. When the axis CNN or
the marker detector is unavailable, the pipeline degrades to Donut-only rather
than crashing -- but a Donut-only run is a *different system* producing a
different number, so the degradation is recorded everywhere the score is,
never inferred silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

#: Full multi-stage pipeline: Donut for chart type and x, CV stages re-derive y.
MODE_FULL = "full"
#: Donut alone: its generated x and y series are used directly, with no
#: detection stages. This is what tuned-donut did.
MODE_DONUT_ONLY = "donut_only"
MODES = (MODE_FULL, MODE_DONUT_ONLY)

#: config field -> the stage it enables
STAGE_REQUIREMENTS = {
    "donut": ("donut_model_dir",),
    "axis": ("x_axis_model_path", "y_axis_model_path"),
    "markers": ("marker_model_path",),
}


@dataclass(frozen=True)
class StageAvailability:
    """Which stages have a usable checkpoint on disk."""

    donut: bool
    axis: bool
    markers: bool
    reasons: dict = field(default_factory=dict)

    @property
    def available(self) -> tuple[str, ...]:
        return tuple(
            name for name in ("donut", "axis", "markers") if getattr(self, name)
        )

    @property
    def skipped(self) -> tuple[str, ...]:
        return tuple(
            name for name in ("donut", "axis", "markers") if not getattr(self, name)
        )

    @property
    def mode(self) -> str:
        """The richest mode these checkpoints support.

        Both detection stages are required for the full pipeline: the decoders
        need marker boxes *and* axis ticks to convert pixels into values, so one
        without the other buys nothing.
        """
        if self.axis and self.markers:
            return MODE_FULL
        return MODE_DONUT_ONLY

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "stages_run": list(self.available),
            "stages_skipped": list(self.skipped),
            "reasons": dict(self.reasons),
        }

    def describe(self) -> str:
        lines = [f"pipeline mode: {self.mode}"]
        for name in ("donut", "axis", "markers"):
            if getattr(self, name):
                lines.append(f"  run     {name}")
            else:
                lines.append(f"  SKIP    {name}: {self.reasons.get(name, 'unavailable')}")
        return "\n".join(lines)


def _stage_ok(paths, fields: tuple[str, ...]) -> tuple[bool, str]:
    for name in fields:
        value = getattr(paths, name, None)
        if value is None:
            return False, f"{name} not configured"
        if not Path(value).exists():
            return False, f"{name} not on disk ({value})"
    return True, ""


def detect_stages(paths) -> StageAvailability:
    """Inspect resolved paths and report which stages can run.

    ``paths`` is anything exposing the checkpoint attributes -- a ResolvedPaths
    or a PipelineConfig, whose field names differ, so both spellings are tried.
    """
    results: dict[str, bool] = {}
    reasons: dict[str, str] = {}

    for stage, fields in STAGE_REQUIREMENTS.items():
        ok, reason = _stage_ok(paths, _resolve_field_names(paths, fields))
        results[stage] = ok
        if not ok:
            reasons[stage] = reason

    return StageAvailability(
        donut=results["donut"],
        axis=results["axis"],
        markers=results["markers"],
        reasons=reasons,
    )


#: ResolvedPaths and PipelineConfig name the same checkpoints differently.
_FIELD_ALIASES = {
    "donut_model_dir": "donut_dir",
    "x_axis_model_path": "x_axis_model",
    "y_axis_model_path": "y_axis_model",
    "marker_model_path": "marker_model",
}


def _resolve_field_names(paths, fields: tuple[str, ...]) -> tuple[str, ...]:
    resolved = []
    for name in fields:
        if hasattr(paths, name):
            resolved.append(name)
        else:
            resolved.append(_FIELD_ALIASES.get(name, name))
    return tuple(resolved)


def resolve_mode(requested: str, availability: StageAvailability) -> str:
    """Pick the run mode, or explain why the requested one is impossible.

    ``requested`` is "auto", "full" or "donut_only". Forcing "full" without the
    detection checkpoints is an error rather than a silent downgrade: a caller
    who asked for the full pipeline should be told they are not getting it.
    """
    if requested not in ("auto", *MODES):
        raise ValueError(f"unknown mode {requested!r}; use auto, full or donut_only")

    if not availability.donut:
        raise ValueError(
            f"Donut checkpoint unavailable ({availability.reasons.get('donut')}); "
            "no mode can run without it"
        )

    if requested == "auto":
        return availability.mode

    if requested == MODE_FULL and availability.mode != MODE_FULL:
        missing = ", ".join(
            f"{stage} ({availability.reasons.get(stage)})"
            for stage in availability.skipped
        )
        raise ValueError(
            f"--mode full requires the detection checkpoints, but: {missing}. "
            "Use --mode auto or --mode donut_only."
        )

    return requested
