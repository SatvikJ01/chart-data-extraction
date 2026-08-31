"""Checkpoint and data path resolution.

The same entrypoint runs locally and on Kaggle, so no path is hardcoded to
``/kaggle/input``. Resolution order, highest precedence first:

  1. explicit argument (CLI flag)
  2. environment variable
  3. JSON config file (``--paths-config``, or ``chart_extraction.paths.json``)
  4. a named preset, only if that environment actually looks present

A path that cannot be resolved is an error naming every place that was checked,
rather than a default that fails later inside a model loader.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

#: field name -> environment variable
ENV_VARS = {
    "data_root": "BENETECH_DATA_ROOT",
    "donut_dir": "BENETECH_DONUT_DIR",
    "x_axis_model": "BENETECH_X_AXIS",
    "y_axis_model": "BENETECH_Y_AXIS",
    "marker_model": "BENETECH_MARKER",
    "results_dir": "BENETECH_RESULTS_DIR",
}

#: Presets are conveniences, not defaults. A preset only applies when its
#: marker directory exists, so a local run never silently resolves to a Kaggle
#: path that is not there.
PRESETS: dict[str, dict[str, str]] = {
    "kaggle": {
        "_marker": "/kaggle/input",
        "data_root": "/kaggle/input/benetech-making-graphs-accessible",
        "donut_dir": "/kaggle/input/benetech-donut",
        "x_axis_model": "/kaggle/input/x-axis-model-10/model (1).pth",
        "y_axis_model": "/kaggle/input/y-axis-model-10/Y_Point_generation_weights_1.0.pth",
        "marker_model": "/kaggle/input/marker-model/Marker_weights.pth",
        "results_dir": "/kaggle/working/results",
    },
}

DEFAULT_CONFIG_FILENAME = "chart_extraction.paths.json"


@dataclass
class ResolvedPaths:
    data_root: Path | None = None
    donut_dir: Path | None = None
    x_axis_model: Path | None = None
    y_axis_model: Path | None = None
    marker_model: Path | None = None
    results_dir: Path | None = None

    #: field name -> where the value came from, for the result file
    origins: dict = None  # type: ignore[assignment]

    @property
    def image_dir(self) -> Path:
        return Path(self.data_root) / "train" / "images"

    @property
    def annotation_dir(self) -> Path:
        return Path(self.data_root) / "train" / "annotations"

    def as_dict(self) -> dict:
        out = {
            key: (str(value) if value is not None else None)
            for key, value in asdict(self).items()
            if key != "origins"
        }
        out["origins"] = dict(self.origins or {})
        return out

    def missing(self, required: tuple[str, ...]) -> list[str]:
        return [name for name in required if getattr(self, name) is None]

    def not_on_disk(self, required: tuple[str, ...]) -> list[tuple[str, Path]]:
        problems = []
        for name in required:
            value = getattr(self, name)
            if value is not None and not Path(value).exists():
                problems.append((name, Path(value)))
        return problems


def active_preset(name: str | None = None) -> str | None:
    """Return the preset that applies, or None.

    ``name`` forces a preset. Otherwise a preset is only selected when its
    marker directory exists on this machine.
    """
    if name:
        if name not in PRESETS:
            raise ValueError(f"unknown preset {name!r}; known: {sorted(PRESETS)}")
        return name
    for preset_name, values in PRESETS.items():
        if Path(values["_marker"]).exists():
            return preset_name
    return None


def load_config_file(path: Path | str | None) -> dict:
    """Read a JSON path config, if one is given or discoverable."""
    if path is not None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"paths config not found: {path}")
        return json.loads(path.read_text())

    for candidate in (Path.cwd() / DEFAULT_CONFIG_FILENAME,):
        if candidate.exists():
            return json.loads(candidate.read_text())
    return {}


def resolve_paths(
    overrides: dict | None = None,
    config_path: Path | str | None = None,
    preset: str | None = None,
    env: dict | None = None,
) -> ResolvedPaths:
    """Resolve every path, recording where each value came from."""
    env = os.environ if env is None else env
    overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
    config = load_config_file(config_path)
    preset_name = active_preset(preset)
    preset_values = PRESETS.get(preset_name, {}) if preset_name else {}

    resolved: dict = {}
    origins: dict = {}

    for field_def in fields(ResolvedPaths):
        name = field_def.name
        if name == "origins":
            continue

        if name in overrides:
            resolved[name] = Path(overrides[name])
            origins[name] = "argument"
            continue

        env_var = ENV_VARS.get(name)
        if env_var and env.get(env_var):
            resolved[name] = Path(env[env_var])
            origins[name] = f"env:{env_var}"
            continue

        if name in config:
            resolved[name] = Path(config[name])
            origins[name] = "config-file"
            continue

        if name in preset_values:
            resolved[name] = Path(preset_values[name])
            origins[name] = f"preset:{preset_name}"
            continue

        resolved[name] = None
        origins[name] = "unresolved"

    paths = ResolvedPaths(**resolved)
    paths.origins = origins
    return paths


def describe(paths: ResolvedPaths, required: tuple[str, ...]) -> str:
    """Human-readable resolution report for the console."""
    lines = []
    for name in required:
        value = getattr(paths, name)
        origin = (paths.origins or {}).get(name, "?")
        if value is None:
            status = "UNRESOLVED"
        elif Path(value).exists():
            status = "ok"
        else:
            status = "NOT ON DISK"
        lines.append(f"  {status:<12} {name:<14} {value}   [{origin}]")
    return "\n".join(lines)


def explain_unresolved(name: str) -> str:
    """Tell the user every place that was checked for one path."""
    env_var = ENV_VARS.get(name, "(none)")
    return (
        f"{name}: pass the CLI flag, set ${env_var}, add it to "
        f"{DEFAULT_CONFIG_FILENAME}, or run somewhere a preset applies "
        f"({sorted(PRESETS)})"
    )
