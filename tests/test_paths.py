"""Path resolution: the same entrypoint must work locally and on Kaggle."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chart_extraction.paths import (
    ENV_VARS, PRESETS, ResolvedPaths, active_preset, describe,
    explain_unresolved, resolve_paths,
)

REQUIRED = ("data_root", "donut_dir", "x_axis_model")


def _kaggle_string_lines(path: Path) -> list[int]:
    """Line numbers of executable string literals containing /kaggle/input.

    Uses the AST so prose in docstrings and comments is not mistaken for a
    hardcoded path.
    """
    import ast

    tree = ast.parse(path.read_text())
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc is not None and node.body:
                docstrings.add(node.body[0].lineno)

    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and "/kaggle/input" in node.value
        and node.lineno not in docstrings
    ]


def _presets_line_span(path: Path) -> range:
    import ast

    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        targets = getattr(node, "targets", []) or ([getattr(node, "target", None)])
        for target in targets:
            if isinstance(target, ast.Name) and target.id == "PRESETS":
                return range(node.lineno, (node.end_lineno or node.lineno) + 1)
    raise AssertionError("PRESETS not found")


def test_kaggle_paths_live_only_in_the_preset():
    """The preset is the single place /kaggle/input may appear as code, and it
    only applies when that directory actually exists."""
    source = Path("chart_extraction/paths.py")
    span = _presets_line_span(source)
    stray = [line for line in _kaggle_string_lines(source) if line not in span]
    assert not stray, f"/kaggle/input hardcoded outside PRESETS at lines {stray}"


@pytest.mark.parametrize(
    "module",
    [
        "chart_extraction/config.py",
        "chart_extraction/eval/runner.py",
        "scripts/run_eval.py",
        "scripts/kaggle_eval.py",
    ],
)
def test_no_module_outside_paths_hardcodes_kaggle(module):
    """The same entrypoint runs locally and on Kaggle, so no other module may
    carry a Kaggle path in executable code."""
    assert _kaggle_string_lines(Path(module)) == []


def test_defaults_are_empty_without_env_config_or_preset():
    resolved = resolve_paths(env={})
    assert resolved.missing(REQUIRED) == list(REQUIRED)


def test_env_vars_resolve():
    env = {var: f"/from/env/{name}" for name, var in ENV_VARS.items()}
    resolved = resolve_paths(env=env)
    assert resolved.donut_dir == Path("/from/env/donut_dir")
    assert resolved.origins["donut_dir"] == "env:BENETECH_DONUT_DIR"


def test_argument_beats_env():
    resolved = resolve_paths(
        overrides={"donut_dir": "/from/arg"},
        env={"BENETECH_DONUT_DIR": "/from/env"},
    )
    assert resolved.donut_dir == Path("/from/arg")
    assert resolved.origins["donut_dir"] == "argument"


def test_env_beats_config_file(tmp_path):
    config = tmp_path / "paths.json"
    config.write_text(json.dumps({"donut_dir": "/from/config"}))
    resolved = resolve_paths(
        config_path=config, env={"BENETECH_DONUT_DIR": "/from/env"}
    )
    assert resolved.donut_dir == Path("/from/env")


def test_config_file_used_when_env_absent(tmp_path):
    config = tmp_path / "paths.json"
    config.write_text(json.dumps({"donut_dir": "/from/config"}))
    resolved = resolve_paths(config_path=config, env={})
    assert resolved.donut_dir == Path("/from/config")
    assert resolved.origins["donut_dir"] == "config-file"


def test_none_overrides_are_ignored():
    """Unset CLI flags arrive as None and must not mask an env var."""
    resolved = resolve_paths(
        overrides={"donut_dir": None}, env={"BENETECH_DONUT_DIR": "/from/env"}
    )
    assert resolved.donut_dir == Path("/from/env")


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_paths(config_path=tmp_path / "absent.json", env={})


def test_preset_only_applies_when_its_marker_exists():
    """A local machine must never silently resolve to a Kaggle path."""
    assert not Path(PRESETS["kaggle"]["_marker"]).exists()
    assert active_preset() is None
    assert resolve_paths(env={}).missing(REQUIRED) == list(REQUIRED)


def test_forced_preset_resolves_kaggle_paths():
    resolved = resolve_paths(preset="kaggle", env={})
    assert resolved.donut_dir == Path("/kaggle/input/benetech-donut")
    assert resolved.origins["donut_dir"] == "preset:kaggle"


def test_unknown_preset_rejected():
    with pytest.raises(ValueError, match="unknown preset"):
        resolve_paths(preset="colab", env={})


def test_derived_directories():
    resolved = resolve_paths(overrides={"data_root": "/data/bench"}, env={})
    assert resolved.image_dir == Path("/data/bench/train/images")
    assert resolved.annotation_dir == Path("/data/bench/train/annotations")


def test_not_on_disk_is_distinguished_from_unresolved(tmp_path):
    real = tmp_path / "donut"
    real.mkdir()
    resolved = resolve_paths(
        overrides={"donut_dir": real, "data_root": "/does/not/exist"}, env={}
    )
    assert resolved.missing(("donut_dir", "data_root")) == []
    absent = dict(resolved.not_on_disk(("donut_dir", "data_root")))
    assert "data_root" in absent and "donut_dir" not in absent


def test_describe_reports_status_and_origin(tmp_path):
    resolved = resolve_paths(overrides={"donut_dir": tmp_path}, env={})
    text = describe(resolved, ("donut_dir", "data_root"))
    assert "ok" in text and "UNRESOLVED" in text
    assert "argument" in text


def test_explain_names_every_mechanism():
    message = explain_unresolved("donut_dir")
    for expected in ["BENETECH_DONUT_DIR", "chart_extraction.paths.json", "kaggle"]:
        assert expected in message


def test_as_dict_is_json_serialisable():
    resolved = resolve_paths(overrides={"donut_dir": "/x"}, env={})
    payload = json.dumps(resolved.as_dict())
    assert json.loads(payload)["donut_dir"] == "/x"
