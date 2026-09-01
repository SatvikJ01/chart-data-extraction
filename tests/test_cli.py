"""CLI argument handling: subset, profiles, precision, limit."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chart_extraction.config import LOCAL_GPU_BATCH_SIZES  # noqa: E402
from chart_extraction.eval.ground_truth import Annotation  # noqa: E402
from scripts.run_eval import (  # noqa: E402
    build_pipeline_config, build_runtime, parse_args, select_subset,
)


def _args(*extra):
    return parse_args(list(extra))


# --- Subset -----------------------------------------------------------------

@pytest.fixture
def annotations():
    return {
        "a": Annotation("a", "extracted", "line", (1,), (2,)),
        "b": Annotation("b", "generated", "line", (1,), (2,)),
        "c": Annotation("c", "extracted", "dot", (1,), (2,)),
        "d": Annotation("d", "generated", "dot", (1,), (2,)),
    }


def test_subset_both_is_the_default(annotations):
    assert _args().subset == "both"
    assert select_subset(["a", "b", "c", "d"], annotations, "both") == ["a", "b", "c", "d"]


def test_subset_extracted_selects_only_extracted(annotations):
    assert select_subset(["a", "b", "c", "d"], annotations, "extracted") == ["a", "c"]


def test_subset_generated_selects_only_generated(annotations):
    assert select_subset(["a", "b", "c", "d"], annotations, "generated") == ["b", "d"]


def test_subset_preserves_split_order(annotations):
    assert select_subset(["d", "c", "b", "a"], annotations, "extracted") == ["c", "a"]


def test_invalid_subset_rejected():
    with pytest.raises(SystemExit):
        _args("--subset", "synthetic")


# --- Runtime profiles -------------------------------------------------------

def test_local_profile_is_fp16():
    runtime = build_runtime(_args("--profile", "local"))
    assert runtime.precision == "fp16"
    assert runtime.device.startswith("cuda")
    assert runtime.oom_retry_enabled


def test_kaggle_profile_is_fp32():
    assert build_runtime(_args("--profile", "kaggle")).precision == "fp32"


def test_cpu_profile():
    runtime = build_runtime(_args("--profile", "cpu"))
    assert runtime.device == "cpu" and runtime.precision == "fp32"


def test_precision_flag_overrides_the_profile():
    assert build_runtime(_args("--profile", "kaggle", "--precision", "fp16")).precision == "fp16"
    assert build_runtime(_args("--profile", "local", "--precision", "fp32")).precision == "fp32"


def test_device_flag_overrides_the_profile():
    assert build_runtime(_args("--profile", "local", "--device", "cuda:1")).device == "cuda:1"


# --- OOM retry configuration ------------------------------------------------

def test_oom_scales_parse():
    runtime = build_runtime(_args("--oom-retry-scales", "0.8,0.6,0.4"))
    assert runtime.oom_retry_scales == (0.8, 0.6, 0.4)
    assert runtime.oom_retry_enabled


def test_empty_oom_scales_disables_rescaling():
    runtime = build_runtime(_args("--oom-retry-scales", ""))
    assert runtime.oom_retry_scales == ()
    assert not runtime.oom_retry_enabled


def test_default_oom_ladder():
    assert build_runtime(_args()).oom_retry_scales == (0.75, 0.5)


# --- Batch sizes ------------------------------------------------------------

def _config(args, tmp_path):
    from chart_extraction.paths import resolve_paths

    resolved = resolve_paths(
        overrides={
            "data_root": tmp_path, "donut_dir": tmp_path,
            "x_axis_model": tmp_path, "y_axis_model": tmp_path,
            "marker_model": tmp_path,
        },
        env={},
    )
    return build_pipeline_config(args, resolved, build_runtime(args))


def test_local_profile_batch_size(tmp_path):
    """Raised from 1 to 8 after a measured run peaked at 482MB of 3771MB --
    batch size 1 was leaving ~87% of the card unused."""
    config = _config(_args("--profile", "local"), tmp_path)
    assert config.donut_batch_size == 8
    assert config.axis_batch_size == 8
    assert config.marker_batch_size == 8
    assert LOCAL_GPU_BATCH_SIZES["donut_batch_size"] == 8


def test_batch_size_flag_overrides_the_local_default(tmp_path):
    config = _config(_args("--profile", "local", "--batch-size", "16"), tmp_path)
    assert config.donut_batch_size == 16


def test_batch_size_one_is_still_reachable(tmp_path):
    """The conservative setting must remain available for a smaller card."""
    config = _config(_args("--profile", "local", "--batch-size", "1"), tmp_path)
    assert config.donut_batch_size == 1


def test_oom_fallback_stays_on_at_the_larger_batch(tmp_path):
    """A bigger batch is only safe because the fallback still catches it."""
    runtime = build_runtime(_args("--profile", "local", "--batch-size", "32"))
    assert runtime.oom_retry_enabled
    assert runtime.oom_retry_scales == (0.75, 0.5)


def test_kaggle_profile_keeps_larger_batches(tmp_path):
    config = _config(_args("--profile", "kaggle"), tmp_path)
    assert config.donut_batch_size > 1


def test_batch_size_flag_applies_to_every_stage(tmp_path):
    config = _config(_args("--profile", "kaggle", "--batch-size", "2"), tmp_path)
    assert (config.donut_batch_size, config.axis_batch_size, config.marker_batch_size) == (2, 2, 2)


def test_paths_flow_into_the_pipeline_config(tmp_path):
    config = _config(_args(), tmp_path)
    assert config.image_dir == tmp_path / "train" / "images"
    assert config.donut_model_dir == tmp_path


# --- Decode labelling -------------------------------------------------------

def test_decode_choices_are_beam_width_only():
    assert _args("--decode", "greedy").decode == "greedy"
    assert _args("--decode", "beam2").decode == "beam2"
    with pytest.raises(SystemExit):
        _args("--decode", "temperature")


def test_limit_flag():
    assert _args("--limit", "25").limit == 25
    assert _args().limit is None


# --- Config paths are never Kaggle defaults ---------------------------------

def test_pipeline_config_has_no_default_paths():
    """A hardcoded /kaggle/input default resolves to nothing on a local box and
    fails later inside a model loader."""
    from chart_extraction.config import PipelineConfig

    config = PipelineConfig()
    for name in [
        "image_dir", "donut_model_dir", "x_axis_model_path",
        "y_axis_model_path", "marker_model_path",
    ]:
        assert getattr(config, name) is None, f"{name} must not default to a path"


def test_require_paths_names_what_is_missing():
    from chart_extraction.config import PipelineConfig

    with pytest.raises(ValueError, match="donut_model_dir"):
        PipelineConfig().require_paths("donut_model_dir")

    with pytest.raises(ValueError, match="BENETECH"):
        PipelineConfig().require_paths("image_dir")


def test_require_paths_passes_when_set(tmp_path):
    from chart_extraction.config import PipelineConfig

    PipelineConfig(donut_model_dir=tmp_path).require_paths("donut_model_dir")
