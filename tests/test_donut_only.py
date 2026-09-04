"""Donut-only mode: it must run cleanly without detection checkpoints, and it
must never be confusable with a full-pipeline run."""

from __future__ import annotations

import json

import numpy as np
import pytest

from chart_extraction.config import GREEDY, PipelineConfig
from chart_extraction.data.images import ImageRef
from chart_extraction.donut.parsing import DonutPrediction
from chart_extraction.eval.ground_truth import Annotation
from chart_extraction.eval.harness import StageTimings, evaluate
from chart_extraction.eval.results import append_result, format_report
from chart_extraction.eval.sanity import (
    NEAR_ZERO, REFERENCE_POINTS, SUSPICIOUS_RATIO, check_against_reference,
)
from chart_extraction.eval.splits import build_validation_split
from chart_extraction.eval.taxonomy import (
    AXIS_MISESTIMATION, MARKER_MISS, applicable_categories, taxonomy_report,
)
from chart_extraction.paths import resolve_paths
from chart_extraction.pipeline import ImageOutcome, decode_all
from chart_extraction.stages import (
    MODE_DONUT_ONLY, MODE_FULL, detect_stages, resolve_mode,
)


# --- Stage detection --------------------------------------------------------

@pytest.fixture
def donut_only_paths(tmp_path):
    donut = tmp_path / "donut"
    donut.mkdir()
    return resolve_paths(overrides={"data_root": tmp_path, "donut_dir": donut}, env={})


@pytest.fixture
def full_paths(tmp_path):
    donut = tmp_path / "donut"
    donut.mkdir()
    overrides = {"data_root": tmp_path, "donut_dir": donut}
    for name in ("x_axis_model", "y_axis_model", "marker_model"):
        path = tmp_path / f"{name}.pth"
        path.touch()
        overrides[name] = path
    return resolve_paths(overrides=overrides, env={})


def test_missing_detection_checkpoints_give_donut_only(donut_only_paths):
    availability = detect_stages(donut_only_paths)
    assert availability.donut
    assert not availability.axis and not availability.markers
    assert availability.mode == MODE_DONUT_ONLY
    assert set(availability.skipped) == {"axis", "markers"}


def test_all_checkpoints_present_gives_full(full_paths):
    availability = detect_stages(full_paths)
    assert availability.mode == MODE_FULL
    assert availability.skipped == ()


def test_reason_is_recorded_for_each_skipped_stage(donut_only_paths):
    availability = detect_stages(donut_only_paths)
    assert "axis" in availability.reasons and "markers" in availability.reasons
    assert "not configured" in availability.reasons["markers"]


def test_a_configured_but_absent_checkpoint_is_distinguished(tmp_path):
    donut = tmp_path / "donut"
    donut.mkdir()
    paths = resolve_paths(
        overrides={
            "data_root": tmp_path, "donut_dir": donut,
            "marker_model": tmp_path / "nope.pth",
        },
        env={},
    )
    availability = detect_stages(paths)
    assert "not on disk" in availability.reasons["markers"]


def test_partial_detection_checkpoints_still_degrade(tmp_path):
    """Markers without axis ticks buys nothing -- the decoders need both."""
    donut = tmp_path / "donut"
    donut.mkdir()
    marker = tmp_path / "marker.pth"
    marker.touch()
    paths = resolve_paths(
        overrides={"data_root": tmp_path, "donut_dir": donut, "marker_model": marker},
        env={},
    )
    assert detect_stages(paths).mode == MODE_DONUT_ONLY


# --- Mode resolution --------------------------------------------------------

def test_auto_picks_donut_only_when_detection_is_absent(donut_only_paths):
    assert resolve_mode("auto", detect_stages(donut_only_paths)) == MODE_DONUT_ONLY


def test_auto_picks_full_when_everything_is_present(full_paths):
    assert resolve_mode("auto", detect_stages(full_paths)) == MODE_FULL


def test_forcing_full_without_checkpoints_errors_rather_than_downgrading(donut_only_paths):
    """A caller who asked for the full pipeline must be told they cannot have
    it, not silently handed a different system."""
    with pytest.raises(ValueError, match="requires the detection checkpoints"):
        resolve_mode("full", detect_stages(donut_only_paths))


def test_donut_only_can_be_forced_even_with_checkpoints_present(full_paths):
    assert resolve_mode("donut_only", detect_stages(full_paths)) == MODE_DONUT_ONLY


def test_missing_donut_is_fatal_in_every_mode(tmp_path):
    paths = resolve_paths(overrides={"data_root": tmp_path}, env={})
    for requested in ("auto", "full", "donut_only"):
        with pytest.raises(ValueError, match="Donut checkpoint unavailable"):
            resolve_mode(requested, detect_stages(paths))


# --- Decoding ---------------------------------------------------------------

def _refs(*ids):
    return [ImageRef(image_id=i, path=f"/tmp/{i}.jpg") for i in ids]


def test_donut_only_uses_donuts_own_series_directly():
    """This reproduces tuned-donut, which is the configuration the
    published leaderboard score refers to."""
    refs = _refs("a")
    predictions = {"a": DonutPrediction("a", "line", x=["Mon", "Tue"], y=[1.5, 2.5])}
    outcomes = decode_all(
        refs, predictions, None, None, PipelineConfig(mode=MODE_DONUT_ONLY)
    )
    assert outcomes["a"].x_series == ["Mon", "Tue"]
    assert outcomes["a"].y_series == [1.5, 2.5]
    assert outcomes["a"].failure_mode is None
    assert outcomes["a"].mode == MODE_DONUT_ONLY


def test_donut_only_runs_without_any_detection_input():
    """The whole point: no crash when the detection stages produced nothing."""
    refs = _refs("a", "b")
    predictions = {
        i: DonutPrediction(i, "scatter", x=[1.0], y=[2.0]) for i in ["a", "b"]
    }
    outcomes = decode_all(
        refs, predictions, {}, {}, PipelineConfig(mode=MODE_DONUT_ONLY)
    )
    assert len(outcomes) == 2
    assert all(o.failure_mode is None for o in outcomes.values())


def test_donut_only_still_reports_malformed_generations():
    refs = _refs("a")
    predictions = {
        "a": DonutPrediction("a", "line", failure_mode="missing_series_delimiters")
    }
    outcomes = decode_all(
        refs, predictions, None, None, PipelineConfig(mode=MODE_DONUT_ONLY)
    )
    assert outcomes["a"].failure_mode == "missing_series_delimiters"


def test_donut_only_flags_an_empty_series():
    refs = _refs("a")
    predictions = {"a": DonutPrediction("a", "line", x=["m"], y=[])}
    outcomes = decode_all(
        refs, predictions, None, None, PipelineConfig(mode=MODE_DONUT_ONLY)
    )
    assert outcomes["a"].failure_mode == "empty_series"


def test_full_mode_still_requires_detection_input():
    """Donut-only must not have loosened the full path."""
    refs = _refs("a")
    predictions = {"a": DonutPrediction("a", "vertical_bar", x=["m"], y=[1.0])}
    outcomes = decode_all(refs, predictions, {}, {}, PipelineConfig(mode=MODE_FULL))
    assert outcomes["a"].failure_mode == "missing_marker_detections"


def test_unknown_mode_rejected():
    with pytest.raises(ValueError, match="unknown pipeline mode"):
        decode_all(_refs("a"), {"a": DonutPrediction("a", "line")},
                   config=PipelineConfig(mode="halfway"))


# --- Taxonomy ---------------------------------------------------------------

def test_detection_categories_are_omitted_not_zeroed():
    """Reporting marker_miss: 0 would read as a clean detection pass rather
    than an absent one."""
    applicable = applicable_categories(MODE_DONUT_ONLY)
    assert MARKER_MISS not in applicable
    assert AXIS_MISESTIMATION not in applicable
    assert MARKER_MISS in applicable_categories(MODE_FULL)


def test_taxonomy_report_lists_what_it_could_not_produce():
    outcomes = {"a": ImageOutcome("a", "line", mode=MODE_DONUT_ONLY)}
    annotations = {"a": Annotation("a", "generated", "line", (1,), (2,))}
    report = taxonomy_report(outcomes, annotations, MODE_DONUT_ONLY)
    assert report["mode"] == MODE_DONUT_ONLY
    assert MARKER_MISS not in report["counts"]
    assert MARKER_MISS in report["not_applicable"]


# --- Sanity check against the reference -------------------------------------

def test_reference_point_is_the_donut_only_leaderboard_score():
    reference = REFERENCE_POINTS[MODE_DONUT_ONLY]
    assert reference.score == 0.44
    assert reference.mode == MODE_DONUT_ONLY


def test_near_zero_score_is_flagged_as_a_loading_problem():
    warnings = check_against_reference(0.001, MODE_DONUT_ONLY, n_instances=500)
    codes = {w["code"] if isinstance(w, dict) else w.code for w in warnings}
    assert "score_near_zero" in codes
    assert any(w.level == "error" for w in warnings)
    assert any("loading" in w.message for w in warnings)


def test_far_below_reference_is_an_error():
    warnings = check_against_reference(0.10, MODE_DONUT_ONLY, n_instances=500)
    assert any(w.code == "far_below_reference" and w.level == "error" for w in warnings)


def test_slightly_below_reference_is_only_a_warning():
    warnings = check_against_reference(0.40, MODE_DONUT_ONLY, n_instances=500)
    levels = {w.code: w.level for w in warnings}
    assert levels.get("below_reference") == "warning"
    assert "far_below_reference" not in levels


def test_above_reference_is_informational_not_success():
    """Scoring above the leaderboard is EXPECTED here -- easier split, possible
    leakage -- and must never be presented as beating it."""
    warnings = check_against_reference(0.75, MODE_DONUT_ONLY, n_instances=500)
    above = [w for w in warnings if w.code == "above_reference"]
    assert above and above[0].level == "info"
    assert "EXPECTED" in above[0].message
    assert "beating the leaderboard" in above[0].message


def test_the_threshold_boundary():
    reference = REFERENCE_POINTS[MODE_DONUT_ONLY].score
    just_above = check_against_reference(
        reference * SUSPICIOUS_RATIO + 0.01, MODE_DONUT_ONLY, 500
    )
    assert not any(w.code == "far_below_reference" for w in just_above)
    just_below = check_against_reference(
        reference * SUSPICIOUS_RATIO - 0.01, MODE_DONUT_ONLY, 500
    )
    assert any(w.code == "far_below_reference" for w in just_below)


def test_small_sample_is_flagged():
    warnings = check_against_reference(0.5, MODE_DONUT_ONLY, n_instances=10)
    assert any(w.code == "small_sample" for w in warnings)


def test_full_mode_has_no_reference_and_is_not_compared():
    """The 0.44 reference is a Donut-only number; applying it to a full run
    would be a category error."""
    warnings = check_against_reference(0.9, MODE_FULL, n_instances=500)
    assert not any(w.code in ("above_reference", "far_below_reference") for w in warnings)


# --- Reporting --------------------------------------------------------------

@pytest.fixture
def donut_only_result():
    ids = ["a", "b"]
    refs = _refs(*ids)
    annotations = {
        "a": Annotation("a", "extracted", "line", ("m", "t"), (1.0, 2.0)),
        "b": Annotation("b", "generated", "line", ("m", "t"), (1.0, 2.0)),
    }
    predictions = {
        i: DonutPrediction(i, "line", x=["m", "t"], y=[1.0, 2.0]) for i in ids
    }
    config = PipelineConfig(generation=GREEDY, mode=MODE_DONUT_ONLY)
    availability = type(
        "A", (), {
            "as_dict": lambda self: {
                "mode": MODE_DONUT_ONLY,
                "stages_run": ["donut"],
                "stages_skipped": ["axis", "markers"],
                "reasons": {"axis": "x_axis_model not configured",
                            "markers": "marker_model not configured"},
            }
        }
    )()
    return evaluate(
        refs=refs, annotations=annotations, donut_predictions=predictions,
        axis_ticks={}, marker_detections={},
        timings=StageTimings(n_images=2, donut_s=2.0),
        models=[{"name": "donut", "parameters": 201_000_000,
                 "trainable_parameters": 201_000_000, "size_mb": 383.5}],
        config=config, split=build_validation_split(annotations, fraction=1.0),
        run_id="donly", stages=availability.as_dict(),
    )


def test_result_records_the_mode_and_skipped_stages(donut_only_result):
    assert donut_only_result.config["mode"] == MODE_DONUT_ONLY
    assert donut_only_result.stages["stages_skipped"] == ["axis", "markers"]
    assert donut_only_result.stages["stages_run"] == ["donut"]


def test_result_carries_a_donut_only_caveat(donut_only_result):
    assert any("DONUT-ONLY" in c for c in donut_only_result.caveats)
    assert any("never be" in c for c in donut_only_result.caveats)


def test_report_leads_with_the_mode_and_skipped_stages(donut_only_result):
    report = format_report(donut_only_result)
    assert "mode=donut_only" in report
    assert "STAGES SKIPPED" in report
    assert "NOT comparable" in report


def test_ablation_row_carries_the_mode(donut_only_result, tmp_path):
    append_result(donut_only_result, tmp_path)
    ablation = (tmp_path / "ablation.md").read_text()
    row = [l for l in ablation.splitlines() if l.startswith("| `donly`")][0]
    assert "donut_only" in row
    assert "| mode |" in ablation
    assert "Never compare a" in ablation


def test_result_json_records_mode_stages_and_warnings(donut_only_result, tmp_path):
    append_result(donut_only_result, tmp_path)
    record = json.loads((tmp_path / "runs.jsonl").read_text())
    assert record["config"]["mode"] == MODE_DONUT_ONLY
    assert record["stages"]["stages_skipped"] == ["axis", "markers"]
    assert isinstance(record["warnings"], list)


def test_skipped_stages_have_no_models_or_latency(donut_only_result):
    names = {m["name"] for m in donut_only_result.models}
    assert names == {"donut"}
    assert donut_only_result.latency["axis_ms"] == 0.0
    assert donut_only_result.latency["markers_ms"] == 0.0


# --- Composition-aware messaging --------------------------------------------

from chart_extraction.eval.sanity import Composition, composition_from_scores  # noqa: E402
from chart_extraction.eval.harness import distribution_caveat  # noqa: E402


def _comp(extracted=0, generated=0):
    counts = {}
    if extracted:
        counts["extracted"] = extracted
    if generated:
        counts["generated"] = generated
    return Composition(counts=counts)


def test_extracted_only_run_is_never_called_synthetic():
    """The bug this guards: an --subset extracted run described as 'mostly
    synthetic' when it scored zero synthetic instances."""
    composition = _comp(extracted=2236)
    assert composition.label == "extracted-only"
    assert not composition.distribution_applies

    warnings = check_against_reference(
        0.5455, MODE_DONUT_ONLY, 2236, composition=composition
    )
    for warning in warnings:
        assert "mostly synthetic" not in warning.message
        assert "split is mostly synthetic" not in warning.message


def test_extracted_only_message_names_leakage_as_the_only_pressure():
    composition = _comp(extracted=2236)
    warning = [
        w for w in check_against_reference(0.5455, MODE_DONUT_ONLY, 2236,
                                           composition=composition)
        if w.code == "above_reference"
    ][0]
    assert "extracted-only" in warning.message
    assert "leakage is the only known upward pressure" in warning.message
    assert "Do not report this as beating the leaderboard" in warning.message


def test_generated_heavy_message_does_cite_synthetic():
    composition = _comp(extracted=500, generated=1500)
    warning = [
        w for w in check_against_reference(0.60, MODE_DONUT_ONLY, 2000,
                                           composition=composition)
        if w.code == "above_reference"
    ][0]
    assert "75%" in warning.message
    assert "synthetic" in warning.message


def test_composition_labels():
    assert _comp(extracted=10).label == "extracted-only"
    assert "fully synthetic" in _comp(generated=10).label
    assert "50%" in _comp(extracted=5, generated=5).label


def test_upward_pressures_count_by_composition():
    """Extracted-only has one known upward pressure; mixed has two."""
    assert len(_comp(extracted=10).upward_pressures()) == 1
    assert len(_comp(extracted=5, generated=5).upward_pressures()) == 2


def test_below_reference_reasoning_adapts():
    extracted = check_against_reference(0.40, MODE_DONUT_ONLY, 100,
                                        composition=_comp(extracted=100))
    message = [w for w in extracted if w.code == "below_reference"][0].message
    assert "easier than the test set" not in message
    assert "extracted-only" in message


def test_distribution_caveat_is_omitted_or_inverted_for_extracted_only():
    caveat = distribution_caveat(_comp(extracted=100))
    assert "does not apply to this number" in caveat
    assert "mostly synthetic" not in caveat


def test_distribution_caveat_warns_hard_on_generated_only():
    caveat = distribution_caveat(_comp(generated=100))
    assert "not indicative of real-world performance" in caveat


def test_distribution_caveat_reports_the_real_fraction():
    caveat = distribution_caveat(_comp(extracted=250, generated=750))
    assert "75%" in caveat


def test_empty_composition_produces_no_distribution_caveat():
    assert distribution_caveat(_comp()) is None


def test_composition_is_built_from_scored_instances_not_the_split():
    """A --subset run evaluates a fraction of its split; describing the split
    would misstate the run."""
    composition = composition_from_scores({"extracted": {"n_instances": 2236}})
    assert composition.counts == {"extracted": 2236}
    assert composition.total == 2236
