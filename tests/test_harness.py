"""Harness, taxonomy and results tests.

No models are loaded: stage outputs are fabricated, so these exercise the
aggregation, taxonomy and persistence logic that produce every reported number.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from chart_extraction.axis.inference import AxisTicks
from chart_extraction.config import BEAM2, GREEDY, PipelineConfig
from chart_extraction.data.images import ImageRef
from chart_extraction.donut.parsing import DonutPrediction
from chart_extraction.eval.ground_truth import Annotation
from chart_extraction.eval.harness import (
    LEAKAGE_CAVEAT, StageTimings, _generation_label, evaluate,
    outcomes_to_prediction_frame, score_breakdown,
)
from chart_extraction.eval.results import append_result, format_report, load_runs
from chart_extraction.eval.splits import build_validation_split
from chart_extraction.eval.taxonomy import (
    AXIS_MISESTIMATION, MALFORMED_SEQUENCE, MARKER_MISS, OK, UNSUPPORTED_CHART_TYPE,
    WRONG_CHART_TYPE, categorise, taxonomy_by_chart_type, taxonomy_counts,
)
from chart_extraction.pipeline import ImageOutcome


# --- Taxonomy ---------------------------------------------------------------

def _annotation(image_id="a", chart_type="line", source="generated"):
    return Annotation(image_id, source, chart_type, (1.0, 2.0), (3.0, 4.0))


@pytest.mark.parametrize(
    "failure_mode,expected",
    [
        ("no_chart_type_token", MALFORMED_SEQUENCE),
        ("missing_series_delimiters", MALFORMED_SEQUENCE),
        ("empty_series", MALFORMED_SEQUENCE),
        ("generation_error", MALFORMED_SEQUENCE),
        ("unusable_axis_calibration", AXIS_MISESTIMATION),
        ("no_markers_decoded", MARKER_MISS),
        ("missing_marker_detections", MARKER_MISS),
        (None, OK),
    ],
)
def test_failure_modes_map_to_brief_categories(failure_mode, expected):
    outcome = ImageOutcome("a", "line", failure_mode=failure_mode)
    assert categorise(outcome, _annotation()) == expected


def test_wrong_chart_type_requires_ground_truth():
    outcome = ImageOutcome("a", "scatter")
    assert categorise(outcome, _annotation(chart_type="line")) == WRONG_CHART_TYPE
    # Without ground truth the pipeline cannot know it is wrong.
    assert categorise(outcome, None) == OK


def test_malformed_sequence_wins_over_wrong_chart_type():
    """The malformed sequence is the upstream cause; reporting the downstream
    symptom would misattribute it."""
    outcome = ImageOutcome("a", "scatter", failure_mode="empty_series")
    assert categorise(outcome, _annotation(chart_type="line")) == MALFORMED_SEQUENCE


def test_axis_failure_on_a_misclassified_chart_is_reported_as_wrong_type():
    outcome = ImageOutcome("a", "scatter", failure_mode="unusable_axis_calibration")
    assert categorise(outcome, _annotation(chart_type="line")) == WRONG_CHART_TYPE
    assert categorise(outcome, _annotation(chart_type="scatter")) == AXIS_MISESTIMATION


def test_horizontal_bar_is_unsupported_not_a_marker_miss():
    """Finding F: it has no decoder, which is a different problem from the
    detector failing."""
    outcome = ImageOutcome("a", "horizontal_bar", failure_mode="no_decoder_for_horizontal_bar")
    got = categorise(outcome, _annotation(chart_type="horizontal_bar"))
    assert got == UNSUPPORTED_CHART_TYPE


def test_taxonomy_counts_include_zero_categories():
    counts = taxonomy_counts({"a": ImageOutcome("a", "line")}, {"a": _annotation()})
    assert counts[OK] == 1
    assert counts[MARKER_MISS] == 0, "zero categories must still be reported"


def test_taxonomy_groups_on_ground_truth_chart_type():
    """Grouping on the prediction would hide wrong-type errors inside whichever
    type the model wrongly guessed."""
    outcomes = {"a": ImageOutcome("a", "scatter")}
    annotations = {"a": _annotation(chart_type="line")}
    by_type = taxonomy_by_chart_type(outcomes, annotations)
    assert "line" in by_type and "scatter" not in by_type
    assert by_type["line"][WRONG_CHART_TYPE] == 1


# --- Score breakdown --------------------------------------------------------

def _outcome(image_id, chart_type="line", x=(1.0, 2.0), y=(3.0, 4.0), failure=None):
    return ImageOutcome(image_id, chart_type, list(x), list(y), failure)


def test_breakdown_aggregations_are_consistent():
    """Overall, per-type and per-source must be aggregations of the same
    per-instance scores."""
    annotations = {
        "a": Annotation("a", "generated", "line", (1.0, 2.0), (3.0, 4.0)),
        "b": Annotation("b", "extracted", "dot", (1.0, 2.0), (3.0, 4.0)),
    }
    outcomes = {"a": _outcome("a", "line"), "b": _outcome("b", "dot")}

    breakdown = score_breakdown(outcomes, annotations, ["a", "b"])
    assert breakdown["overall"] == pytest.approx(1.0)
    assert breakdown["by_source"]["extracted"]["score"] == pytest.approx(1.0)
    assert breakdown["by_source"]["generated"]["score"] == pytest.approx(1.0)
    assert breakdown["n_instances"] == 4
    per_instance = breakdown["per_instance"]
    assert per_instance["score"].mean() == pytest.approx(breakdown["overall"])


def test_breakdown_separates_extracted_from_generated():
    annotations = {
        "a": Annotation("a", "generated", "line", (1.0, 2.0), (3.0, 4.0)),
        "b": Annotation("b", "extracted", "line", (1.0, 2.0), (3.0, 4.0)),
    }
    outcomes = {
        "a": _outcome("a", "line"),                       # perfect
        "b": _outcome("b", "scatter"),                    # wrong chart type -> 0
    }
    breakdown = score_breakdown(outcomes, annotations, ["a", "b"])
    assert breakdown["by_source"]["generated"]["score"] == pytest.approx(1.0)
    assert breakdown["by_source"]["extracted"]["score"] == 0.0
    assert breakdown["chart_type_accuracy"] == pytest.approx(0.5)


def test_prediction_frame_has_two_rows_per_image():
    frame = outcomes_to_prediction_frame({"a": _outcome("a")}, ["a"])
    assert list(frame.index) == ["a_x", "a_y"]


# --- Full evaluate() --------------------------------------------------------

def _refs(*ids):
    return [ImageRef(image_id=i, path=f"/tmp/{i}.jpg") for i in ids]


def _stage_inputs(ids, chart_type="vertical_bar"):
    donut = {
        i: DonutPrediction(i, chart_type, x=["a", "b"], y=[10.0, 20.0, 30.0]) for i in ids
    }
    ticks = {
        i: AxisTicks(
            i,
            np.array([[15.0, 210.0], [45.0, 210.0]]),
            np.array([[40.0, 100.0], [40.0, 120.0], [40.0, 140.0]]),
        )
        for i in ids
    }
    from chart_extraction.markers.inference import MarkerDetections

    markers = {
        i: MarkerDetections(
            i,
            np.array([[10, 95, 20, 105], [40, 115, 50, 125]], dtype=float),
            np.array([3, 3]),
            np.array([0.9, 0.9]),
        )
        for i in ids
    }
    return donut, ticks, markers


@pytest.fixture
def evaluation():
    ids = ["a", "b", "c"]
    refs = _refs(*ids)
    annotations = {
        "a": Annotation("a", "generated", "vertical_bar", ("a", "b"), (1.0, 2.0)),
        "b": Annotation("b", "extracted", "vertical_bar", ("a", "b"), (1.0, 2.0)),
        "c": Annotation("c", "extracted", "horizontal_bar", ("a", "b"), (1.0, 2.0)),
    }
    donut, ticks, markers = _stage_inputs(ids)
    donut["c"] = DonutPrediction("c", "horizontal_bar", x=["a", "b"], y=[1.0, 2.0])

    split = build_validation_split(annotations, fraction=1.0)
    timings = StageTimings(n_images=3, donut_s=3.0, axis_s=1.5, markers_s=0.75)
    models = [{"name": "donut", "parameters": 200_000_000, "trainable_parameters": 200_000_000, "size_mb": 800.0}]

    return evaluate(
        refs=refs,
        annotations=annotations,
        donut_predictions=donut,
        axis_ticks=ticks,
        marker_detections=markers,
        timings=timings,
        models=models,
        config=PipelineConfig(generation=GREEDY),
        split=split,
        run_id="testrun",
    )


def test_evaluate_emits_every_requested_number(evaluation):
    """One pass must produce all of Phase 1 and Phase 2 together."""
    assert evaluation.run_id == "testrun"
    for key in ["overall", "by_chart_type", "by_source", "chart_type_accuracy"]:
        assert key in evaluation.scores
    assert "counts" in evaluation.taxonomy
    assert "by_ground_truth_chart_type" in evaluation.taxonomy
    assert evaluation.latency["total_ms"] > 0
    assert evaluation.models[0]["parameters"] == 200_000_000
    assert evaluation.split["n_images"] == 3


def test_latency_is_per_image(evaluation):
    # 3.0s donut over 3 images = 1000 ms/image.
    assert evaluation.latency["donut_ms"] == pytest.approx(1000.0)
    assert evaluation.latency["axis_ms"] == pytest.approx(500.0)
    assert evaluation.latency["markers_ms"] == pytest.approx(250.0)
    assert evaluation.latency["total_ms"] >= 1750.0


def test_finding_f_population_is_reported(evaluation):
    assert evaluation.populations["horizontal_bar_ground_truth"] == 1
    assert evaluation.populations["horizontal_bar_predicted"] == 1
    assert evaluation.populations["by_ground_truth_chart_type"]["horizontal_bar"] == 1


def test_headline_is_the_extracted_slice(evaluation):
    assert evaluation.headline == evaluation.scores["by_source"]["extracted"]["score"]
    assert evaluation.generated_score == evaluation.scores["by_source"]["generated"]["score"]


def test_leakage_caveat_travels_with_every_result(evaluation):
    assert LEAKAGE_CAVEAT in evaluation.caveats
    assert LEAKAGE_CAVEAT in evaluation.to_dict()["caveats"]


def test_result_dict_is_json_serialisable(evaluation):
    payload = json.dumps(evaluation.to_dict(), sort_keys=True)
    assert json.loads(payload)["run_id"] == "testrun"
    assert "per_instance" not in json.loads(payload)["scores"]


# --- Generation labelling ---------------------------------------------------

def test_generation_labels_never_claim_sampling_tuning():
    """Phase 0 finding E: temperature/top_k/top_p were inert because do_sample
    was never set. Labels must say beam search."""
    greedy = _generation_label(PipelineConfig(generation=GREEDY))
    beam = _generation_label(PipelineConfig(generation=BEAM2))
    assert greedy == "greedy"
    assert beam == "beam_search(num_beams=2)"
    for label in (greedy, beam):
        assert "temperature" not in label
        assert "top_p" not in label
        assert "nucleus" not in label


def test_sampling_label_only_appears_when_sampling_is_enabled():
    from chart_extraction.config import GenerationConfig

    sampling = PipelineConfig(
        generation=GenerationConfig(do_sample=True, temperature=0.9, top_p=0.4, top_k=1)
    )
    assert _generation_label(sampling).startswith("sampling(")


# --- Results persistence ----------------------------------------------------

def test_results_append_rather_than_overwrite(evaluation, tmp_path):
    append_result(evaluation, tmp_path)
    append_result(evaluation, tmp_path)

    runs = load_runs(tmp_path)
    assert len(runs) == 2, "runs.jsonl must append, not overwrite"

    ablation = (tmp_path / "ablation.md").read_text()
    assert ablation.count("| `testrun` ") == 2
    assert ablation.count("# Ablation table") == 1, "header written once"


def test_ablation_row_reports_extracted_and_generated_separately(evaluation, tmp_path):
    append_result(evaluation, tmp_path)
    ablation = (tmp_path / "ablation.md").read_text()
    assert "extracted" in ablation and "generated" in ablation
    assert LEAKAGE_CAVEAT in ablation


def test_ablation_never_labels_a_row_as_temperature_tuning(evaluation, tmp_path):
    append_result(evaluation, tmp_path)
    ablation = (tmp_path / "ablation.md").read_text()
    row = [l for l in ablation.splitlines() if l.startswith("| `testrun`")][0]
    assert "greedy" in row
    assert "temperature" not in row and "top_p" not in row


def test_per_instance_scores_are_written(evaluation, tmp_path):
    written = append_result(evaluation, tmp_path)
    assert written["per_instance"].exists()
    assert written["per_instance"].read_text().startswith("instance_id,")


def test_format_report_covers_the_required_sections(evaluation):
    report = format_report(evaluation)
    for expected in [
        "HEADLINE (extracted)", "generated", "per chart type",
        "error taxonomy", "latency", "models", "finding F", "caveats",
    ]:
        assert expected in report
