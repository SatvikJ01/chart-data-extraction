"""Tests for the annotation loader and the source-keyed split builder."""

from __future__ import annotations

import json

import pytest

from chart_extraction.eval.ground_truth import (
    Annotation, annotations_to_frame, load_annotations, parse_annotation, summarise,
)
from chart_extraction.eval.splits import (
    build_validation_split, holdout_complement,
)


def _payload(**overrides):
    payload = {
        "source": "generated",
        "chart-type": "vertical_bar",
        "data-series": [{"x": "Mon", "y": 1.0}, {"x": "Tue", "y": 2.0}],
    }
    payload.update(overrides)
    return payload


def test_parses_hyphenated_keys():
    annotation = parse_annotation("img", _payload())
    assert annotation.source == "generated"
    assert annotation.chart_type == "vertical_bar"
    assert annotation.x_series == ("Mon", "Tue")
    assert annotation.y_series == (1.0, 2.0)


@pytest.mark.parametrize("key", ["source", "chart-type", "data-series"])
def test_missing_required_key_raises(key):
    payload = _payload()
    del payload[key]
    with pytest.raises(KeyError, match=key):
        parse_annotation("img", payload)


def test_unexpected_source_raises():
    """A silently defaulted source would corrupt the split and every per-source
    number built on it."""
    with pytest.raises(ValueError, match="unexpected source"):
        parse_annotation("img", _payload(source="synthetic"))


def test_unexpected_chart_type_raises():
    with pytest.raises(ValueError, match="unexpected chart-type"):
        parse_annotation("img", _payload(**{"chart-type": "pie"}))


def test_loads_from_disk_and_restricts_to_requested_ids(tmp_path):
    for name, source in [("a", "generated"), ("b", "extracted"), ("c", "generated")]:
        (tmp_path / f"{name}.json").write_text(json.dumps(_payload(source=source)))

    everything = load_annotations(tmp_path)
    assert set(everything) == {"a", "b", "c"}

    subset = load_annotations(tmp_path, image_ids=["a", "b"])
    assert set(subset) == {"a", "b"}


def test_missing_annotation_is_an_error_by_default(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps(_payload()))
    with pytest.raises(FileNotFoundError):
        load_annotations(tmp_path, image_ids=["a", "missing"])
    relaxed = load_annotations(tmp_path, image_ids=["a", "missing"], strict=False)
    assert set(relaxed) == {"a"}


def test_frame_has_two_instances_per_image():
    annotations = [parse_annotation("img", _payload())]
    frame = annotations_to_frame(annotations)
    assert list(frame.index) == ["img_x", "img_y"]
    assert frame.loc["img_x", "data_series"] == ["Mon", "Tue"]
    assert frame.loc["img_y", "data_series"] == [1.0, 2.0]
    assert (frame["chart_type"] == "vertical_bar").all()


def test_summarise_counts_by_source_and_type():
    annotations = [
        Annotation("a", "generated", "line", (1,), (2,)),
        Annotation("b", "generated", "line", (1,), (2,)),
        Annotation("c", "extracted", "dot", (1,), (2,)),
    ]
    summary = summarise(annotations).set_index(["source", "chart_type"])["count"]
    assert summary[("generated", "line")] == 2
    assert summary[("extracted", "dot")] == 1


# --- Split ------------------------------------------------------------------

@pytest.fixture
def population():
    annotations = {}
    for i in range(600):
        source = "extracted" if i % 60 == 0 else "generated"
        annotations[f"img{i:04d}"] = Annotation(
            f"img{i:04d}", source, "line", (1,), (2,)
        )
    return annotations


def test_split_is_deterministic(population):
    first = build_validation_split(population, fraction=0.1)
    second = build_validation_split(population, fraction=0.1)
    assert first.image_ids == second.image_ids


def test_split_changes_with_salt(population):
    default = build_validation_split(population, fraction=0.1)
    other = build_validation_split(population, fraction=0.1, salt="different")
    assert default.image_ids != other.image_ids


def test_split_is_stratified_by_source(population):
    split = build_validation_split(population, fraction=0.1)
    by_source = split.by_source(population)
    # All extracted are taken by default; extracted is scarce and is the headline.
    assert len(by_source["extracted"]) == 10
    # Generated is sampled near the requested fraction.
    assert 0.05 <= len(by_source["generated"]) / 590 <= 0.18


def test_include_all_extracted_can_be_disabled(population):
    split = build_validation_split(population, fraction=0.1, include_all_extracted=False)
    assert len(split.by_source(population)["extracted"]) < 10


def test_split_and_complement_partition_the_population(population):
    split = build_validation_split(population, fraction=0.1)
    complement = holdout_complement(population, split)
    assert not set(split.image_ids) & set(complement)
    assert len(split) + len(complement) == len(population)


def test_composition_is_recorded_for_reproducibility(population):
    split = build_validation_split(population, fraction=0.1)
    assert split.composition["total"] == len(split)
    assert set(split.composition["by_source"]) == {"extracted", "generated"}
    assert split.salt and split.fraction == 0.1


def test_invalid_fraction_rejected(population):
    for bad in [0.0, -0.1, 1.5]:
        with pytest.raises(ValueError, match="fraction"):
            build_validation_split(population, fraction=bad)
