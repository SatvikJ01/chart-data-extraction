"""Fine-tuning: target serialization, the held-out partition, and oversampling.

The load-bearing property is that the held-out 40% never reaches the optimiser.
That is asserted structurally here, not just documented.
"""

from __future__ import annotations

import json

import pytest

from chart_extraction.donut.parsing import string2preds
from chart_extraction.eval.ground_truth import Annotation
from chart_extraction.eval.harness import holdout_caveat
from chart_extraction.train.serialization import (
    format_value, roundtrip_ok, serialize_annotation,
)
from chart_extraction.train.splits import (
    build_extracted_split, load_split, save_split,
)


def _ann(image_id="img", source="extracted", chart_type="vertical_bar",
         x=("Mon", "Tue"), y=(1.0, 2.0)):
    return Annotation(image_id, source, chart_type, tuple(x), tuple(y))


# --- Serialization must round-trip through the production parser ------------

@pytest.mark.parametrize(
    "chart_type", ["line", "scatter", "dot", "vertical_bar", "horizontal_bar"]
)
def test_every_chart_type_round_trips(chart_type):
    """If the target format drifts from string2preds, training converges and
    every prediction parses to nothing."""
    annotation = _ann(chart_type=chart_type, x=("a", "b", "c"), y=(1.0, 2.0, 3.0))
    parsed = string2preds(serialize_annotation(annotation), "img")
    assert parsed.is_well_formed
    assert parsed.chart_type == chart_type
    assert len(parsed.x) == 3 and len(parsed.y) == 3
    assert roundtrip_ok(annotation)


def test_categorical_values_survive():
    annotation = _ann(x=("Monday", "Tuesday"), y=(10, 20))
    parsed = string2preds(serialize_annotation(annotation), "img")
    assert parsed.x == ["Monday", "Tuesday"]


def test_numeric_values_survive_round_trip():
    annotation = _ann(x=(1.0, 2.0), y=(-5.0, 3.25))
    parsed = string2preds(serialize_annotation(annotation), "img")
    assert parsed.y == [-5, 3.25]


def test_separator_inside_a_label_is_neutralised():
    """A ';' in a label would create a phantom value and change the series
    length, which the metric scores as zero."""
    annotation = _ann(x=("a;b", "c"), y=(1, 2))
    parsed = string2preds(serialize_annotation(annotation), "img")
    assert len(parsed.x) == 2


def test_target_contains_every_token_the_parser_needs():
    text = serialize_annotation(_ann())
    for token in ("<|BOS|>", "<vertical_bar>", "<x_start>", "<x_end>",
                  "<y_start>", "<y_end>"):
        assert token in text


def test_unknown_chart_type_raises():
    with pytest.raises(ValueError, match="no chart-type token"):
        serialize_annotation(_ann(chart_type="pie"))


def test_mismatched_series_lengths_raise():
    """The metric scores unequal lengths as zero, so this cannot be a target."""
    with pytest.raises(ValueError, match="differ in length"):
        serialize_annotation(_ann(x=(1, 2, 3), y=(1, 2)))


@pytest.mark.parametrize(
    "value,expected",
    [(1.0, "1"), (2.5, "2.5"), (7, "7"), (True, "1"),
     (float("nan"), "0"), (float("inf"), "0"), (1e5, "100000")],
)
def test_value_formatting(value, expected):
    assert format_value(value) == expected


# --- The held-out partition -------------------------------------------------

@pytest.fixture
def annotations():
    mix = {"line": 423, "vertical_bar": 457, "scatter": 165, "horizontal_bar": 73}
    out = {}
    index = 0
    for chart_type, count in mix.items():
        for _ in range(count):
            out[f"e{index:05d}"] = _ann(f"e{index:05d}", "extracted", chart_type)
            index += 1
    for j in range(500):
        out[f"g{j:05d}"] = _ann(f"g{j:05d}", "generated", "line")
    return out


def test_split_is_60_40_of_the_extracted_images(annotations):
    split = build_extracted_split(annotations, train_fraction=0.6, seed=1234)
    total = 1118
    fit = len(split.train_ids) + len(split.val_ids)
    assert fit + len(split.holdout_ids) == total
    assert len(split.holdout_ids) / total == pytest.approx(0.40, abs=0.01)


def test_holdout_is_disjoint_from_everything(annotations):
    split = build_extracted_split(annotations, seed=1234)
    split.assert_disjoint()
    assert not set(split.holdout_ids) & set(split.train_ids)
    assert not set(split.holdout_ids) & set(split.val_ids)


def test_validation_comes_from_the_train_side_not_the_holdout(annotations):
    """Selecting the best epoch on the held-out set would leak it as surely as
    training on it."""
    split = build_extracted_split(annotations, seed=1234)
    assert set(split.val_ids).isdisjoint(split.holdout_ids)
    assert len(split.val_ids) > 0


def test_split_excludes_generated_images_entirely(annotations):
    split = build_extracted_split(annotations, seed=1234)
    everything = set(split.train_ids) | set(split.val_ids) | set(split.holdout_ids)
    assert not any(i.startswith("g") for i in everything)


def test_split_is_deterministic_for_a_seed(annotations):
    a = build_extracted_split(annotations, seed=1234)
    b = build_extracted_split(annotations, seed=1234)
    assert a.holdout_ids == b.holdout_ids and a.train_ids == b.train_ids


def test_different_seed_gives_a_different_holdout(annotations):
    a = build_extracted_split(annotations, seed=1)
    b = build_extracted_split(annotations, seed=2)
    assert a.holdout_ids != b.holdout_ids


def test_split_is_stratified_by_chart_type(annotations):
    split = build_extracted_split(annotations, seed=1234)
    holdout = split.composition["holdout"]["by_chart_type"]
    train = split.composition["train"]["by_chart_type"]
    assert set(holdout) == set(train)
    for chart_type in holdout:
        assert holdout[chart_type] > 0


def test_split_persists_and_reloads(annotations, tmp_path):
    split = build_extracted_split(annotations, seed=1234)
    path = save_split(split, tmp_path / "split.json")
    reloaded = load_split(path)
    assert reloaded.holdout_ids == split.holdout_ids
    assert reloaded.seed == 1234
    assert json.loads(path.read_text())["seed"] == 1234


def test_overlapping_split_is_rejected():
    from chart_extraction.train.splits import ExtractedSplit

    bad = ExtractedSplit(
        seed=0, train_fraction=0.6, val_fraction_of_train=0.1,
        train_ids=("a", "b"), val_ids=(), holdout_ids=("b",),
    )
    with pytest.raises(ValueError, match="overlap"):
        bad.assert_disjoint()


def test_no_extracted_images_is_an_error():
    with pytest.raises(ValueError, match="no extracted images"):
        build_extracted_split({"g": _ann("g", "generated")})


# --- Provenance and caveats -------------------------------------------------

def test_holdout_provenance_records_the_limit_of_the_claim(annotations):
    provenance = build_extracted_split(annotations, seed=1234).provenance()
    assert provenance["held_out_from_finetune"] is True
    assert provenance["held_out_from_base_checkpoint"] is False


def test_holdout_caveat_does_not_overclaim():
    """The base checkpoint may still have trained on these images, so the
    absolute score is not clean -- only the difference between models is."""
    caveat = holdout_caveat({
        "held_out_from_finetune": True,
        "held_out_from_base_checkpoint": False,
        "seed": 1234, "n_holdout": 447,
    })
    assert "PARTIALLY LEAKAGE-FREE" in caveat
    assert "NOT held out from the base checkpoint" in caveat
    assert "DIFFERENCE" in caveat


def test_fully_held_out_caveat_is_available():
    caveat = holdout_caveat({
        "held_out_from_finetune": True,
        "held_out_from_base_checkpoint": True,
        "seed": 1, "n_holdout": 100,
    })
    assert "leakage-free" in caveat


def test_no_provenance_means_no_holdout_caveat():
    assert holdout_caveat(None) is None
    assert holdout_caveat({"held_out_from_finetune": False}) is None
