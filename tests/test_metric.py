"""Hand-constructed cases for the competition metric.

The metric is asymmetric by design; these pin both branches and every gate.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from chart_extraction.eval.levenshtein import levenshtein_distance
from chart_extraction.eval.metric import (
    benetech_score, normalized_levenshtein_score, normalized_rmse,
    score_instance, score_series, sigmoid,
)


# --- Levenshtein -----------------------------------------------------------

@pytest.mark.parametrize(
    "a,b,expected",
    [
        ("kitten", "sitting", 3),   # the canonical case
        ("", "", 0),
        ("", "abc", 3),
        ("abc", "", 3),
        ("abc", "abc", 0),
        ("flaw", "lawn", 2),
        ("a", "b", 1),
    ],
)
def test_levenshtein_known_distances(a, b, expected):
    assert levenshtein_distance(a, b) == expected
    assert levenshtein_distance(b, a) == expected, "distance must be symmetric"


# --- The squashing function ------------------------------------------------

def test_sigmoid_endpoints_and_monotonicity():
    assert sigmoid(0.0) == pytest.approx(1.0)
    assert sigmoid(1e9) == pytest.approx(0.0, abs=1e-9)
    values = [sigmoid(x) for x in [0.0, 0.5, 1.0, 2.0, 5.0]]
    assert values == sorted(values, reverse=True)
    assert all(0.0 <= v <= 1.0 for v in values)


# --- Numeric branch --------------------------------------------------------

def test_numeric_exact_match_scores_one():
    assert score_series([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_numeric_error_reduces_score_monotonically():
    truth = [1.0, 2.0, 3.0, 4.0]
    scores = [
        score_series(truth, [1.0, 2.0, 3.0, 4.0]),
        score_series(truth, [1.1, 2.1, 3.1, 4.1]),
        score_series(truth, [2.0, 3.0, 4.0, 5.0]),
        score_series(truth, [100.0, 200.0, 300.0, 400.0]),
    ]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == pytest.approx(1.0)
    assert scores[-1] == pytest.approx(0.0, abs=1e-6)


def test_numeric_normalisation_is_against_the_mean_baseline():
    """RMSE is normalised by the RMSE of predicting the ground-truth mean, so
    predicting the mean itself lands at sigmoid(1)."""
    truth = [1.0, 2.0, 3.0, 4.0]
    mean_prediction = [2.5] * 4
    assert normalized_rmse(truth, mean_prediction) == pytest.approx(sigmoid(1.0))


def test_numeric_zero_variance_ground_truth():
    """No variance means no scale for partial credit: exact or nothing."""
    assert score_series([5.0, 5.0, 5.0], [5.0, 5.0, 5.0]) == pytest.approx(1.0)
    assert score_series([5.0, 5.0, 5.0], [5.0, 5.0, 6.0]) == 0.0


def test_numeric_ground_truth_with_uncoercible_prediction_scores_zero():
    assert score_series([1.0, 2.0], ["abc", "def"]) == 0.0


def test_numeric_ground_truth_accepts_numeric_strings():
    """Donut emits strings; a numerically-valid string must not be penalised."""
    assert score_series([1.0, 2.0], ["1.0", "2.0"]) == pytest.approx(1.0)


def test_non_finite_prediction_scores_zero():
    assert score_series([1.0, 2.0], [float("nan"), 2.0]) == 0.0
    assert score_series([1.0, 2.0], [float("inf"), 2.0]) == 0.0


# --- Categorical branch ----------------------------------------------------

def test_categorical_exact_match_scores_one():
    assert score_series(["Mon", "Tue"], ["Mon", "Tue"]) == pytest.approx(1.0)


def test_categorical_partial_credit_scales_with_edit_distance():
    truth = ["Monday", "Tuesday"]
    near = score_series(truth, ["Munday", "Tuesday"])   # 1 edit of 13 chars
    far = score_series(truth, ["xxxxxx", "yyyyyyy"])    # 13 edits of 13 chars
    assert 0.0 < far < near < 1.0


def test_categorical_normalisation_is_by_total_truth_length():
    truth = ["abcd"]
    # 2 edits over 4 characters -> ratio 0.5
    assert normalized_levenshtein_score(truth, ["abxy"]) == pytest.approx(sigmoid(0.5))


def test_series_type_is_decided_by_ground_truth_first_element():
    """The official scorer tests isinstance(y_true[0], str) -- the first
    element alone selects the branch for the whole series."""
    # Numeric truth -> RMSE branch, so a numeric-string prediction matches.
    assert score_series([1.0, 2.0], ["1", "2"]) == pytest.approx(1.0)
    # String truth -> Levenshtein branch, so "1" vs "1.0" costs edits.
    assert score_series(["1", "2"], ["1.0", "2.0"]) < 1.0


# --- Gates -----------------------------------------------------------------

def test_length_mismatch_scores_zero():
    assert score_series([1.0, 2.0, 3.0], [1.0, 2.0]) == 0.0
    assert score_series([1.0], [1.0, 2.0]) == 0.0
    assert score_series(["a", "b"], ["a"]) == 0.0


def test_placeholder_series_scores_zero_on_length_alone():
    """The notebooks emitted a 2-element '0;0' placeholder. Against any series
    of a different length that is zero regardless of values -- which is why the
    Phase 0 line-chart stub was worth nothing, not 'a little'."""
    truth = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert score_series(truth, [0.0, 0.0]) == 0.0


def test_empty_series():
    """Two empty series match. The official scorer would raise IndexError on
    y_true[0]; this is a documented, tested deviation."""
    assert score_series([], []) == pytest.approx(1.0)
    assert score_series([], [1.0]) == 0.0
    assert score_series([1.0], []) == 0.0


def test_wrong_chart_type_scores_zero_regardless_of_series():
    """The chart-type gate precedes everything, so a perfect series still
    scores zero if the type is wrong."""
    assert score_instance([1.0, 2.0], "line", [1.0, 2.0], "scatter") == 0.0
    assert score_instance([1.0, 2.0], "line", [1.0, 2.0], "line") == pytest.approx(1.0)


def test_wrong_chart_type_beats_nothing_even_with_perfect_values():
    perfect_but_wrong = score_instance(["a", "b"], "dot", ["a", "b"], "line")
    assert perfect_but_wrong == 0.0


# --- Aggregation -----------------------------------------------------------

def _frame(rows):
    return pd.DataFrame(rows).set_index("id")


def test_benetech_score_averages_over_instances():
    truth = _frame([
        {"id": "a_x", "data_series": ["p", "q"], "chart_type": "line"},
        {"id": "a_y", "data_series": [1.0, 2.0], "chart_type": "line"},
    ])
    predictions = _frame([
        {"id": "a_x", "data_series": ["p", "q"], "chart_type": "line"},
        {"id": "a_y", "data_series": [0.0, 0.0], "chart_type": "line"},
    ])
    score = benetech_score(truth, predictions)
    # One perfect instance and one poor one.
    assert 0.0 < score < 1.0
    assert score == pytest.approx(
        (1.0 + score_series([1.0, 2.0], [0.0, 0.0])) / 2
    )


def test_benetech_score_perfect_and_worst():
    truth = _frame([
        {"id": "a_x", "data_series": ["p"], "chart_type": "dot"},
        {"id": "a_y", "data_series": [1.0], "chart_type": "dot"},
    ])
    assert benetech_score(truth, truth) == pytest.approx(1.0)

    wrong = _frame([
        {"id": "a_x", "data_series": ["p"], "chart_type": "line"},
        {"id": "a_y", "data_series": [1.0], "chart_type": "line"},
    ])
    assert benetech_score(truth, wrong) == 0.0


def test_benetech_score_requires_matching_index():
    truth = _frame([{"id": "a_x", "data_series": [1.0], "chart_type": "dot"}])
    predictions = _frame([{"id": "b_x", "data_series": [1.0], "chart_type": "dot"}])
    with pytest.raises(ValueError, match="exactly one prediction"):
        benetech_score(truth, predictions)
