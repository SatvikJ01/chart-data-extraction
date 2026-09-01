"""Reportable subsampling (--sample/--seed) and the progress reporter."""

from __future__ import annotations

import collections
import logging

import pytest

from chart_extraction.eval.ground_truth import Annotation
from chart_extraction.eval.splits import stratified_sample
from chart_extraction.progress import ProgressReporter, _format_duration

CHART_TYPES = ["line", "scatter", "dot", "vertical_bar", "horizontal_bar"]


@pytest.fixture
def population():
    annotations = {}
    for i in range(1000):
        annotations[f"i{i:04d}"] = Annotation(
            f"i{i:04d}",
            "extracted" if i % 5 == 0 else "generated",
            CHART_TYPES[i % 5],
            (1,), (2,),
        )
    return annotations


def _mix(ids, annotations, attr="chart_type"):
    counts = collections.Counter(getattr(annotations[i], attr) for i in ids)
    return {k: round(v / len(ids), 2) for k, v in sorted(counts.items())}


# --- Determinism ------------------------------------------------------------

def test_same_seed_gives_the_same_images(population):
    first, _ = stratified_sample(list(population), population, 200, seed=42)
    second, _ = stratified_sample(list(population), population, 200, seed=42)
    assert first == second


def test_different_seed_gives_different_images(population):
    first, _ = stratified_sample(list(population), population, 200, seed=1)
    second, _ = stratified_sample(list(population), population, 200, seed=2)
    assert first != second
    assert len(first) == len(second) == 200


def test_input_order_does_not_affect_the_result(population):
    """Determinism must not depend on dict or filesystem ordering."""
    forward = list(population)
    first, _ = stratified_sample(forward, population, 150, seed=7)
    second, _ = stratified_sample(list(reversed(forward)), population, 150, seed=7)
    assert first == second


# --- Stratification ---------------------------------------------------------

def test_sample_preserves_the_chart_type_mix(population):
    """Chart types score very differently (scatter ~0.11 vs vertical_bar ~0.69),
    so an unstratified draw would move the aggregate by mix alone."""
    selected, _ = stratified_sample(list(population), population, 200, seed=3)
    assert _mix(selected, population) == _mix(list(population), population)


def test_sample_preserves_the_source_mix(population):
    selected, _ = stratified_sample(list(population), population, 200, seed=3)
    assert _mix(selected, population, "source") == _mix(
        list(population), population, "source"
    )


def test_exact_sample_size_even_with_awkward_remainders(population):
    for n in (7, 13, 99, 101, 337):
        selected, record = stratified_sample(list(population), population, n, seed=5)
        assert len(selected) == n
        assert record["n_selected"] == n


def test_small_strata_are_not_starved():
    """Largest-remainder allocation must not systematically drop rare types."""
    annotations = {}
    for i in range(100):
        annotations[f"a{i:03d}"] = Annotation(f"a{i:03d}", "generated", "line", (1,), (2,))
    for i in range(3):
        annotations[f"z{i}"] = Annotation(f"z{i}", "extracted", "dot", (1,), (2,))

    selected, _ = stratified_sample(list(annotations), annotations, 40, seed=0)
    assert any(annotations[i].chart_type == "dot" for i in selected)


# --- Record -----------------------------------------------------------------

def test_record_captures_size_and_seed(population):
    _, record = stratified_sample(list(population), population, 200, seed=42)
    assert record["sampled"] is True
    assert record["n_requested"] == 200
    assert record["n_selected"] == 200
    assert record["n_population"] == 1000
    assert record["seed"] == 42
    assert record["fraction"] == pytest.approx(0.2)
    assert "stratified" in record["method"]


def test_record_has_per_stratum_counts(population):
    _, record = stratified_sample(list(population), population, 200, seed=1)
    assert record["strata"]
    assert sum(v["selected"] for v in record["strata"].values()) == 200


def test_sample_larger_than_population_is_not_a_sample(population):
    """Asking for more than exists evaluates everything and must not be
    labelled a subsample."""
    selected, record = stratified_sample(list(population), population, 5000, seed=1)
    assert len(selected) == 1000
    assert record["sampled"] is False
    assert "reason" in record


def test_non_positive_sample_rejected(population):
    for bad in (0, -5):
        with pytest.raises(ValueError, match="must be positive"):
            stratified_sample(list(population), population, bad)


# --- Progress reporter ------------------------------------------------------

def test_progress_emits_start_and_finish(caplog):
    with caplog.at_level(logging.INFO):
        with ProgressReporter(10, "donut", interval_s=999) as progress:
            progress.update(10)
    messages = [r.getMessage() for r in caplog.records]
    assert any("starting, 10 images" in m for m in messages)
    assert any("done 10/10" in m for m in messages)


def test_progress_emits_periodically(caplog):
    with caplog.at_level(logging.INFO):
        progress = ProgressReporter(100, "donut", interval_s=0.0).start()
        for _ in range(5):
            progress.update()
    messages = [r.getMessage() for r in caplog.records]
    updates = [m for m in messages if "eta" in m]
    assert len(updates) >= 3


def test_progress_reports_counts_out_of_total(caplog):
    with caplog.at_level(logging.INFO):
        progress = ProgressReporter(1118, "donut", interval_s=0.0).start()
        progress.update(320)
    messages = [r.getMessage() for r in caplog.records]
    assert any("320/1118" in m for m in messages)


def test_progress_reports_where_it_got_to_on_failure(caplog):
    """Knowing a run died at image 900 of 1118 is the point."""
    with caplog.at_level(logging.INFO):
        with pytest.raises(RuntimeError):
            with ProgressReporter(1118, "donut", interval_s=999) as progress:
                progress.update(900)
                raise RuntimeError("boom")
    messages = [r.getMessage() for r in caplog.records]
    assert any("done 900/1118" in m for m in messages)


def test_progress_survives_a_zero_total(caplog):
    with caplog.at_level(logging.INFO):
        with ProgressReporter(0, "donut", interval_s=0.0) as progress:
            progress.update(0)


@pytest.mark.parametrize(
    "seconds,expected", [(5, "5s"), (90, "1m30s"), (3700, "1h01m"), (-1, "?")]
)
def test_duration_formatting(seconds, expected):
    assert _format_duration(seconds) == expected
