"""Tests for the local-GPU runtime path: allocator, precision, OOM recovery."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from chart_extraction.config import GREEDY
from chart_extraction.data.images import ImageRef
from chart_extraction.donut.inference import (
    current_size, model_dtype, prepare_donut_model, processor_size, run_donut,
)
from chart_extraction.runtime import (
    ALLOC_CONF_VAR, EXPANDABLE_SEGMENTS, OomEvent, OomPolicy, configure_allocator,
    is_out_of_memory, scaled_size,
)


# --- Allocator --------------------------------------------------------------

def test_allocator_set_when_unset(monkeypatch):
    monkeypatch.delenv(ALLOC_CONF_VAR, raising=False)
    assert configure_allocator() == EXPANDABLE_SEGMENTS


def test_allocator_appends_and_preserves_existing(monkeypatch):
    monkeypatch.setenv(ALLOC_CONF_VAR, "max_split_size_mb:128")
    value = configure_allocator()
    assert "max_split_size_mb:128" in value
    assert EXPANDABLE_SEGMENTS in value


def test_allocator_respects_an_explicit_opposing_choice(monkeypatch):
    """A caller who deliberately disabled it must not be overridden."""
    monkeypatch.setenv(ALLOC_CONF_VAR, "expandable_segments:False")
    assert configure_allocator() == "expandable_segments:False"


# --- OOM detection ----------------------------------------------------------

@pytest.mark.parametrize(
    "exc,expected",
    [
        (RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"), True),
        (RuntimeError("out of memory"), True),
        (RuntimeError("shape mismatch"), False),
        (ValueError("something else"), False),
    ],
)
def test_is_out_of_memory(exc, expected):
    assert is_out_of_memory(exc) is expected


def test_torch_oom_class_is_recognised():
    assert is_out_of_memory(torch.cuda.OutOfMemoryError("CUDA out of memory"))


# --- Size scaling -----------------------------------------------------------

def test_scaled_size_rounds_to_multiple():
    scaled = scaled_size({"height": 1280, "width": 960}, 0.75, multiple=32)
    assert scaled == {"height": 960, "width": 704}
    assert scaled["height"] % 32 == 0 and scaled["width"] % 32 == 0


def test_scaled_size_never_degenerates():
    assert scaled_size({"height": 40, "width": 40}, 0.01, multiple=32) == {
        "height": 32, "width": 32
    }


# --- Fakes ------------------------------------------------------------------

class FakeImageProcessor:
    def __init__(self, height=1280, width=960):
        self.size = {"height": height, "width": width}


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1


class FakeProcessorOutput:
    def __init__(self, pixel_values):
        self.pixel_values = pixel_values


class FakeProcessor:
    def __init__(self, height=1280, width=960):
        self.image_processor = FakeImageProcessor(height, width)
        self.tokenizer = FakeTokenizer()

    def __call__(self, arr, random_padding=False, return_tensors=None):
        size = self.image_processor.size
        # Encode the resolution into the tensor so the model can react to it.
        return FakeProcessorOutput(
            torch.zeros(1, 3, size["height"] // 32, size["width"] // 32)
        )

    def batch_decode(self, sequences):
        return [
            "<vertical_bar><x_start>a;b<x_end><y_start>1;2<y_end>"
            for _ in range(len(sequences))
        ]


class FakeConfig:
    decoder_start_token_id = 2


class FakeDonut(torch.nn.Module):
    """Raises OOM until the input is small enough.

    ``oom_above`` is a threshold on the tensor's height, which tracks the
    processor's configured resolution.
    """

    def __init__(self, oom_above: int = 0, oom_batches_above: int = 10**9):
        super().__init__()
        self.config = FakeConfig()
        self.linear = torch.nn.Linear(2, 2)
        self.oom_above = oom_above
        self.oom_batches_above = oom_batches_above
        self.calls: list[tuple[int, int]] = []

    def generate(self, pixel_values, **kwargs):
        batch, _, height, _ = pixel_values.shape
        self.calls.append((batch, height))
        if batch > self.oom_batches_above:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        if height > self.oom_above:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return type("Out", (), {"sequences": torch.zeros(batch, 4, dtype=torch.long)})()


@pytest.fixture
def images(tmp_path):
    refs = []
    for name in ["a", "b", "c"]:
        path = tmp_path / f"{name}.jpg"
        Image.fromarray(np.full((64, 64, 3), 200, dtype=np.uint8)).save(path)
        refs.append(ImageRef(image_id=name, path=path))
    return refs


# --- processor_size ---------------------------------------------------------

def test_processor_size_restores_on_exit():
    processor = FakeProcessor()
    original = dict(processor.image_processor.size)
    with processor_size(processor, {"height": 320, "width": 320}):
        assert processor.image_processor.size == {"height": 320, "width": 320}
    assert processor.image_processor.size == original


def test_processor_size_restores_even_when_body_raises():
    """One retried image must not leave the whole run at reduced resolution."""
    processor = FakeProcessor()
    original = dict(processor.image_processor.size)
    with pytest.raises(RuntimeError):
        with processor_size(processor, {"height": 320, "width": 320}):
            raise RuntimeError("boom")
    assert processor.image_processor.size == original


def test_current_size_reads_the_processor():
    assert current_size(FakeProcessor(640, 480)) == {"height": 640, "width": 480}


# --- Happy path -------------------------------------------------------------

def test_no_oom_means_no_events(images):
    policy = OomPolicy()
    model = FakeDonut(oom_above=10**9)
    predictions = run_donut(
        images, model, FakeProcessor(), GREEDY, device="cpu",
        batch_size=1, num_workers=0, oom_policy=policy,
    )
    assert set(predictions) == {"a", "b", "c"}
    assert all(p.is_well_formed for p in predictions.values())
    assert policy.events == []
    assert policy.summary()["n_recovered"] == 0


# --- OOM recovery -----------------------------------------------------------

def test_batch_oom_falls_back_to_single_images(images):
    """A batch that will not fit is retried one image at a time before any
    resolution is sacrificed."""
    policy = OomPolicy()
    model = FakeDonut(oom_above=10**9, oom_batches_above=1)
    predictions = run_donut(
        images, model, FakeProcessor(), GREEDY, device="cpu",
        batch_size=3, num_workers=0, oom_policy=policy,
    )
    assert all(p.is_well_formed for p in predictions.values())
    # Recovered by batch splitting alone -- resolution was never reduced.
    assert policy.events == []
    assert model.calls[0][0] == 3, "first attempt used the full batch"
    assert all(call[0] == 1 for call in model.calls[1:])


def test_oom_retries_at_lower_resolution_and_records_it(images):
    # Full res is 1280/32 = 40 rows; 0.75 scale -> 960/32 = 30 rows.
    policy = OomPolicy(retry_scales=(0.75, 0.5))
    model = FakeDonut(oom_above=30)
    predictions = run_donut(
        images, model, FakeProcessor(), GREEDY, device="cpu",
        batch_size=1, num_workers=0, oom_policy=policy,
    )

    assert all(p.is_well_formed for p in predictions.values())
    assert len(policy.events) == 3, "one recorded event per degraded image"
    assert all(e.recovered for e in policy.events)
    assert all(e.scale == 0.75 for e in policy.events)
    assert all(e.height == 960 and e.width == 704 for e in policy.events)

    summary = policy.summary()
    assert summary["n_recovered"] == 3
    assert summary["degraded_image_ids"] == ["a", "b", "c"]


def test_oom_walks_down_the_scale_ladder(images):
    """When 0.75 still will not fit, 0.5 is tried."""
    policy = OomPolicy(retry_scales=(0.75, 0.5))
    model = FakeDonut(oom_above=20)  # 0.75 -> 30 rows fails, 0.5 -> 20 rows fits
    predictions = run_donut(
        images[:1], model, FakeProcessor(), GREEDY, device="cpu",
        batch_size=1, num_workers=0, oom_policy=policy,
    )
    assert predictions["a"].is_well_formed
    assert [e.scale for e in policy.events] == [0.5]
    assert policy.events[0].height == 640


def test_unrecoverable_oom_is_recorded_not_swallowed(images):
    """No scale fits: the image gets an explicit 'oom' failure mode, which the
    taxonomy can count, rather than a silent placeholder."""
    policy = OomPolicy(retry_scales=(0.75, 0.5))
    model = FakeDonut(oom_above=0)
    predictions = run_donut(
        images[:1], model, FakeProcessor(), GREEDY, device="cpu",
        batch_size=1, num_workers=0, oom_policy=policy,
    )
    assert predictions["a"].failure_mode == "oom"
    assert not predictions["a"].is_well_formed
    summary = policy.summary()
    assert summary["n_unrecovered"] == 1
    assert summary["failed_image_ids"] == ["a"]


def test_retry_can_be_disabled(images):
    policy = OomPolicy(enabled=False)
    model = FakeDonut(oom_above=0)
    predictions = run_donut(
        images[:1], model, FakeProcessor(), GREEDY, device="cpu",
        batch_size=1, num_workers=0, oom_policy=policy,
    )
    assert predictions["a"].failure_mode == "oom"
    assert len(model.calls) == 2, "batch attempt then single attempt, no rescaling"


def test_non_oom_errors_are_not_retried_at_lower_resolution(images):
    """A shape bug must not be misreported as a memory problem."""

    class Broken(FakeDonut):
        def generate(self, pixel_values, **kwargs):
            raise RuntimeError("shape mismatch in decoder")

    policy = OomPolicy()
    predictions = run_donut(
        images[:1], Broken(), FakeProcessor(), GREEDY, device="cpu",
        batch_size=1, num_workers=0, oom_policy=policy,
    )
    assert predictions["a"].failure_mode == "generation_error"
    assert policy.events == []


def test_every_image_gets_a_prediction_even_when_all_fail(images):
    policy = OomPolicy(retry_scales=(0.5,))
    predictions = run_donut(
        images, FakeDonut(oom_above=0), FakeProcessor(), GREEDY, device="cpu",
        batch_size=2, num_workers=0, oom_policy=policy,
    )
    assert set(predictions) == {"a", "b", "c"}


# --- Precision --------------------------------------------------------------

def test_prepare_model_keeps_fp32_by_default():
    model = FakeDonut()
    prepared = prepare_donut_model(model, precision="fp32", device="cpu")
    assert model_dtype(prepared) == torch.float32


def test_fp16_on_cpu_stays_fp32_with_a_warning(caplog):
    """Half precision on CPU is slow and poorly supported; silently producing
    it would be worse than declining."""
    model = FakeDonut()
    prepared = prepare_donut_model(model, precision="fp16", device="cpu")
    assert model_dtype(prepared) == torch.float32
    assert any("fp16" in r.message for r in caplog.records)


def test_unknown_precision_rejected():
    with pytest.raises(ValueError, match="unknown precision"):
        prepare_donut_model(FakeDonut(), precision="bf8", device="cpu")


def test_inputs_are_cast_to_the_models_dtype(images):
    """A halved model must receive halved inputs; the caller should not have to
    track which happened."""

    class DtypeRecorder(FakeDonut):
        seen = None

        def generate(self, pixel_values, **kwargs):
            DtypeRecorder.seen = pixel_values.dtype
            return super().generate(pixel_values, **kwargs)

    model = DtypeRecorder(oom_above=10**9).half()
    run_donut(
        images[:1], model, FakeProcessor(), GREEDY, device="cpu",
        batch_size=1, num_workers=0,
    )
    assert DtypeRecorder.seen == torch.float16
