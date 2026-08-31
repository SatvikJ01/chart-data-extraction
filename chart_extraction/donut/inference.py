"""Donut generation stage.

Donut is a Swin Transformer encoder feeding a BART-style autoregressive decoder,
trained OCR-free: it reads the image and emits the structured token sequence
directly, with no text-detection step.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from chart_extraction.config import GenerationConfig
from chart_extraction.data.images import ImageRef
from chart_extraction.donut.parsing import DonutPrediction, string2preds
from chart_extraction.runtime import (
    OomEvent, OomPolicy, empty_cache, is_out_of_memory, scaled_size,
)

logger = logging.getLogger(__name__)


class DonutImageDataset(Dataset):
    """Applies the DonutProcessor to each image.

    AUDIT NOTE: the notebooks passed ``random_padding=True`` at inference.
    Random padding is a *training* augmentation -- it jitters where the image
    sits inside the padded canvas -- so predictions were not reproducible
    between runs of identical code. Defaults to False here; see
    PipelineConfig.donut_random_padding.
    """

    def __init__(
        self, refs: Sequence[ImageRef], processor, random_padding: bool = False
    ) -> None:
        self.refs = list(refs)
        self.processor = processor
        self.random_padding = random_padding

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int):
        ref = self.refs[index]
        arr = np.array(Image.open(ref.path))
        # Some images in the set are single-channel; the processor needs 3.
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        pixel_values = self.processor(
            arr, random_padding=self.random_padding, return_tensors="pt"
        ).pixel_values
        return pixel_values.squeeze(0), ref.image_id


def _collate(batch):
    tensors, ids = zip(*batch)
    return torch.stack(tensors), list(ids)



@contextmanager
def processor_size(processor, size: dict | None):
    """Temporarily override the processor's target input size.

    Restored on exit even if generation raises, so one retried image cannot
    leave the whole run at reduced resolution.
    """
    if size is None:
        yield
        return
    image_processor = processor.image_processor
    original = image_processor.size
    image_processor.size = size
    try:
        yield
    finally:
        image_processor.size = original


def current_size(processor) -> dict | None:
    size = getattr(processor.image_processor, "size", None)
    if isinstance(size, dict) and "height" in size and "width" in size:
        return {"height": int(size["height"]), "width": int(size["width"])}
    return None


def model_dtype(model) -> "torch.dtype":
    """The dtype the model's weights actually are.

    Inputs are cast to this rather than to a configured precision, so a model
    that was .half()-ed and one that was not both receive matching inputs
    without the caller tracking which happened.
    """
    try:
        return next(model.parameters()).dtype
    except StopIteration:  # pragma: no cover - a model with no parameters
        return torch.float32


def _prepare_pixel_values(image_path, processor, random_padding: bool):
    arr = np.array(Image.open(image_path))
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return processor(
        arr, random_padding=random_padding, return_tensors="pt"
    ).pixel_values


@torch.no_grad()
def _generate(model, processor, pixel_values, generation, device) -> list[str]:
    """One generate call. Inputs are cast to the model's own dtype."""
    pixel_values = pixel_values.to(device=device, dtype=model_dtype(model))
    decoder_input_ids = torch.full(
        (pixel_values.shape[0], 1), model.config.decoder_start_token_id, device=device
    )
    outputs = model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        return_dict_in_generate=True,
        **generation.to_generate_kwargs(),
    )
    return processor.batch_decode(outputs.sequences)


def _retry_at_lower_resolution(
    ref: ImageRef,
    model,
    processor,
    generation: GenerationConfig,
    device,
    random_padding: bool,
    policy: OomPolicy,
) -> str | None:
    """Retry one image at progressively lower Donut input resolution.

    Returns the generated string, or None if no scale fit.

    Reducing Donut's input size is not always safe: a Swin encoder configured
    with absolute position embeddings is tied to the size it was trained at and
    will raise on a different one. That is caught per-scale and treated the same
    as a failed attempt, so a checkpoint that cannot be down-scaled degrades to
    a recorded failure rather than taking down the run.
    """
    base_size = current_size(processor)
    if base_size is None:
        policy.record(OomEvent(ref.image_id, "donut", None, None, None, recovered=False))
        return None

    for scale in policy.retry_scales:
        empty_cache()
        target = scaled_size(base_size, scale, policy.size_multiple)
        try:
            with processor_size(processor, target):
                pixel_values = _prepare_pixel_values(ref.path, processor, random_padding)
                generated = _generate(model, processor, pixel_values, generation, device)
            policy.record(
                OomEvent(
                    ref.image_id, "donut", scale,
                    target["height"], target["width"], recovered=True,
                )
            )
            return generated[0]
        except Exception as exc:
            if is_out_of_memory(exc):
                logger.info(
                    "%s still OOM at scale %.2f (%dx%d)",
                    ref.image_id, scale, target["height"], target["width"],
                )
            else:
                logger.info(
                    "%s failed at scale %.2f (%dx%d): %s",
                    ref.image_id, scale, target["height"], target["width"], exc,
                )
            continue

    empty_cache()
    policy.record(OomEvent(ref.image_id, "donut", None, None, None, recovered=False))
    return None


@torch.no_grad()
def run_donut(
    refs: Sequence[ImageRef],
    model,
    processor,
    generation: GenerationConfig,
    device: str | torch.device = "cpu",
    batch_size: int = 4,
    num_workers: int = 2,
    random_padding: bool = False,
    apply_cleaning: bool = True,
    oom_policy: OomPolicy | None = None,
) -> dict[str, DonutPrediction]:
    """Generate and parse for every image, keyed on image id.

    AUDIT NOTE: the notebooks wrapped generation in a bare ``except:`` that
    swallowed the exception and emitted empty strings, so a batch could fail for
    any reason -- OOM, a corrupt image, a shape error -- and leave no trace
    beyond placeholder rows. Failures are logged with the exception and the
    affected ids here, and recorded as a failure_mode on the prediction.

    OOM handling: when a batch runs out of memory the batch is retried one image
    at a time, and any image that still will not fit is retried at progressively
    lower Donut input resolution (see ``OomPolicy``). Every retry is logged and
    recorded, because a run where some images were silently processed at reduced
    resolution is not comparable with one where none were -- the count belongs
    in the result file, not just in the logs.
    """
    policy = oom_policy if oom_policy is not None else OomPolicy()

    loader = DataLoader(
        DonutImageDataset(refs, processor, random_padding=random_padding),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate,
    )
    refs_by_id = {ref.image_id: ref for ref in refs}
    predictions: dict[str, DonutPrediction] = {}
    model.eval()

    def _record(image_id: str, generated: str) -> None:
        predictions[image_id] = string2preds(
            generated, image_id=image_id, apply_cleaning=apply_cleaning
        )

    def _fail(image_id: str, mode: str) -> None:
        predictions[image_id] = DonutPrediction(
            image_id=image_id, chart_type="", failure_mode=mode
        )

    for pixel_values, image_ids in loader:
        try:
            for image_id, generated in zip(
                image_ids, _generate(model, processor, pixel_values, generation, device)
            ):
                _record(image_id, generated)
            continue
        except Exception as exc:
            if not is_out_of_memory(exc):
                logger.exception("Donut generation failed for batch %s", image_ids)
                for image_id in image_ids:
                    _fail(image_id, "generation_error")
                continue

            logger.warning(
                "OOM on a batch of %d; retrying image-by-image", len(image_ids)
            )
            del pixel_values
            empty_cache()

        # Batch-level OOM: fall back to one image at a time, then to lower
        # resolution for whichever images still will not fit.
        for image_id in image_ids:
            ref = refs_by_id[image_id]
            try:
                single = _prepare_pixel_values(ref.path, processor, random_padding)
                generated = _generate(model, processor, single, generation, device)
                _record(image_id, generated[0])
                continue
            except Exception as exc:
                if not is_out_of_memory(exc):
                    logger.exception("Donut generation failed for %s", image_id)
                    _fail(image_id, "generation_error")
                    continue
                empty_cache()

            if not policy.enabled:
                policy.record(
                    OomEvent(image_id, "donut", None, None, None, recovered=False)
                )
                _fail(image_id, "oom")
                continue

            generated = _retry_at_lower_resolution(
                ref, model, processor, generation, device, random_padding, policy
            )
            if generated is None:
                _fail(image_id, "oom")
            else:
                _record(image_id, generated)

    missing = set(refs_by_id) - set(predictions)
    for image_id in missing:  # pragma: no cover - defensive
        _fail(image_id, "generation_error")

    return predictions


def prepare_donut_model(model, precision: str = "fp32", device: str = "cpu"):
    """Move Donut to the device and, for fp16, halve its weights.

    ``.half()`` rather than autocast for Donut specifically: autocast keeps
    fp32 weights resident and only casts activations, so it does nothing for the
    ~800MB of parameters that dominate this model's footprint. Halving the
    weights is what makes it fit on a small local card. Inputs are matched to
    the resulting dtype automatically by ``_generate``.
    """
    if precision not in ("fp32", "fp16"):
        raise ValueError(f"unknown precision {precision!r}; use 'fp32' or 'fp16'")

    model = model.to(device)
    if precision == "fp16":
        if not str(device).startswith("cuda"):
            logger.warning(
                "precision=fp16 requested on device %s; half precision on CPU is "
                "slow and poorly supported. Staying in fp32.", device,
            )
        else:
            model = model.half()
    model.eval()
    return model
