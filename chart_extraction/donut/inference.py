"""Donut generation stage.

Donut is a Swin Transformer encoder feeding a BART-style autoregressive decoder,
trained OCR-free: it reads the image and emits the structured token sequence
directly, with no text-detection step.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from chart_extraction.config import GenerationConfig
from chart_extraction.data.images import ImageRef
from chart_extraction.donut.parsing import DonutPrediction, string2preds

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
) -> dict[str, DonutPrediction]:
    """Generate and parse for every image, keyed on image id.

    AUDIT NOTE: the notebooks wrapped generation in a bare ``except:`` that
    swallowed the exception and emitted empty strings, so a batch could fail for
    any reason -- OOM, a corrupt image, a shape error -- and leave no trace
    beyond placeholder rows. Failures are logged with the exception and the
    affected ids here, and recorded as a failure_mode on the prediction.
    """
    loader = DataLoader(
        DonutImageDataset(refs, processor, random_padding=random_padding),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate,
    )

    decoder_start_token_id = model.config.decoder_start_token_id
    generate_kwargs = generation.to_generate_kwargs()

    predictions: dict[str, DonutPrediction] = {}
    model.eval()

    for pixel_values, image_ids in loader:
        pixel_values = pixel_values.to(device)
        decoder_input_ids = torch.full(
            (pixel_values.shape[0], 1), decoder_start_token_id, device=device
        )

        try:
            outputs = model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
                return_dict_in_generate=True,
                **generate_kwargs,
            )
            generations = processor.batch_decode(outputs.sequences)
        except Exception:
            logger.exception("Donut generation failed for batch %s", image_ids)
            for image_id in image_ids:
                predictions[image_id] = DonutPrediction(
                    image_id=image_id,
                    chart_type="",
                    failure_mode="generation_error",
                )
            continue

        for image_id, generated in zip(image_ids, generations):
            predictions[image_id] = string2preds(
                generated, image_id=image_id, apply_cleaning=apply_cleaning
            )

    return predictions
