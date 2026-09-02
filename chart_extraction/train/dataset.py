"""Oversampled training set for the specialization phase.

The specialization phase's whole point is to shift the model toward the
``extracted`` distribution it will be judged on, so extracted images are
repeated and generated images are subsampled. Both counts are recorded, because
the ratio is the experiment's main knob.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from chart_extraction.eval.ground_truth import Annotation
from chart_extraction.train.serialization import roundtrip_ok, serialize_annotation

logger = logging.getLogger(__name__)

#: Labels at this value are ignored by the cross-entropy loss.
IGNORE_INDEX = -100


@dataclass
class DatasetRecipe:
    """What went into a training set, for the run record."""

    n_extracted_unique: int
    oversample: int
    n_extracted_rows: int
    n_generated_rows: int
    n_rows: int
    n_dropped_unserialisable: int
    n_dropped_too_long: int
    max_target_tokens: int

    def as_dict(self) -> dict:
        return {
            "n_extracted_unique": self.n_extracted_unique,
            "oversample": self.oversample,
            "n_extracted_rows": self.n_extracted_rows,
            "n_generated_rows": self.n_generated_rows,
            "n_rows": self.n_rows,
            "extracted_share": round(self.n_extracted_rows / self.n_rows, 4)
            if self.n_rows else 0.0,
            "n_dropped_unserialisable": self.n_dropped_unserialisable,
            "n_dropped_too_long": self.n_dropped_too_long,
            "max_target_tokens": self.max_target_tokens,
        }


def build_rows(
    extracted_ids: Sequence[str],
    generated_ids: Sequence[str],
    annotations: Mapping[str, Annotation],
    tokenizer,
    oversample: int = 6,
    max_target_tokens: int = 512,
) -> tuple[list[str], DatasetRecipe]:
    """Build the (repeated) id list for one epoch, dropping unusable targets.

    Two filters, both of which would otherwise corrupt training silently:

    * a target the parser cannot read back teaches a format that scores zero
    * a target longer than ``max_target_tokens`` gets truncated, teaching the
      model to emit sequences that never terminate

    Dropped counts are returned rather than logged and forgotten, because a
    large drop rate changes what the resulting number means.
    """
    usable_extracted: list[str] = []
    dropped_parse = 0
    dropped_long = 0

    def usable(image_id: str) -> bool:
        nonlocal dropped_parse, dropped_long
        annotation = annotations[image_id]
        if not roundtrip_ok(annotation):
            dropped_parse += 1
            return False
        text = serialize_annotation(annotation)
        if len(tokenizer(text, add_special_tokens=False).input_ids) > max_target_tokens:
            dropped_long += 1
            return False
        return True

    for image_id in extracted_ids:
        if usable(image_id):
            usable_extracted.append(image_id)

    usable_generated = [image_id for image_id in generated_ids if usable(image_id)]

    rows = list(usable_extracted) * oversample + list(usable_generated)

    recipe = DatasetRecipe(
        n_extracted_unique=len(usable_extracted),
        oversample=oversample,
        n_extracted_rows=len(usable_extracted) * oversample,
        n_generated_rows=len(usable_generated),
        n_rows=len(rows),
        n_dropped_unserialisable=dropped_parse,
        n_dropped_too_long=dropped_long,
        max_target_tokens=max_target_tokens,
    )
    if dropped_parse or dropped_long:
        logger.warning(
            "dropped %d unserialisable and %d over-long targets from %d candidates",
            dropped_parse, dropped_long,
            len(extracted_ids) + len(generated_ids),
        )
    return rows, recipe


class DonutFineTuneDataset(Dataset):
    """Image -> (pixel_values, labels) for teacher-forced training."""

    def __init__(
        self,
        image_ids: Sequence[str],
        annotations: Mapping[str, Annotation],
        image_dir: Path | str,
        processor,
        max_target_tokens: int = 512,
        augment: bool = False,
    ) -> None:
        self.image_ids = list(image_ids)
        self.annotations = annotations
        self.image_dir = Path(image_dir)
        self.processor = processor
        self.max_target_tokens = max_target_tokens
        # random_padding jitters the image inside its canvas. It is a genuine
        # train-time augmentation here, unlike at inference where the notebooks
        # left it on by mistake (Phase 0).
        self.augment = augment

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        annotation = self.annotations[image_id]

        arr = np.array(Image.open(self.image_dir / f"{image_id}.jpg"))
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        pixel_values = self.processor(
            arr, random_padding=self.augment, return_tensors="pt"
        ).pixel_values.squeeze(0)

        target = serialize_annotation(annotation)
        encoding = self.processor.tokenizer(
            target,
            add_special_tokens=False,
            max_length=self.max_target_tokens,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = encoding.input_ids.squeeze(0).clone()
        # Padding must not contribute to the loss.
        labels[labels == self.processor.tokenizer.pad_token_id] = IGNORE_INDEX

        return {"pixel_values": pixel_values, "labels": labels, "image_id": image_id}


def collate(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "image_ids": [b["image_id"] for b in batch],
    }
