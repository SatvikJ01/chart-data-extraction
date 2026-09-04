"""Marker detection stage.

Preprocessing preserved from ``inference-3``: cv2 grayscale read ->
3-channel expand -> float32 /255 -> albumentations Resize(256, 256) ->
ToTensorV2. Unlike the axis stage there is no mean/std normalisation, matching
how the checkpoint was trained.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from chart_extraction.data.images import ImageRef
from chart_extraction.progress import ProgressReporter


@dataclass
class MarkerDetections:
    """Raw detector output for one image, in 256x256 pixel space.

    Boxes, labels and scores are held together on one object so they cannot be
    sourced from different images -- the structural fix for bug 1.
    """

    image_id: str
    boxes: np.ndarray  # (N, 4)
    labels: np.ndarray  # (N,)
    scores: np.ndarray  # (N,)

    def __post_init__(self) -> None:
        n = len(self.boxes)
        if not (len(self.labels) == len(self.scores) == n):
            raise ValueError(
                f"detection arrays disagree for {self.image_id!r}: "
                f"boxes={n} labels={len(self.labels)} scores={len(self.scores)}"
            )


def get_marker_transform(size: int = 256):
    """Build the albumentations test transform (imported lazily)."""
    import albumentations as A
    from albumentations.pytorch.transforms import ToTensorV2

    return A.Compose([A.Resize(size, size), ToTensorV2(p=1.0)])


class MarkerImageDataset(Dataset):
    def __init__(self, refs: Sequence[ImageRef], transforms=None) -> None:
        self.refs = list(refs)
        self.transforms = transforms if transforms is not None else get_marker_transform()

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int):
        import cv2

        ref = self.refs[index]
        image = cv2.imread(str(ref.path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"could not read image: {ref.path}")
        if image.ndim == 2:
            image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2)
        image = image.astype(np.float32) / 255.0
        image = self.transforms(image=image)["image"]
        return image, ref.image_id


def _collate(batch):
    return tuple(zip(*batch))


@torch.no_grad()
def detect_markers(
    refs: Sequence[ImageRef],
    model,
    device: str | torch.device = "cpu",
    batch_size: int = 4,
    num_workers: int = 2,
    progress_interval_s: float = 15.0,
) -> dict[str, MarkerDetections]:
    """Run the marker detector, returning results keyed on image id."""
    loader = DataLoader(
        MarkerImageDataset(refs),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate,
    )

    results: dict[str, MarkerDetections] = {}
    model.eval()
    progress = ProgressReporter(
        len(refs), "markers", interval_s=progress_interval_s
    ).start()

    for images, image_ids in loader:
        images = [image.to(device) for image in images]
        outputs = model(images)
        for i, image_id in enumerate(image_ids):
            results[image_id] = MarkerDetections(
                image_id=image_id,
                boxes=outputs[i]["boxes"].detach().cpu().numpy(),
                labels=outputs[i]["labels"].detach().cpu().numpy(),
                scores=outputs[i]["scores"].detach().cpu().numpy(),
            )
        progress.update(len(image_ids))

    progress.finish()
    return results
