"""Axis tick detection stage.

Preprocessing is preserved exactly from ``inference-3``: PIL open ->
RGB -> Resize(256, 256) -> ToTensor -> Normalize(mean=.5, std=.5). This differs
from the marker stage's preprocessing (grayscale, unnormalised); both are kept
as-is because the two checkpoints were trained under different pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from chart_extraction.data.images import ImageRef
from chart_extraction.progress import ProgressReporter

AXIS_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

# Head slot is a real tick when argmax over the 4 class logits equals this.
VALID_TICK_CLASS = 1


@dataclass
class AxisTicks:
    """Detected tick positions for one image, in 256x256 pixel space."""

    image_id: str
    x_points: np.ndarray  # (N, 2)
    y_points: np.ndarray  # (M, 2)

    @property
    def x_pixels(self) -> list[float]:
        """x-coordinates of x-axis ticks."""
        return [float(p[0]) for p in self.x_points]

    @property
    def y_pixels(self) -> list[float]:
        """y-coordinates (rows) of y-axis ticks."""
        return [float(p[1]) for p in self.y_points]


class AxisImageDataset(Dataset):
    def __init__(self, refs: Sequence[ImageRef], transform=AXIS_TRANSFORM) -> None:
        self.refs = list(refs)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int):
        ref = self.refs[index]
        image = Image.open(ref.path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return self.transform(image), ref.image_id


def _collate(batch):
    tensors, ids = zip(*batch)
    return torch.stack(tensors), list(ids)


@torch.no_grad()
def detect_axis_ticks(
    refs: Sequence[ImageRef],
    model_x,
    model_y,
    device: str | torch.device = "cpu",
    batch_size: int = 32,
    num_workers: int = 2,
    progress_interval_s: float = 15.0,
) -> dict[str, AxisTicks]:
    """Run both axis models over every image.

    Returns a dict keyed on image id (bug 3): downstream stages join on the id,
    never on position.
    """
    loader = DataLoader(
        AxisImageDataset(refs),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate,
    )

    results: dict[str, AxisTicks] = {}
    model_x.eval()
    model_y.eval()
    progress = ProgressReporter(
        len(refs), "axis", interval_s=progress_interval_s
    ).start()

    for images, image_ids in loader:
        images = images.to(device)
        out_x = model_x(images)
        out_y = model_y(images)

        for i, image_id in enumerate(image_ids):
            px = out_x["points"][i].cpu().numpy()
            lx = torch.argmax(out_x["labels"][i], dim=1).cpu().numpy()
            py = out_y["points"][i].cpu().numpy()
            ly = torch.argmax(out_y["labels"][i], dim=1).cpu().numpy()

            results[image_id] = AxisTicks(
                image_id=image_id,
                x_points=px[lx == VALID_TICK_CLASS],
                y_points=py[ly == VALID_TICK_CLASS],
            )
        progress.update(len(image_ids))

    progress.finish()
    return results
