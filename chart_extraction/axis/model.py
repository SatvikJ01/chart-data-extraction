"""ResNet-18 backbone with two regression/classification heads for axis ticks.

Architecture preserved from ``inference-3.ipynb`` so existing checkpoints load
without conversion: a ResNet-18 trunk (final FC stripped) feeding a shared
128-d bottleneck, which forks into a point-regression head
(``max_num_points x 2``) and a per-point class head (``max_num_points x 4``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class AxisTickCNN(nn.Module):
    """Predicts axis tick positions and per-slot validity labels.

    AUDIT NOTE: the notebook defined this as a ``pl.LightningModule`` purely to
    inherit ``training_step``/``configure_optimizers``, none of which run during
    inference. That pulled pytorch_lightning into the inference dependency set
    for no benefit. It is a plain ``nn.Module`` here; the training hooks belong
    with training code, which does not exist yet.

    AUDIT NOTE: the original ``configure_optimizers`` read a module-level global
    ``model`` rather than ``self.parameters()`` -- it would have optimised
    whatever happened to be bound to that name. Dropped along with the rest of
    the training scaffolding rather than carried forward broken.
    """

    def __init__(self, max_num_points: int = 25) -> None:
        super().__init__()
        self.max_num_points = max_num_points
        resnet = models.resnet18(weights=None)
        self.resnet_layers = nn.Sequential(*list(resnet.children())[:-1])
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, max_num_points * 2)
        self.fc3 = nn.Linear(128, max_num_points * 4)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.resnet_layers(x)
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        points = self.fc2(x).view(x.size(0), -1, 2)
        labels = self.fc3(x).view(x.size(0), -1, 4)
        return {"points": points, "labels": labels}


def load_axis_model(
    checkpoint_path, max_num_points: int = 25, device: str | torch.device = "cpu"
) -> AxisTickCNN:
    """Load an axis CNN checkpoint saved from the Lightning module.

    Lightning state dicts share key names with the plain module here because the
    submodule attribute names are unchanged, so they load directly. Any
    training-only keys are reported rather than silently ignored.
    """
    model = AxisTickCNN(max_num_points=max_num_points)
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"axis checkpoint missing keys: {sorted(missing)[:8]}")
    if unexpected:
        # Not fatal (e.g. Lightning bookkeeping), but must not pass unnoticed.
        print(f"[axis] ignoring unexpected checkpoint keys: {sorted(unexpected)[:8]}")
    model.eval()
    return model.to(device)
