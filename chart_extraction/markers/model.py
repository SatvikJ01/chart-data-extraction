"""Faster R-CNN marker detector.

Two-stage detector (RPN proposals + ROI head). Architecture and class count
preserved from ``inference-3`` so the existing checkpoint loads.
"""

from __future__ import annotations

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_marker_model(num_classes: int = 4):
    """MobileNetV3-Large FPN backbone with the box predictor resized."""
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights=None, weights_backbone=None
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def load_marker_model(
    checkpoint_path, num_classes: int = 4, device: str | torch.device = "cpu"
):
    model = build_marker_model(num_classes=num_classes)
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model.to(device)
