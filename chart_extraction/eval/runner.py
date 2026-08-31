"""Model loading and the timed inference pass.

Separated from harness.py so the aggregation logic stays testable without
torch, transformers or any checkpoint on disk.
"""

from __future__ import annotations

import logging
from typing import Sequence

from chart_extraction.config import PipelineConfig
from chart_extraction.data.images import ImageRef
from chart_extraction.eval.harness import StageTimings, _timed, model_size

logger = logging.getLogger(__name__)


def synchronize(device: str) -> None:
    """Flush async CUDA work so stage timings are not attributed to the next
    stage. Without this, GPU latency numbers are meaningless."""
    if str(device).startswith("cuda"):
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()


def load_all_models(config: PipelineConfig, device: str):
    """Load every stage's model. Returns (models_dict, size_records)."""
    from transformers import DonutProcessor, VisionEncoderDecoderModel

    from chart_extraction.axis.model import load_axis_model
    from chart_extraction.markers.model import load_marker_model

    logger.info("loading Donut from %s", config.donut_model_dir)
    donut = VisionEncoderDecoderModel.from_pretrained(config.donut_model_dir).to(device)
    donut.eval()
    processor = DonutProcessor.from_pretrained(config.donut_model_dir)

    logger.info("loading axis models")
    model_x = load_axis_model(config.x_axis_model_path, config.axis_max_num_points, device)
    model_y = load_axis_model(config.y_axis_model_path, config.axis_max_num_points, device)

    logger.info("loading marker detector")
    markers = load_marker_model(config.marker_model_path, config.marker_num_classes, device)

    models = {
        "donut": donut,
        "processor": processor,
        "axis_x": model_x,
        "axis_y": model_y,
        "markers": markers,
    }
    sizes = [
        model_size(donut, "donut"),
        model_size(model_x, "axis_x"),
        model_size(model_y, "axis_y"),
        model_size(markers, "marker_rcnn"),
    ]
    return models, sizes


def run_stages(
    refs: Sequence[ImageRef],
    models: dict,
    config: PipelineConfig,
    device: str,
) -> tuple[dict, dict, dict, StageTimings]:
    """Run the three inference stages, timing each.

    Decode timing is added later by ``evaluate``; this returns only the model
    stages so that decode cost is attributed separately.
    """
    from chart_extraction.axis.inference import detect_axis_ticks
    from chart_extraction.donut.inference import run_donut
    from chart_extraction.markers.inference import detect_markers

    timings = StageTimings(n_images=len(refs))

    # Each stage syncs before leaving its timed block. CUDA kernels are async,
    # so without this the tail of one stage's GPU work lands in the next
    # stage's wall-clock measurement and every per-stage latency is wrong.
    with _timed(timings, "donut_s"):
        donut_predictions = run_donut(
            refs,
            models["donut"],
            models["processor"],
            config.generation,
            device=device,
            batch_size=config.donut_batch_size,
            num_workers=config.num_workers,
            random_padding=config.donut_random_padding,
        )
        synchronize(device)

    with _timed(timings, "axis_s"):
        axis_ticks = detect_axis_ticks(
            refs,
            models["axis_x"],
            models["axis_y"],
            device=device,
            batch_size=config.axis_batch_size,
            num_workers=config.num_workers,
        )
        synchronize(device)

    with _timed(timings, "markers_s"):
        marker_detections = detect_markers(
            refs,
            models["markers"],
            device=device,
            batch_size=config.marker_batch_size,
            num_workers=config.num_workers,
        )
        synchronize(device)

    return donut_predictions, axis_ticks, marker_detections, timings

