"""Model loading and the timed inference pass.

Separated from harness.py so the aggregation logic stays testable without
torch, transformers or any checkpoint on disk.
"""

from __future__ import annotations

import logging
from typing import Sequence

from chart_extraction.config import PipelineConfig, RuntimeConfig
from chart_extraction.data.images import ImageRef
from chart_extraction.eval.harness import StageTimings, _timed, model_size
from chart_extraction.runtime import OomPolicy, autocast_context
from chart_extraction.stages import MODE_DONUT_ONLY

logger = logging.getLogger(__name__)


def synchronize(device: str) -> None:
    """Flush async CUDA work so stage timings are not attributed to the next
    stage. Without this, GPU latency numbers are meaningless."""
    if str(device).startswith("cuda"):
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()


def describe_device(runtime: RuntimeConfig) -> dict:
    """Record what the run actually executed on, for the result file."""
    info = {
        "requested_device": runtime.device,
        "precision": runtime.precision,
        "oom_retry_enabled": runtime.oom_retry_enabled,
        "oom_retry_scales": list(runtime.oom_retry_scales),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        available = torch.cuda.is_available()
        info["cuda_available"] = available
        if available and str(runtime.device).startswith("cuda"):
            index = torch.device(runtime.device).index or 0
            props = torch.cuda.get_device_properties(index)
            info["gpu_name"] = props.name
            info["gpu_total_mb"] = round(props.total_memory / (1024 * 1024), 1)
    except Exception:  # pragma: no cover - diagnostics must never break a run
        logger.debug("could not describe device", exc_info=True)
    return info


def peak_memory_mb(device: str) -> float | None:
    """Peak allocated CUDA memory since the last reset, in MB."""
    if not str(device).startswith("cuda"):
        return None
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 1)
    except Exception:  # pragma: no cover
        return None


def reset_peak_memory(device: str) -> None:
    if not str(device).startswith("cuda"):
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
    except Exception:  # pragma: no cover
        pass


def load_all_models(config: PipelineConfig, runtime: RuntimeConfig):
    """Load every stage's model. Returns (models_dict, size_records).

    Donut is moved to the device and halved here when precision is fp16, so
    every later caller sees a model already in its final dtype and the reported
    model sizes reflect what is actually resident.
    """
    from transformers import DonutProcessor, VisionEncoderDecoderModel

    from chart_extraction.axis.model import load_axis_model
    from chart_extraction.donut.inference import prepare_donut_model
    from chart_extraction.markers.model import load_marker_model

    donut_only = config.mode == MODE_DONUT_ONLY
    config.require_paths("donut_model_dir")
    if not donut_only:
        config.require_paths(
            "x_axis_model_path", "y_axis_model_path", "marker_model_path",
        )
    device = runtime.device

    logger.info("loading Donut from %s (precision=%s)", config.donut_model_dir, runtime.precision)
    donut = VisionEncoderDecoderModel.from_pretrained(config.donut_model_dir)
    donut = prepare_donut_model(donut, precision=runtime.precision, device=device)
    processor = DonutProcessor.from_pretrained(config.donut_model_dir)

    models = {"donut": donut, "processor": processor}
    sizes = [model_size(donut, "donut")]

    if donut_only:
        logger.warning(
            "mode=donut_only: skipping the axis CNN and marker detector. "
            "Donut's generated series is used directly; no detection stage "
            "runs. This is a different system from the full pipeline and its "
            "score is not comparable with a full-pipeline score."
        )
        return models, sizes

    logger.info("loading axis models")
    model_x = load_axis_model(config.x_axis_model_path, config.axis_max_num_points, device)
    model_y = load_axis_model(config.y_axis_model_path, config.axis_max_num_points, device)

    logger.info("loading marker detector")
    markers = load_marker_model(config.marker_model_path, config.marker_num_classes, device)

    models.update({"axis_x": model_x, "axis_y": model_y, "markers": markers})
    sizes += [
        model_size(model_x, "axis_x"),
        model_size(model_y, "axis_y"),
        model_size(markers, "marker_rcnn"),
    ]
    return models, sizes


def run_stages(
    refs: Sequence[ImageRef],
    models: dict,
    config: PipelineConfig,
    runtime: RuntimeConfig,
    oom_policy: OomPolicy | None = None,
) -> tuple[dict, dict, dict, StageTimings, OomPolicy]:
    """Run the three inference stages, timing each.

    Decode timing is added later by ``evaluate``; this returns only the model
    stages so that decode cost is attributed separately.
    """
    from chart_extraction.axis.inference import detect_axis_ticks
    from chart_extraction.donut.inference import run_donut
    from chart_extraction.markers.inference import detect_markers

    device = runtime.device
    policy = oom_policy if oom_policy is not None else OomPolicy(
        enabled=runtime.oom_retry_enabled, retry_scales=runtime.oom_retry_scales
    )
    timings = StageTimings(n_images=len(refs))
    reset_peak_memory(device)

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
            oom_policy=policy,
        )
        synchronize(device)

    if config.mode == MODE_DONUT_ONLY:
        # Neither detection stage runs. Their timings stay at zero, which is
        # correct and distinguishable from "ran and was instant" because the
        # skipped stages are recorded by name in the result.
        return donut_predictions, {}, {}, timings, policy

    with _timed(timings, "axis_s"):
        with autocast_context(device, runtime.precision):
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
        with autocast_context(device, runtime.precision):
            marker_detections = detect_markers(
                refs,
                models["markers"],
                device=device,
                batch_size=config.marker_batch_size,
                num_workers=config.num_workers,
            )
        synchronize(device)

    return donut_predictions, axis_ticks, marker_detections, timings, policy
