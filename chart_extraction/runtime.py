"""GPU runtime concerns: allocator tuning, precision, and OOM recovery.

Deliberately importable without torch, so configuration can be applied *before*
torch is imported. That ordering matters -- see :func:`configure_allocator`.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

ALLOC_CONF_VAR = "PYTORCH_CUDA_ALLOC_CONF"
EXPANDABLE_SEGMENTS = "expandable_segments:True"


def configure_allocator(setting: str = EXPANDABLE_SEGMENTS) -> str:
    """Enable expandable segments in the CUDA caching allocator.

    ``expandable_segments:True`` lets the allocator grow a segment in place
    instead of reserving fixed-size blocks, which is what makes a long run of
    variable-resolution images survive on a small card: without it, fragmented
    reserved blocks accumulate and a later allocation fails even though the
    total free memory would have covered it.

    MUST be called before ``import torch`` -- the value is read when the
    caching allocator initialises, and a process that has already touched CUDA
    will silently ignore a later change. Entrypoints call this at module top,
    ahead of every torch import.

    Existing settings are preserved: the value is appended rather than replaced,
    so a caller who set e.g. ``max_split_size_mb`` keeps it.
    """
    current = os.environ.get(ALLOC_CONF_VAR, "").strip()

    key = setting.split(":", 1)[0]
    if current:
        entries = [e.strip() for e in current.split(",") if e.strip()]
        if any(e.split(":", 1)[0] == key for e in entries):
            return current  # caller already expressed an opinion; respect it
        entries.append(setting)
        value = ",".join(entries)
    else:
        value = setting

    os.environ[ALLOC_CONF_VAR] = value

    if "torch" in _imported_modules():
        logger.warning(
            "%s set to %r after torch was already imported; the CUDA allocator "
            "may have initialised and will ignore it. Set it before importing "
            "torch to be sure it takes effect.",
            ALLOC_CONF_VAR,
            value,
        )
    return value


def _imported_modules() -> set[str]:
    import sys

    return set(sys.modules)


def is_out_of_memory(exc: BaseException) -> bool:
    """True for a CUDA OOM, however the installed torch chooses to raise it.

    Newer torch raises ``torch.cuda.OutOfMemoryError``; older versions raise a
    plain ``RuntimeError``. Both are checked, and the message is checked last so
    this keeps working if the class moves again.
    """
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except (ImportError, AttributeError):  # pragma: no cover
        pass

    message = str(exc).lower()
    return "out of memory" in message or "cuda oom" in message


def empty_cache() -> None:
    """Release cached blocks back to the driver after an OOM."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:  # pragma: no cover
        pass


@dataclass(frozen=True)
class OomEvent:
    """One image that hit an OOM and what was done about it."""

    image_id: str
    stage: str
    scale: float | None
    height: int | None
    width: int | None
    recovered: bool

    def as_dict(self) -> dict:
        return {
            "image_id": self.image_id,
            "stage": self.stage,
            "scale": self.scale,
            "height": self.height,
            "width": self.width,
            "recovered": self.recovered,
        }


@dataclass
class OomPolicy:
    """How to recover when an image will not fit.

    On OOM the batch is retried image-by-image at progressively lower Donut
    input resolution. Each attempt is logged and recorded, because a run that
    silently degraded some images to a lower resolution is not comparable with
    one that did not -- the count belongs in the result file.
    """

    enabled: bool = True
    retry_scales: tuple[float, ...] = (0.75, 0.5)
    #: Donut's Swin encoder downsamples in stages; keeping both sides a multiple
    #: of this avoids a shape mismatch part-way through the encoder.
    size_multiple: int = 32
    events: list[OomEvent] = field(default_factory=list)

    def record(self, event: OomEvent) -> None:
        self.events.append(event)
        if event.recovered:
            logger.warning(
                "OOM on %s (%s): recovered at scale %.2f -> %dx%d. This image "
                "was processed at reduced resolution and is not directly "
                "comparable with the rest of the run.",
                event.image_id, event.stage, event.scale, event.height, event.width,
            )
        else:
            logger.error(
                "OOM on %s (%s): no retry scale fit; emitting a failure for "
                "this image.", event.image_id, event.stage,
            )

    def summary(self) -> dict:
        recovered = [e for e in self.events if e.recovered]
        return {
            "enabled": self.enabled,
            "retry_scales": list(self.retry_scales),
            "n_events": len(self.events),
            "n_recovered": len(recovered),
            "n_unrecovered": len(self.events) - len(recovered),
            "degraded_image_ids": sorted({e.image_id for e in recovered}),
            "failed_image_ids": sorted(
                {e.image_id for e in self.events if not e.recovered}
            ),
            "events": [e.as_dict() for e in self.events],
        }


def scaled_size(size: dict, scale: float, multiple: int = 32) -> dict:
    """Scale a processor size dict, rounded to a usable multiple.

    Never returns a side smaller than ``multiple``; a degenerate size would fail
    in the encoder rather than saving memory.
    """
    height = int(size["height"] * scale)
    width = int(size["width"] * scale)
    height = max(multiple, (height // multiple) * multiple)
    width = max(multiple, (width // multiple) * multiple)
    return {"height": height, "width": width}


def autocast_context(device, precision: str):
    """Autocast for the detection stages, or a no-op.

    Donut uses ``.half()`` instead (see ``prepare_donut_model``) because its
    parameters dominate its footprint. The axis CNN and Faster R-CNN are small
    enough that halving weights buys little, and torchvision's detection models
    are known to be fragile under hard fp16 -- autocast keeps their master
    weights in fp32 while still running the heavy convolutions in half, which is
    the safer trade for them.
    """
    import contextlib

    if precision != "fp16" or not str(device).startswith("cuda"):
        return contextlib.nullcontext()

    import torch

    return torch.autocast(device_type="cuda", dtype=torch.float16)
