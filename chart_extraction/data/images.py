"""Image discovery.

AUDIT NOTE (Phase 0, bug 3) -- LATENT, NOT ACTIVE
--------------------------------------------------
The original ``inference-3.ipynb`` built its image-ID list three separate times:
once via ``Path.glob("*.jpg")`` for the Donut stage, and twice via
``os.listdir`` for the axis-CNN and marker stages. The three per-stage result
frames were then joined **by positional index** (``df['x_val'][i]``,
``df2['x_points'][i]``, ``df3['boxes'][i]``).

This was reported as an active misalignment bug. It is not. Both ``Path.glob``
and ``os.listdir`` delegate to ``os.scandir`` and returned identical order in a
200-file shuffled-creation-order test at audit time; measured positional
mismatch was 0/200. Nothing was being corrupted.

It is nonetheless a real contract violation: neither function documents an
ordering guarantee, and the positional join silently depends on one. The fix
here is to enumerate **once** and key every downstream join on the image ID.

Because this bug was latent, it MUST NOT be credited with any part of a
Phase 0 -> Phase 1 score delta. By definition it cannot have changed output.
See docs/PHASE0_AUDIT.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ImageRef:
    """One image, identified by its stem."""

    image_id: str
    path: Path


def discover_images(image_dir: Path | str, pattern: str = "*.jpg") -> list[ImageRef]:
    """Enumerate images once, in a deterministic (sorted) order.

    Sorting is not required for correctness -- every downstream join is keyed on
    ``image_id`` -- but it makes runs reproducible and diffable, which matters
    for the ablation table.
    """
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise NotADirectoryError(f"image_dir does not exist: {image_dir}")

    refs = [ImageRef(image_id=p.stem, path=p) for p in image_dir.glob(pattern)]
    refs.sort(key=lambda r: r.image_id)

    ids = [r.image_id for r in refs]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(
            f"duplicate image ids in {image_dir}: {sorted(duplicates)}. "
            "Image ids must be unique -- downstream stages are joined on them."
        )
    return refs


def require_all_ids(
    stage_name: str,
    produced: Iterable[str],
    expected: Sequence[str],
) -> None:
    """Assert a stage emitted a result for every expected image id.

    The notebooks had no such check: a stage that silently dropped images would
    shorten a frame and shift every positional join after it.
    """
    produced_set = set(produced)
    expected_set = set(expected)
    missing = expected_set - produced_set
    extra = produced_set - expected_set
    if missing or extra:
        raise ValueError(
            f"stage {stage_name!r} id mismatch: "
            f"{len(missing)} missing (e.g. {sorted(missing)[:3]}), "
            f"{len(extra)} unexpected (e.g. {sorted(extra)[:3]})"
        )
