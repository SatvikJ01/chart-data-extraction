"""Held-out validation split, keyed on the annotation ``source`` field.

WHY THE SPLIT IS BUILT ON ``source``
====================================
Benetech assembled the dataset from **synthetic** (``generated``) and **real
textbook** (``extracted``) charts. Training is overwhelmingly generated; the
competition's test set skewed far more toward extracted. Top teams scored
0.88 public / 0.72 private, and local validation was notoriously optimistic
precisely because it was measured on the generated-heavy distribution.

So the extracted slice is the headline number and the generated slice is
reported alongside it as a deliberate contrast. The gap between them is the
finding, not a nuisance.

LEAKAGE CAVEAT -- READ BEFORE QUOTING ANY NUMBER FROM THIS SPLIT
================================================================
This split is held out with respect to *future* training in this repo. It is
**not** guaranteed to be unseen by the checkpoints currently being evaluated.
Those checkpoints (Donut, the axis CNNs, the marker detector) were fine-tuned
elsewhere on ``train/``, and no record of their train/validation partition
exists in this repo. Any image in this split may have been in their training
data.

Scores measured here are therefore **optimistic for the existing checkpoints**,
on top of the synthetic-vs-extracted optimism described above. Both effects push
the same direction. This caveat is carried into every emitted result file so a
number cannot be quoted without it.

The split is deterministic (hash of image id, fixed salt), so it is stable
across runs and machines without needing to be stored -- but it is also written
into each result file for reproducibility.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from chart_extraction.eval.ground_truth import Annotation

DEFAULT_SALT = "benetech-phase1-validation"


def _bucket(image_id: str, salt: str) -> float:
    """Deterministic uniform value in [0, 1) for one image id."""
    digest = hashlib.md5(f"{salt}:{image_id}".encode()).hexdigest()
    # Top 8 hex digits give 32 bits of resolution, plenty for a split.
    return int(digest[:8], 16) / 0xFFFFFFFF


@dataclass(frozen=True)
class Split:
    """A named set of image ids plus the composition that produced it."""

    name: str
    image_ids: tuple[str, ...]
    salt: str
    fraction: float
    composition: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.image_ids)

    def by_source(self, annotations: Mapping[str, Annotation]) -> dict[str, list[str]]:
        """Partition this split's ids into extracted / generated."""
        out: dict[str, list[str]] = {"extracted": [], "generated": []}
        for image_id in self.image_ids:
            out[annotations[image_id].source].append(image_id)
        return out


def build_validation_split(
    annotations: Mapping[str, Annotation],
    fraction: float = 0.05,
    salt: str = DEFAULT_SALT,
    include_all_extracted: bool = True,
    name: str = "val",
) -> Split:
    """Select a deterministic held-out validation split.

    The split is stratified by ``source``: the hash threshold is applied within
    each stratum, so the generated slice is sampled at ``fraction`` regardless
    of how the two populations compare in size.

    ``include_all_extracted`` defaults True because extracted images are scarce
    in ``train/`` -- sampling 5% of an already-small population yields a slice
    too small for a headline number to mean anything. Since this repo does not
    train on the extracted slice, taking all of it costs nothing and buys
    statistical power. Set False if a later phase starts training on extracted
    data, at which point holding some back becomes necessary.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    selected: list[str] = []
    for image_id in sorted(annotations):
        source = annotations[image_id].source
        if source == "extracted" and include_all_extracted:
            selected.append(image_id)
        elif _bucket(image_id, salt) < fraction:
            selected.append(image_id)

    composition = _composition(selected, annotations)
    return Split(
        name=name,
        image_ids=tuple(selected),
        salt=salt,
        fraction=fraction,
        composition=composition,
    )


def _composition(image_ids: Sequence[str], annotations: Mapping[str, Annotation]) -> dict:
    """Counts by source and by (source, chart_type), for the result file."""
    by_source: dict[str, int] = {}
    by_source_type: dict[str, int] = {}
    for image_id in image_ids:
        annotation = annotations[image_id]
        by_source[annotation.source] = by_source.get(annotation.source, 0) + 1
        key = f"{annotation.source}/{annotation.chart_type}"
        by_source_type[key] = by_source_type.get(key, 0) + 1
    return {
        "total": len(image_ids),
        "by_source": dict(sorted(by_source.items())),
        "by_source_and_chart_type": dict(sorted(by_source_type.items())),
    }


def holdout_complement(
    annotations: Mapping[str, Annotation], split: Split
) -> tuple[str, ...]:
    """Everything not in the split.

    Provided so a future training phase can assert it is not training on
    validation ids. Nothing in Phase 1/2 trains, so nothing calls this yet --
    it exists to make the guarantee checkable rather than assumed.
    """
    held = set(split.image_ids)
    return tuple(image_id for image_id in sorted(annotations) if image_id not in held)
