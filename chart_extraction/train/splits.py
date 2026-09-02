"""The train/held-out partition for the specialization fine-tune.

WHY THIS EXISTS SEPARATELY FROM eval/splits.py
==============================================
``eval/splits.py`` builds a validation split for *scoring an existing
checkpoint*. This builds a partition for *training a new one*, which carries a
much stronger obligation: the held-out portion must never reach the optimiser,
and that has to be checkable after the fact rather than asserted in a docstring.

So the partition is written to disk with its seed and its exact id lists, the
trainer asserts disjointness against it before the first step, and evaluation
reads the same file. A reader can verify the claim without rerunning anything.

THREE-WAY, NOT TWO-WAY
======================
The brief asks for 60/40 train/test. That is what is built -- but the 60% is
then subdivided again into an inner train and an inner validation slice, and
**only the inner validation slice is used for loss monitoring and checkpoint
selection**.

Selecting the best epoch on the held-out 40% would leak it just as surely as
training on it: the reported number would be the best of N draws against that
set rather than an honest estimate. The 40% is touched exactly once, at the very
end, by the evaluation harness.

WHAT "LEAKAGE-FREE" DOES AND DOES NOT MEAN HERE
===============================================
The held-out 40% is never seen by *this fine-tune*. It may still have been in
the training data of the **base checkpoint** being fine-tuned, whose own
partition is unrecorded (see eval/splits.py). So a score on this held-out set is
free of leakage from the specialization phase only.

That is still worth having: the baseline and the fine-tuned model are scored on
the same images and inherit the same base-checkpoint history, so the *difference*
between them is attributable to the fine-tune even though neither absolute
number is provably clean.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from chart_extraction.eval.ground_truth import Annotation

SPLIT_FORMAT_VERSION = 1


@dataclass
class ExtractedSplit:
    """A recorded three-way partition of the extracted images."""

    seed: int
    train_fraction: float
    val_fraction_of_train: float
    train_ids: tuple[str, ...]
    val_ids: tuple[str, ...]
    holdout_ids: tuple[str, ...]
    composition: dict = field(default_factory=dict)
    version: int = SPLIT_FORMAT_VERSION

    @property
    def fit_ids(self) -> tuple[str, ...]:
        """Everything the optimiser is allowed to see."""
        return self.train_ids

    def assert_disjoint(self) -> None:
        """Fail loudly if any image appears in more than one part."""
        train, val, holdout = set(self.train_ids), set(self.val_ids), set(self.holdout_ids)
        for a, b, name in (
            (train, holdout, "train/holdout"),
            (val, holdout, "val/holdout"),
            (train, val, "train/val"),
        ):
            overlap = a & b
            if overlap:
                raise ValueError(
                    f"{name} overlap of {len(overlap)} ids, e.g. "
                    f"{sorted(overlap)[:3]} -- the held-out set must never be "
                    "trained on or selected against"
                )

    def as_dict(self) -> dict:
        payload = asdict(self)
        for key in ("train_ids", "val_ids", "holdout_ids"):
            payload[key] = list(payload[key])
        return payload

    def provenance(self) -> dict:
        """The record the evaluation harness reads to justify its caveats."""
        return {
            "kind": "extracted_finetune_holdout",
            "seed": self.seed,
            "train_fraction": self.train_fraction,
            "n_train": len(self.train_ids),
            "n_val": len(self.val_ids),
            "n_holdout": len(self.holdout_ids),
            "held_out_from_finetune": True,
            # The honest limit of the claim.
            "held_out_from_base_checkpoint": False,
        }


def build_extracted_split(
    annotations: Mapping[str, Annotation],
    train_fraction: float = 0.6,
    val_fraction_of_train: float = 0.1,
    seed: int = 0,
) -> ExtractedSplit:
    """Partition the extracted images, stratified by chart type.

    Stratified because measured per-type scores on this pipeline span roughly
    0.11 to 0.69: an unstratified draw could hand the held-out set a different
    type mix from the training set and make the comparison meaningless.
    """
    extracted = {
        image_id: annotation
        for image_id, annotation in annotations.items()
        if annotation.source == "extracted"
    }
    if not extracted:
        raise ValueError("no extracted images in the supplied annotations")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")

    strata: dict[str, list[str]] = {}
    for image_id in sorted(extracted):
        strata.setdefault(extracted[image_id].chart_type, []).append(image_id)

    rng = random.Random(seed)
    train_ids: list[str] = []
    val_ids: list[str] = []
    holdout_ids: list[str] = []

    for chart_type in sorted(strata):
        members = sorted(strata[chart_type])
        shuffled = list(members)
        rng.shuffle(shuffled)

        n_fit_total = round(len(shuffled) * train_fraction)
        # Never let a stratum contribute to only one side when it could split.
        if len(shuffled) >= 2:
            n_fit_total = max(1, min(len(shuffled) - 1, n_fit_total))
        fit_part = shuffled[:n_fit_total]
        holdout_ids.extend(shuffled[n_fit_total:])

        n_val = int(round(len(fit_part) * val_fraction_of_train))
        if len(fit_part) >= 2:
            n_val = max(1, min(len(fit_part) - 1, n_val))
        val_ids.extend(fit_part[:n_val])
        train_ids.extend(fit_part[n_val:])

    split = ExtractedSplit(
        seed=seed,
        train_fraction=train_fraction,
        val_fraction_of_train=val_fraction_of_train,
        train_ids=tuple(sorted(train_ids)),
        val_ids=tuple(sorted(val_ids)),
        holdout_ids=tuple(sorted(holdout_ids)),
    )
    split.composition = _composition(split, extracted)
    split.assert_disjoint()
    return split


def _composition(split: ExtractedSplit, extracted: Mapping[str, Annotation]) -> dict:
    def by_type(ids: Sequence[str]) -> dict:
        counts: dict[str, int] = {}
        for image_id in ids:
            chart_type = extracted[image_id].chart_type
            counts[chart_type] = counts.get(chart_type, 0) + 1
        return dict(sorted(counts.items()))

    return {
        "n_extracted_total": len(extracted),
        "train": {"n": len(split.train_ids), "by_chart_type": by_type(split.train_ids)},
        "val": {"n": len(split.val_ids), "by_chart_type": by_type(split.val_ids)},
        "holdout": {
            "n": len(split.holdout_ids),
            "by_chart_type": by_type(split.holdout_ids),
        },
    }


def save_split(split: ExtractedSplit, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split.as_dict(), indent=2, sort_keys=True))
    return path


def load_split(path: Path | str) -> ExtractedSplit:
    payload = json.loads(Path(path).read_text())
    split = ExtractedSplit(
        seed=payload["seed"],
        train_fraction=payload["train_fraction"],
        val_fraction_of_train=payload["val_fraction_of_train"],
        train_ids=tuple(payload["train_ids"]),
        val_ids=tuple(payload["val_ids"]),
        holdout_ids=tuple(payload["holdout_ids"]),
        composition=payload.get("composition", {}),
        version=payload.get("version", SPLIT_FORMAT_VERSION),
    )
    split.assert_disjoint()
    return split
