"""Sanity checks against known reference scores.

The point of these is to catch a *loading* problem -- a checkpoint that did not
actually load, a tokenizer mismatch, a special-token schema that no longer
matches the decoder -- which shows up as a score collapsing toward zero rather
than as an exception.

WHY THE COMPARISON IS ASYMMETRIC
================================
The reference is a competition **leaderboard** score, measured on the hidden
test set. Our number is measured on a held-out slice of ``train/``. These are
not the same distribution, and up to two known differences push our number
*up*:

  * **distribution** -- the test set skewed toward ``extracted`` (real textbook
    charts). A ``generated``-heavy split is markedly easier. This applies only
    in proportion to how much generated data the run actually scored.
  * **leakage** -- the checkpoint was fine-tuned on ``train/`` with no partition
    recorded in this repo, so validation images may have been in its training
    data. This applies to every run.

Which of these applies depends on **what the run actually evaluated**, not on
what the split contained. An ``--subset extracted`` run scores no synthetic data
at all, so the distribution argument does not apply to it and saying "this split
is mostly synthetic" would be simply false. Every message below is built from
the measured composition of the scored instances.

Scoring **above** the reference is expected and is not evidence of anything
being right. Scoring **far below** it is the informative signal. Only the low
side is treated as a warning; the high side is reported as context.
"""

from __future__ import annotations

from dataclasses import dataclass

from chart_extraction.stages import MODE_DONUT_ONLY, MODE_FULL


@dataclass(frozen=True)
class ReferencePoint:
    """A known score for a given configuration."""

    score: float
    mode: str
    source: str
    measured_on: str

    def as_dict(self) -> dict:
        return {
            "score": self.score,
            "mode": self.mode,
            "source": self.source,
            "measured_on": self.measured_on,
        }


#: Known reference scores, keyed by pipeline mode.
REFERENCE_POINTS: dict[str, ReferencePoint] = {
    MODE_DONUT_ONLY: ReferencePoint(
        score=0.44,
        mode=MODE_DONUT_ONLY,
        source="reported competition leaderboard score for this Donut checkpoint",
        measured_on="hidden competition test set",
    ),
}

#: Below reference * this, the difference is larger than distribution effects
#: plausibly explain in the unfavourable direction.
SUSPICIOUS_RATIO = 0.5
#: Below this absolute score, a loading failure is the overwhelmingly likely
#: cause regardless of the reference.
NEAR_ZERO = 0.05


@dataclass(frozen=True)
class Composition:
    """What a run actually scored, by source.

    Built from the per-source instance counts of the scored instances, never
    from the split definition -- a --subset run evaluates a fraction of its
    split, and describing the split would misstate the run.
    """

    counts: dict

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def generated_fraction(self) -> float:
        return self.counts.get("generated", 0) / self.total if self.total else 0.0

    @property
    def extracted_fraction(self) -> float:
        return self.counts.get("extracted", 0) / self.total if self.total else 0.0

    @property
    def label(self) -> str:
        if not self.total:
            return "empty"
        if self.counts.get("generated", 0) == 0:
            return "extracted-only"
        if self.counts.get("extracted", 0) == 0:
            return "generated-only (fully synthetic)"
        return (
            f"mixed ({self.generated_fraction:.0%} generated, "
            f"{self.extracted_fraction:.0%} extracted)"
        )

    @property
    def distribution_applies(self) -> bool:
        """Whether the easier-distribution argument applies to this run."""
        return self.counts.get("generated", 0) > 0

    def upward_pressures(self) -> list[str]:
        """The known reasons this run's score may sit above the reference."""
        reasons = []
        if self.distribution_applies:
            if self.counts.get("extracted", 0) == 0:
                reasons.append(
                    "this run scored only synthetic (generated) charts, which "
                    "are markedly easier than the test set"
                )
            else:
                reasons.append(
                    f"{self.generated_fraction:.0%} of the instances scored were "
                    "synthetic (generated), which are easier than the test set"
                )
        reasons.append(
            "the checkpoint was fine-tuned on train/ and may have seen these "
            "images"
        )
        return reasons

    def rationale(self) -> str:
        reasons = self.upward_pressures()
        if len(reasons) == 1:
            return reasons[0]
        return "; ".join(reasons[:-1]) + "; and " + reasons[-1]

    def contrast_note(self) -> str:
        """Extra context when the easier-distribution argument does NOT apply."""
        if self.distribution_applies:
            return ""
        return (
            " Note this run is extracted-only -- the same kind of real chart the "
            "test set was weighted toward -- so the easier-distribution effect "
            "does not apply here and leakage is the only known upward pressure."
        )

    def as_dict(self) -> dict:
        return {
            "counts": dict(self.counts),
            "label": self.label,
            "generated_fraction": round(self.generated_fraction, 4),
        }


def composition_from_scores(by_source: dict | None) -> Composition:
    """Build a Composition from the harness's per-source breakdown."""
    counts = {
        str(source): int(entry.get("n_instances", 0))
        for source, entry in (by_source or {}).items()
    }
    return Composition(counts=counts)


@dataclass(frozen=True)
class SanityWarning:
    level: str          # "error" | "warning" | "info"
    code: str
    message: str

    def as_dict(self) -> dict:
        return {"level": self.level, "code": self.code, "message": self.message}


def check_against_reference(
    observed: float,
    mode: str,
    n_instances: int = 0,
    reference: ReferencePoint | None = None,
    composition: Composition | None = None,
) -> list[SanityWarning]:
    """Compare an observed overall score against the reference for this mode.

    ``composition`` describes what the run actually scored; every message that
    reasons about distribution is built from it, so an extracted-only run is
    never described as synthetic.
    """
    reference = reference or REFERENCE_POINTS.get(mode)
    composition = composition or Composition(counts={})
    warnings: list[SanityWarning] = []

    if observed < NEAR_ZERO:
        warnings.append(
            SanityWarning(
                "error",
                "score_near_zero",
                f"Overall score {observed:.4f} is near zero. This is almost "
                "always a loading problem rather than a modelling result: check "
                "that the checkpoint weights actually loaded (no silently "
                "ignored keys), that the DonutProcessor and tokenizer come from "
                "the same checkpoint directory, and that the special-token "
                "schema (<x_start>, <y_start>, ...) matches what the decoder "
                "emits. Inspect a few raw generations before believing this "
                "number.",
            )
        )

    if reference is None:
        return warnings

    if reference.mode != mode:
        warnings.append(
            SanityWarning(
                "info",
                "reference_mode_mismatch",
                f"Reference score {reference.score:.2f} is for mode "
                f"{reference.mode!r}, but this run is {mode!r}. Not comparable.",
            )
        )
        return warnings

    ratio = observed / reference.score if reference.score else float("inf")

    scored = f"Scored {composition.total} instances ({composition.label})."

    if observed < reference.score * SUSPICIOUS_RATIO:
        warnings.append(
            SanityWarning(
                "error",
                "far_below_reference",
                f"Overall score {observed:.4f} is {ratio:.0%} of the reference "
                f"{reference.score:.2f} ({reference.source}). {scored} A "
                "shortfall this large is not explained by the known differences, "
                f"which push the other way: {composition.rationale()}. Suspect a "
                "loading problem -- checkpoint weights, processor/tokenizer "
                "pairing, or special-token schema.",
            )
        )
    elif observed < reference.score:
        detail = (
            "this run is easier than the test set the reference was measured on"
            if composition.distribution_applies
            else "leakage would usually push a correctly-loaded checkpoint higher"
        )
        warnings.append(
            SanityWarning(
                "warning",
                "below_reference",
                f"Overall score {observed:.4f} is below the reference "
                f"{reference.score:.2f}, which is mildly surprising: {detail}. "
                f"{scored}{composition.contrast_note()} Worth a look, but within "
                "the range distribution differences could explain.",
            )
        )
    else:
        warnings.append(
            SanityWarning(
                "info",
                "above_reference",
                f"Overall score {observed:.4f} exceeds the reference "
                f"{reference.score:.2f} ({reference.measured_on}). {scored} This "
                "is EXPECTED and is not evidence the pipeline is correct: "
                f"{composition.rationale()}."
                f"{composition.contrast_note()} Do not report this as beating "
                "the leaderboard.",
            )
        )

    if n_instances and n_instances < 50:
        warnings.append(
            SanityWarning(
                "warning",
                "small_sample",
                f"Only {n_instances} instances scored; the comparison against "
                "the reference is noisy at this size.",
            )
        )

    return warnings
