"""Sanity checks against known reference scores.

The point of these is to catch a *loading* problem -- a checkpoint that did not
actually load, a tokenizer mismatch, a special-token schema that no longer
matches the decoder -- which shows up as a score collapsing toward zero rather
than as an exception.

WHY THE COMPARISON IS ASYMMETRIC
================================
The reference is a competition **leaderboard** score, measured on the hidden
test set. Our number is measured on a held-out slice of ``train/``. These are
not the same distribution and the comparison is not apples-to-apples in either
direction, but the two sources of difference push the *same* way:

  * the test set skewed toward ``extracted`` (real textbook charts); our split
    is dominated by ``generated`` (synthetic), which is markedly easier
  * the checkpoint was fine-tuned on ``train/`` with no partition recorded in
    this repo, so validation images may have been in its training data

So scoring **above** the reference is expected and is not evidence of anything
being right. Scoring **far below** it is the informative signal, because none of
the known differences would produce that. Only the low side is treated as a
warning; the high side is reported as context.
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
) -> list[SanityWarning]:
    """Compare an observed overall score against the reference for this mode."""
    reference = reference or REFERENCE_POINTS.get(mode)
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

    if observed < reference.score * SUSPICIOUS_RATIO:
        warnings.append(
            SanityWarning(
                "error",
                "far_below_reference",
                f"Overall score {observed:.4f} is {ratio:.0%} of the reference "
                f"{reference.score:.2f} ({reference.source}). A shortfall this "
                "large is not explained by the known differences, which all "
                "push the other way: this split is mostly synthetic and "
                "therefore easier, and the checkpoint may have trained on these "
                "images. Suspect a loading problem -- checkpoint weights, "
                "processor/tokenizer pairing, or special-token schema.",
            )
        )
    elif observed < reference.score:
        warnings.append(
            SanityWarning(
                "warning",
                "below_reference",
                f"Overall score {observed:.4f} is below the reference "
                f"{reference.score:.2f}, which is mildly surprising: this split "
                "is easier than the test set the reference was measured on, so "
                "a correctly-loaded checkpoint would usually score higher. "
                "Worth a look, but within the range distribution differences "
                "could explain.",
            )
        )
    else:
        warnings.append(
            SanityWarning(
                "info",
                "above_reference",
                f"Overall score {observed:.4f} exceeds the reference "
                f"{reference.score:.2f} ({reference.measured_on}). This is "
                "EXPECTED and is not evidence the pipeline is correct: this "
                "split is mostly synthetic and the checkpoint may have trained "
                "on these images. Do not report this as beating the leaderboard.",
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
