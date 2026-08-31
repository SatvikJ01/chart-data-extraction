"""Levenshtein edit distance.

The official competition scorer used ``rapidfuzz.distance.Levenshtein.distance``.
That is standard unit-cost edit distance (insert/delete/substitute all cost 1),
so a direct implementation is numerically identical. rapidfuzz is used when
present purely for speed; the pure-Python path is the reference and is what the
unit tests pin.
"""

from __future__ import annotations

try:  # pragma: no cover - exercised only when the optional dep is installed
    from rapidfuzz.distance import Levenshtein as _rf_levenshtein

    _HAVE_RAPIDFUZZ = True
except ImportError:  # pragma: no cover
    _HAVE_RAPIDFUZZ = False


def levenshtein_distance(a: str, b: str) -> int:
    """Unit-cost edit distance between two strings."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    # Two-row dynamic programme; O(min(len)) memory.
    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,           # deletion
                    current[j - 1] + 1,        # insertion
                    previous[j - 1] + (ca != cb),  # substitution
                )
            )
        previous = current
    return previous[-1]


def distance(a: str, b: str) -> int:
    """Edit distance, using rapidfuzz when available."""
    if _HAVE_RAPIDFUZZ:  # pragma: no cover
        return int(_rf_levenshtein.distance(a, b))
    return levenshtein_distance(a, b)
