r"""Repair of numeric strings emitted by Donut.

AUDIT NOTE (Phase 0, bug 5)
---------------------------
The brief recorded this as "clean_preds() never runs -- call is commented out in
both notebooks". That is correct for ``tuned-donut`` but **wrong for
``inference-3``**, where the call is live::

    x, y = clean_preds(x, y)          # inference-3, uncommented

inference-3 is, however, the worse of the two, because its copy of the function
has the load-bearing line commented out::

    #temp = re.sub(r"[^0-9\.\-eE]", "", temp)

Without that strip, a value that fails the initial cast is never repaired: it
falls through the ``multiple_*`` branches unchanged and then fails the second
cast, landing in ``except ValueError: temp = 0``. Measured at audit time:

    input : ['11', '1E', '3.14', '-5', '1e5']
    output: [11,    0,   3.14,   -5,   0   ]

So inference-3 called a function that converted salvageable values to zero,
while tuned-donut had an intact function it never called. Both are fixed here:
the strip is restored and the function is wired into the parse path.
"""

from __future__ import annotations

import re
from typing import List, Sequence

# Characters that can legitimately appear in a number Donut might emit.
_NUMERIC_KEEP = re.compile(r"[^0-9\.\-eE]")
_WHITESPACE = re.compile(r"\s")


def _repair_token(token: str) -> float | int:
    """Coerce one generated token to a number, repairing what can be repaired.

    Returns 0 only when nothing numeric survives -- not, as in the inference-3
    version, whenever the first cast happens to fail.
    """
    dtype = int if "." not in token else float

    # Fast path: "10 0" -> 100 (Donut sometimes inserts spaces mid-number).
    try:
        return dtype(_WHITESPACE.sub("", token))
    except ValueError:
        pass

    # Restored strip (this is the line commented out in inference-3).
    temp = _NUMERIC_KEEP.sub("", token)
    if not temp:
        return 0

    # A stray leading/trailing sign or exponent marker is common; normalise.
    if len(re.findall(r"-", temp)) > 1:
        temp = "-" + temp.replace("-", "")
    if len(re.findall(r"\.", temp)) > 1:
        head, *rest = temp.split(".")
        temp = head + "." + "".join(rest)
    if len(re.findall(r"[eE]", temp)) > 1:
        while temp.lower().startswith("e"):
            temp = temp[1:]
        while temp.lower().endswith("e"):
            temp = temp[:-1]
        chunks = temp.split("e") if "e" in temp else temp.split("E")
        if len(chunks) > 1:
            # Keep the LAST exponent marker: "1e2e-5" -> "12e-5".
            temp = "".join(chunks[:-1]) + "e" + chunks[-1]

    # An exponent marker with nothing after it is not castable.
    temp = temp.rstrip("eE")
    if temp in ("", "-", ".", "-."):
        return 0

    for candidate in (dtype, float, int):
        try:
            return candidate(temp)
        except (ValueError, TypeError):
            continue
    return 0


def clean_numeric_series(values: Sequence[str]) -> List[float | int]:
    """Repair every token in a series that has been judged numeric."""
    return [_repair_token(v) for v in values]


def _numeric_fraction(values: Sequence[str]) -> float:
    r"""Fraction of characters across the whole series that are digits.

    AUDIT NOTE: the notebook computed this as
    ``len(re.sub(r"[^\d]", "", joined)) / len(joined)`` with no guard, so an
    all-empty series raised ZeroDivisionError. That exception was then swallowed
    by a bare ``except:``, demoting the entire image to a placeholder row. An
    empty series is treated as non-numeric (0.0) instead.
    """
    joined = "".join(values)
    if not joined:
        return 0.0
    return len(re.sub(r"[^\d]", "", joined)) / len(joined)


def clean_preds(
    x: Sequence[str],
    y: Sequence[str],
    numeric_threshold: float = 0.5,
) -> tuple[list, list]:
    """Clean an (x, y) series pair.

    Each axis is independently judged numeric or categorical by digit density,
    then either repaired numerically or merely whitespace-stripped. Categorical
    axes must not be pushed through numeric repair -- doing so would turn
    "Monday" into 0.
    """
    if _numeric_fraction(x) >= numeric_threshold:
        new_x = clean_numeric_series(x)
    else:
        new_x = [s.strip() for s in x]

    if _numeric_fraction(y) >= numeric_threshold:
        new_y = clean_numeric_series(y)
    else:
        new_y = [s.strip() for s in y]

    return new_x, new_y
