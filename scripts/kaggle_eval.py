#!/usr/bin/env python3
"""Kaggle convenience wrapper: install missing deps, then run both decode
configs through the same harness as a local run.

    !git clone https://github.com/<user>/<repo>.git /kaggle/working/repo
    %run /kaggle/working/repo/scripts/kaggle_eval.py

Pass --dry-run to verify paths and report what would be installed, without
installing anything or loading a model.

This is a thin wrapper. It owns no paths of its own -- resolution is
``chart_extraction.paths``, identical to a local run, so the only difference
between here and a workstation is which preset happens to apply and which
environment variables are set.

Attach these datasets (or set the BENETECH_* environment variables):
    benetech-making-graphs-accessible, benetech-donut,
    x-axis-model-10, y-axis-model-10, marker-model
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Before any torch import, including transitively through the package.
from chart_extraction.runtime import configure_allocator  # noqa: E402

configure_allocator()

from chart_extraction.paths import describe, resolve_paths  # noqa: E402

# Kaggle images ship torch/torchvision/pandas/numpy/PIL and usually cv2, but
# not always transformers at the version this needs. Install only what is
# actually missing so a warm image is not reinstalled for nothing.
REQUIRED = {
    "transformers": "transformers>=4.30",
    "albumentations": "albumentations>=1.3",
    "cv2": "opencv-python>=4.7",
}

REQUIRED_PATHS = (
    "data_root", "donut_dir", "x_axis_model", "y_axis_model", "marker_model",
)


def install_missing() -> None:
    missing = [req for module, req in REQUIRED.items() if not _importable(module)]
    if not missing:
        print("[deps] all present, nothing to install")
        return
    print(f"[deps] installing: {' '.join(missing)}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *missing])
    print("[deps] done")


def _importable(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    dry_run = "--dry-run" in argv

    # Paths are checked BEFORE installing anything. Installing several hundred
    # MB of wheels and only then discovering the datasets are not attached
    # wastes minutes of a session for no reason.
    resolved = resolve_paths()
    print("paths:")
    print(describe(resolved, REQUIRED_PATHS))

    if resolved.missing(REQUIRED_PATHS) or resolved.not_on_disk(REQUIRED_PATHS):
        print(
            "\n[paths] Attach the missing datasets, or set the BENETECH_* "
            "environment variables to where they are mounted. "
            "Nothing was installed."
        )
        return 2

    if dry_run:
        missing = [req for module, req in REQUIRED.items() if not _importable(module)]
        print(f"[dry-run] paths OK; would install: {missing or 'nothing'}")
        return 0

    install_missing()

    from scripts.run_eval import main as run_eval_main

    profile = os.environ.get("BENETECH_PROFILE", "kaggle")
    subset = os.environ.get("BENETECH_SUBSET", "both")
    fraction = os.environ.get("BENETECH_VAL_FRACTION", "0.05")
    limit = os.environ.get("BENETECH_LIMIT")

    # Both decode configs run through the identical harness and split, so the
    # only difference between the two rows is beam width.
    for decode in ("greedy", "beam2"):
        print("\n" + "=" * 72)
        print(f"  decode config: {decode}")
        print("=" * 72)
        argv = [
            "--profile", profile,
            "--decode", decode,
            "--subset", subset,
            "--fraction", fraction,
        ]
        if limit:
            argv += ["--limit", limit]

        code = run_eval_main(argv)
        if code != 0:
            print(f"[run] {decode} failed with exit code {code}")
            return code

    print("\n[done] results appended; copy them out of the notebook and commit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
