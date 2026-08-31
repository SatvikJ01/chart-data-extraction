#!/usr/bin/env python3
"""Kaggle-runnable entrypoint: installs missing deps, then runs both decode
configs through the same harness.

Paste into a Kaggle notebook cell (GPU on, internet on for the pip install):

    !git clone https://github.com/<user>/<repo>.git /kaggle/working/repo
    %run /kaggle/working/repo/scripts/kaggle_eval.py

Requires these datasets attached to the notebook:
    benetech-making-graphs-accessible   (the competition data)
    benetech-donut                      (fine-tuned Donut checkpoint)
    x-axis-model-10, y-axis-model-10    (axis tick CNNs)
    marker-model                        (Faster R-CNN marker detector)

Checkpoint paths are the ones the original notebooks used. Override with the
environment variables named in PATHS below if a dataset is mounted elsewhere.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Kaggle images ship torch/torchvision/pandas/numpy/PIL and usually cv2, but
# not always transformers at the version this needs. Install only what is
# actually missing so a warm image is not reinstalled for nothing.
REQUIRED = {
    "transformers": "transformers>=4.30",
    "albumentations": "albumentations>=1.3",
    "cv2": "opencv-python>=4.7",
}

PATHS = {
    "data_root": os.environ.get(
        "BENETECH_DATA_ROOT", "/kaggle/input/benetech-making-graphs-accessible"
    ),
    "donut_dir": os.environ.get("BENETECH_DONUT_DIR", "/kaggle/input/benetech-donut"),
    "x_axis_model": os.environ.get(
        "BENETECH_X_AXIS", "/kaggle/input/x-axis-model-10/model (1).pth"
    ),
    "y_axis_model": os.environ.get(
        "BENETECH_Y_AXIS",
        "/kaggle/input/y-axis-model-10/Y_Point_generation_weights_1.0.pth",
    ),
    "marker_model": os.environ.get(
        "BENETECH_MARKER", "/kaggle/input/marker-model/Marker_weights.pth"
    ),
}


def install_missing() -> None:
    missing = []
    for module, requirement in REQUIRED.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(requirement)

    if not missing:
        print("[deps] all present, nothing to install")
        return

    print(f"[deps] installing: {' '.join(missing)}")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", *missing]
    )
    print("[deps] done")


def check_paths() -> bool:
    ok = True
    for name, raw in PATHS.items():
        path = Path(raw)
        exists = path.exists()
        print(f"[paths] {'OK ' if exists else 'MISSING'}  {name:<14} {path}")
        if not exists:
            ok = False
    if not ok:
        print(
            "\n[paths] Attach the missing datasets to the notebook, or set the "
            "BENETECH_* environment variables to where they are mounted."
        )
    return ok


def main() -> int:
    install_missing()
    if not check_paths():
        return 2

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from scripts.run_eval import main as run_eval_main

    fraction = float(os.environ.get("BENETECH_VAL_FRACTION", "0.05"))
    limit = os.environ.get("BENETECH_LIMIT")
    results_dir = os.environ.get("BENETECH_RESULTS_DIR", str(REPO_ROOT / "results"))

    # Both decode configs run through the identical harness and split, so the
    # only difference between the two rows is beam width.
    for decode in ("greedy", "beam2"):
        print("\n" + "=" * 72)
        print(f"  decode config: {decode}")
        print("=" * 72)
        argv = [
            "--data-root", PATHS["data_root"],
            "--donut-dir", PATHS["donut_dir"],
            "--x-axis-model", PATHS["x_axis_model"],
            "--y-axis-model", PATHS["y_axis_model"],
            "--marker-model", PATHS["marker_model"],
            "--decode", decode,
            "--fraction", str(fraction),
            "--results-dir", results_dir,
        ]
        if limit:
            argv += ["--limit", limit]

        code = run_eval_main(argv)
        if code != 0:
            print(f"[run] {decode} failed with exit code {code}")
            return code

    print(f"\n[done] results appended under {results_dir}")
    print("[done] copy results/ back out of the notebook and commit it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
