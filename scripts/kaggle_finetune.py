#!/usr/bin/env python3
"""Kaggle entrypoint: fine-tune, then score baseline and fine-tuned on the SAME
held-out extracted images, appending both to the ablation table.

    !git clone https://github.com/<user>/<repo>.git /kaggle/working/repo
    %run /kaggle/working/repo/scripts/kaggle_finetune.py

Pass --benchmark-only to measure throughput and project the runtime without
training. Do that first.

WHY IT SCORES THE BASELINE TOO
==============================
The held-out 40% was never seen by the fine-tune, but it was not held out from
the *base* checkpoint, whose partition is unrecorded. So neither absolute score
is provably clean. Scoring both models on the identical images makes the
DIFFERENCE attributable to the fine-tune, because both inherit the same base
history. One row on its own would not support that claim.

Stage budget (T4, defaults): benchmark ~2 min, training <= --max-hours,
two evaluations of 447 images each. Keep the total inside the GPU quota.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chart_extraction.runtime import configure_allocator  # noqa: E402

configure_allocator()

from chart_extraction.paths import describe, resolve_paths  # noqa: E402

REQUIRED = {
    "transformers": "transformers>=4.30",
    "albumentations": "albumentations>=1.3",
    "cv2": "opencv-python>=4.7",
}
REQUIRED_PATHS = ("data_root", "donut_dir")


def _importable(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


def install_missing() -> None:
    missing = [req for module, req in REQUIRED.items() if not _importable(module)]
    if not missing:
        print("[deps] all present")
        return
    print(f"[deps] installing: {' '.join(missing)}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *missing])


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    benchmark_only = "--benchmark-only" in argv

    resolved = resolve_paths()
    print("paths:")
    print(describe(resolved, REQUIRED_PATHS))
    if resolved.missing(REQUIRED_PATHS) or resolved.not_on_disk(REQUIRED_PATHS):
        print("\n[paths] attach the datasets or set BENETECH_* env vars. "
              "Nothing was installed.")
        return 2

    install_missing()

    seed = int(os.environ.get("BENETECH_SPLIT_SEED", "1234"))
    out_dir = Path(os.environ.get("BENETECH_RUN_DIR", "/kaggle/working/finetune"))
    results_dir = os.environ.get("BENETECH_RESULTS_DIR", str(REPO_ROOT / "results"))
    max_hours = os.environ.get("BENETECH_MAX_HOURS", "2.0")
    epochs = os.environ.get("BENETECH_EPOCHS", "3")
    oversample = os.environ.get("BENETECH_OVERSAMPLE", "6")
    n_generated = os.environ.get("BENETECH_N_GENERATED", "2000")
    model_tag = f"ft-extracted-s{seed}"

    from scripts.run_eval import main as run_eval_main
    from scripts.train_donut import main as train_main

    print("\n" + "=" * 72)
    print("  STAGE 1/3  fine-tune")
    print("=" * 72)
    train_argv = [
        "--data-root", str(resolved.data_root),
        "--donut-dir", str(resolved.donut_dir),
        "--output-dir", str(out_dir),
        "--split-seed", str(seed),
        "--epochs", epochs,
        "--oversample", oversample,
        "--n-generated", n_generated,
        "--max-hours", max_hours,
    ]
    if benchmark_only:
        train_argv.append("--benchmark-only")

    code = train_main(train_argv)
    if code != 0:
        print(f"[train] failed with exit code {code}")
        return code
    if benchmark_only:
        print("\n[done] benchmark only -- no training, no evaluation, no results row")
        return 0

    split_path = out_dir / "extracted_split.json"
    best_ckpt = out_dir / "best"
    if not best_ckpt.exists():
        print(f"[eval] no checkpoint at {best_ckpt}")
        return 3

    # Both evaluations use the identical --image-ids-file, so the two rows differ
    # only by checkpoint. That is the whole point of running the baseline again
    # rather than reusing an earlier row measured on a different image set.
    common = [
        "--data-root", str(resolved.data_root),
        "--image-ids-file", str(split_path),
        "--eval-provenance", str(split_path),
        "--profile", "kaggle",
        "--decode", "greedy",
        "--results-dir", results_dir,
    ]

    print("\n" + "=" * 72)
    print("  STAGE 2/3  evaluate BASE checkpoint on the held-out extracted set")
    print("=" * 72)
    code = run_eval_main(common + [
        "--donut-dir", str(resolved.donut_dir),
        "--model-tag", "base",
        "--run-id", f"holdout-base-s{seed}",
    ])
    if code != 0:
        return code

    print("\n" + "=" * 72)
    print("  STAGE 3/3  evaluate FINE-TUNED checkpoint on the same images")
    print("=" * 72)
    code = run_eval_main(common + [
        "--donut-dir", str(best_ckpt),
        "--model-tag", model_tag,
        "--run-id", f"holdout-{model_tag}",
    ])
    if code != 0:
        return code

    print(f"\n[done] two rows appended to {results_dir}/ablation.md")
    print("[done] compare them: same images, same decode, different checkpoint.")
    print("[done] the DIFFERENCE is the finding; neither absolute is leakage-free.")

    run_json = out_dir / "training_run.json"
    if run_json.exists():
        record = json.loads(run_json.read_text())
        print(f"[done] training took {record.get('total_hms')}, "
              f"best epoch {record.get('best_epoch')} "
              f"(val_loss {record.get('best_val_loss')})")
        if record.get("stopped_on_budget"):
            print("[done] NOTE: training stopped on the wall-clock ceiling, "
                  "so it ran fewer epochs than configured. Record that.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
