#!/usr/bin/env python3
"""Run one evaluation pass over a held-out validation split.

Single pass, every Phase 1 and Phase 2 number emitted together. The same
entrypoint runs locally and on Kaggle -- no path is hardcoded.

Local single GPU:

    export BENETECH_DATA_ROOT=~/data/benetech
    export BENETECH_DONUT_DIR=~/models/donut
    export BENETECH_X_AXIS=~/models/x_axis.pth
    export BENETECH_Y_AXIS=~/models/y_axis.pth
    export BENETECH_MARKER=~/models/marker.pth
    python scripts/run_eval.py --profile local --subset extracted

Kaggle (paths resolve from the kaggle preset when /kaggle/input exists):

    python scripts/run_eval.py --profile kaggle --decode greedy

Use --decode beam2 for the second generation config. Only beam width varies:
neither notebook set do_sample, so temperature/top_k/top_p were inert (Phase 0
finding E). Results are labelled as beam search, never as sampling tuning.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The CUDA allocator reads PYTORCH_CUDA_ALLOC_CONF when it initialises, so this
# must run before anything imports torch. Keep it above the torch-touching
# imports below -- moving it under them silently disables expandable segments.
from chart_extraction.runtime import configure_allocator  # noqa: E402

if "--no-expandable-segments" not in sys.argv:
    configure_allocator()

from chart_extraction.config import (  # noqa: E402
    BEAM2, GREEDY, LOCAL_GPU_BATCH_SIZES, PipelineConfig, RuntimeConfig,
)
from chart_extraction.data.images import ImageRef  # noqa: E402
from chart_extraction.eval.ground_truth import load_annotations, summarise  # noqa: E402
from chart_extraction.eval.harness import evaluate  # noqa: E402
from chart_extraction.eval.results import append_result, format_report  # noqa: E402
from chart_extraction.paths import (  # noqa: E402
    describe, explain_unresolved, resolve_paths,
)
from chart_extraction.runtime import OomPolicy  # noqa: E402
from chart_extraction.stages import (  # noqa: E402
    MODE_DONUT_ONLY, detect_stages, resolve_mode,
)

DECODE_CONFIGS = {"greedy": GREEDY, "beam2": BEAM2}
#: Without these nothing can run at all.
REQUIRED_PATHS = ("data_root", "donut_dir")
#: Needed only for the full pipeline; absent means a donut_only run.
DETECTION_PATHS = ("x_axis_model", "y_axis_model", "marker_model")
ALL_PATHS = REQUIRED_PATHS + DETECTION_PATHS


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    paths = parser.add_argument_group("paths (CLI > env var > config file > preset)")
    paths.add_argument("--data-root", type=Path, default=None,
                       help="competition root containing train/images and train/annotations "
                            "[$BENETECH_DATA_ROOT]")
    paths.add_argument("--donut-dir", type=Path, default=None, help="[$BENETECH_DONUT_DIR]")
    paths.add_argument("--x-axis-model", type=Path, default=None, help="[$BENETECH_X_AXIS]")
    paths.add_argument("--y-axis-model", type=Path, default=None, help="[$BENETECH_Y_AXIS]")
    paths.add_argument("--marker-model", type=Path, default=None, help="[$BENETECH_MARKER]")
    paths.add_argument("--results-dir", type=Path, default=None, help="[$BENETECH_RESULTS_DIR]")
    paths.add_argument("--paths-config", type=Path, default=None,
                       help="JSON file of path overrides")
    paths.add_argument("--preset", choices=["kaggle"], default=None,
                       help="force a path preset instead of auto-detecting")

    split = parser.add_argument_group("split")
    split.add_argument("--subset", choices=["extracted", "generated", "both"],
                       default="both",
                       help="which source stratum to evaluate (default: both). "
                            "extracted is the headline slice")
    split.add_argument("--fraction", type=float, default=0.05,
                       help="fraction of the generated stratum to hold out")
    split.add_argument("--limit", type=int, default=None,
                       help="cap the split size for smoke runs; refuses to write results")

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--profile", choices=["local", "kaggle", "cpu"], default="kaggle",
                         help="local: fp16 + batch size 1. kaggle: fp32 + larger batches")
    runtime.add_argument("--device", default=None, help="overrides the profile's device")
    runtime.add_argument("--precision", choices=["fp32", "fp16"], default=None,
                         help="overrides the profile's precision")
    runtime.add_argument("--batch-size", type=int, default=None,
                         help="apply one batch size to every stage")
    runtime.add_argument("--num-workers", type=int, default=None)
    runtime.add_argument("--oom-retry-scales", default=None,
                         help="comma-separated Donut resolution scales to retry after an "
                              "OOM, e.g. '0.75,0.5'. Empty string disables rescaling")
    runtime.add_argument("--no-expandable-segments", action="store_true",
                         help="do not set PYTORCH_CUDA_ALLOC_CONF")

    parser.add_argument("--mode", choices=["auto", "full", "donut_only"], default="auto",
                        help="auto (default) runs the full pipeline when the axis and "
                             "marker checkpoints are present and falls back to Donut-only "
                             "when they are not. full errors out rather than silently "
                             "downgrading")
    parser.add_argument("--decode", choices=sorted(DECODE_CONFIGS), default="greedy",
                        help="Donut decoding strategy. Only beam width varies.")
    parser.add_argument("--axis-label-source", default="donut_series",
                        help="AxisLabelSource implementation (Phase 3 registers more)")
    parser.add_argument("--run-id", default=None)
    return parser.parse_args(argv)


def build_runtime(args) -> RuntimeConfig:
    base = {
        "local": RuntimeConfig.local_gpu(),
        "kaggle": RuntimeConfig(),
        "cpu": RuntimeConfig.cpu(),
    }[args.profile]

    scales = base.oom_retry_scales
    enabled = base.oom_retry_enabled
    if args.oom_retry_scales is not None:
        raw = args.oom_retry_scales.strip()
        if not raw:
            scales, enabled = (), False
        else:
            scales = tuple(float(part) for part in raw.split(","))

    return RuntimeConfig(
        device=args.device or base.device,
        precision=args.precision or base.precision,
        oom_retry_scales=scales,
        oom_retry_enabled=enabled,
        expandable_segments=not args.no_expandable_segments,
    )


def build_pipeline_config(args, resolved, runtime, mode="full") -> PipelineConfig:
    batch_sizes = dict(LOCAL_GPU_BATCH_SIZES) if args.profile == "local" else {}
    if args.batch_size is not None:
        batch_sizes = {
            "donut_batch_size": args.batch_size,
            "axis_batch_size": args.batch_size,
            "marker_batch_size": args.batch_size,
        }
    if args.num_workers is not None:
        batch_sizes["num_workers"] = args.num_workers

    return PipelineConfig(
        image_dir=resolved.image_dir,
        donut_model_dir=resolved.donut_dir,
        x_axis_model_path=resolved.x_axis_model,
        y_axis_model_path=resolved.y_axis_model,
        marker_model_path=resolved.marker_model,
        generation=DECODE_CONFIGS[args.decode],
        axis_label_source=args.axis_label_source,
        mode=mode,
        **batch_sizes,
    )


def select_subset(image_ids, annotations, subset: str) -> list[str]:
    if subset == "both":
        return list(image_ids)
    return [i for i in image_ids if annotations[i].source == subset]


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    log = logging.getLogger("run_eval")

    resolved = resolve_paths(
        overrides={
            "data_root": args.data_root,
            "donut_dir": args.donut_dir,
            "x_axis_model": args.x_axis_model,
            "y_axis_model": args.y_axis_model,
            "marker_model": args.marker_model,
            "results_dir": args.results_dir,
        },
        config_path=args.paths_config,
        preset=args.preset,
    )
    print("paths:")
    print(describe(resolved, ALL_PATHS))
    print()

    unresolved = resolved.missing(REQUIRED_PATHS)
    if unresolved:
        log.error("could not resolve %d required path(s):", len(unresolved))
        for name in unresolved:
            log.error("  %s", explain_unresolved(name))
        return 2

    absent = resolved.not_on_disk(REQUIRED_PATHS)
    if absent:
        for name, path in absent:
            log.error("%s resolved to a path that does not exist: %s", name, path)
        return 2

    availability = detect_stages(resolved)
    try:
        mode = resolve_mode(args.mode, availability)
    except ValueError as exc:
        log.error("%s", exc)
        return 2

    print(availability.describe())
    print()
    if mode == MODE_DONUT_ONLY:
        log.warning(
            "running DONUT-ONLY: %s unavailable. Donut's generated series is "
            "used directly, no detection stage runs, and the resulting score is "
            "not comparable with a full-pipeline score.",
            " and ".join(availability.skipped),
        )

    runtime = build_runtime(args)
    log.info("runtime: %s", runtime)

    annotation_dir = resolved.annotation_dir
    image_dir = resolved.image_dir
    for path in (image_dir, annotation_dir):
        if not path.is_dir():
            log.error("missing required directory: %s", path)
            return 2

    log.info("loading annotations from %s", annotation_dir)
    annotations = load_annotations(annotation_dir)
    log.info("loaded %d annotations", len(annotations))
    log.info("dataset composition:\n%s", summarise(annotations.values()).to_string(index=False))

    from chart_extraction.eval.splits import build_validation_split

    split = build_validation_split(annotations, fraction=args.fraction)
    image_ids = select_subset(split.image_ids, annotations, args.subset)
    if not image_ids:
        log.error("subset %r selected 0 images from the split", args.subset)
        return 2
    log.info("subset=%s -> %d images", args.subset, len(image_ids))

    if args.limit:
        image_ids = image_ids[: args.limit]
        log.warning(
            "SMOKE RUN: capped at %d images -- not a reportable number, and not "
            "written to the results file", len(image_ids),
        )

    refs = [ImageRef(image_id=i, path=image_dir / f"{i}.jpg") for i in image_ids]
    missing = [r.image_id for r in refs if not r.path.exists()]
    if missing:
        log.error("%d split images missing on disk, e.g. %s", len(missing), missing[:3])
        return 2

    split_annotations = {i: annotations[i] for i in image_ids}
    config = build_pipeline_config(args, resolved, runtime, mode)

    from chart_extraction.eval.runner import (
        describe_device, load_all_models, peak_memory_mb, run_stages,
    )

    models, sizes = load_all_models(config, runtime)
    donut_predictions, axis_ticks, marker_detections, timings, policy = run_stages(
        refs, models, config, runtime,
        oom_policy=OomPolicy(
            enabled=runtime.oom_retry_enabled, retry_scales=runtime.oom_retry_scales
        ),
    )

    runtime_info = describe_device(runtime)
    runtime_info["peak_memory_mb"] = peak_memory_mb(runtime.device)
    runtime_info["subset"] = args.subset
    runtime_info["limit"] = args.limit
    runtime_info["allocator"] = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    runtime_info["mode"] = mode

    result = evaluate(
        refs=refs,
        annotations=split_annotations,
        donut_predictions=donut_predictions,
        axis_ticks=axis_ticks,
        marker_detections=marker_detections,
        timings=timings,
        models=sizes,
        config=config,
        split=split,
        run_id=args.run_id,
        runtime=runtime_info,
        oom_policy=policy,
        stages=availability.as_dict() | {"mode": mode},
    )

    print()
    print(format_report(result))
    print()

    for warning in result.warnings:
        if warning["level"] == "error":
            log.error("SANITY [%s] %s", warning["code"], warning["message"])
        elif warning["level"] == "warning":
            log.warning("SANITY [%s] %s", warning["code"], warning["message"])

    if args.limit:
        log.warning("smoke run -- NOT written to the results file")
    else:
        results_dir = resolved.results_dir or (REPO_ROOT / "results")
        written = append_result(result, results_dir)
        for label, path in written.items():
            log.info("wrote %s -> %s", label, path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
