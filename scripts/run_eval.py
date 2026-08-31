#!/usr/bin/env python3
"""Run one evaluation pass over a held-out validation split.

Single pass, every Phase 1 and Phase 2 number emitted together.

    python scripts/run_eval.py \
        --data-root /kaggle/input/benetech-making-graphs-accessible \
        --donut-dir /kaggle/input/benetech-donut \
        --decode greedy

Use --decode beam2 for the second generation config. Only beam width varies:
neither notebook set do_sample, so temperature/top_k/top_p were inert (Phase 0
finding E). Results are labelled as beam search, never as sampling tuning.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chart_extraction.config import BEAM2, GREEDY, PipelineConfig  # noqa: E402
from chart_extraction.data.images import ImageRef  # noqa: E402
from chart_extraction.eval.ground_truth import load_annotations, summarise  # noqa: E402
from chart_extraction.eval.harness import evaluate  # noqa: E402
from chart_extraction.eval.results import append_result, format_report  # noqa: E402
from chart_extraction.eval.runner import load_all_models, run_stages  # noqa: E402
from chart_extraction.eval.splits import build_validation_split  # noqa: E402

DECODE_CONFIGS = {"greedy": GREEDY, "beam2": BEAM2}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="competition root containing train/images and train/annotations")
    parser.add_argument("--donut-dir", type=Path, required=True)
    parser.add_argument("--x-axis-model", type=Path,
                        default=Path("/kaggle/input/x-axis-model-10/model (1).pth"))
    parser.add_argument("--y-axis-model", type=Path,
                        default=Path("/kaggle/input/y-axis-model-10/Y_Point_generation_weights_1.0.pth"))
    parser.add_argument("--marker-model", type=Path,
                        default=Path("/kaggle/input/marker-model/Marker_weights.pth"))
    parser.add_argument("--decode", choices=sorted(DECODE_CONFIGS), default="greedy",
                        help="Donut decoding strategy. Only beam width varies.")
    parser.add_argument("--axis-label-source", default="donut_series",
                        help="AxisLabelSource implementation (Phase 3 registers more)")
    parser.add_argument("--fraction", type=float, default=0.05,
                        help="fraction of the generated stratum to hold out")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap the split size, for smoke runs")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="override the Donut batch size")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    log = logging.getLogger("run_eval")

    image_dir = args.data_root / "train" / "images"
    annotation_dir = args.data_root / "train" / "annotations"
    for path in (image_dir, annotation_dir):
        if not path.is_dir():
            log.error("missing required directory: %s", path)
            return 2

    log.info("loading annotations from %s (this reads the full train set once)", annotation_dir)
    annotations = load_annotations(annotation_dir)
    log.info("loaded %d annotations", len(annotations))
    log.info("dataset composition:\n%s", summarise(annotations.values()).to_string(index=False))

    split = build_validation_split(annotations, fraction=args.fraction)
    image_ids = list(split.image_ids)
    if args.limit:
        image_ids = image_ids[: args.limit]
        log.warning("SMOKE RUN: split capped at %d images -- not a reportable number", len(image_ids))
    log.info("validation split: %d images %s", len(image_ids), split.composition["by_source"])

    refs = [ImageRef(image_id=i, path=image_dir / f"{i}.jpg") for i in image_ids]
    missing = [r.image_id for r in refs if not r.path.exists()]
    if missing:
        log.error("%d split images missing on disk, e.g. %s", len(missing), missing[:3])
        return 2

    split_annotations = {i: annotations[i] for i in image_ids}

    config = PipelineConfig(
        image_dir=image_dir,
        donut_model_dir=args.donut_dir,
        x_axis_model_path=args.x_axis_model,
        y_axis_model_path=args.y_axis_model,
        marker_model_path=args.marker_model,
        generation=DECODE_CONFIGS[args.decode],
        axis_label_source=args.axis_label_source,
        **({"donut_batch_size": args.batch_size} if args.batch_size else {}),
    )

    models, sizes = load_all_models(config, args.device)
    donut_predictions, axis_ticks, marker_detections, timings = run_stages(
        refs, models, config, args.device
    )

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
    )

    print()
    print(format_report(result))
    print()

    if args.limit:
        log.warning("smoke run -- NOT written to the results file")
    else:
        written = append_result(result, args.results_dir)
        for label, path in written.items():
            log.info("wrote %s -> %s", label, path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
