#!/usr/bin/env python3
"""Specialization fine-tune of the Donut checkpoint on oversampled extracted data.

    python scripts/train_donut.py --profile local --benchmark-only
    python scripts/train_donut.py --output-dir runs/ft-s1234

--benchmark-only measures throughput on this machine over a dozen real steps
and projects the full runtime, WITHOUT committing to the run. Do that first:
the projection comes from the hardware rather than from an estimate.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chart_extraction.runtime import configure_allocator  # noqa: E402

configure_allocator()

from chart_extraction.eval.ground_truth import load_annotations, summarise  # noqa: E402
from chart_extraction.eval.splits import stratified_sample  # noqa: E402
from chart_extraction.paths import describe, resolve_paths  # noqa: E402
from chart_extraction.train.dataset import DonutFineTuneDataset, build_rows  # noqa: E402
from chart_extraction.train.finetune import (  # noqa: E402
    TrainConfig, benchmark_throughput, prepare_for_training, project_runtime, train,
)
from chart_extraction.train.splits import build_extracted_split, save_split  # noqa: E402

REQUIRED_PATHS = ("data_root", "donut_dir")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--donut-dir", type=Path, default=None,
                   help="base checkpoint to fine-tune from")
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "runs" / "finetune")

    p.add_argument("--split-seed", type=int, default=1234,
                   help="seed for the 60/40 extracted partition. RECORDED with the run")
    p.add_argument("--train-fraction", type=float, default=0.6)
    p.add_argument("--val-fraction", type=float, default=0.1,
                   help="fraction of the TRAIN side reserved for loss monitoring and "
                        "checkpoint selection. Never the held-out 40%%")

    p.add_argument("--oversample", type=int, default=6,
                   help="how many times each extracted training image is repeated")
    p.add_argument("--n-generated", type=int, default=2000,
                   help="generated images mixed in, stratified by chart type")

    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-hours", type=float, default=2.0,
                   help="hard wall-clock ceiling; stops cleanly and still saves")
    p.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="cuda:0")

    p.add_argument("--benchmark-only", action="store_true",
                   help="measure throughput, project runtime, then exit without training")
    p.add_argument("--benchmark-steps", type=int, default=12)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    log = logging.getLogger("train_donut")

    resolved = resolve_paths(
        overrides={"data_root": args.data_root, "donut_dir": args.donut_dir}
    )
    print("paths:")
    print(describe(resolved, REQUIRED_PATHS))
    if resolved.missing(REQUIRED_PATHS) or resolved.not_on_disk(REQUIRED_PATHS):
        log.error("resolve --data-root and --donut-dir (or set BENETECH_* env vars)")
        return 2

    log.info("loading annotations")
    annotations = load_annotations(resolved.annotation_dir)
    log.info("dataset composition:\n%s", summarise(annotations.values()).to_string(index=False))

    split = build_extracted_split(
        annotations,
        train_fraction=args.train_fraction,
        val_fraction_of_train=args.val_fraction,
        seed=args.split_seed,
    )
    split.assert_disjoint()
    output_dir = Path(args.output_dir)
    split_path = save_split(split, output_dir / "extracted_split.json")
    log.info(
        "extracted split (seed %d): train %d, val %d, HELD OUT %d -> %s",
        args.split_seed, len(split.train_ids), len(split.val_ids),
        len(split.holdout_ids), split_path,
    )

    generated_ids = [i for i, a in annotations.items() if a.source == "generated"]
    sampled_generated, generated_record = stratified_sample(
        generated_ids, annotations, min(args.n_generated, len(generated_ids)),
        seed=args.split_seed,
    )

    from transformers import DonutProcessor, VisionEncoderDecoderModel

    log.info("loading base checkpoint from %s", resolved.donut_dir)
    processor = DonutProcessor.from_pretrained(resolved.donut_dir)
    model = VisionEncoderDecoderModel.from_pretrained(resolved.donut_dir)

    config = TrainConfig(
        epochs=args.epochs, batch_size=args.batch_size, grad_accum=args.grad_accum,
        learning_rate=args.lr, max_hours=args.max_hours, precision=args.precision,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        num_workers=args.num_workers, seed=args.split_seed,
    )

    rows, recipe = build_rows(
        split.train_ids, sampled_generated, annotations, processor.tokenizer,
        oversample=args.oversample, max_target_tokens=config.max_target_tokens,
    )
    log.info("training rows: %s", json.dumps(recipe.as_dict()))

    # A held-out image reaching the optimiser would invalidate the entire
    # experiment, so it is checked against the actual row list, not assumed.
    holdout = set(split.holdout_ids)
    leaked = holdout & set(rows)
    if leaked:
        log.error("ABORT: %d held-out ids in the training rows, e.g. %s",
                  len(leaked), sorted(leaked)[:3])
        return 3
    log.info("verified: 0 of %d held-out ids appear in the %d training rows",
             len(holdout), len(rows))

    model = prepare_for_training(model, config, args.device)

    bench_set = DonutFineTuneDataset(
        rows[: max(64, args.benchmark_steps * args.batch_size * 2)],
        annotations, resolved.image_dir, processor,
        max_target_tokens=config.max_target_tokens, augment=config.augment,
    )
    rate = benchmark_throughput(model, bench_set, config, args.device, args.benchmark_steps)
    projection = project_runtime(rate, len(rows), args.epochs)
    log.info("projected runtime: %s", json.dumps(projection))

    print()
    print("=" * 68)
    print(f"  measured throughput : {projection.get('samples_per_s')} samples/s")
    print(f"  rows per epoch      : {len(rows)}")
    print(f"  projected per epoch : {projection.get('epoch_hms')}")
    print(f"  projected total     : {projection.get('total_hms')} "
          f"for {args.epochs} epochs")
    print(f"  wall-clock ceiling  : {args.max_hours} h (stops cleanly and saves)")
    print("=" * 68)
    print()

    if args.benchmark_only:
        (output_dir / "benchmark.json").write_text(
            json.dumps({"projection": projection, "recipe": recipe.as_dict(),
                        "rows": len(rows)}, indent=2, sort_keys=True)
        )
        log.info("benchmark only -- exiting without training")
        return 0

    run = train(
        model=model, processor=processor,
        train_ids=rows, val_ids=list(split.val_ids), annotations=annotations,
        image_dir=resolved.image_dir, config=config, device=args.device,
        output_dir=output_dir,
        recipe=recipe.as_dict() | {"generated_sample": generated_record},
    )
    run.throughput_samples_per_s = projection.get("samples_per_s")
    (output_dir / "training_run.json").write_text(
        json.dumps(run.as_dict(), indent=2, sort_keys=True)
    )

    print()
    print("epoch   train_loss   val_loss    time")
    for record in run.epochs:
        print(f"  {record.epoch:<5} {record.train_loss:<12.4f} "
              f"{record.val_loss:<11.4f} {record.seconds:.0f}s")
    print(f"\nbest epoch {run.best_epoch} (val_loss {run.best_val_loss})")
    print(f"checkpoints: {output_dir / 'best'} and {output_dir / 'last'}")
    print(f"held-out ids for evaluation: {split_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
