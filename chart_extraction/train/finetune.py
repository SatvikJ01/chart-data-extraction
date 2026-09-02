"""Specialization fine-tune of the Donut checkpoint on oversampled extracted data.

Follows the *strategy* of the 2nd-place solution (rbiswasfc/benetech-mga):
a second-stage fine-tune that starts from an already-trained checkpoint and
specializes it on real extracted charts, oversampled against synthetic ones.

It does NOT reproduce that solution. Theirs is built on ``google/matcha-base``
with per-chart-type models (separate scatter and non-scatter runs), and its
config files are not published in the repository README, so the hyperparameters
here are chosen for this checkpoint and this budget rather than copied.

Targets a Kaggle T4 (16GB): fp16 autocast with a gradient scaler, gradient
checkpointing, small batch with accumulation.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch.utils.data import DataLoader

from chart_extraction.progress import ProgressReporter, _format_duration
from chart_extraction.train.dataset import DonutFineTuneDataset, collate

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Hyperparameters. Defaults chosen for a T4 and a ~3 GPU-hour budget."""

    epochs: int = 3
    batch_size: int = 2
    grad_accum: int = 8
    # Low for a specialization phase: the base checkpoint has already converged,
    # and a large step would undo it rather than shift it toward extracted.
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    max_target_tokens: int = 512

    precision: str = "fp16"
    gradient_checkpointing: bool = True
    num_workers: int = 2
    augment: bool = True

    #: Hard wall-clock ceiling. Training stops cleanly at this point, saves, and
    #: still hands a usable checkpoint to evaluation. Without it a slower-than-
    #: expected machine silently eats the whole GPU budget before producing
    #: anything.
    max_hours: float = 2.0
    seed: int = 1234

    @property
    def effective_batch(self) -> int:
        return self.batch_size * self.grad_accum

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["effective_batch"] = self.effective_batch
        return payload


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    val_loss: float
    seconds: float
    samples: int
    stopped_early: bool = False


@dataclass
class TrainingRun:
    config: dict
    recipe: dict
    epochs: list = field(default_factory=list)
    best_epoch: int | None = None
    best_val_loss: float | None = None
    total_seconds: float = 0.0
    stopped_on_budget: bool = False
    throughput_samples_per_s: float | None = None

    def as_dict(self) -> dict:
        return {
            "config": self.config,
            "recipe": self.recipe,
            "epochs": [asdict(e) for e in self.epochs],
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "total_seconds": round(self.total_seconds, 1),
            "total_hms": _format_duration(self.total_seconds),
            "stopped_on_budget": self.stopped_on_budget,
            "throughput_samples_per_s": self.throughput_samples_per_s,
        }


def prepare_for_training(model, config: TrainConfig, device: str):
    """Move to device and enable the memory-saving settings.

    Weights stay fp32 while autocast handles the casting: this is training, not
    inference, and pure-fp16 master weights make the optimiser numerically
    unstable at these learning rates. Contrast prepare_donut_model(), which
    halves weights outright because inference has no optimiser state to corrupt.
    """
    model = model.to(device)
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # Checkpointing and the KV cache are mutually exclusive; transformers
        # warns and disables the cache anyway, so do it explicitly.
        model.config.use_cache = False
    model.train()
    return model


def _build_loader(dataset, config: TrainConfig, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        collate_fn=collate,
        pin_memory=True,
        drop_last=False,
    )


@torch.no_grad()
def evaluate_loss(model, loader, device: str, config: TrainConfig) -> float:
    """Mean teacher-forced loss over a loader."""
    model.eval()
    total, count = 0.0, 0
    autocast = torch.autocast(
        device_type="cuda" if str(device).startswith("cuda") else "cpu",
        dtype=torch.float16,
        enabled=(config.precision == "fp16" and str(device).startswith("cuda")),
    )
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with autocast:
            out = model(pixel_values=pixel_values, labels=labels)
        total += float(out.loss.detach()) * len(labels)
        count += len(labels)
    model.train()
    return total / count if count else float("nan")


def benchmark_throughput(
    model, dataset, config: TrainConfig, device: str, steps: int = 12
) -> float:
    """Measure samples/second on a handful of real steps.

    Runs before committing to a full run so the projected duration comes from
    this machine rather than from an estimate. Includes a few warm-up steps,
    since the first pass pays cuDNN autotuning and allocator growth.
    """
    loader = _build_loader(dataset, config, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scaler = torch.amp.GradScaler(enabled=_amp_enabled(config, device))
    autocast = _autocast(config, device)

    warmup = min(3, max(1, steps // 4))
    seen, started = 0, None

    for index, batch in enumerate(loader):
        if index >= steps:
            break
        if index == warmup:
            _sync(device)
            started = time.perf_counter()

        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with autocast:
            loss = model(pixel_values=pixel_values, labels=labels).loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if started is not None:
            seen += len(labels)

    _sync(device)
    elapsed = time.perf_counter() - started if started else float("nan")
    rate = seen / elapsed if elapsed and elapsed > 0 else float("nan")

    # Leave no optimiser state behind for the real run.
    optimizer.zero_grad(set_to_none=True)
    del optimizer, scaler
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    logger.info(
        "benchmark: %.2f samples/s over %d timed samples (batch %d)",
        rate, seen, config.batch_size,
    )
    return rate


def project_runtime(rate: float, n_rows: int, epochs: int) -> dict:
    """Project epoch and total training time from a measured rate."""
    if not rate or rate != rate or rate <= 0:
        return {"samples_per_s": None, "epoch_s": None, "total_s": None}
    epoch_s = n_rows / rate
    return {
        "samples_per_s": round(rate, 3),
        "epoch_s": round(epoch_s, 1),
        "epoch_hms": _format_duration(epoch_s),
        "total_s": round(epoch_s * epochs, 1),
        "total_hms": _format_duration(epoch_s * epochs),
    }


def _amp_enabled(config: TrainConfig, device: str) -> bool:
    return config.precision == "fp16" and str(device).startswith("cuda")


def _autocast(config: TrainConfig, device: str):
    return torch.autocast(
        device_type="cuda" if str(device).startswith("cuda") else "cpu",
        dtype=torch.float16,
        enabled=_amp_enabled(config, device),
    )


def _sync(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def train(
    model,
    processor,
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    annotations: Mapping,
    image_dir: Path | str,
    config: TrainConfig,
    device: str,
    output_dir: Path | str,
    recipe: dict | None = None,
) -> TrainingRun:
    """Run the fine-tune, saving the best and last checkpoints."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)

    train_set = DonutFineTuneDataset(
        train_ids, annotations, image_dir, processor,
        max_target_tokens=config.max_target_tokens, augment=config.augment,
    )
    val_set = DonutFineTuneDataset(
        val_ids, annotations, image_dir, processor,
        max_target_tokens=config.max_target_tokens, augment=False,
    )
    train_loader = _build_loader(train_set, config, shuffle=True)
    val_loader = _build_loader(val_set, config, shuffle=False)

    steps_per_epoch = math.ceil(len(train_loader) / config.grad_accum)
    total_steps = max(1, steps_per_epoch * config.epochs)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    warmup_steps = max(1, int(total_steps * config.warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler(enabled=_amp_enabled(config, device))
    autocast = _autocast(config, device)

    run = TrainingRun(config=config.as_dict(), recipe=recipe or {})
    budget_s = config.max_hours * 3600.0
    started = time.perf_counter()
    stop = False

    for epoch in range(1, config.epochs + 1):
        epoch_started = time.perf_counter()
        progress = ProgressReporter(
            len(train_set), f"train epoch {epoch}/{config.epochs}", log=logger
        ).start()
        running, seen = 0.0, 0
        optimizer.zero_grad(set_to_none=True)

        for index, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with autocast:
                loss = model(pixel_values=pixel_values, labels=labels).loss
            scaler.scale(loss / config.grad_accum).backward()

            if (index + 1) % config.grad_accum == 0 or (index + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            running += float(loss.detach()) * len(labels)
            seen += len(labels)
            progress.update(len(labels))

            if time.perf_counter() - started > budget_s:
                logger.warning(
                    "wall-clock budget of %.2f h reached mid-epoch %d; stopping "
                    "cleanly and saving. The run is shorter than configured -- "
                    "record it as such.", config.max_hours, epoch,
                )
                stop = True
                break

        progress.finish()
        train_loss = running / seen if seen else float("nan")
        val_loss = evaluate_loss(model, val_loader, device, config)
        elapsed = time.perf_counter() - epoch_started

        run.epochs.append(
            EpochRecord(epoch, round(train_loss, 6), round(val_loss, 6),
                        round(elapsed, 1), seen, stopped_early=stop)
        )
        logger.info(
            "epoch %d/%d  train_loss %.4f  val_loss %.4f  %s  (%d samples)",
            epoch, config.epochs, train_loss, val_loss,
            _format_duration(elapsed), seen,
        )

        if run.best_val_loss is None or val_loss < run.best_val_loss:
            run.best_val_loss = round(val_loss, 6)
            run.best_epoch = epoch
            _save(model, processor, output_dir / "best")
            logger.info("epoch %d is the new best (val_loss %.4f); saved", epoch, val_loss)

        _save(model, processor, output_dir / "last")

        if stop:
            run.stopped_on_budget = True
            break

    run.total_seconds = time.perf_counter() - started
    (output_dir / "training_run.json").write_text(
        json.dumps(run.as_dict(), indent=2, sort_keys=True)
    )
    logger.info(
        "training finished in %s; best epoch %s (val_loss %s)",
        _format_duration(run.total_seconds), run.best_epoch, run.best_val_loss,
    )
    return run


def _save(model, processor, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    processor.save_pretrained(path)
