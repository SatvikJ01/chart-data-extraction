# Specialization fine-tune

A second-stage fine-tune that starts from the existing Donut checkpoint and
specializes it on real `extracted` charts, oversampled against synthetic ones.

## Relationship to the 2nd-place solution

Follows the *strategy* of [rbiswasfc/benetech-mga](https://github.com/rbiswasfc/benetech-mga):
domain adaptation on synthetic data, then a **specialization phase** that
fine-tunes on real extracted plots with those plots over-sampled.

It is **not** a reproduction, and should not be described as one:

| | Their solution | This |
|---|---|---|
| backbone | `google/matcha-base` | the existing Donut checkpoint |
| models | separate scatter / non-scatter | one model |
| phase 1 | 36–50 h on A100/A6000 | not run; we start from an existing checkpoint |
| hyperparameters | in `conf/`, not published in the README | chosen here |

The oversampling ratio, learning rate, schedule and epoch count are **ours**.
Their README states extracted plots were over-sampled but not by how much.

## The partition, and what "leakage-free" does and does not mean

The 1118 extracted images are split **60/40** with a recorded seed, stratified
by chart type. The 60% is then subdivided again:

```
1118 extracted
├── 60%  671 ── train 605   → the optimiser sees only these
│            └── val    66  → loss curves and checkpoint selection
└── 40%  447 ── HELD OUT    → touched once, by the final evaluation
```

The inner validation slice exists because **selecting the best epoch on the
held-out 40% would leak it as surely as training on it** — the reported number
would be the best of N draws against that set rather than an honest estimate.

The partition is written to `extracted_split.json` with its seed and exact id
lists. The trainer checks the held-out ids against the actual training row list
and aborts if any appear, so the claim is verified rather than asserted.

**The honest limit.** The held-out 447 were never seen by *this fine-tune*. They
were **not** held out from the base checkpoint, whose own partition is
unrecorded. So the absolute score is still optimistic. What is defensible is the
**difference** between the baseline and the fine-tuned model scored on the same
447 images, since both inherit the same base-checkpoint history. That is why
`kaggle_finetune.py` scores both and appends two rows rather than one.

## Measured throughput

Forward+backward on the real checkpoint, 560×560, 512-token targets, fp16
autocast, measured on an RTX 2050:

| config | samples/s | peak (no optimizer states) |
|---|---:|---:|
| bs=1, gradient checkpointing | 2.34 | 2083 MB |
| bs=2, gradient checkpointing | 2.53 | 2698 MB |
| bs=1, no checkpointing | 3.10 | 2725 MB |

Two things follow. Batch scaling is weak (+8% from bs 1→2), so this is
compute-bound rather than launch-bound and a larger batch buys little.
Gradient checkpointing costs **24%** throughput to save 642 MB — worth it on a
4 GB card, a poor trade on a 16 GB T4. It is the default because it is the safe
setting; pass `--no-gradient-checkpointing` on a T4 to get that 24% back.

Add AdamW states (~2.1 GB for 261M fp32 params) for the real memory figure:
roughly 5.5 GB at bs=2 without checkpointing, comfortably inside a T4.

## Running it

Measure on the actual machine before committing the budget — this takes about
two minutes and replaces every estimate above with a real number:

```bash
python scripts/train_donut.py --benchmark-only
```

Then the full experiment (train, score baseline, score fine-tuned):

```bash
python scripts/kaggle_finetune.py
```

`--max-hours` is a hard wall-clock ceiling. Training stops cleanly at that
point, saves, and still hands a usable checkpoint to evaluation, so a
slower-than-expected machine cannot silently consume the whole GPU quota
without producing anything. A run that stopped early is recorded as such.

## Overfitting risk

At the defaults each of the 605 extracted training images is seen 18 times
(oversample 6 × 3 epochs). That is deliberate — specialization is the point —
but it is enough repetition to memorise. The per-epoch validation loss is the
guard, and the best-epoch checkpoint is what gets evaluated. If validation loss
turns up after epoch 1, lower `--oversample` or `--epochs` rather than ignoring
it.
