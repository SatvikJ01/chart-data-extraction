# Chart Data Extraction

Converts chart images (bar, line, scatter, dot) into structured data series,
built on the Kaggle competition *Benetech — Making Graphs Accessible*. This
repository is an audit, correction, instrumentation and evaluation of an
inference pipeline that originally came from public Kaggle notebooks.

## Provenance

**The pipeline did not start here.** It began as two public Kaggle notebooks,
which are other authors' work and are **not redistributed in this repository**:

| Component | Source | Author |
|---|---|---|
| Donut inference | [Tuned Donut](https://www.kaggle.com/code/cody11null/tuned-donut) | Cody_Null |
| Donut training / checkpoint | [🍩 donut-train [benetech]](https://www.kaggle.com/code/nbroad/donut-train-benetech) | nbroad |
| 0.44 reference score | [🍩 donut-infer (LB 0.44) [benetech]](https://www.kaggle.com/code/nbroad/donut-infer-lb-0-44-benetech) | nbroad |
| Multi-stage extension | a derivative adding an axis-regression CNN and a Faster R-CNN marker detector | unidentified |

The fine-tuning design follows the *strategy* of the 2nd-place solution,
[rbiswasfc/benetech-mga](https://github.com/rbiswasfc/benetech-mga) — domain
adaptation, then specialization on oversampled real data. It is not a
reproduction: that solution uses `matcha-base` with per-chart-type models, and
its hyperparameters are not published.

**What is mine** is everything in `chart_extraction/`, `scripts/`, `tests/` and
`docs/`: the bug audit, the modular rewrite, the scorer, the validation
splits, the evaluation harness, the error taxonomy, the runtime work, and the
fine-tuning code. The original notebooks contained no evaluation code at all —
every number below comes from code in this repository.

## Results

Donut-only mode (the detection-stage checkpoints are unavailable — see
Limitations). Greedy decoding, fp16, batch size 8. Full records in
[`results/`](results/).

| Slice | Images | Score | Chart-type accuracy | ms/image |
|---|---:|---:|---:|---:|
| **extracted** (real) | 1118 | **0.5446** | 0.9830 | 526 |
| **generated** (synthetic) | 400 of 2988, seed 42 | **0.9130** ±0.0185 | 1.0000 | 446 |

Per chart type:

| Chart type | Extracted | n | Generated | n |
|---|---:|---:|---:|---:|
| vertical_bar | 0.6926 | 914 | 0.9665 | 250 |
| line | 0.6165 | 846 | 0.9507 | 334 |
| scatter | **0.1096** | 330 | 0.7260 | 150 |
| horizontal_bar | **0.1854** | 146 | — | 0 |
| dot | — | 0 | 0.9449 | 66 |

Instance counts are per (image, axis) pair — the competition's scoring
granularity — so they are twice the image count. The ±0.0185 is a 95% interval
clustered by image; the extracted runs predate that instrumentation and carry no
interval.

## Findings

**1. A ~0.37 synthetic-to-real gap.** 0.9130 on synthetic against 0.5446 on real
extracted charts. Two runs of an identical configuration differing only in batch
size span 0.0009, so the gap is orders of magnitude larger than the measurement
noise floor. This
is the distribution shift the competition was known for — models that looked
strong on training-like data did not transfer — reproduced here with local
measurement rather than inferred from the leaderboard.

**2. Scatter and horizontal_bar are the failure modes, not classification.**
Chart-type accuracy on the extracted slice is 0.9830, so the model almost always
knows what it is looking at. Scatter scores 0.1096 and horizontal_bar 0.1854
against 0.69 and 0.62 for bar and line. Scatter also drops furthest between
distributions (0.9449 → 0.1096). The failure is series extraction, not
recognition.

**3. Beam search buys nothing measurable.** Beam search (`num_beams=2`) scored
0.5421 against 0.5446 for greedy — no gain, marginally worse — and dropped
chart-type accuracy from 0.9830 to 0.9776. It ran at 1463 ms/image against 526,
though those runs used different batch sizes (4 vs 8), so the decode-only cost
is below that 2.8× ratio. What is controlled is the score: there is no gain to
pay for.

Related: the original notebooks set `temperature`, `top_k` and `top_p` but never
set `do_sample`, which defaults to `False`. Those three parameters were inert.
The only real difference between the two published configurations was beam
width.

**4. The dataset itself is asymmetric by chart type.** Across the validation
split:

| | dot | horizontal_bar |
|---|---:|---:|
| extracted (all 1118) | **0** | 73 |
| generated (2988 sampled) | 250 | **0** |

There is no real `dot` chart anywhere in the extracted data, and no synthetic
`horizontal_bar` in 2988 sampled generated images. So neither type can be
evaluated across the distribution shift at all, and any model trained on
synthetic data has never seen a horizontal bar chart. The extracted figure is
exact — all 1118 extracted images are in the split. The generated figure is from
a 5% sample, so it bounds the true rate near zero rather than proving it.

## Limitations

**Base-checkpoint leakage.** The Donut checkpoint was fine-tuned on `train/`
with no recorded train/validation partition, so validation images may have been
in its training data. Every score above is therefore optimistic, and the amount
cannot be recovered after the fact. The fine-tuning experiment
(`docs/FINETUNE.md`) builds a properly held-out split to address this, but only
for the specialization phase — the base checkpoint's history still applies.

**Donut-only mode.** The axis-regression CNN and Faster R-CNN marker detector
checkpoints are not obtainable, so the detection stages cannot run. All results
above use Donut's generated series directly. The multi-stage decoding path is
implemented and tested but unmeasured, and these scores must not be compared
with a full-pipeline score.

**Small extracted samples.** Horizontal bar has 73 extracted images and scatter
165. Per-type scores at those sizes are indicative, not precise, and the
extracted runs predate the confidence-interval instrumentation.

**Synthetic slice is a sample.** The generated figure is 400 images of 2988,
stratified and seeded; reproduce it exactly with `--sample 400 --seed 42`.

**Nothing here beats the leaderboard.** The 0.44 reference is a hidden-test-set
score. Ours is a held-out slice of `train/` that is both easier and possibly
leaked. Scoring above it is expected and is not evidence of quality.

## Reproducing

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Paths resolve from CLI flag → `BENETECH_*` env var → `chart_extraction.paths.json`
→ preset, so the same commands run locally and on Kaggle. Nothing is hardcoded.

```bash
export BENETECH_DATA_ROOT=~/data/benetech-making-graphs-accessible
export BENETECH_DONUT_DIR=~/models/benetech-donut
```

The competition data comes from
`kaggle competitions download -c benetech-making-graphs-accessible` (accept the
rules first). The Donut checkpoint is nbroad's, linked above.

```bash
# the two results rows above
python scripts/run_eval.py --profile local --subset extracted
python scripts/run_eval.py --profile local --subset generated --sample 400 --seed 42

# beam search comparison
python scripts/run_eval.py --profile local --subset extracted --decode beam2

# specialization fine-tune: measure runtime first, then train + evaluate
python scripts/train_donut.py --benchmark-only
python scripts/kaggle_finetune.py
```

`--sample N --seed S` is reportable and written to `results/`; `--limit N` is a
throwaway smoke option that refuses to write. Results append — nothing is
rewritten in place.

```bash
python -m pytest -q
```

## Layout

```
chart_extraction/
  config.py       PipelineConfig, GenerationConfig, RuntimeConfig
  data/           id-keyed image discovery, submission assembly
  donut/          token schema, numeric repair, parsing, generation
  axis/           tick CNN, calibration, AxisLabelSource seam
  markers/        Faster R-CNN wrapper, box geometry
  decoding/       per-chart-type decoders + registry
  eval/           metric, ground truth, splits, harness, taxonomy, results
  train/          specialization fine-tune: serialization, split, dataset
  paths.py        CLI > env > config-file > preset resolution
  runtime.py      allocator, precision, OOM recovery
scripts/          evaluation, training, Kaggle entrypoints
docs/             PHASE0_AUDIT.md (12 findings), FINETUNE.md
results/          append-only run records and ablation table
```

## Engineering notes

Twelve defects were found in the original notebooks; all are recorded in
[`docs/PHASE0_AUDIT.md`](docs/PHASE0_AUDIT.md) with runnable reproductions,
split into **active** (was corrupting output) and **latent** (a real contract
violation that was not). Two of the six originally suspected turned out to be
latent, and those are excluded from any before/after claim, because a latent bug
cannot have changed a score.

One design error is **deliberately preserved**: the axis calibration is fed
Donut's predicted data series as if it were axis tick labels. It sits behind a
config-selectable seam so a corrected implementation can be swapped in and
measured as a single variable, rather than being quietly fixed and losing the
comparison.

305 tests. Every result carries its own caveats into the record — leakage,
sampling error, degraded images, skipped stages — so a number cannot be lifted
out of `results/` without the context that qualifies it.

## Licence

[MIT](LICENSE). Applies to the code in this repository. The Kaggle notebooks it
derives from, the competition data, and the model checkpoints are covered by
their own terms.
