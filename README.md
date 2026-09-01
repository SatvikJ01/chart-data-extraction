# Chart Data Extraction

Converts chart images (bar, line, scatter, dot) into structured tabular data
series. Built on the Kaggle competition *Benetech — Making Graphs Accessible*.

**Status: Phases 0–2 built, not yet run.** The pipeline is a correct, modular
baseline and the evaluation harness is complete and tested. **There are still no
accuracy numbers**, because the checkpoints and competition data live in Kaggle
datasets that are not on the development machine. `results/` is empty until
`scripts/kaggle_eval.py` runs there. Nothing in this repo should be quoted as a
result yet.

## Pipeline

```
image
  ├─ Donut (Swin encoder + BART decoder, OCR-free)  → chart type, x series, y series
  ├─ Axis CNN (ResNet-18, two heads)                → x/y tick pixel positions
  └─ Faster R-CNN (MobileNetV3-L FPN)               → data-point marker boxes
                          │
                    AxisLabelSource ──→ AxisCalibration (pixel → value)
                          │
                    per-chart-type decoder → y data series
                          │
                       submission
```

The x series comes from Donut for every chart type; only the y series is
re-derived from the CV stages. That is the original design, preserved.

## Layout

```
chart_extraction/
  config.py            PipelineConfig, GenerationConfig
  data/                image discovery (id-keyed), submission assembly
  donut/               token schema, numeric repair, parsing, generation
  axis/                tick CNN, calibration, AxisLabelSource seam
  markers/             Faster R-CNN wrapper, box geometry
  decoding/            per-chart-type decoders + registry
  pipeline.py          stage orchestration, failure taxonomy
  eval/                metric, ground truth, splits, harness, taxonomy, results
  paths.py             CLI > env > config-file > preset path resolution
  runtime.py           allocator config, precision, OOM recovery policy
scripts/run_eval.py    one evaluation pass over a split (local or Kaggle)
scripts/kaggle_eval.py Kaggle wrapper: installs deps, runs both decode configs
docs/PHASE0_AUDIT.md   what was broken, what was fixed, what was preserved
notebooks/             the original Kaggle notebooks, unmodified
results/               ablation table + per-run JSON (empty until a real run)
```

## Running

```bash
pip install -r requirements.txt
```

```bash
python -m pytest tests/ -q
```

The full pipeline needs the checkpoints, which live in Kaggle datasets and are
not in this repo (`benetech-donut`, `x-axis-model-10`, `y-axis-model-10`,
`marker-model`). Point `PipelineConfig` at them and call `run_pipeline`.

## What Phase 0 changed

Twelve defects were found across the two notebooks. Six were in the original
brief; two of those were wrong as stated. Full detail in
[docs/PHASE0_AUDIT.md](docs/PHASE0_AUDIT.md).

The findings are split into **ACTIVE** (was corrupting output) and **LATENT** (a
real contract violation that was not corrupting output). This matters for
Phase 1: a latent bug cannot have changed any score, so only the active fixes
may be credited with a Phase 0 → Phase 1 delta.

| Finding | Status | Fixed? |
|---|---|---|
| 2 — line charts hardcoded to `0.0;0.0` | ACTIVE | yes |
| 4 — `isdigit()` zeroed negative/decimal ticks | ACTIVE | yes |
| 5 — numeric repair gutted, salvageable values → 0 | ACTIVE | yes |
| A — tick list sorted in place, desyncing labels | ACTIVE | yes |
| C — negative-index wraparound in axis scaling | ACTIVE | yes |
| D — line decoder read a global, no score filter | ACTIVE (dead code) | yes |
| F — `horizontal_bar` had no decode branch | ACTIVE | registered, decode deferred |
| 6 — chained pandas assignment, bare `except:` | ACTIVE | yes |
| 1 — global `scores` instead of `self.scores` | **LATENT** | yes, excluded from delta |
| 3 — positional joins across rebuilt ID lists | **LATENT** | yes, excluded from delta |
| E — sampling parameters were inert | ACTIVE (framing) | made explicit |
| B — axis labels fed the wrong quantity | ACTIVE | **deliberately preserved** |

### Finding B is preserved on purpose

The axis calibration is fed Donut's predicted y *data series* as if it were the
y-axis *tick labels*. Those are different quantities, which makes the numeric
branch circular. Nothing in the pipeline reads axis tick text at all.

It is preserved bit-for-bit behind the `AxisLabelSource` seam, because Phase 0's
job is a faithful refactor rather than a better model. Fixing it here would
leave Phase 3's OCR component with nothing to demonstrate and would cost the
ablation table its cleanest row. Phase 3 registers a second implementation and
flips one config key.

### Finding E affects how Phase 1 can be described

Neither notebook set `do_sample`, so `temperature`, `top_k` and `top_p` were
inert in both. The two generation configs differ **only** in `num_beams` (1 vs
2) — greedy vs 2-beam search. The comparison is still worth running; it just
cannot be called temperature or nucleus-sampling tuning.

## Evaluation (Phases 1–2)

The competition metric is asymmetric by design, and knowing why is the point:

- **Categorical** series → summed Levenshtein distance normalised by total
  ground-truth string length. Nearly-right labels earn partial credit.
- **Numeric** series → RMSE normalised against the RMSE of predicting the
  ground-truth mean. A prediction is measured against the trivial "guess the
  average" baseline; a zero-variance series cannot earn partial credit.

Both squash through `2 - 2/(1 + exp(-x))`. Two gates precede everything and
award exactly 0: wrong chart type, and wrong series length. The length gate is
why the old `0;0` placeholder scored nothing rather than "a little".

One pass over a split emits every number together — overall, per-chart-type,
per-source, latency, model sizes and the error taxonomy — because they share a
single inference run.

```bash
python scripts/run_eval.py --profile local --subset extracted --decode greedy
```

### Pipeline modes

The pipeline needs three separate checkpoints. When the axis CNN or the marker
detector is unavailable it degrades to **Donut-only** rather than crashing:
Donut's generated x and y series are used directly, with no axis calibration,
no marker detection and no per-chart-type decoder. That is exactly what
`tuned-donut.ipynb` did.

`--mode auto` (default) picks the richest mode the available checkpoints
support. `--mode full` **errors out** rather than silently downgrading — a
caller who asked for the full pipeline should be told they are not getting it.
`--mode donut_only` forces the reduced path even when everything is present.

A Donut-only run is a **different system**, not a degraded full run, so the mode
is recorded everywhere the score is: a `mode` column in the ablation table, a
`stages_skipped` list in the JSON, a banner at the top of the console report,
and a caveat on the row. In Donut-only mode the `marker_miss` and
`axis_misestimation` taxonomy categories are **omitted rather than reported as
zero** — a zero there would read as a clean detection pass instead of an absent
one.

### Sanity check against the reference score

The reported leaderboard score for this Donut checkpoint is **0.44**, measured
Donut-only on the hidden test set. Runs are compared against it, but the
comparison is deliberately **asymmetric**:

- Scoring **far below** 0.44 is flagged as an error, because none of the known
  differences explain it. That is the signal for a loading problem — weights not
  actually loaded, processor and tokenizer from different checkpoints, or a
  special-token schema the decoder no longer matches.
- Scoring **above** 0.44 is reported as information only, never as success. Up
  to two known effects push our number up, and **which ones apply depends on
  what the run actually scored**, not on what the split contained: a
  `--subset extracted` run scores no synthetic data, so the easier-distribution
  argument does not apply to it and only possible `train/` leakage does. Every
  message is built from the measured per-source composition of the scored
  instances. A higher number is expected and must not be reported as beating the
  leaderboard.

Result records report the composition **actually evaluated** under
`split.evaluated_composition`, with the split it was drawn from retained
separately as `split.source_split_composition` — a `--subset` run evaluates a
fraction of its split, so reporting the split's totals would overstate it.

### Configuration and paths

No path is hardcoded, so the same entrypoint runs locally and on Kaggle.
Resolution order, highest precedence first: **CLI flag → `BENETECH_*` env var →
`chart_extraction.paths.json` → preset**. A preset only applies when its
environment is actually present, so a local run can never silently resolve to a
Kaggle mount. An unresolved path is an error naming every mechanism that was
checked, rather than a default that fails later inside a model loader.

```bash
export BENETECH_DATA_ROOT=~/data/benetech
export BENETECH_DONUT_DIR=~/models/donut
export BENETECH_X_AXIS=~/models/x_axis.pth
export BENETECH_Y_AXIS=~/models/y_axis.pth
export BENETECH_MARKER=~/models/marker.pth
```

`--subset extracted|generated|both` picks the source stratum.

Two different ways to evaluate less than everything, and the distinction is
deliberate:

- **`--sample N --seed S` is reportable.** A deterministic stratified random
  subsample of N images, stratified on `(source, chart_type)` — chart types
  score very differently from one another, so an unstratified draw would move
  the aggregate by mix alone. The run is written to `results/` recorded with its
  size, seed and per-stratum counts, and appears in the ablation table as
  `N/P s=SEED` rather than `full`. Reproduce it exactly by passing the same N
  and seed.
- **`--limit N` is a throwaway.** It takes the first N images and refuses to
  write results at all. Use it to check a run starts, never for a number you
  intend to quote.

Every row carries a `+/-95%` column: the 95% confidence interval on the overall
score, **clustered by image**. The x and y instances of one image share a chart
type, a generation and a failure mode, so treating 2N instances as 2N
independent draws would understate the error by up to √2. Two rows differing by
less than their intervals are not distinguishable at their sample sizes.

### Progress

All three inference stages log progress on a fixed time interval —
`donut: 320/1118 (28.6%) 2.13 img/s elapsed 2m30s eta 6m20s` — so a slow run is
distinguishable from a hung one. It logs a complete line rather than using
tqdm's carriage-return rendering, which turns a captured Kaggle log or redirected
file into thousands of unreadable lines. If a pass raises, the reporter still
logs where it got to.

### Local single-GPU path

`--profile local` targets one small card: fp16, batch size 8 on every stage, and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set before torch is imported
so the allocator can grow segments in place instead of fragmenting.

Donut gets `.half()` rather than autocast, because autocast keeps fp32 master
weights resident and does nothing for the parameters that dominate its
footprint. The axis CNN and Faster R-CNN use autocast instead — torchvision's
detection models are fragile under hard fp16, and halving weights buys little
for models that small.

Batch size was originally 1 on the assumption that Donut's encoder activations
would dominate. A measured `donut_only` run peaked at **482 MB of 3771 MB** on an
RTX 2050 — about 13% utilisation — so that was far too conservative and cost
throughput for nothing. The default is now 8, chosen to sit well inside the
measured headroom rather than to saturate it; `--batch-size N` overrides it, and
the OOM fallback below means an over-large value degrades rather than fails.

**OOM recovery.** A batch that will not fit is retried image-by-image; an image
that still will not fit is retried at progressively lower Donut input resolution
(default ladder `0.75, 0.5`, tunable with `--oom-retry-scales`, disabled with
`--oom-retry-scales ''`). Every retry is logged *and counted into the result
file*, because an image processed at reduced resolution has a different
prediction — a run containing degraded images is not directly comparable with
one that has none, so the `oom` column and an extra caveat travel with the row.
An image that fits at no resolution gets an explicit `oom` failure mode rather
than a silent placeholder.

Reducing Donut's input size is not universally safe: a Swin encoder with
absolute position embeddings is tied to its trained resolution and will raise on
a different one. That is caught per-scale and treated as a failed attempt, so
such a checkpoint degrades to a recorded failure instead of taking down the run.

### The split, and why extracted is the headline

Benetech built the dataset from synthetic (`generated`) and real textbook
(`extracted`) charts. Training is overwhelmingly generated; the competition's
test set skewed far more toward extracted, and top teams scored 0.88 public
against 0.72 private. Local validation in this competition was notoriously
optimistic.

So the split is keyed on the annotation `source` field, **extracted is reported
as the headline**, generated is reported beside it, and the gap between them is
treated as a finding rather than noise.

### Leakage caveat, carried on every result row

The split is held out with respect to *future* training in this repo. It is
**not** guaranteed unseen by the checkpoints being evaluated — those were
fine-tuned elsewhere on `train/` with no recorded partition. Any validation
image may have been in their training data. Scores will be optimistic for these
checkpoints, on top of the synthetic-vs-extracted optimism. Both effects push
the same direction, and neither can be corrected after the fact — only stated.

### Decoding configs

`--decode greedy` and `--decode beam2` run through the identical harness and
split. Per Phase 0 finding E, these differ **only in beam width**; results are
labelled as beam search and never as temperature or nucleus-sampling tuning.

## Roadmap

- **Phase 0** — correct, modular baseline ✅ built
- **Phase 1** — metric, split, harness ✅ built, ⏳ not yet run
- **Phase 2** — error taxonomy ✅ built, ⏳ not yet run
- **Phase 3** — YOLOv8 vs Faster R-CNN, U-Net crop, Pytesseract axis OCR — each measured
- **Phase 4** — zero-shot Qwen2.5-VL benchmark on the same split and metric
- **Phase 5** — ablation table, README, latency profile

## Ground rule

Every claim that reaches a CV must trace to code that ran and produced a logged
number. A component that does not help gets reported as not helping.
