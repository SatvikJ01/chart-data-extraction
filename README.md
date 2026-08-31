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

`--subset extracted|generated|both` picks the source stratum; `--limit N` caps
the split for smoke runs and refuses to write to the results file, so a smoke
number cannot be mistaken for a reportable one.

### Local single-GPU path

`--profile local` targets one small card: fp16, batch size 1 on every stage, and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set before torch is imported
so the allocator can grow segments in place instead of fragmenting.

Donut gets `.half()` rather than autocast, because autocast keeps fp32 master
weights resident and does nothing for the parameters that dominate its
footprint. The axis CNN and Faster R-CNN use autocast instead — torchvision's
detection models are fragile under hard fp16, and halving weights buys little
for models that small.

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
