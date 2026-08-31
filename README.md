# Chart Data Extraction

Converts chart images (bar, line, scatter, dot) into structured tabular data
series. Built on the Kaggle competition *Benetech — Making Graphs Accessible*.

**Status: Phase 0 complete.** The pipeline is a correct, modular baseline. There
are no accuracy numbers in this repo yet, because there is no scorer yet — that
is Phase 1. Nothing here should be quoted as a result.

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
docs/PHASE0_AUDIT.md   what was broken, what was fixed, what was preserved
notebooks/             the original Kaggle notebooks, unmodified
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

## Roadmap

- **Phase 0** — correct, modular baseline ✅
- **Phase 1** — official metric, extracted-vs-generated validation split, first honest number
- **Phase 2** — per-chart-type error taxonomy
- **Phase 3** — YOLOv8 vs Faster R-CNN, U-Net crop, Pytesseract axis OCR — each measured
- **Phase 4** — zero-shot Qwen2.5-VL benchmark on the same split and metric
- **Phase 5** — ablation table, README, latency profile

## Ground rule

Every claim that reaches a CV must trace to code that ran and produced a logged
number. A component that does not help gets reported as not helping.
