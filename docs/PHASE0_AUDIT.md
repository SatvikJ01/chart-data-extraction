# Phase 0 Audit

Audit of `tuned-donut.ipynb` and `inference-3.ipynb` prior to refactoring.
Every claim below was checked against notebook source; claims about *behaviour*
were checked with runnable reproductions rather than by reading.

## Classification

Findings are split into **ACTIVE** (was corrupting output at audit time) and
**LATENT** (a real contract violation that was not corrupting output).

**This distinction is load-bearing for Phase 1.** A latent bug cannot, by
definition, have changed any score. Only the ACTIVE fixes may be credited with a
Phase 0 → Phase 1 delta:

> Phase 0 → Phase 1 delta is attributable **only** to bugs 2, 4, 5, A, C and D.
> Bugs 1 and 3 are excluded.

## The six bugs from the brief

| # | Claim | Verdict | Where |
|---|---|---|---|
| 1 | Global `scores` instead of `self.scores` | **LATENT** — real, but mechanism differs | code cells 32, 34 |
| 2 | Line charts stubbed | **ACTIVE** — confirmed | cell 39 |
| 3 | Image-ID misalignment | **LATENT** — does not reproduce | cells 4, 9, 21 |
| 4 | `isdigit()` drops non-integer ticks | **ACTIVE** — confirmed | cell 19 |
| 5 | `clean_preds()` never runs | **Wrong as stated** — see below | both notebooks |
| 6 | Chained assignment, bare `except:` | **ACTIVE** — confirmed | cell 38, and both |

### Bug 1 — LATENT

`PredBarPlot.generate_output` and `PredScatterPlot.generate_output` filter with
the bare global `scores`:

```python
marker = self.marker[np.logical_and(self.labels == 3, scores >= self.detection_threshold)]
```

The brief states this makes every bar/scatter prediction after the first use the
previous image's confidences. **It does not.** The final prediction loop rebinds
the module-level name `scores = df3['scores'][i]` inside every branch *before*
constructing the decoder, so at call time the global equals `self.scores`.
Verified: identity held on every iteration.

It is still a genuine defect — it detonates the moment the code moves into
modules (i.e. this refactor) or the branch order changes. Fixed structurally:
boxes, labels and scores travel together on `MarkerDetections`, and
`filter_by_label_and_score` takes them as explicit arguments with a length check.

Regression test: `tests/test_regressions.py::test_bug1_no_cross_talk_between_decoder_instances`

### Bug 3 — LATENT

Three ID lists built two ways (`Path.glob` vs `os.listdir`), joined positionally.
Both functions delegate to `os.scandir`. Tested on a 200-file directory with
shuffled creation order: **identical order, 0/200 positional mismatches**,
neither sorted. Nothing was being corrupted.

Still fixed, because nothing *guarantees* that agreement: images are enumerated
once and every join is keyed on `image_id`.

Regression test: `tests/test_regressions.py::test_bug3_submission_joins_are_id_keyed_not_positional`

### Bug 5 — wrong as stated, and the real problem is worse

The call is commented out in `tuned-donut.ipynb` only. In `inference-3.ipynb` it
is **live**. But inference-3's copy of the function has the load-bearing line
commented out:

```python
#temp = re.sub(r"[^0-9\.\-eE]", "", temp)
```

Without the strip, a value that fails the first cast is never repaired — it
falls through unchanged and lands in `except ValueError: temp = 0`. Measured:

```
input : ['11', '1E', '3.14', '-5', '1e5']
output: [11,    0,   3.14,   -5,   0   ]
```

So one notebook had an intact function it never called, and the other called a
function that zeroed salvageable values. Both fixed.

## Additional findings

| ID | Finding | Verdict |
|---|---|---|
| A | `find_element_above` sorts the caller's list in place | **ACTIVE**, silent |
| B | Axis calibration is fed the wrong quantity | **ACTIVE** design error — *preserved* |
| C | Negative-index wraparound in `least_count` | **ACTIVE**, silent |
| D | `PredLinePlot` reads a global `y_points`; threshold `>= 0.0` | in dead code (bug 2) |
| E | `temperature`/`top_k`/`top_p` are inert | **ACTIVE**, affects Phase 1 framing |
| F | No `horizontal_bar` branch in the decode loop | **ACTIVE** |

### A — list mutation through a shared reference

`find_element_above` begins `lst.sort()`, sorting the caller's `self.y_points`
in place while `self.y_labels` keeps its original order. The pairing is
destroyed on the first call:

```
before: [(200.0, 0.0), (150.0, 10.0), (100.0, 20.0), (50.0, 30.0)]
after : [(50.0,  0.0), (100.0, 10.0), (150.0, 20.0), (200.0, 30.0)]
```

Every value decoded afterwards is calibrated against mispaired labels. Fixed
structurally by `AxisCalibration`, which stores bound `(pixel, value)` pairs as
immutable tuples.

### B — the axis calibration is fed the wrong quantity (PRESERVED)

```python
labels = [float(label) if label.isdigit() else 0.0 for label in df['y_val'][i]]
extended_y.append(extend_y_axis(y, labels))
```

`df['y_val'][i]` is **Donut's predicted y data series**. It is passed as
`y_labels`, the **y-axis tick labels**. Different quantities; they coincide only
by accident. Downstream, marker pixels are converted to values by interpolating
against Donut's own output — the numeric branch is circular.

Nothing in the pipeline reads axis tick *text*. That is the gap Phase 3's OCR
component closes, and it is the measured justification for that component
existing.

**Decision: preserved bit-for-bit** as `DonutSeriesAxisLabelSource`, the default
behind the `AxisLabelSource` seam. Phase 0 is a faithful refactor, not a better
model — if the baseline already had correct axis labels, the Phase 3 OCR row in
the ablation table would have nothing to demonstrate. Phase 3 registers a second
implementation and flips `PipelineConfig.axis_label_source`: one variable.

Bug 4 (the `isdigit` parse) *is* fixed inside this preserved source. Parsing and
wrong-input are independent problems.

### C — negative-index wraparound

On the out-of-range branch the decoders set `index = 0` then read
`y_labels[index - 1]` → `y_labels[-1]`, wrapping to the opposite end of the axis.
Produces a plausible wrong scale (measured `least_count = -0.2`) rather than
raising. Fixed by explicit linear extrapolation.

### E — the two generation configs differ only in `num_beams`

Neither notebook sets `do_sample`, which defaults to `False`. `temperature`,
`top_k` and `top_p` are therefore inert in both. `tuned-donut`'s
`temperature=0.9, top_k=1, top_p=0.4` does nothing.

The two configs are **greedy vs. 2-beam search**. Phase 1 should still score
them against each other, but the result cannot be described as tuning
temperature or nucleus sampling — three of the four differing parameters have no
effect. `GenerationConfig.to_generate_kwargs` now omits sampling parameters
unless `do_sample=True`, so an inert setting cannot be mistaken for a tuned one.

*(Verified by reading, not execution — `transformers` was not installed on the
audit machine.)*

### F — `horizontal_bar` was never decoded

The final loop branches on `line`, `vertical_bar`, `dot` and `scatter` only.
`horizontal_bar` is in the token schema and `string2preds` can return it, so
those images fell through every branch to `result = []` and the `0;0`
placeholder. `HorizontalBarDecoder` is registered but deliberately returns an
empty series: decoding it correctly needs an **x**-axis calibration, which no
stage produces. Phase 2 should quantify how many images this affects before
Phase 3 decides whether to build one.

## Minor

- `random_padding=True` at inference (both notebooks) — a train-time
  augmentation, making predictions non-reproducible. Now defaults to `False`.
- `clean_preds` divides by `len("".join(x))` with no guard → `ZeroDivisionError`
  on an empty series, swallowed by the bare `except` into a placeholder row.
- Cell 19 indexes `y_points[1]` with no guard → `IndexError` when the axis model
  returns fewer than two ticks.
- `if len(x) == 0` after `"".split(";")` is unreachable — `split` returns `['']`.
- `configure_optimizers` closed over a module-level `model` instead of
  `self.parameters()`. Dropped with the rest of the training scaffolding.
- `PLACEHOLDER_CHART_TYPE = "line"` means generation failures are *also*
  labelled line — a second path to the same placeholder as bug 2.

## Not verifiable in Phase 0

The brief notes the old CV claimed U-Net, YOLO, Pytesseract and a 21% accuracy
improvement. Confirmed by keyword search: **zero references** to any of the
three models, and no metric computation of any kind, in either notebook. There
is no local number to reproduce because both notebooks only ever pointed at
`test/`, whose labels are hidden.
