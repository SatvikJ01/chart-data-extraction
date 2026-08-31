# Chart Data Extraction — Project Brief

## 1. What this project is

An end-to-end system that converts chart images (bar, line, scatter, dot, horizontal bar)
into structured tabular data series. Built on the Kaggle competition
**Benetech — Making Graphs Accessible**.

This is a CV portfolio project for an IIT Kharagpur dual-degree student. It sits on the
**AI/ML–Data CV** as the *vision and multimodal* anchor project. It does not need to
bridge to aerospace.

The objective is not to win the competition (it closed in 2023). The objective is a
**technically deep, honestly measured, interviewable project** with numbers that can be
defended under questioning.

## 2. Non-negotiable ground rule

**Every claim that ends up on the CV must be traceable to code that actually ran and
produced a logged number.**

This project is being rebuilt precisely because the existing CV bullets outran the code.
Do not fabricate, estimate, or round-up metrics. If a component doesn't help, the correct
outcome is to report that it didn't help — a negative result honestly measured is more
defensible in an interview than an invented positive one.

Corollary: **do not add a model just to make an old CV bullet true.** Add it only if it
has a real job in the pipeline and its contribution is measured.

## 3. Current state — what exists

Two Kaggle notebooks, both **inference-only**. No training code, no evaluation code.

### `tuned-donut.ipynb` (~9.5k chars)
Standalone Donut inference.
- `VisionEncoderDecoderModel` + `DonutProcessor` from a fine-tuned checkpoint
- Generation config: `num_beams=2, temperature=0.9, top_k=1, top_p=0.4`
- `string2preds()` parses generated token string → chart type + x/y series via special
  tokens (`<x_start>`, `<x_end>`, `<y_start>`, `<y_end>`, `<dot>`, `<line>`, etc.)
- `clean_preds()` is defined but **never called** (call is commented out)
- Writes `submission.csv`

### `inference-3.ipynb` (~32k chars)
Multi-stage pipeline.
1. **Donut** — same as above but `num_beams=1`, no top-p. This looks like the *untuned*
   baseline generation config.
2. **Custom CNN** — ResNet-18 backbone, two regression heads, predicts x-axis and y-axis
   tick positions.
3. **Faster R-CNN** — `fasterrcnn_mobilenet_v3_large_fpn`, detects data-point markers
   (bars, dots, scatter points).
4. **Per-chart-type classes** — `PredBarPlot`, `PredDotPlot`, `PredScatterPlot`,
   `PredLinePlot` convert marker pixel positions → data values by interpolating against
   detected axis ticks.

### What is NOT in either notebook (verified by keyword search over all code cells)
- **U-Net** — zero references
- **YOLO** — zero references
- **Pytesseract / Tesseract** — zero references
- **Any accuracy or evaluation metric** — no `sklearn.metrics`, no Levenshtein, no scoring
  of any kind

The old CV bullets claim all four of these plus a "21% accuracy improvement". None of it
is currently supported. That gap is the reason for this rebuild.

## 4. Known bugs to fix first (Phase 0)

These are in `inference-3.ipynb` unless noted. Both notebooks parse as valid Python; these
are logic bugs, not syntax errors. Several fail silently — they produce wrong output
without raising.

1. **`self.scores` vs global `scores`** — in `PredBarPlot.generate_output` and
   `PredScatterPlot.generate_output`, the marker filter uses the bare global `scores`
   instead of `self.scores`:
   ```python
   marker = self.marker[np.logical_and(self.labels == 3, scores >= self.detection_threshold)]
   ```
   `scores` resolves to leftover global state from the last image in the Faster R-CNN loop,
   so every bar/scatter prediction after the first filters using the wrong image's
   confidences. `PredDotPlot` and `PredLinePlot` correctly use `self.scores`.

2. **Line charts are stubbed** — in the final prediction loop the `PredLinePlot` call is
   commented out and replaced with a hardcoded `result = [0.0, 0.0]`. Every line chart
   gets a placeholder.

3. **Image-ID misalignment risk** — the ID list is rebuilt three times using two different
   methods (`Path.glob` for the Donut stage, `os.listdir` for the CNN and Faster R-CNN
   stages), then joined by positional index (`df['x_val'][i]`, `df2['x_points'][i]`,
   `df3['boxes'][i]`). `os.listdir` order is not guaranteed to match `Path.glob` order.
   Fix by keying on image ID, not position.

4. **Axis-label parsing drops non-integer ticks** — `extended_y` construction filters with
   `label.isdigit()`, which is `False` for `"-5"`, `"3.14"`, `"1e5"`. Any chart with
   negative or decimal tick labels gets a zeroed axis scale.

5. **`clean_preds()` never runs** — call is commented out inside `string2preds` in **both**
   notebooks.

6. Minor: chained pandas assignment (`sub_df['data_series'][i] = ...` → use `.loc`), and
   bare `except:` blocks that swallow generation failures without logging.

## 5. Dataset facts

Official competition data: `kaggle competitions download -c benetech-making-graphs-accessible`
(requires accepting comp rules on the Kaggle page first, and `~/.kaggle/kaggle.json`).

Structure: `train/images`, `train/annotations`, `test/images`.

- **Training set: 60,578 images** with paired JSON annotations containing ground-truth
  data series, chart type, and axis info. The old CV said 65,000 — that is wrong, use
  60,578.
- **Test set labels are hidden.** This is why the existing notebooks have no evaluation
  code — they only ever pointed at `test/`. The 21% figure, if it was real at all, came
  from the leaderboard, not from local code.
- **Ground truth was available the whole time** in `train/annotations`. Validation should
  be built from a held-out slice of `train/`.

### The distribution-shift trap — this matters a lot

Benetech built the dataset from a mix of **synthetic** and **extracted** (real textbook)
images, because real-world data was scarce. The training set is overwhelmingly synthetic;
the test set skewed much more toward real extracted charts.

Evidence of the gap: top teams scored **0.88 public / 0.72 private**. Local validation in
this competition was notoriously optimistic.

**Therefore:** check the `source` field in the annotation JSONs, split `extracted` from
`generated`, and **report the headline metric on the extracted-only validation set.**
Report the generated-set number too, as a deliberate contrast — the gap between them is
itself a finding worth putting in the README.

## 6. Evaluation protocol

Implement the official competition metric: **Levenshtein-based similarity for categorical
data series, RMSE-based for numeric**, matched per series and averaged over all instances.
The competition published a scoring utility; reimplement it faithfully and unit-test it
against a few hand-constructed cases.

Every experiment logs: overall score, **per-chart-type score**, inference latency per
image, and model size. Results go into a single ablation table that is version-controlled.

## 7. Plan

### Phase 0 — Correct baseline
Fix the six bugs above. Refactor the two notebooks into a proper repo (modules for donut
inference, axis regression, marker detection, per-chart-type decoding, scoring). No new
models yet.

### Phase 1 — Real numbers
Build the scorer, build the extracted/generated validation split, score the fixed pipeline.
This produces the first honest number and replaces the unsupported 21%.

Also score the two generation configs against each other — `num_beams=1` (inference-3) vs
`num_beams=2, top_p=0.4` (tuned-donut). This is the only "hyperparameter tuning" claim
with any actual code behind it, so measure it properly and report whatever it gives.

### Phase 2 — Error taxonomy
Per-chart-type breakdown of failures. Expect scatter and line to be weakest. This is what
*motivates* the detection branch existing — without it, the multi-stage design looks
arbitrary. Categorize failure modes (malformed sequence, wrong chart type, axis
misestimation, marker miss).

### Phase 3 — Earn the additional models
Only add these if Phase 2 shows they address a measured weakness:
- **YOLOv8 (`ultralytics`)** as a drop-in replacement for the Faster R-CNN marker
  detector. Gives a real comparison: mAP@0.5, mAP@0.5:0.95, latency. Not a bare mention.
- **U-Net** for plot-area / axis-region segmentation as a preprocessing crop before marker
  detection. Distinct role from detection, testable in isolation: does cropping improve
  marker precision?
- **Pytesseract** for chart title / legend / category-label OCR — a task Donut is *not*
  currently doing (Donut only extracts the data series). This makes it complementary
  rather than redundant.

Each gets its own row in the ablation table. If one doesn't help, say so.

### Phase 4 — Modern VLM benchmark (highest-value addition)
Zero-shot / few-shot **Qwen2.5-VL** (or a frontier API model) on the same validation set,
same metric. The question: does a 2025 general-purpose VLM beat a fine-tuned 2022
OCR-free specialist at chart derendering, and at what cost and latency?

Report accuracy, per-chart-type, latency, and cost per 1k images. This is a genuinely
open question, it modernizes the project, and it earns current keywords honestly.

### Phase 5 — Consolidate
Ablation table + README + per-chart-type results + latency profile. This is the artifact
that makes the project look worked rather than assembled.

## 8. Repo hygiene

Real git history with meaningful, incremental commits matters here — the repo will be
linked from the CV, and commit history is itself evidence of sustained work. Do not
squash everything into one commit.

Include a README with: problem statement, pipeline diagram, ablation table, the
synthetic-vs-extracted validation finding, and honest limitations.

## 9. Environment constraints

- GPU availability at IIT KGP needs confirming before Phase 3/4 training runs.
- If GPU access is Kaggle/Colab rather than an SSH-able node, develop locally and push
  training scripts up; pull results back for analysis.
- Donut checkpoints referenced by the old notebooks live in Kaggle datasets
  (`/kaggle/input/benetech-donut`, `/kaggle/input/x-axis-model-10`,
  `/kaggle/input/marker-model`) and are not in this repo.

## 10. Technical details worth knowing for interviews

- Donut = **Swin Transformer encoder + BART-style autoregressive text decoder**, trained
  OCR-free. It reads the image and generates a structured token sequence directly.
- The special-token schema (`<x_start>...<x_end>`) is how a generative model is coerced
  into structured output — worth being able to explain why that's fragile and what
  constrained decoding would fix.
- Faster R-CNN is two-stage (RPN + ROI head); YOLO is single-stage. The accuracy/latency
  trade-off between them is the point of the Phase 3 comparison.
- The competition metric is asymmetric across data types — know why Levenshtein for
  categorical and RMSE for numeric, and what that means for which errors get punished.

## 11. Out of scope for this repo

CV bullet writing and decisions about what is defensible to claim happen separately, in
chat, once the ablation table exists. Do not write CV bullets here.
