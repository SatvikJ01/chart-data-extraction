# Ablation table

Appended one row per evaluation run. Never rewritten -- a superseded number
stays in the record and is corrected by a later row.

**`mode` gates every comparison.** `full` is Donut + axis CNN + marker detector.
`donut_only` is Donut alone, with its generated series used directly and no
detection stage -- a different system, not a degraded full run. Never compare a
`donut_only` row with a `full` row, and never report one as the other.

**Headline is the `extracted` column.** The `generated` column is reported
beside it as a deliberate contrast: it is a far easier, mostly-synthetic
distribution, and the gap between the two is itself the finding.

> **Caveat carried on every row.** Validation ids are held out with respect to future training in this repo only. The checkpoints under evaluation were fine-tuned elsewhere on train/ with no recorded partition, so these images may have been in their training data. Scores are optimistic for these checkpoints. Separately, the generated slice is a far easier distribution than the competition's test set; the extracted slice is the headline number.

`decode` names the Donut decoding strategy. Only beam width varies -- neither
notebook set `do_sample`, so temperature/top_k/top_p were inert (Phase 0
finding E). No row here is temperature or nucleus-sampling tuning.

`prec`/`bs` are runtime settings and should not change scores -- except `oom`,
which counts images processed at reduced Donut input resolution after an
out-of-memory retry. A non-zero `oom` means that row contains degraded images
and is not directly comparable with a row that has none.

| run | mode | decode | axis labels | extracted | generated | overall | type acc | ms/img | n img | prec | bs | oom |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| `20260901T003911Z` | donut_only | greedy | donut_series | 0.5455 | 0.0000 | 0.5455 | 0.9830 | 2516.5 | 4106 | fp16 | 1 | 0 |
