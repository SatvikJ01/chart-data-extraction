"""Persist evaluation results.

Two artefacts, both version-controlled:

  ``results/runs.jsonl``  - one JSON object per run, appended. Line-delimited
                            so concurrent runs and git merges append cleanly
                            instead of conflicting on a rewritten array.
  ``results/ablation.md`` - the human-readable table, appended a row at a time.

Neither is ever rewritten in place. A run that produced a bad number stays in
the record; the correct response is another row, not a deletion.
"""

from __future__ import annotations

import json
from pathlib import Path

from chart_extraction.eval.harness import LEAKAGE_CAVEAT, EvaluationResult

DEFAULT_RESULTS_DIR = Path("results")
RUNS_FILENAME = "runs.jsonl"
ABLATION_FILENAME = "ablation.md"
PER_INSTANCE_DIRNAME = "per_instance"

_ABLATION_HEADER = f"""# Ablation table

Appended one row per evaluation run. Never rewritten -- a superseded number
stays in the record and is corrected by a later row.

**`sample` says whether the row measures the whole subset.** `full` means every
image of the selected subset was evaluated. `N/P s=SEED` means a deterministic
stratified random subsample of N images from a population of P -- a legitimate
measurement, reproducible with `--sample N --seed SEED`, but carrying the
sampling error shown in `+/-95%`. Two rows differing by less than that interval
are not distinguishable at their sample sizes. (A `--limit` smoke run is never
written here at all.)

**`mode` gates every comparison.** `full` is Donut + axis CNN + marker detector.
`donut_only` is Donut alone, with its generated series used directly and no
detection stage -- a different system, not a degraded full run. Never compare a
`donut_only` row with a `full` row, and never report one as the other.

**Headline is the `extracted` column.** The `generated` column is reported
beside it as a deliberate contrast: it is a far easier, mostly-synthetic
distribution, and the gap between the two is itself the finding.

> **Caveat carried on every row.** {LEAKAGE_CAVEAT}

`decode` names the Donut decoding strategy. Only beam width varies -- neither
notebook set `do_sample`, so temperature/top_k/top_p were inert (Phase 0
finding E). No row here is temperature or nucleus-sampling tuning.

`prec`/`bs` are runtime settings and should not change scores -- except `oom`,
which counts images processed at reduced Donut input resolution after an
out-of-memory retry. A non-zero `oom` means that row contains degraded images
and is not directly comparable with a row that has none.

| run | mode | subset | sample | decode | axis labels | extracted | generated | overall | +/-95% | type acc | ms/img | n img | prec | bs | oom |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
"""


def _wrap(text: str, width: int) -> list[str]:
    import textwrap

    return textwrap.wrap(text, width=width) or [""]


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def ablation_row(result: EvaluationResult) -> str:
    scores = result.scores
    sampling = (result.split or {}).get("sampling") or {}

    if sampling.get("sampled"):
        sample_cell = (
            f"{sampling['n_selected']}/{sampling['n_population']} "
            f"s={sampling['seed']}"
        )
    else:
        sample_cell = "full"

    return (
        f"| `{result.run_id}` "
        f"| {result.config.get('mode', '?')} "
        f"| {result.runtime.get('subset', '?')} "
        f"| {sample_cell} "
        f"| {result.config.get('generation', '?')} "
        f"| {result.config.get('axis_label_source', '?')} "
        f"| {_fmt(result.headline)} "
        f"| {_fmt(result.generated_score)} "
        f"| {_fmt(scores.get('overall', 0.0))} "
        f"| {1.96 * scores.get('stderr', 0.0):.4f} "
        f"| {_fmt(scores.get('chart_type_accuracy', 0.0))} "
        f"| {result.latency.get('total_ms', 0.0):.1f} "
        f"| {result.split.get('n_images', 0)} "
        f"| {result.runtime.get('precision', '?')} "
        f"| {result.config.get('donut_batch_size', '?')} "
        f"| {result.oom.get('n_recovered', 0)} |\n"
    )


def append_result(
    result: EvaluationResult,
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
    write_per_instance: bool = True,
) -> dict[str, Path]:
    """Append one run to both artefacts. Returns the paths written."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    runs_path = results_dir / RUNS_FILENAME
    with open(runs_path, "a") as handle:
        handle.write(json.dumps(result.to_dict(), sort_keys=True) + "\n")

    ablation_path = results_dir / ABLATION_FILENAME
    if not ablation_path.exists():
        ablation_path.write_text(_ABLATION_HEADER)
    with open(ablation_path, "a") as handle:
        handle.write(ablation_row(result))

    written = {"runs": runs_path, "ablation": ablation_path}

    if write_per_instance and result.per_instance is not None:
        per_instance_dir = results_dir / PER_INSTANCE_DIRNAME
        per_instance_dir.mkdir(parents=True, exist_ok=True)
        path = per_instance_dir / f"{result.run_id}.csv"
        result.per_instance.to_csv(path, index=False)
        written["per_instance"] = path

    return written


def load_runs(results_dir: Path | str = DEFAULT_RESULTS_DIR) -> list[dict]:
    """Read every recorded run, oldest first."""
    path = Path(results_dir) / RUNS_FILENAME
    if not path.exists():
        return []
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def format_report(result: EvaluationResult) -> str:
    """Console summary of one run."""
    scores = result.scores
    mode = result.config.get("mode", "?")
    lines = [
        f"run {result.run_id}  [mode={mode}, {result.config.get('generation')}]",
    ]

    sampling = (result.split or {}).get("sampling") or {}
    if sampling.get("sampled"):
        lines += [
            "",
            f"  ** SUBSAMPLE: {sampling['n_selected']} of "
            f"{sampling['n_population']} images "
            f"({sampling.get('fraction', 0):.1%}), seed {sampling['seed']}",
            f"     stratified on (source, chart_type); reproduce with "
            f"--sample {sampling['n_requested']} --seed {sampling['seed']}",
        ]

    stages = result.stages or {}
    if stages.get("stages_skipped"):
        lines += [
            "",
            f"  !! STAGES SKIPPED: {', '.join(stages['stages_skipped'])}",
        ]
        for stage, reason in (stages.get("reasons") or {}).items():
            lines.append(f"       {stage}: {reason}")
        lines.append(
            "     This score is NOT comparable with a full-pipeline score."
        )

    lines += [
        "",
        f"  HEADLINE (extracted) : {result.headline:.4f}"
        f"   n={scores.get('by_source', {}).get('extracted', {}).get('n_instances', 0)}",
        f"  generated            : {result.generated_score:.4f}"
        f"   n={scores.get('by_source', {}).get('generated', {}).get('n_instances', 0)}",
        f"  overall              : {scores.get('overall', 0.0):.4f}"
        f"  +/-{1.96 * scores.get('stderr', 0.0):.4f} (95% CI, clustered by image)",
        f"  chart-type accuracy  : {scores.get('chart_type_accuracy', 0.0):.4f}",
        "",
        "  per chart type:",
    ]
    for chart_type, entry in sorted(scores.get("by_chart_type", {}).items()):
        lines.append(
            f"    {chart_type:<16} {entry['score']:.4f}  (n={entry['n_instances']})"
        )

    lines += ["", "  error taxonomy:"]
    total = sum(result.taxonomy.get("counts", {}).values()) or 1
    for category, count in result.taxonomy.get("counts", {}).items():
        if count:
            lines.append(f"    {category:<24} {count:>6}  ({100.0 * count / total:.1f}%)")
    not_applicable = result.taxonomy.get("not_applicable") or []
    if not_applicable:
        lines.append(
            f"    (not applicable in {result.taxonomy.get('mode', '?')} mode: "
            f"{', '.join(not_applicable)})"
        )

    lines += ["", "  latency (ms/image):"]
    for key, value in result.latency.items():
        if key.endswith("_ms"):
            lines.append(f"    {key:<16} {value:.1f}")

    lines += ["", "  runtime:"]
    runtime = result.runtime
    lines.append(
        f"    device           {runtime.get('gpu_name', runtime.get('requested_device', '?'))}"
    )
    lines.append(f"    precision        {runtime.get('precision', '?')}")
    # Only stages that ran; a batch size for a skipped stage is noise.
    stages_run = set((result.stages or {}).get("stages_run") or
                     ["donut", "axis", "markers"])
    batch_parts = [f"donut={result.config.get('donut_batch_size', '?')}"]
    if "axis" in stages_run:
        batch_parts.append(f"axis={result.config.get('axis_batch_size', '?')}")
    if "markers" in stages_run:
        batch_parts.append(f"markers={result.config.get('marker_batch_size', '?')}")
    lines.append("    batch sizes      " + " ".join(batch_parts))
    if runtime.get("peak_memory_mb") is not None:
        lines.append(f"    peak GPU memory  {runtime['peak_memory_mb']:.1f} MB")
    if runtime.get("gpu_total_mb") is not None:
        lines.append(f"    GPU total        {runtime['gpu_total_mb']:.1f} MB")

    oom = result.oom
    if oom:
        recovered = oom.get("n_recovered", 0)
        unrecovered = oom.get("n_unrecovered", 0)
        if recovered or unrecovered:
            lines += ["", "  OOM recovery:"]
            lines.append(f"    degraded (lower res)  {recovered}")
            lines.append(f"    unrecoverable         {unrecovered}")
            for event in oom.get("events", [])[:10]:
                if event.get("recovered"):
                    lines.append(
                        f"      {event['image_id']}: scale {event['scale']}"
                        f" -> {event['height']}x{event['width']}"
                    )
                else:
                    lines.append(f"      {event['image_id']}: no scale fit")
            if len(oom.get("events", [])) > 10:
                lines.append(f"      ... {len(oom['events']) - 10} more")

    lines += ["", "  models:"]
    for model in result.models:
        lines.append(
            f"    {model['name']:<16} {model['parameters'] / 1e6:>8.1f}M params"
            f"  {model['size_mb']:>8.1f} MB"
        )

    populations = result.populations
    if result.warnings:
        lines += ["", "  sanity checks:"]
        for warning in result.warnings:
            marker = {"error": "!!", "warning": " !", "info": "  "}.get(
                warning.get("level"), "  "
            )
            lines.append(f"    {marker} [{warning.get('code')}]")
            for chunk in _wrap(warning.get("message", ""), 68):
                lines.append(f"         {chunk}")

    lines += [
        "",
        "  finding F population:",
        f"    horizontal_bar in ground truth : {populations.get('horizontal_bar_ground_truth', 0)}",
        f"    horizontal_bar predicted       : {populations.get('horizontal_bar_predicted', 0)}",
        "",
        "  caveats:",
    ]
    for caveat in result.caveats:
        lines.append(f"    - {caveat}")
    return "\n".join(lines)
