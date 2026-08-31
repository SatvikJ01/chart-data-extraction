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

**Headline is the `extracted` column.** The `generated` column is reported
beside it as a deliberate contrast: it is a far easier, mostly-synthetic
distribution, and the gap between the two is itself the finding.

> **Caveat carried on every row.** {LEAKAGE_CAVEAT}

`decode` names the Donut decoding strategy. Only beam width varies -- neither
notebook set `do_sample`, so temperature/top_k/top_p were inert (Phase 0
finding E). No row here is temperature or nucleus-sampling tuning.

| run | decode | axis labels | extracted | generated | overall | type acc | ms/img | n img |
|---|---|---|---:|---:|---:|---:|---:|---:|
"""


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def ablation_row(result: EvaluationResult) -> str:
    scores = result.scores
    by_source = scores.get("by_source", {})
    return (
        f"| `{result.run_id}` "
        f"| {result.config.get('generation', '?')} "
        f"| {result.config.get('axis_label_source', '?')} "
        f"| {_fmt(result.headline)} "
        f"| {_fmt(result.generated_score)} "
        f"| {_fmt(scores.get('overall', 0.0))} "
        f"| {_fmt(scores.get('chart_type_accuracy', 0.0))} "
        f"| {result.latency.get('total_ms', 0.0):.1f} "
        f"| {result.split.get('n_images', 0)} |\n"
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
    lines = [
        f"run {result.run_id}  [{result.config.get('generation')}]",
        "",
        f"  HEADLINE (extracted) : {result.headline:.4f}"
        f"   n={scores.get('by_source', {}).get('extracted', {}).get('n_instances', 0)}",
        f"  generated            : {result.generated_score:.4f}"
        f"   n={scores.get('by_source', {}).get('generated', {}).get('n_instances', 0)}",
        f"  overall              : {scores.get('overall', 0.0):.4f}",
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

    lines += ["", "  latency (ms/image):"]
    for key, value in result.latency.items():
        if key.endswith("_ms"):
            lines.append(f"    {key:<16} {value:.1f}")

    lines += ["", "  models:"]
    for model in result.models:
        lines.append(
            f"    {model['name']:<16} {model['parameters'] / 1e6:>8.1f}M params"
            f"  {model['size_mb']:>8.1f} MB"
        )

    populations = result.populations
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
