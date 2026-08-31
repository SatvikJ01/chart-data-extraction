# Results

**Empty until a real evaluation run happens.** No numbers have been produced
yet: the checkpoints and competition data live in Kaggle datasets and are not
available on the development machine.

Populated by `scripts/run_eval.py` (or `scripts/kaggle_eval.py` on Kaggle):

- `runs.jsonl` — one JSON object per run, appended. Line-delimited so runs and
  git merges append cleanly instead of conflicting on a rewritten array.
- `ablation.md` — the human-readable table, one row appended per run.
- `per_instance/<run_id>.csv` — per-instance scores for error analysis.

Nothing here is ever rewritten in place. A run that produced a bad number stays
in the record and is corrected by a later row.

## Before quoting any number from this directory

Every result carries a leakage caveat, and it is not boilerplate:

1. The split is held out with respect to *future* training in this repo only.
   The checkpoints being evaluated were fine-tuned elsewhere on `train/` with no
   recorded partition, so these images may have been in their training data.
2. The `generated` slice is a much easier distribution than the competition's
   test set.

Both effects push scores the same way — optimistic. The `extracted` column is
the headline for that reason, and the gap between the two columns is itself a
reportable finding.
