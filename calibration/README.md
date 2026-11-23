# Calibration of stages 5 and 6

The code that chose the stage-5 weights and the stage-6 settings now in `concept_pipeline`
(`CALIBRATED_WEIGHTS`, `T_MAX_LOG2_RATIO`, `LOCAL_DISCOUNT`, `IMPORTANCE_WEIGHT`,
`ENTRY_THRESHOLD`, `REPLACE_MARGIN`). The data is not in this repository: point
`--data-dir` or `TACTIMIND_DATA_DIR` at a folder laid out like this:

- `training_set/<course>/`: the course files; the summary's file name contains the Hebrew
  word `סיכום` ("summary").
- `training_set_gold/<course>/concepts.gold.json`: dev gold from the model committee
  (`scripts/concept_committee.py`), plus the model caches (`_normalization_cache`, ...).
- `training_set_heldout_gold/<course>/concepts.gold.json`: held-out gold, never tuned on.
- `stage56_results/`: written by these scripts (`dumps/<split>/*.pkl`, `runs/*.json`).

Order of the runs:

1. `dump.py --split dev` and `dump.py --split heldout`: per-course evidence and labels.
   `--mode algorithmic` (default) needs no model; `--mode replay` reads the recorded model
   answers and stops on a missing one (record them with `scripts/stage4_merging_eval.py --send`).
2. `s5_run.py` (and `--label label_relaxed`): the stage-5 grid, weights fitted leave-one-course-out.
3. `paired.py`: each stage-5 change against the design's opening values, paired by course.
4. `s6_run.py --config configs/stage5_chosen.json`: the stage-6 grid.
5. `final_eval.py --split heldout` and `ceiling_eval.py`: the chosen settings, measured once.

`configs/final.json` holds the chosen settings. `s5_run.py` and `s6_run.py` use 2 processes
unless `--workers` says otherwise.
