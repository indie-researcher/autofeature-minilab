# Phase 3 method notes

- **SEALED loader (reused, not rewritten):** `load_dev_window()` / `window_arrays()` / `Track` from the Phase 1 runner,
  with `WINDOW_DAY = 2017-11-08`, `DAY_START_ROW = 68,941,878`, `SKIP_MARGIN = 6,000,000` (Phase 0 SEALED window,
  08:00-16:00). Fold sizes: train 1,469,995 / early-stop 591,583 / eval 2,198,357 (eval positives 14,798).
- **Baseline guard:** raw-feature baseline measured once with `lab.full.fit([], 0)`; abort if it differs from the frozen
  0.9742099952759418 by more than 0.0002.
- **Real-feature replay:** for each of the 30 embedded runs, the accepted real specs are recomputed on SEALED in their
  original order and stacked on the 4 raw columns.
- **Three-seed re-measurement:** baseline and final AUC are each the mean of three fresh LightGBM fits at seeds
  100, 101, 102. Gap closed = 100 * (final - baseline) / (frozen SEALED ceiling - baseline). Results are cached by the
  hash of the ordered accepted specs, so identical stacks are fit once.
- **Matched-seed decoy delta:** for each embedded gate-tested decoy, the underlying spec is computed on the full SEALED
  window, permuted with `make_decoy` (seed 7,000,000 + 1000 * run_index + decoy_index), and added to that run's final
  accepted stack. Delta = mean over seeds 100/101/102 of AUC(stack + decoy) - AUC(stack) at the same seed.
- **Decoy reporting is descriptive and post hoc:** naive_pass = delta > 0; materiality_pass = delta >= the frozen
  materiality floor. For FDR arms "accepted" = materiality_pass; for naive arms "accepted" = naive_pass. These are checks
  against each run's FINAL stack, not the stack at proposal time.
- **LORD++ was NOT re-simulated** because the original trial-level FDR state is not embedded; any re-simulation would
  be fabricated.
- The SEALED baseline was measured in Phase 0 and again in the validation run.
- **Disclosure:** an earlier invocation of this script aborted at the first run with a KeyError (embedded accepted
  records are specs, not {'spec': ...} wrappers) before any feature AUC was computed on SEALED; only the baseline was
  observed. The fix is a bug fix with no change to frozen references, seeds, thresholds or evaluation logic.
- No Phase 3 feature results were inspected before the final run.
- Embedded `tokens` are cumulative counters across runs within an arm; `tokens_prompt`/`tokens_completion` are
  per-run deltas (run 1 taken as-is, assuming the counter started at 0), raw values kept in `*_cumulative` columns.
  Tokens and `n_dup_total` are null for R and L-naive (random / not logged) and written as NaN.
- R and L-naive never logged `decoys_proposed` / `decoys_screened`. They are DERIVED: proposed = the per-run-number decoy
  schedule (10, 15, 10, 16, 9; identical across the four arms that logged it), screened = proposed - gate-tested.
  Flagged in the `decoy_counts_source` column.
- An earlier skeleton invocation wrote a `started` lock on 2026-11-19 13:06 (local results/phase3) before failing; no
  SEALED AUC from any feature set was computed or inspected.
