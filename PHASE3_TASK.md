GOAL
Build ONE self-contained Kaggle script:
    code/af_phase3_sealed_combined.py
It must perform the pre-registered Phase 3 SEALED evaluation and run successfully with NO arguments on Kaggle.

AVAILABLE FILES
- code/af_phase1_R_naive.py: the Phase 1 runner. It contains the feature grammar, compute(), make_decoy(),
  Track, Lab, replay logic, LightGBM fitting, LORD FDR gate, and final selection-free re-measurement logic.
- code/af_phase3_sealed.py: contains EMBEDDED_INPUTS (30 runs across 6 arms) and EMBEDDED_SPEC_FREEZE,
  but it is only an inspection script. It imports af_phase1_R_naive and has no main().
- The repository may contain the Phase 0 script that originally computed the SEALED baseline and ceiling.

NON-NEGOTIABLE RULES
1. Inline the COMPLETE source of code/af_phase1_R_naive.py into the combined script so its functions
   are available without importing af_phase1_R_naive. Preserve the namespace name `p12` for the Phase 3 code.
2. Remove `import af_phase1_R_naive as p12`.
3. Preserve EMBEDDED_INPUTS and EMBEDDED_SPEC_FREEZE exactly; do not edit their contents.
4. Do not alter any frozen value: SEALED baseline 0.9742099952759418, SEALED ceiling 0.985440922359235,
   DEV references, feature grammar, seeds, thresholds, model parameters, or evaluation logic.
5. Find and reuse the EXACT Phase 0 SEALED window-loading implementation from this repository.
   Search for `2017-11-08`, `SEALED`, `DAY_START_ROW`, and `SKIP_MARGIN`.
   Do NOT invent a SEALED start row, skip margin, date, fold boundary, or preprocessing rule.
   If the exact loader cannot be found, create a file named PHASE3_BLOCKED.txt explaining exactly
   what is missing and do not create a runnable main() that would open SEALED with guessed parameters.
6. The final script must run with no arguments on Kaggle and execute the real SEALED Phase 3 run.
7. It must not require af_phase1_R_naive.py, final_*.json, trials_*.jsonl, or spec_freeze.json at runtime.
8. It may read only the Kaggle competition file:
   /kaggle/input/competitions/talkingdata-adtracking-fraud-detection/train.csv
9. Write all outputs under /kaggle/working. Create a completion lock only after every required
   computation succeeds.

PHASE 3 ALGORITHM
For each of the 30 embedded runs:
- Use only the run's `accepted` real feature specs, in the exact order given.
- Do NOT use the DEV proxy screen, proposer, DEV feedback, or DEV gate decisions.
- Build the feature stack on the SEALED window using the same compute() and Track/Lab machinery
  as af_phase1_R_naive.py.
- Perform the same selection-free re-measurement used at the end of Phase 1:
  * baseline = mean AUC over 3 fresh fits with seeds 100, 101, 102 on the raw SEALED features;
  * final = mean AUC over 3 fresh fits with seeds 100, 101, 102 after adding the run's accepted real features;
  * gap_closed_pct = 100 * (final - baseline) / (SEALED_ceiling - baseline).
- Use the SEALED baseline and ceiling from EMBEDDED_SPEC_FREEZE.
- Record per-run results in /kaggle/working/phase3_runs.csv and /kaggle/working/phase3_runs.json.
- Aggregate by arm into /kaggle/working/phase3_arms.csv and /kaggle/working/phase3_arms.md.
- For every decoy listed in a run's `decoys` array, re-permute it with a fresh deterministic seed,
  compute it on SEALED, and evaluate it through the same gate type as its arm:
  * naive arms: accept if SEALED full-track AUC increases;
  * FDR arms: use a fresh LORD object for that run, because the original trial-level FDR state
    is not embedded; document this reconstruction clearly.
- Record decoys tested, decoys accepted, and decoy FPR by arm.
- Create /kaggle/working/phase3_lock.json only after all 30 runs and all listed decoys are processed.
- Create /kaggle/working/PHASE3_METHOD_NOTES.md documenting:
  * the exact SEALED loader source file and lines reused;
  * the exact replay and 3-seed measurement procedure;
  * the decoy re-permutation seeds;
  * the FDR-state reconstruction limitation, if applicable.

VALIDATION BEFORE FINISHING
1. Run: python3 -m py_compile code/af_phase3_sealed_combined.py
2. Confirm the file does NOT contain `import af_phase1_R_naive`.
3. Confirm it contains EMBEDDED_INPUTS, EMBEDDED_SPEC_FREEZE, SEALED, `if __name__ == "__main__":`, and `main()`.
4. Run this smoke test, which must use synthetic TEST data and must NOT open the real SEALED window:
   AF_TEST=1 python3 code/af_phase3_sealed_combined.py
   It must complete without error and must not write a phase3_lock.json claiming a real SEALED run.
5. Do not claim success unless every validation command passes.
