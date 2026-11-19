# PHASE 3 SEALED RUNNER — IMPLEMENTATION BRIEF

Create ONE self-contained Kaggle script:

    kaggle/phase3/af_phase3_sealed_combined.py

It must perform the pre-registered Phase 3 SEALED evaluation when run with no arguments.

It must not require af_phase1_R_naive.py, af_phase3_sealed.py, final_*.json,
trials_*.jsonl, or spec_freeze.json at runtime.

It may read only:

    /kaggle/input/competitions/talkingdata-adtracking-fraud-detection/train.csv

Write outputs to:

    /kaggle/working/phase3/

## Sources

- code/af_phase1_R_naive.py contains the Phase 1 evaluation machinery:
  Lab, Track, compute, make_decoy, block_aucs, signflip_p, Lord, replay,
  LightGBM fitting, and final selection-free re-measurement.
- code/af_phase0.py contains the exact frozen window loader:
  load_window(day), window_arrays(df), Fold, fit_eval(...).
- code/af_phase3_sealed.py contains EMBEDDED_INPUTS and EMBEDDED_SPEC_FREEZE,
  but is inspection-only: it imports af_phase1_R_naive and has no main().
- code/af_phase3_sealed_combined.py is currently inspection-only and invalid.
- code/af_phase3_sealed_combined.py.bak contains the Phase 1 runner plus embedded
  Phase 3 data; it runs Phase 1, not Phase 3.

## Non-negotiable rules

1. Inline the complete Phase 1 runner source.
2. Remove `import af_phase1_R_naive as p12`.
3. Preserve EMBEDDED_INPUTS and EMBEDDED_SPEC_FREEZE exactly.
4. Do not alter frozen references, SEALED baseline/ceiling, dates, feature grammar,
   seeds, thresholds, model parameters, or evaluation logic.
5. The script must run with no arguments on Kaggle and perform the real SEALED run.
6. Do not include inspection/source-printing code or --inspect behavior by default.
7. Do not weaken or bypass the SEALED baseline tolerance check.
8. Do not fabricate results.
9. Create the completion lock only after all 30 runs and all 29 gate-tested decoys succeed.
10. After any failure, make only a pure bug fix unrelated to results. Do not change
    frozen values, embedded inputs, evaluation semantics, thresholds, seeds, or metrics.

## Frozen references

SEALED_BASELINE_FROZEN = 0.9742099952759418
SEALED_CEILING = 0.985440922359235
DEV_CEILING = 0.9847143092101822
PHASE3_BASE_TOL = 0.0002

Expected canonical SHA-256 of EMBEDDED_INPUTS:

dd1d6f41cb851536776b3f83cadf2bcbc922c300cbf27a2b3ec6c16e84b68c7e

Compute it as:

hashlib.sha256(
    json.dumps(EMBEDDED_INPUTS, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()

Assert it matches before starting.

## SEALED loader

Use the Phase 0 SEALED window exactly:

SEALED_DAY = "2017-11-08"
SEALED_DAY_START_ROW = 68_941_878
SKIP_MARGIN = 6_000_000

The Phase 1 loader is hard-coded to DEV. Set:

WINDOW_DAY = "2017-11-08"
DAY_START_ROW = 68_941_878

Then:

W = get_window(cache_dir)

Before evaluating runs:

measured_baseline = float(lab.full.fit([], 0)["auc"])
assert abs(measured_baseline - SEALED_BASELINE_FROZEN) <= PHASE3_BASE_TOL

This has already been validated on Kaggle:

SEALED measured baseline: 0.9742100
SEALED frozen baseline:  0.9742100
full train/ES/eval [1469995, 591583, 2198357] (eval pos 14,798)

## Embedded inputs

EMBEDDED_INPUTS["runs"] is a dictionary, not a list.

Iterate:

for run_index, run in enumerate(EMBEDDED_INPUTS["runs"].values()):

There are 30 runs across six arms:

R, R-naive, L-naive, L-FDR, L-FDR-3B, L-FDR-14B

Assert six arms and five runs per arm.

For each run, assert:

n_accepted_real == run["accepted_real"]

## Real-feature replay

For each run:

1. Take only accepted records where is_decoy is false or absent.
2. Preserve accepted order.
3. Compute each accepted feature on the SEALED full window.
4. Add it to the evaluation stack.
5. Measure baseline and final AUC using three fresh fits at seeds 100, 101, 102.
6. Use the mean of the three fits for both baseline and final AUC.
7. Compute:

gap_closed_pct = 100 * (final_auc - baseline_auc) / (SEALED_CEILING - baseline_auc)

Use the Phase 1 API exactly:

af = lab.full_cols(spec, False, 0)
extras = [lab.full.col(af)]
fit = lab.full.fit(extras, seed)

Use `full_cols`, not `fullcols`.
`ref` belongs to `Lab`, not `Track`.
For the baseline guard, use `lab.full.fit([], 0)`, not `lab.ref`.

## Caching

Measure the SEALED baseline once and cache it under the empty-set hash.

Create one Lab per distinct accepted-feature stack, keyed by final_set_hash.

Cache the three per-seed AUCs and their mean by final_set_hash.

Assert that no feature state leaks between stacks.

## Decoys

The embedded decoys are exactly the 29 gate-tested decoys.

Assert:

sum(len(run.get("decoys", [])) for run in EMBEDDED_INPUTS["runs"].values()) == 29

Arm counts:

R: 6
R-naive: 3
L-naive: 0
L-FDR: 6
L-FDR-3B: 1
L-FDR-14B: 13

For each decoy:

1. Compute the decoy spec’s underlying real feature on the full SEALED window.
2. Apply Phase 1’s exact transformation:

decoy_array = make_decoy(real_feature_array, dseed)

3. Use:

dseed = 7_000_000 + 1000 * run_index + decoy_index

4. Evaluate against that run’s final accepted stack.
5. Use matched seeds: compare the three-seed mean AUC of final stack + decoy
   against the three-seed mean AUC of the final stack alone.
6. Report:
   - naive_pass: delta_auc > 0
   - materiality_pass: delta_auc >= F["materiality"]
7. Do not reconstruct or re-simulate LORD++.
8. Decoy results are descriptive post hoc checks.

## Required outputs

Write:

phase3_runs.csv
phase3_decoys.csv
phase3_arms.csv
phase3_arms.md
phase3_summary.json
phase3_lock.json
PHASE3_METHOD_NOTES.md

phase3_runs.csv columns:

arm, run, final_set_hash, n_accepted_real, n_decoys_tested, n_decoys_accepted,
decoy_fpr, sealed_baseline, sealed_final_auc, sealed_gap_closed_pct,
dev_baseline, dev_final_auc, dev_gap_closed_pct, optimism_gap, model, minutes,
tokens_prompt, tokens_completion, n_proposals, n_full_tests, decoys_proposed,
decoys_screened, dups

dups comes from embedded n_dup_total. Preserve null/NaN where embedded.

dev_gap_closed_pct:

100 * (dev_final_auc - dev_baseline) / (DEV_CEILING - dev_baseline)

optimism_gap:

dev_gap_closed_pct - sealed_gap_closed_pct

phase3_decoys.csv columns:

arm, run, decoy_index, decoy_idx, spec, repermute_seed,
sealed_stack_auc, sealed_stack_decoy_auc, delta_auc,
naive_pass, materiality_pass

phase3_arms.csv columns:

arm, runs, n_effective_outcomes, mean_gap_closed, median_gap_closed,
min_gap_closed, max_gap_closed, mean_final_auc, mean_dev_gap_closed,
mean_optimism_gap, mean_real_accepted, decoys_tested, decoys_accepted, decoy_fpr

n_effective_outcomes is the number of distinct final_set_hash values per arm.

## Lock file

Use three states:

1. started: after baseline guard and embedded-input hash assertion pass.
   Include embedded_inputs_sha256, script_sha256, and started_at_utc.
2. Flush each distinct stack result immediately after it completes.
3. complete: only after all 30 runs and all 29 decoys succeed.

Use atomic writes: .tmp file, flush, fsync, os.replace.

## AF_TEST requirement

AF_TEST=1 must exercise the full Phase 3 path:

- synthetic window
- at least two fake embedded runs
- full real-feature replay
- full decoy evaluation
- all required output files written to a temporary directory
- assertion that every required CSV column exists
- assertion that the lock reaches complete
- assertion that no real SEALED data was read

Do not treat --dry-run as sufficient Phase 3 validation.

## Kaggle metadata

Create:

kaggle/phase3/kernel-metadata.json

{
  "id": "yoka33/af-minilab-phase3-sealed",
  "title": "af-minilab-phase3-sealed",
  "code_file": "af_phase3_sealed_combined.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": false,
  "enable_tpu": false,
  "enable_internet": false,
  "dataset_sources": [],
  "competition_sources": [
    "talkingdata-adtracking-fraud-detection"
  ],
  "kernel_sources": []
}

Push only from:

kaggle/phase3

with:

kaggle kernels push -p kaggle/phase3

## Known prior failures

1. Do not import af_phase1_R_naive.
2. Do not leave inspection code in the default path.
3. Use lab.full_cols, not lab.fullcols.
4. Do not use lab.full.ref; ref belongs to Lab.
5. Do not use lab.ref["fullauc"]; that key does not exist.
6. Do not iterate EMBEDDED_INPUTS["runs"] as a list.
7. Do not use --dry-run as proof that Phase 3 works.
8. Do not use a loose baseline tolerance.
9. Do not compare decoy and stack AUCs across mismatched seeds.
10. Do not modify EMBEDDED_INPUTS before the push.
