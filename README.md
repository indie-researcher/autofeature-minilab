# AutoFeature Mini-Lab

A small study (Phases 0-3 pre-registered; Phase 4 is a pilot plus post hoc audits) of **agentic feature search**: when an agent proposes features and a gate decides what to keep, which layer earns the trust: the proposer, the gate, or the sealed holdout day?

**Short answer:** on this task the sealed day confirmed DEV almost exactly (within about 6 points for every arm), so overfitting was not a problem at about 10 full tests per run. The gate still matters for noise: in a gate-only canary (Phase 4b), a keep-if-better gate accepted a pure-noise feature in 13/13 runs and LORD++ in 0/13, though the noise cost only about 2% of the gap. In the full pipeline the proxy screen removed most decoys before the gate. Random search matched or beat the LLM, and random's lead traces to one feature that one random run happened to propose (about a 12% draw). The grammar is dense (86% of random proposals clear the floor), so the proposer barely matters. A boundary-condition result, reported as it came out. Phase 4a (larger grammar, LLM vs random) was left incomplete.

## Question

1. Does the evaluation harness (proxy screen, statistical gate, sealed day) matter more than the proposer (random vs LLM)?
2. Does LLM size (Qwen2.5 3B / 7B / 14B) change the result much?

Idea worth stealing: **canary decoys**. Insert known-null hypotheses (permuted real features) into the agent's queue and count how many it "discovers". That is a live estimate of the pipeline's false-discovery rate. (Same family as Boruta shadow features and knockoffs. No novelty claimed on online FDR; this is a controlled empirical case study.)

## Design

- **Data:** TalkingData AdTracking Fraud Detection. DEV = 2017-11-09, 08:00-16:00 (25.7M rows). SEALED = 2017-11-08, same hours, opened once, at the end.
- **Task:** the agent picks features from a fixed grammar of 6 operations (count, nunique, cumcount, lag, roll, ratio) over keys c0-c4 and h. Selection, not invention.
- **Pipeline per proposal:** proxy screen (10% IP subsample, promote if AUC gain >= 0.0005) -> full DEV test -> gate -> stack.
- **Decoys:** 20% of proposals are permuted-real features (known noise).
- **Gates:** naive (keep if DEV AUC goes up) vs LORD++ online FDR (alpha 0.10, sign-flip p over 12 blocks, materiality floor 0.000219).
- **Arms (5 runs x 60 proposals each):** random + FDR, random + naive, Qwen 7B + FDR, Qwen 7B + naive, plus Qwen 3B and 14B with FDR.
- **Reference points:** baseline (4 raw ID columns) and a human ceiling. DEV 0.97040 / 0.98471, SEALED 0.97421 / 0.98544.
- **Metric:** gap closed (%) = (AUC - baseline) / (ceiling - baseline). Optimism gap = DEV gap closed minus SEALED gap closed. SEALED baseline and final AUC are 3-seed means.

**Hypotheses (frozen before any SEALED result was read)**
- **H1:** the naive gate shows a larger DEV-to-SEALED optimism gap than the FDR gate. (Reframed before unsealing: the proxy screen removes most decoys before the gate, so H1 is judged on the optimism gap of accepted real features, not on decoy counts.)
- **H2:** LLM proposals close more of the SEALED gap than random proposals, under this fixed grammar and budget.
- **H3:** proposer size changes the SEALED gap less than the gate does.

With 5 runs per arm, claims are **descriptive** (mean, median, range). Exact statement allowed: if all 5 runs of one arm beat all 5 of another, one-sided exact permutation p = 1/252 (about 0.004). Arms with fewer than 5 distinct final feature sets count as that many effective outcomes and are excluded from permutation tests.

## Results

### DEV (agent feedback, so optimistic)

| Arm | Mean gap closed | Range | Real accepted | Decoys gate-tested / accepted |
|---|---|---|---|---|
| Random + FDR | 55.6% | 44.3-91.0 | 4.0 | 6 / 0 |
| Random + naive | 56.4% | 41.6-90.4 | 6.6 | 3 / 0 |
| Qwen 7B + FDR | 37.1% | 29.9-47.9 | 4.4 | 6 / 0 |
| Qwen 7B + naive | 40.3% | 27.7-50.1 | 5.8 | 0 / 0 |
| Qwen 3B + FDR | 29.2% | 28.7-31.3 | 1.2 | 1 / 0 |
| Qwen 14B + FDR | 37.6% | 33.5-41.6 | 3.8 | 13 / 0 |

### SEALED (opened once)

| Arm | SEALED gap closed, mean (median) | Range | Optimism gap (mean) | Effective outcomes |
|---|---|---|---|---|
| Random + FDR | 56.6 (48.1) | 42.8-93.3 | -1.0 | 5 |
| Random + naive | 55.3 (47.8) | 39.1-90.5 | +1.0 | 5 |
| Qwen 7B + FDR | 32.9 (31.3) | 26.5-45.1 | +4.2 | 5 |
| Qwen 7B + naive | 45.1 (43.7) | 32.4-56.7 | -4.8 | 5 |
| Qwen 3B + FDR | 34.7 (35.4) | 31.9-35.4 | -5.5 | 2 |
| Qwen 14B + FDR | 40.1 (39.8) | 39.5-40.8 | -2.5 | 5 |

Negative optimism means SEALED did better than DEV. Source: `results/phase3/phase3_runs.csv`, `phase3_arms.csv`.

### Verdicts (pre-registered rules)

- **H1: not supported.** The naive gate did not show a larger optimism gap. For random proposals the two gates are indistinguishable (-1.0 vs +1.0). For 7B the naive arm was better on SEALED (-4.8 vs +4.2). Per the pre-registered fail rule: *the gate was not needed on this task.*
- **H2: not supported.** Random beat the LLM on means (56.6 vs 32.9 with FDR; 55.3 vs 45.1 with naive). On medians, 7B + naive (43.7) is close to random (47.8-48.1). Random's mean is lifted by one run (93% in one arm, 90% in the other); without that run random sits at 47.4 and 46.5, and 7B + naive at 48.2 (post hoc sensitivity check). Per the fail rule: *grammar + gate carry the result; the LLM adds little here.* Note: R and R-naive share the proposer seed for each run number; the trial logs confirm run 3 is the same underlying draw in both arms (same feature, same proposal index), so it is one lucky draw shown twice, not two independent hits.
- **H3: not supported by the rule, but a near tie.** Size effect (range of means over 3B/7B/14B) = 7.2 points; gate effect (mean |naive - FDR|) = 6.7 points. The gate effect is uneven (about 1 point for random, about 12 for 7B) and size is not monotonic (3B 34.7 > 7B 32.9; 3B is effectively two outcomes). 14B is the best and most stable LLM arm.
- **No exact permutation claim is available:** in every valid pair the ranges overlap, so no arm beat another 5-against-5. 3B is excluded from permutation tests by rule.

### Three numbers

1. DEV to SEALED optimism is within about 6 points for every arm: the sealed day confirmed DEV.
2. Naive vs FDR gate with 7B proposals: 45.1% vs 32.9% SEALED gap closed (+12.2 points for the naive gate, with no overfitting penalty).
3. Canary decoys: 28 of 29 gate-tested decoys lowered SEALED AUC; 0 of 29 passed the materiality floor.

### Post hoc decoy check (descriptive, not a pre-registered FDR replication)

The 29 gate-tested decoys were re-permuted on SEALED (fresh seeds) and added to each run's final stack, comparing 3-seed means with matched seeds. Mean delta AUC was -0.000219 (median -0.000201). One decoy had a positive delta (+0.000134, below the 0.000219 floor); a naive "AUC went up" rule would have accepted 1 of 29 (3.4%), the materiality floor 0 of 29. LORD++ was not re-simulated because the trial-level FDR state is not embedded. Per arm, gate-tested decoys: random + FDR 6, random + naive 3, 7B + FDR 6, 7B + naive 0, 3B 1, 14B 13: too few for a naive-vs-FDR decoy comparison.

### Exploratory (post hoc, not pre-registered)

Diagnostics from the trial logs (`results/diagnostics/`, script `code/af_phase3_analyze.py`). These do not change the verdicts above.

- **One feature explains random's lead.** The 93% (and 90% in R-naive) run 3 traces to `lag(c0, c1, c3; k=1, dir=next)`, the largest-gain accepted feature in the best SEALED run (gain +0.0066 when accepted, at proposal 23). Random proposed it in 1 of 5 runs; R and R-naive hit it in the same run number (same draw). The four LLM arms (3B, 7B x 2 gates, 14B; 15 runs, 960 proposals) never proposed it, nor any `next`-lag over those keys with any k. Run-level Fisher exact p = 0.25 (1 of 5 vs 0 of 15, counting R and R-naive as one draw; 0.15 if counted as independent). Descriptive only.
- **Without that run, random and 7B + naive are about equal** on SEALED (47.4 / 46.5 vs 48.2 gap closed; post hoc exclusion).
- **The LLMs did explore the lag region, just not that spec.** Lag proposals (non-decoy, 240 per arm): random 56 (28 next / 28 prev), 7B + naive 43 (37 / 6), 7B + FDR 42 (41 / 1), 14B 41 (3 / 38), 3B 4 (0 / 4). Different arms leaned on different directions; none reached the needle's key set.
- **3B collapsed onto a narrow set of operations** (0.94 bits of operation entropy vs 2.46-2.57 for the other arms; 6 operations would be at most 2.58). 7B and 14B were not narrower than random at the operation level.
- **LLM arms accept something immediately** (median first accepted proposal index 1, vs 6 for random): fast early gain, then a plateau.
- Caveats: the needle's gain is conditional on the stack at the time it was accepted, and it is one feature on one dataset. Trial logs record failed proposals unevenly across arms (cross-arm duplicate counts come from the run summaries, not the logs), so per-run unique-spec counts are not reported.

Figures: `figures/` (`fig1_design_diagram`, `fig2_noise_kept_and_sealed_drop`, `fig3_proposer_scaling`, `fig4_cost_frontier`; exploratory: `diag_diversity_needle`, `diag_discovery_curves`).


## Phase 4a: extended grammar (DEV only, incomplete; sealed day not opened)

**Plan.** Grow the grammar (rung 1: time-bucket keys, 3.8x atomic specs; rung 2: more windows/lags, `quantile`, `burst`, 10.4x) and compare random vs Qwen 14B under a fixed gate (24 blocks, 30-test cap), judged on a fresh sealed day (2017-11-07). See `PHASE4_PLAN.md`.

**Status.** Only the random arm completed (5 runs at each of rungs 1 and 2). The 14B arms finished 1 run (L-FDR-14B, rung 1) and 2 runs (LF-FDR-14B, no operator nudge, rung 1), with 2 further runs incomplete and none at rung 2. The 2017-11-07 day was never opened (no lock file). H4-1 and H4-1b (LLM vs random across rungs) were not tested. H4-3 (FDR false discoveries) is covered by the Phase 4b canary (0/390).

**DEV observations (not comparable to Phase 3: different gate and test cap).** Random mean gap closed: 31.5% at rung 1 (22.1-39.8), 35.0% at rung 2 (28.1-41.9); every random run accepted at least one real feature. L-14B rung 1: 34.0% (n = 1). LF-14B rung 1: 17.9% (15.7-20.2, n = 2). Random accepted a new-axis feature in 3/5 (rung 1) and 4/5 (rung 2) runs; the sealed gain was not measured, so H4-2 is untested.

**Decision.** Not completed: the LLM arms cost about 20+ GPU-hours to finish and Phase 4b suggests a dense grammar where the proposer barely matters. The sealed day stays unopened for future work. Provenance caveat: `PHASE4_PLAN.md` was never committed or tagged before the runs and was revised mid-way (superseded runs kept under `results/phase4_prerevision_nov21` and `results/phase4_prev_20261123_221501`), so Phase 4a is not pre-registered in the git sense and is reported as a pilot.


## Phase 4b: gate stress tests (post hoc, CPU only, no new sealed day)

**Canary (tests 1-2).** 13 paired runs x 30 permuted decoys per gate (same decoys per run id; decoys bypass the proxy, so this isolates the gate). Naive accepted exactly one decoy in 13/13 runs; LORD++ accepted none (0/390, Wilson 95% upper bound 1.0%). Paired sign test over 13 runs: p = 2^-13 = 1.2e-4 (decoys are shared per run id, so runs are paired, not independent). The accepted noise "closed" 0.1-3.1% of the DEV gap (mean 2.0%). The pooled decoy rate (13/390 = 3.3%) is diluted by a ratchet. On a clean stack 16% of decoys raised DEV AUC (48/300, FDR-canary logs, runs 4-13; 13% before the first naive acceptance, 10/76). After the first noise acceptance, 0 of 224 decoys did, and their mean delta AUC fell from -0.00036 to -0.00141, so the reference AUC appears inflated by the lucky win (winner's curse; consistent with, not proven). The floor alone is not enough: 5% of clean-stack decoys (15/300) clear 0.000219, yet LORD++ accepted none. Per-test false acceptance for keep-if-better is about 13-16% on a clean stack; do not use 3.3%.

**Rejected-candidate audit (test 3; post hoc, sealed day already used).** Of the 63 material candidates LORD++ rejected, 78% had a positive SEALED delta, but only 37% (23/63) cleared the 0.000219 floor. Median +0.000155 vs decoy median -0.000201. Material share by arm: 3B 4/4, 7B 8/12, 14B 9/33, random 2/14. Candidates sit in 17 stacks (not independent), and beating permuted decoys is a low bar, so the floor share is the cost estimate.

**Grammar census (test 4; DEV, relative to the 4-column baseline).** 86% of random proposals clear the proxy floor (83% of ratio specs, 88% of non-ratio). The needle ranks 2 of 796 non-ratio specs and is drawn with p = 0.056% per proposal: about 12% chance that 5 random runs hit it, and an expected 0.4 hits across the 720 LLM proposals (P(0) = 67%). The needle was mostly luck. Ratio-spec rank is unknown (300 of about 211k sampled).

**Reading.** The strict gate is conservative both ways: it blocks noise (naive: about one false acceptance per run once enough decoys are tested) at the price of discarding some real gain. With about 10 full tests per run, only 3 decoys reached a naive gate in Phase 3, so that trade did not pay here. The grammar is dense, so the proposer barely matters. Boundary condition, not a general claim.

**Provenance.** Plan and notebooks frozen at tag `phase4b-freeze` (`8c6b36b`, 2026-11-24). Results and report committed at tag `phase4b-sealed`. Per-job Kaggle start times were not logged (log files carry a collection timestamp of 2026-11-24 23:47 only). No deviations known. Report: `results/p4b/P4B_REPORT.md`.

## Honest limits

- One dataset, one DEV/SEALED day pair, 5 runs per arm.
- The grammar contains the known good operations: this tests selection, not invention.
- DEV was used for agent feedback, so DEV numbers are optimistic by construction.
- With about 10 full tests per run on 25M rows, there was little room for winner's curse. The null on H1 may not hold with many more candidates per run.
- Qwen2.5 family only. Other families are future work.
- Possible model contamination. The contamination probes (3B, 7B, 14B) named other datasets (Avazu/KDD Cup, Amazon/Yahoo, Criteo), which is weak evidence of low contamination.
- 3B is effectively two outcomes (low proposal diversity), so no size trend is claimed from it alone.
- The 7B proposer saturated in a duplicate loop in one run; a patch (more attempts, temperature ramp, duplicate feedback in the prompt) was applied and all runs then reached 60 proposals. Protocol deviation, disclosed here.
- The random-vs-LLM difference rests largely on one feature found once (n = 1 event); a different set of random seeds could have missed it.
- Post hoc, DEV only: across the four FDR arms, 63 non-decoy candidates cleared the materiality floor but were rejected by LORD++ (51 with nominal p <= 0.05; 12 for 7B, 14 for random, 33 for 14B, 4 for 3B). None of the 29 gate-tested decoys did both (0/29, Wilson 95% upper bound 11.7%), so the near-misses are more consistent with real marginal signal than with noise. The strict gate probably cost some real gain, which may contribute to the naive-over-FDR gap for 7B; it does not explain why the same gate cost random proposals about 1 point. The gate never died (0 of 258 tests had alpha_t below 2^-12).
- Decoy FPR in Phase 3 is weak evidence: the proxy screen removes most decoys first, so only 29 reached the gate. The Phase 4b canary bypasses the proxy to measure the gate alone.
- Token counters were cumulative across runs and were converted to per-run deltas (run 1 of each arm taken as-is, unverified). R and L-naive decoy proposal counts are derived from the shared per-run schedule, not logged; R and L-naive have no token or duplicate data.
- **Provenance timeline (UTC):** spec and DEV results committed (`619cd08`, tag `phase0-freeze-v2`) before the Kaggle SEALED job started at 22:25 on 2026-11-19. The runner script was committed 10-16 minutes after the job started (`e3ff20f`, `3e0bac9`); the lock records its sha256 (`c5d8e730...`). The job finished at 02:09 on 2026-11-20 (`results/phase3/phase3_lock.json`, tag `phase3-sealed`). An earlier invocation crashed on the first run (a parsing bug) before any feature AUC was computed on SEALED, and an earlier skeleton wrote a `started` lock that was archived under `archive/`. No SEALED feature result was read before the final run.

## Reproduce

Everything runs on Kaggle (CPU for Phase 3; the LLM arms need a GPU with a local Qwen server). See `RUN_ORDER.md` for the full order.

```bash
# Phase 0: freeze references (baseline, ceiling, materiality)
# Phase 1: random and naive arms            kaggle/1B_FDR, kaggle/1C_naive, kaggle/R_naive
# Phase 2: Qwen 3B / 14B arms               kaggle/phase2_3B, kaggle/phase2_14B
# Phase 3: open SEALED once, recompute accepted features and decoys
kaggle kernels push -p kaggle/phase3
# Analysis: figures + exploratory diagnostics from the Phase 3 outputs and trial logs
python3 code/af_phase3_analyze.py
# Phase 4a (pilot, DEV only):   kaggle/phase4_a_cpu, phase4_a_gpu, phase4_b_gpu
# Phase 4b (CPU):               kaggle/p4b_1_canary_naive, p4b_2_canary_fdr, p4b_3_audit, p4b_4_census; p4b_collect.sh builds results/p4b/P4B_REPORT.md
```

Layout: `code/` (pipeline), `kaggle/` (one folder per notebook), `results/` (all `final_*.json`, trials, summaries, Phase 3 outputs), `figures/`, `docs/PHASE3_BRIEF.md`, `archive/` (aborted attempts). Git tags: `phase0-freeze-v2`, `phase3-sealed`, `phase4b-freeze`, `phase4b-sealed`. Default branch: `master`.
