# AutoFeature Mini-Lab

A small, pre-registered study of **agentic feature search**: when an agent proposes features and a gate decides what to keep, which layer earns the trust: the proposer, the gate, or the sealed holdout day?

**Short answer:** on this task, the sealed day confirmed DEV almost exactly, so the strict gate bought no protection and, with LLM proposals, cost some gap closed. Random search over the same grammar matched or beat the LLM. A boundary-condition result, reported as it came out. Random's lead traces to a single feature that one random run happened to propose (see Exploratory).

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
- Decoy FPR is weak evidence: the proxy screen removes most decoys first, so only 29 reached the gate.
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
```

Layout: `code/` (pipeline), `kaggle/` (one folder per notebook), `results/` (all `final_*.json`, trials, summaries, Phase 3 outputs), `figures/`, `docs/PHASE3_BRIEF.md`, `archive/` (aborted attempts). Git tags: `phase0-freeze-v2`, `phase3-sealed`. Default branch: `master`.
