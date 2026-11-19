# AutoFeature Mini-Lab

A small, pre-registered study of **agentic feature search**: when an agent proposes features and a gate decides what to keep, which layer earns the trust: the proposer, the gate, or the sealed holdout day?

## Question

1. Does the evaluation harness (proxy screen, statistical gate, sealed day) matter more than the proposer (random vs LLM)?
2. Does LLM size (Qwen2.5 3B / 7B / 14B) change the result much?

Idea worth stealing: **canary decoys**. Insert known-null hypotheses (permuted real features) into the agent's queue and count how many it "discovers". That is a live estimate of the pipeline's false-discovery rate. (Same family as Boruta shadow features and knockoffs. No novelty claimed on online FDR; this is a controlled empirical case study.)

## Design

- **Data:** TalkingData AdTracking Fraud Detection. DEV = 2017-11-09, 08:00-16:00 (25.7M rows). SEALED = 2017-11-08, same hours, opened once, at the end.
- **Task:** agent picks features from a fixed grammar of 6 operations (count, nunique, cumcount, lag, roll, ratio) over keys c0-c4 and h. Selection, not invention.
- **Pipeline per proposal:** proxy screen (10% IP subsample, promote if AUC gain >= 0.0005) -> full DEV test -> gate -> stack.
- **Decoys:** 20% of proposals are permuted-real features (known noise).
- **Gates:** naive (keep if DEV AUC goes up) vs LORD++ online FDR (alpha 0.10, sign-flip p over 12 blocks, materiality floor 0.000219).
- **Arms (5 runs x 60 proposals each):** random + FDR, random + naive, Qwen 7B + FDR, Qwen 7B + naive, plus Qwen 3B and 14B with FDR.
- **Reference points:** baseline (4 raw ID columns) and a human ceiling. DEV 0.97040 / 0.98471, SEALED 0.97421 / 0.98544.
- **Metric:** gap closed (%) = (AUC - baseline) / (ceiling - baseline). Optimism gap = DEV gap closed minus SEALED gap closed.

**Hypotheses (frozen before SEALED was opened)**
- **H1:** the naive gate shows a larger DEV-to-SEALED optimism gap than the FDR gate. (Reframed: the proxy screen removes most decoys before the gate, so H1 is judged on the optimism gap of accepted real features, not on decoy counts.)
- **H2:** LLM proposals close more of the SEALED gap than random proposals, under this fixed grammar and budget.
- **H3:** proposer size changes the SEALED gap less than the gate does.

With 5 runs per arm, claims are **descriptive** (mean, median, range). Exact statement used: if all 5 runs of one arm beat all 5 of another, one-sided exact permutation p = 1/252 (about 0.004). Arms with fewer than 5 distinct final feature sets count as that many effective outcomes and are excluded from permutation tests.

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

Observations: the proxy screen removes most decoys before any gate sees them (54 of 60 for 7B + FDR), so decoy false-positive rates are undefined or rest on tiny n. Random beat every LLM arm on DEV. One random run (run 3) reached about 91% in both random arms, so medians matter more than means. 3B converged to the same single feature in 4 of 5 runs. 14B wastes far fewer proposals on duplicates (77 vs 272 for 7B, 305 for 3B).

### SEALED (opened once)

> TODO: fill from `results/phase3/phase3_arms.csv` after the Kaggle run. Replace the placeholders; do not edit the DEV table above.

| Arm | SEALED gap closed (mean / median / range) | Optimism gap | Effective outcomes |
|---|---|---|---|
| Random + FDR | | | |
| Random + naive | | | |
| Qwen 7B + FDR | | | |
| Qwen 7B + naive | | | |
| Qwen 3B + FDR | | | 2 |
| Qwen 14B + FDR | | | |

**Verdicts (rule outcomes from the analysis script):** H1 = TODO, H2 = TODO, H3 = TODO.

**Three numbers:** TODO, TODO, TODO.

**Post hoc decoy check (descriptive, not a pre-registered FDR replication):** the 29 gate-tested decoys were re-permuted on SEALED and added to each run's final stack. LORD++ was not re-simulated because the trial-level FDR state is not embedded. TODO: counts passing naive / materiality.

Figures: `figures/` (design diagram, noise kept and DEV-to-SEALED drop, proposer scaling, cost frontier).

## Honest limits

- One dataset, one DEV/SEALED day pair, 5 runs per arm.
- The grammar contains the known good operations: this tests selection, not invention.
- DEV was used for agent feedback, so DEV numbers are optimistic by construction.
- Qwen2.5 family only. Other families are future work.
- Possible model contamination. The contamination probes (3B, 7B, 14B) named other datasets, which is weak evidence of low contamination.
- 3B is effectively one outcome (low proposal diversity), so no size trend is claimed from it alone.
- The 7B proposer saturated in a duplicate loop in one run; a patch (more attempts, temperature ramp, duplicate feedback in the prompt) was applied and all runs then reached 60 proposals. Protocol deviation, disclosed here.
- Decoy FPR is weak evidence because the proxy screen removes most decoys first (gate-tested decoys: 29 in total).
- Token counters were cumulative across runs and were converted to per-run deltas. R and L-naive decoy proposal counts are derived from the shared schedule, not logged.
- The Phase 3 script was committed after the run started and is recorded by md5 (`results/phase3/PUSHED_MD5.txt`). An earlier crash happened before any feature AUC was computed on SEALED.

## Reproduce

Everything runs on Kaggle (CPU for Phase 3; LLM arms need a GPU with Ollama). See `RUN_ORDER.md` for the full order.

```bash
# Phase 0: freeze references (baseline, ceiling, materiality)        kaggle/phase0_1A
# Phase 1: random and naive arms                                      kaggle/1B_FDR, kaggle/1C_naive, kaggle/R_naive
# Phase 2: Qwen 3B / 14B arms                                         kaggle/phase2_3B, kaggle/phase2_14B
# Phase 3: open SEALED once, recompute accepted features and decoys
kaggle kernels push -p kaggle/phase3
```

Layout: `code/` (pipeline), `kaggle/` (one folder per notebook), `results/` (all `final_*.json`, trials, summaries, Phase 3 outputs), `figures/`, `docs/PHASE3_BRIEF.md`. Git tags: `phase0-freeze-v2`, `phase3-script-pushed`.
