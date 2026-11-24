# Phase 4b plan (freeze before launching; tag `phase4b-freeze`)

**Closure of Phase 4.** The GPU arms (14B, no-nudge) and the fresh-sealed-day headline H4-1 are NOT run. Reason: cost versus expected
information after Phase 3 (LLM prior added nothing under the fixed grammar). Existing Phase 4 results (random rungs 1-2, canary pilot,
1-2 partial 14B runs) are reported as exploratory pilots.

All four tests are CPU-only, descriptive, and use data already seen (DEV, or the Phase 3 sealed day, which is NOT a fresh holdout).

1. **Canary, naive gate** (rung 2; run ids 4-13, 30 decoys per run, proxy bypassed) and
2. **Canary, FDR gate** (same run ids = same decoys, paired). Pooled with the 3+3 pilot runs. Report per-gate acceptance rate with a
   Wilson 95% interval and a one-sided Fisher exact test (naive > FDR). Verdict rule: "naive passes noise that FDR rejects" if
   Fisher p < 0.05; otherwise "not distinguishable at this n". Also report the DEV gap "closed" by accepted noise.
3. **Audit of FDR-rejected candidates** (63 non-decoy candidates that cleared the materiality floor but were rejected by LORD++ in
   Phase 3), scored on the Phase 3 sealed window (3-seed matched deltas, added to the run's final stack), compared with the 29 decoys.
   Verdict rule: "gate discarded real signal" if >= 50% have delta > 0 AND the median delta exceeds the decoy 95th percentile; otherwise
   "rejections consistent with noise / inconclusive".
4. **Grammar census** (Phase 3 grammar, DEV window): proxy gain of every non-ratio spec and a sampler-distributed sample of ratio specs;
   sampler probability of hitting a useful spec; rank and prior probability of the Phase 3 needle. Descriptive only.

Not claimed: any novelty on online FDR; any LLM-vs-random statement beyond Phase 3.
