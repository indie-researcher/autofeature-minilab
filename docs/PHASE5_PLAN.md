# Phase 5 plan (freeze before launching; tag `phase5-freeze`)

**Q2 - regime map.** When does a statistical gate (LORD++) pay off versus keep-if-better? Phase 3 found no benefit at 25M rows and ~10 full tests per run.
Design: random proposals from the Phase 3 grammar, same sequence under both gates (paired); 20% permuted-real decoys; no proxy screen;
training size varied by IP subsample (K = 200, 50, 10, 2 -> roughly 7k to 740k training rows); proposals per run B = 30 / 60 / 120.
Metric: optimism = DEV gain (3 fresh seeds) - holdout gain (same final stack re-fitted on 2017-11-08, 3 seeds), AUC units; also holdout gain (net benefit),
accepted decoys, fraction of tests where the gate could not accept (alpha_t < 2^-12).
- **H5-1:** the naive gate's excess optimism (naive - FDR, paired) grows as n shrinks. Supported if Spearman(d_optimism, log n) <= -0.6 AND the paired 95% bootstrap CI at the smallest n has lower bound > 0.
- **H5-2:** at the smallest n, FDR has the higher holdout gain (paired CI of naive - FDR holdout gain has upper bound < 0). Reported either way.
- Fail rule: "no regime in this grid where the gate matters; keep-if-better is harmless on this task."
- Caveats (disclosed): holdout day 2017-11-08 was already used for Phase 3 verdicts (not fresh; 2017-11-07 stays reserved); random proposer only;
  12-block gate can die after ~8 tests; materiality floor is fixed in absolute AUC units; one dataset; IP-subsample (not row-subsample) changes n.

**Q1 - calibration (post-processing, no notebook).** From the canary trial logs: ECDF of decoy p-values versus nominal levels (Wilson CIs), decoy d_full distribution and the
share above the materiality floor. Descriptive.
