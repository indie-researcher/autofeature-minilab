# Phase 5b plan (freeze before launching; tag `phase5b-freeze`)

Phase 5 result: keep-if-better has the higher net holdout gain than the 12-block LORD++ gate in all six cells, but the 12-block gate was dead (alpha_t < 2^-12)
for 45-94% of tests, so that comparison mixes "FDR control" with "this gate design starves". Phase 5b asks: **does any better-designed gate pay off?**

Arms (same proposals, IP subsamples and holdout as Phase 5; paired by (K, B, run)):
- **fdr24:** LORD++ alpha 0.10 with an exact 24-block sign-flip test (min p = 2^-24, gate cannot die within a run), materiality floor as before.
- **decoyq:** accept if the candidate's gain is in the top 5% of a rolling null of the last 40 decoy gains (empirical p = (1 + #{null >= d}) / (1 + n)), after a 20-decoy warm-up
  evaluated against the empty stack (warm-up tests are extra and not counted in n_tests), and gain >= materiality floor. The null mixes stack states (approximation, disclosed).
- **naive:** replicated in the cheap cells to verify that the pairing/determinism reproduces Phase 5 (max |delta hold gain| reported).
Cells: K=200/B=120, K=50/B=120 (cheap), K=10/B=120 (~144k rows), K=2/B=60 (~712k rows, the crossover region).

- **Primary rule:** a gate "wins" if the paired 95% bootstrap CI of (gate - naive) net holdout gain has lower bound > 0 in at least one cell. If none: "no gate tried beats keep-if-better below ~1M rows".
- **Secondary (descriptive):** optimism, accepted decoys (false-discovery proxy), accepted real features per run (power), frac_tests_dead.
- Caveats: random proposer; one dataset; holdout day 2017-11-08 reused (not fresh); decoyq warm-up adds ~20 tests; 4 runs in the K=2 cell.
