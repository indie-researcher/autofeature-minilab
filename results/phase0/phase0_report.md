# AutoFeature Mini-Lab - Phase 0 report

Windows: DEV 2017-11-09 / SEALED 2017-11-08, hours 8-15, obfuscated. Eval = hours 13-14.

| window | baseline (4 raw) | ceiling (human set) | gap | train/ES/eval rows | eval pos |
|---|---|---|---|---|---|
| DEV | 0.97040 | 0.98471 | 0.01432 | 1,424,757/572,391/2,082,017 | 16,821 |
| SEALED | 0.97421 | 0.98544 | 0.01123 | 1,469,995/591,583/2,198,357 | 14,798 |

**Noise:** ceiling seed sd 0.00011 -> materiality threshold **0.00022**.

**Proxy check:** Spearman 0.792 over 44 features (need >= 0.7) -> **PASS: use the proxy**.

**Grammar:** 53 random specs executed, ops covered ['count', 'cumcount', 'lag', 'nunique', 'ratio', 'roll'], mean finite fraction 0.986; invalid spec rejected: True.
End-to-end (proxy fold): real lag feature +0.00984 dAUC vs permuted decoy +0.00074.

## Before Phase 1
1. Run the contamination probe and log the answer.
2. `git tag phase0-freeze` (spec, grammar, thresholds are now frozen).
3. Do not look at SEALED again until Phase 3.