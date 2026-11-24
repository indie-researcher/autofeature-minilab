# Phase 4b report (generated)

## Tests 1-2: canary (gate-only decoy stress test)

- **C-naive**: 13 accepted of 390 decoys over 13 runs -> 0.0333 (Wilson 95% 0.0196-0.0562); mean DEV gap 'closed' by runs that accepted noise: 2.02%
- **C-FDR**: 0 accepted of 390 decoys over 13 runs -> 0.0000 (Wilson 95% 0.0000-0.0098)

One-sided Fisher exact (naive > FDR): p = 0.0001103. Pre-registered rule: p < 0.05 -> 'naive passes noise that FDR rejects'; else 'not distinguishable at this n'. **Outcome: naive passes noise that FDR rejects.**

Paired runs (same run id = same decoys): 13; naive accepted noise while FDR did not in 13; the reverse in 0.

## Test 3: audit of FDR-rejected candidates (Phase 3 sealed window; post hoc, descriptive)

```
{
  "n_candidates": 63,
  "n_stacks": 17,
  "sealed_baseline_3seed": 0.9741439506655306,
  "share_positive": 0.7777777777777778,
  "share_material": 0.36507936507936506,
  "median_delta": 0.00015544771045583605,
  "mean_delta": 0.00018893296678859097,
  "decoy_n": 29,
  "decoy_median": -0.00020144185366451536,
  "decoy_p95": -2.6601795751912695e-05,
  "decoy_share_positive": 0.034482758620689655,
  "share_above_decoy_p95": 0.8095238095238095,
  "prob_cand_gt_decoy": 0.8856048166392994,
  "by_arm": {
    "L-FDR": {
      "n": 12,
      "median": 0.0003559648699832015,
      "n_positive": 12,
      "n_material": 8
    },
    "L-FDR-14B": {
      "n": 33,
      "median": 0.00010566961159241828,
      "n_positive": 24,
      "n_material": 9
    },
    "L-FDR-3B": {
      "n": 4,
      "median": 0.000515046680907516,
      "n_positive": 4,
      "n_material": 4
    },
    "R": {
      "n": 14,
      "median": 0.00010225292055228893,
      "n_positive": 9,
      "n_material": 2
    }
  },
  "rule": "gate discarded real signal if >=50% of candidates have delta>0 AND median candidate delta > decoy p95; else rejections are consistent with noise",
  "rule_outcome": "discarded real signal"
}
```

## Test 4: grammar census (DEV window; descriptive)

```
{
  "n_nonratio": 796,
  "n_ratio_sampled": 300,
  "proxy_theta": 0.0005,
  "nonratio_good": 697,
  "nonratio_good_share": 0.8756281407035176,
  "ratio_good": 249,
  "ratio_good_share": 0.83,
  "ratio_good_wilson95": [
    0.7833848085580865,
    0.8682705276434078
  ],
  "sampler_p_ratio": 0.14904666666666666,
  "sampler_p_good_nonratio": 0.7348300000000001,
  "sampler_p_good_total": 0.8585387333333334,
  "expected_good_in_48_random_proposals": 41.209859200000004,
  "needle": {
    "found_in_census": true,
    "d_proxy": 0.008208450345841167,
    "rank_proxy_nonratio": 2,
    "sampler_p": 0.0005566666666666667,
    "mc_n": 300000,
    "d_full_single": 0.005603888600278162
  },
  "good_share_by_op": {
    "count": 0.9107142857142857,
    "cumcount": 0.6785714285714286,
    "lag": 0.9255952380952381,
    "nunique": 0.9388888888888889,
    "ratio": 0.83,
    "roll": 0.7619047619047619
  },
  "minutes": 326.4
}
```

