# Phase 4 report (generated)

19 finished runs (13 search, 6 canary) across 4 notebooks; 2 incomplete runs listed at the end. All gap numbers use DEV feedback (optimistic); SEALED is not used in Phase 4.

## Search arms by rung

| rung | arm | runs | proposals | capped | full_tests | accepted_real | gap_mean | gap_median | gap_min | gap_max | distinct_outcomes | needle_accepted | new_axis | dups | minutes | decoys_tested | decoys_accepted |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | L-FDR-14B | 1 | 60.00 | 0 | 13.00 | 2.00 | 34.01 | 34.01 | 34.01 | 34.01 | 1 | 0 | 1.00 | 5.00 | 35.54 | 2 | 0 |
| 1 | LF-FDR-14B | 2 | 60.00 | 0 | 3.50 | 2.00 | 17.94 | 17.94 | 15.66 | 20.21 | 2 | 0 | 1.50 | 8.00 | 22.48 | 1 | 0 |
| 1 | R-FDR | 5 | 59.00 | 2 | 21.60 | 2.60 | 31.54 | 33.30 | 22.06 | 39.76 | 5 | 0 | 1.20 | 2.00 | 43.60 | 17 | 0 |
| 2 | R-FDR | 5 | 60.00 | 0 | 9.00 | 2.60 | 34.95 | 32.27 | 28.14 | 41.94 | 5 | 0 | 1.00 | 0.40 | 27.26 | 4 | 0 |

## Per run (search arms)

| rung | arm | run | n_proposals | capped | n_full_tests | accepted_real | accepted_decoys | decoys_tested | dev_gap_closed_pct | minutes | needle_accepted | accepted_new_axis | n_dup_total | notebook |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | L-FDR-14B | 1 | 60 | False | 13 | 2 | 0 | 2 | 34.01 | 35.54 | False | 1 | 5 | a_gpu |
| 1 | LF-FDR-14B | 1 | 60 | False | 1 | 1 | 0 | 0 | 15.66 | 19.25 | False | 1 | 5 | b_gpu |
| 1 | LF-FDR-14B | 2 | 60 | False | 6 | 3 | 0 | 1 | 20.21 | 25.70 | False | 2 | 11 | b_gpu |
| 1 | R-FDR | 1 | 56 | True | 30 | 4 | 0 | 4 | 37.96 | 50.19 | False | 2 | 3 | a_cpu |
| 1 | R-FDR | 2 | 60 | False | 14 | 3 | 0 | 2 | 33.30 | 37.09 | False | 1 | 1 | a_cpu |
| 1 | R-FDR | 3 | 59 | True | 30 | 3 | 0 | 7 | 22.06 | 55.66 | False | 3 | 1 | a_cpu |
| 1 | R-FDR | 4 | 60 | False | 21 | 1 | 0 | 4 | 39.76 | 44.16 | False | 0 | 4 | a_cpu |
| 1 | R-FDR | 5 | 60 | False | 13 | 2 | 0 | 0 | 24.62 | 30.88 | False | 0 | 1 | a_cpu |
| 2 | R-FDR | 1 | 60 | False | 9 | 2 | 0 | 1 | 41.94 | 26.78 | False | 0 | 0 | a_cpu |
| 2 | R-FDR | 2 | 60 | False | 5 | 3 | 0 | 1 | 28.14 | 24.95 | False | 2 | 0 | a_cpu |
| 2 | R-FDR | 3 | 60 | False | 15 | 3 | 0 | 2 | 30.71 | 36.59 | False | 1 | 1 | a_cpu |
| 2 | R-FDR | 4 | 60 | False | 3 | 2 | 0 | 0 | 41.70 | 16.96 | False | 1 | 1 | a_cpu |
| 2 | R-FDR | 5 | 60 | False | 13 | 3 | 0 | 0 | 32.27 | 31.02 | False | 1 | 0 | a_cpu |

## Accepted operations (counts over runs)

- r1 L-FDR-14B: {'ratio': 1, 'lag': 1}
- r1 LF-FDR-14B: {'ratio': 4}
- r1 R-FDR: {'lag': 6, 'ratio': 4, 'roll': 1, 'count': 2}
- r2 R-FDR: {'count': 4, 'nunique': 3, 'ratio': 1, 'cumcount': 1, 'roll': 1, 'lag': 3}

## First accepted feature per run

- r1 L-FDR-14B #1: `{"den": {"keys": ["c0", "m15"], "op": "count"}, "num": {"keys": ["c0", "h"], "op": "nunique", "target": "c1"}, "op": "ratio"}`
- r1 LF-FDR-14B #1: `{"den": {"keys": ["c0", "m15"], "op": "count"}, "num": {"keys": ["c0", "h"], "op": "nunique", "target": "c1"}, "op": "ratio"}`
- r1 LF-FDR-14B #2: `{"den": {"keys": ["c1", "m5"], "op": "roll", "w": 60}, "num": {"keys": ["c0", "h"], "op": "nunique", "target": "c1"}, "op": "ratio"}`
- r1 R-FDR #1: `{"dir": "prev", "k": 3, "keys": ["c0", "c2", "c4"], "op": "lag"}`
- r1 R-FDR #2: `{"dir": "next", "k": 1, "keys": ["c1", "c2", "c4", "m1"], "op": "lag"}`
- r1 R-FDR #3: `{"keys": ["c0", "m1"], "op": "roll", "w": 600}`
- r1 R-FDR #4: `{"keys": ["c0", "c2"], "op": "count"}`
- r1 R-FDR #5: `{"keys": ["c4"], "op": "count"}`
- r2 R-FDR #1: `{"keys": ["c4"], "op": "count"}`
- r2 R-FDR #2: `{"keys": ["c0", "c3", "h"], "op": "nunique", "target": "c2"}`
- r2 R-FDR #3: `{"keys": ["c0", "m5"], "op": "cumcount"}`
- r2 R-FDR #4: `{"keys": ["c1", "c2", "c3", "c4"], "op": "nunique", "target": "m5"}`
- r2 R-FDR #5: `{"dir": "prev", "k": 3, "keys": ["c1", "c4", "m5"], "op": "lag"}`

## Canary: gate-only decoy stress test (proxy bypassed)

| rung | arm | runs | decoys_tested | decoys_accepted | fpr | wilson95_low | wilson95_high | mean_dev_gap_from_noise |
|---|---|---|---|---|---|---|---|---|
| 2 | C-FDR | 3 | 90 | 0 | 0.0000 | 0.0000 | 0.0409 | 0.0000 |
| 2 | C-naive | 3 | 90 | 3 | 0.0333 | 0.0114 | 0.0935 | 2.3198 |

_fpr = accepted decoys / decoys tested. Wilson 95% interval. mean_dev_gap_from_noise = DEV gap 'closed' by accepted noise._

## Incomplete runs (trials without a final file; not averaged)

| notebook | rung | arm | run | rows_logged | accepted_so_far |
|---|---|---|---|---|---|
| a_gpu | 1 | L-FDR-14B | 2 | 59 | 3 |
| b_gpu | 1 | LF-FDR-14B | 3 | 52 | 1 |

