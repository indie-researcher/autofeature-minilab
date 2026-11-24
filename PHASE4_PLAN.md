# AutoFeature Phase 4 — plan (draft for pre-registration)

## Headline
**Does the LLM earn its keep once random search can't brute-force the space?**
Same pipeline as Phase 3, but the grammar grows (rung 0 → 1 → 2), measured on a **fresh sealed day**, with a **canary audit** of the gate.

Why this one: Phase 3 ended with "random ≈ LLM, and random's lead is one lucky draw of one feature". The natural reviewer question is "your space is too small; random is brute force". Phase 4 removes that excuse. A null is still a clean curve; a crossover is the headline.

## Space sizes (from `--dry-run`, effective = after removing redundant specs)
| Rung | Adds | Atomic specs | Total incl. ratios | Random hits needle / run |
|---|---|---|---|---|
| 0 | Phase 3 grammar | 796 | 2.1e5 | 7.0% |
| 1 | time-bucket keys m15 / m5 / m1 / s10 | 2,986 (3.8×) | 3.5e6 (17×) | 0.7% |
| 2 | 5 windows, 5 lags, `quantile`, `burst` | 8,287 (10.4×) | 4.1e7 (196×) | 1.0% |

Use **total specs (log scale)** as the x-axis. The draft claimed ~12× / ~70× atomic; nested time buckets make most of that redundant (nominal 10.8k / 40k).

## What the pilots showed (draft script, DEV only; kept as pilots, not as data)
| Finding | Evidence | Change |
|---|---|---|
| **The FDR gate dies after ~8 tests.** With 12 blocks the smallest attainable p is 2^-12 = 2.4e-4; LORD++ alpha_t falls below that at test 8 unless something was accepted early. | R-FDR rung 1: 4 of 7 runs accepted nothing (up to 46 tests); rung 2, 180 proposals: run 101 tested 127 and accepted nothing (264 min). Accepting runs got their features in the first ~7 tests. | 24 blocks, exact sign-flip test (meet-in-the-middle). `gate_alive` logged; canary counts only alive tests. |
| So random vs LLM is not readable yet. | Random DEV gap: 0 in 4 of 7 runs, but 22 / 42 / 53 % when the gate stayed alive (LLM 34%, n = 1). | Compare only under the fixed gate; report `zero_acc` per arm. |
| **Cost is the full test (~2 min), not the run.** | Proxy passes 50–77% of random proposals when the stack is empty: runs took 23–74 min (rung 1), 140–264 min (rung 2, 180). | Cap 30 full tests per run. Budget-axis (180 proposals) dropped. |
| Proxy is a weak filter at this size. | Spearman(d_proxy, d_full) = 0.08 (n = 55). A threshold of 0.0013 would have kept all 4 accepted features and cut tests 55 → 25. | Keep 0.0005 (frozen); `--proxy-theta` stays as a sensitivity flag only. |
| **The operator nudge decides the LLM's operator mix.** | 14B proposals over 8 operators: 15/15/15/14/14/14/14/14. Same pattern as Phase 3's "not narrower than random". Past ~90 proposals it produced 32 failed proposals in 26 slots. | New arm `LF-FDR-14B` without the nudge. |
| **The new axes carry signal.** | 3 of 4 accepted LLM features and the first feature of every accepting random run use a time bucket (`c0,m15`, `c0,m1`, `c0,m5`, `c1,s10`). Best DEV gap 52.6% (random, rung 1), above Phase 3's non-needle random (~47%). | H4-2 stays; DEV-only until sealed. |
| Canary audit was partly vacuous. | Decoys tested after gate death cannot be accepted. | Alive-only counts. |
| Token counts were cumulative across runs. | Pilot run 5 showed 450k prompt tokens. | Per-run deltas. |

## Arms (same constants as Phase 3 except blocks = 24 and the 30-test cap; 20% decoys; 5 runs each)
| Tier | Arm | Rungs | Role |
|---|---|---|---|
| A (headline) | R-FDR vs L-FDR-14B (Phase 3 prompt) | 1, 2 | proposer comparison, gate fixed |
| B | LF-FDR-14B (no operator nudge) | 1, 2 | is it the LLM prior or the operator rotation? |
| C | C-naive, C-FDR: every proposal a decoy, proxy bypassed; 30 tests × 3 runs | 2 | false-discovery rate of each gate |
| Bridge | Phase 3 final stacks (12-block gate, as run) re-evaluated on the new sealed day | 0 | rung 0 on the same day, no re-search |

Disclosed deviation: the Phase 3 gate (12 blocks) is replaced for Phase 4 arms; rung 0 comes only from Phase 3's own stacks.

## Hypotheses (freeze before the seal runs)
- **H4-1 (headline).** (L-14B − R) median sealed ΔAUC is ≥ 0.0005 larger at rung 2 than at rung 0. Fail rule: *the LLM prior adds nothing even at ~200× total space; grammar + gate carry the result.*
- **H4-2 (extended axes).** ≥ 3 of 5 runs of some arm accept a new-axis feature (time bucket, `quantile`, `burst`), and median `new_axis_gain` (sealed AUC of final stack minus same stack without new-axis features) is > 0. Fail rule: *the new axes were noise or duplicates.*
- **H4-3 (canary).** FDR-gate false-discovery rate on canary arms: 0 accepted of ≥ 59 gate-alive tests (95% upper bound ≤ 5%). Naive-gate rate reported, no threshold. Phase 3 had 0/29.
- **H4-1b.** LF − L (no nudge vs nudge) at rung 2, same contrast and threshold; reported either way.
- Exploratory only: proxy-pass dynamics (best `d_proxy` per proposal, share of proposals passing the screen, per arm and rung), needle replication on the new day, any run above the human ceiling (needs `--ceiling-specs`), op entropy, first-accept index.
- Claims stay descriptive (mean, median, range). With 5 runs per cell, an exact claim needs a 5-vs-5 separation (p = 1/252); report `distinct_stacks` next to every mean.

## Metric
Sealed **ΔAUC vs baseline** (3-seed means). Gap-closed % only if a ceiling spec file is supplied for the new day.

## Fresh sealed day
Phase 3 spent 2017-11-08. Phase 4 seals **2017-11-07, 08:00–16:00**, opened once by the `seal` notebook (lock file with script sha256; refuses a second run).

## Kaggle limits and how the plan fits
- 12 h per session (CPU/GPU), ~30 GPU-h/week (floating), 20 GB `/kaggle/working`. LightGBM runs on CPU in every notebook; the GPU is only for Ollama.
- Script stops cleanly at 620 min and resumes: attach the previous output with `KSRC=` and re-push.
- GPU notebooks need internet on (Ollama install + model pull). CPU notebooks run with internet off.
- Time: ~2 min per full test, so a 30-test run is ≤ ~70 min. a_cpu, a_gpu, b_gpu: 10 runs each ≈ 8–10 h (one session each; chain with `KSRC` if a notebook stops at 620 min).

## Order of work
1. Local: `python code/af_phase4.py --dry-run` (10 s). Confirm the table above.
2. Upload Phase 3 `results/final_*.json` as dataset `af-phase3-results` (bridge). Optional: `W_DEV.npz` as `af-w-dev`.
3. `KUSER=<you> ./push_phase4.sh a_cpu` and `a_gpu` (wave 1).
4. Then `b_gpu` and `c_cpu`.
5. Commit script + this plan, tag `phase4-freeze`.
6. `KSRC=<all finished notebooks> P3DATA=<you>/af-phase3-results ./push_phase4.sh seal`.

## Changes vs the draft `af_ph4.py` (why it was not safe as-is)
1. No sealed step, and 2017-11-08 was already spent → added `--seal` on 2017-11-07 with one-shot lock.
2. Nested time buckets made many specs identical (nominal 13.6× / 50× vs effective 3.8× / 10.4×) and defeated duplicate detection → canonical keys, one bucket per spec.
3. `quantile` / `nunique` over a coarser-or-equal time bucket gave constant features; `quantile` over ID codes has no meaning → legal-target rule, quantile over time buckets only.
4. Random arms were not paired (seed included the arm index) and resume replayed the RNG wrongly → per-proposal RNG keyed on (run, rung, i).
5. Ollama context 4096 with accumulating retries could silently drop the grammar prompt at rung 2 → 8192, only the last rejection kept.
6. Needle-probability stats ignored op weights, any-k, decoys → Monte Carlo over the real sampler.
7. `skiprows=range(...)` built a 100M+ element set → integer skip.
8. After the pilot: per-run sampling seed, random tie-break in the prompt nudge, per-run token deltas, `--proxy-theta` sensitivity flag (separate `--work-dir`), diagnostics (`stack_sig`, `proxy_pass_frac`, `best_d_proxy`).
9. Added canary arms, new-axis accounting, multi-rung runs on one data load, output chaining, 620-min guard.

## Not verified
Tested end-to-end only on synthetic data (`AF_TEST=1`, mock LLM). Not run on real data or Ollama. Day row offsets are asserted at load (start before 08:00, > 5M rows). Phase 3 `final_*.json` field names are assumed to match (`accepted`, `dev_final_auc`, `dev_baseline_remeasured`).
