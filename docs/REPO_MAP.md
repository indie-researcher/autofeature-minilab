# Repository map

| Path | What it is |
|---|---|
| `README.md` | project write-up (Phase 3 headline results + exploratory) |
| `RUN_ORDER.md` | order in which the notebooks were run |
| `code/` | pipeline code: `af_phase0.py` (references), `af_phase1*`/`af_phase2*` (arms), `af_phase3_sealed_combined.py` (one-shot sealed run), `af_phase3_analyze.py`, `af_phase4*.py`, `af_phase5*.py` |
| `kaggle/` | one folder per Kaggle notebook (`kernel-metadata.json` + generated `run.py`); `phase*` live runs, `p4b_*` / `p5_*` / `p5b_*` CPU notebooks |
| `scripts/` | shell scripts: `push_*`, `p4b_*`, `p5_*`, `p5b_*` (prepare / push / status / collect); run them from anywhere, they cd to the repo root |
| `results/` | all outputs: arm folders `R`, `R-naive`, `L-naive`, `L-FDR`, `L-FDR-3B`, `L-FDR-14B` (Phase 1-2), `phase0`, `phase3`, `diagnostics`, `phase4`, `p4b`, `p5`, `p5b` (see `results/INDEX.md`) |
| `figures/` | figures for the README / post |
| `docs/` | plans frozen before runs (`PHASE4_PLAN`, `PHASE4B_PLAN`, `PHASE5_PLAN`, `PHASE5B_PLAN`), `PHASE3_BRIEF`, this map |
| `archive/` | aborted attempts, superseded code, stale notebooks and result folders (kept for provenance, not used by any script) |
| `out_*/`, `af_lab_r*/` | raw Kaggle downloads and local work dirs (git-ignored; re-created by the `collect` scripts) |
