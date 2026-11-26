# results/ index
- `R`, `R-naive`, `L-naive`, `L-FDR`, `L-FDR-3B`, `L-FDR-14B`: Phase 1-2 arms (5 runs each: `final_*.json`, `trials_*.jsonl`, `summary_*`), DEV only.
- `phase0`: frozen references (`spec_freeze.json`), proxy check, grammar self-test.
- `phase3`: the one-shot SEALED run (30 runs), decoy check, lock, method notes. Tag `phase3-sealed`.
- `diagnostics`: exploratory needle / diversity diagnostics from the Phase 1-2 trial logs.
- `phase4`: Phase 4 (random rungs 1-2, canary pilot, partial 14B): exploratory pilots, `PHASE4_REPORT.md`.
- `p4b`: Phase 4b CPU tests (canary expansion, audit of FDR-rejected candidates, grammar census), `P4B_REPORT.md`.
- `p5`: Phase 5 regime map (naive vs 12-block LORD++), `P5_REPORT.md`, figures.
- `p5b`: Phase 5b (24-block gate, decoy-null gate), `P5B_REPORT.md` (once collected).
