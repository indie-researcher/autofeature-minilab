#!/usr/bin/env bash
# tidy_repo.sh - reorganize the AutoFeature repo without breaking the pipelines. Dry run by default.
#   bash tidy_repo.sh            show the plan (changes nothing)
#   bash tidy_repo.sh --apply    do it (uses `git mv` for tracked files, so history is kept), then print a summary
# Moves only things that are NOT part of a live pipeline path: root scripts -> scripts/, session notes -> archive/notes/, superseded code,
# stale Kaggle notebooks and stale result folders -> archive/. code/, results/<arm>/, results/phase*/ , kaggle/<live notebooks> stay where they are.
set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || dirname "$0")"
APPLY=0; [ "${1:-}" = "--apply" ] && APPLY=1
say() { printf '%s\n' "$*"; }
[ -d .git ] || { say "ERROR: not a git repository root"; exit 1; }
if [ "$APPLY" = 1 ] && [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  say "ERROR: tracked files have uncommitted changes. Commit or stash first (so this tidy-up is its own commit)."; exit 1; fi

mvg() {   # src dest
  [ -e "$1" ] || return 0
  say "  move  $1  ->  $2"
  [ "$APPLY" = 1 ] || return 0
  mkdir -p "$(dirname "$2")"
  if git ls-files --error-unmatch "$1" >/dev/null 2>&1 || [ -n "$(git ls-files "$1" 2>/dev/null | head -1)" ]; then git mv "$1" "$2"; else mv "$1" "$2"; fi
}

say "== 1. session notes -> archive/notes/"
for f in p4b_gather.txt phase5_getwindow.txt phase5_handoff.txt phase5_needed.txt struct.txt; do mvg "$f" "archive/notes/$f"; done

say "== 2. plans -> docs/"
mvg PHASE4_PLAN.md docs/PHASE4_PLAN.md

say "== 3. root shell scripts -> scripts/ (paths patched to run from the repo root)"
SCRIPTS="gather_phase4_results.sh p4b_collect.sh p4b_gather.sh p4b_prepare.sh p4b_push.sh p5_prepare.sh p5_run.sh p5b_prepare.sh p5b_run.sh push_phase4.sh run_phase4.sh"
for f in $SCRIPTS; do mvg "$f" "scripts/$f"; done

say "== 4. superseded code -> archive/code_superseded/"
mvg code/af_ph4.py archive/code_superseded/af_ph4.py                       # old third-party draft; the revised runner is code/af_phase4.py
mvg code/af_phase4_summarize.py archive/code_superseded/af_phase4_summarize.py   # replaced by code/af_phase4_report.py

say "== 5. stale Kaggle notebooks -> archive/kaggle_stale/ (never pushed or superseded: phase4_14B, phase4_R, phase4_seal, phase4_b_cpu pilot)"
for d in phase4_14B phase4_R phase4_seal phase4_b_cpu; do mvg "kaggle/$d" "archive/kaggle_stale/$d"; done

say "== 6. stale result folders -> archive/results_stale/"
mvg results/phase4_prerevision_nov21 archive/results_stale/phase4_prerevision_nov21
for d in results/phase4_prev_*; do [ -e "$d" ] && mvg "$d" "archive/results_stale/$(basename "$d")"; done
if [ -f results/contamination_probe_answer.json ] && [ -f results/L-FDR/contamination_probe_answer.json ] && cmp -s results/contamination_probe_answer.json results/L-FDR/contamination_probe_answer.json; then
  mvg results/contamination_probe_answer.json archive/results_stale/contamination_probe_answer_dup_of_L-FDR.json
else say "  (results/contamination_probe_answer.json left in place: not a byte-identical duplicate of results/L-FDR's)"; fi

say "== 7. raw downloads and caches: stop tracking, keep on disk =="
for d in out_p4b out_p5 out_p5b out_phase4 af_lab_r1 af_lab_r2; do [ -e "$d" ] && say "  untrack/ignore  $d"; done
find . -name __pycache__ -not -path "./.git/*" | head -3 | sed 's/^/  delete  /'; say "  delete  all __pycache__ / *.pyc"

if [ "$APPLY" != 1 ]; then say; say "DRY RUN only. Review, then: bash tidy_repo.sh --apply"; exit 0; fi

# ---- apply the non-move steps
touch .gitignore
for pat in '__pycache__/' '*.pyc' 'out_p4b/' 'out_p5/' 'out_p5b/' 'out_phase4/' 'af_lab_r1/' 'af_lab_r2/' '*.npz'; do grep -qxF "$pat" .gitignore || echo "$pat" >> .gitignore; done
git rm -r -q --cached --ignore-unmatch out_p4b out_p5 out_p5b out_phase4 af_lab_r1 af_lab_r2 >/dev/null 2>&1 || true
git rm -r -q --cached --ignore-unmatch -- $(git ls-files | grep -E '(__pycache__/|\.pyc$)' || true) >/dev/null 2>&1 || true
find . -name __pycache__ -not -path "./.git/*" -prune -exec rm -rf {} + 2>/dev/null || true

# ---- patch moved scripts so they still resolve the repo root and each other
if [ -d scripts ]; then
  for f in scripts/*.sh; do
    sed -i -e 's|cd "$(dirname "$0")"|cd "$(dirname "$0")/.."|' \
           -e 's|bash \(p4b_\|p5_\|p5b_\)\([a-z]*\)\.sh|bash scripts/\1\2.sh|g' \
           -e 's|\./push_phase4\.sh|./scripts/push_phase4.sh|g' \
           -e 's|chmod +x push_phase4\.sh|chmod +x scripts/push_phase4.sh|' \
           -e 's|\[ -f "push_phase4\.sh" \]|[ -f "scripts/push_phase4.sh" ]|' \
           -e 's|cat > p5b_run\.sh|cat > scripts/p5b_run.sh|' -e 's|chmod +x p5b_run\.sh|chmod +x scripts/p5b_run.sh|' \
           -e 's| p5b_prepare\.sh p5b_run\.sh| scripts/p5b_prepare.sh scripts/p5b_run.sh|' "$f"
    bash -n "$f" || say "WARNING: syntax check failed for $f"
  done
  chmod +x scripts/*.sh
fi

# ---- maps
mkdir -p docs
cat > docs/REPO_MAP.md <<'MD'
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
MD
cat > results/INDEX.md <<'MD'
# results/ index
- `R`, `R-naive`, `L-naive`, `L-FDR`, `L-FDR-3B`, `L-FDR-14B`: Phase 1-2 arms (5 runs each: `final_*.json`, `trials_*.jsonl`, `summary_*`), DEV only.
- `phase0`: frozen references (`spec_freeze.json`), proxy check, grammar self-test.
- `phase3`: the one-shot SEALED run (30 runs), decoy check, lock, method notes. Tag `phase3-sealed`.
- `diagnostics`: exploratory needle / diversity diagnostics from the Phase 1-2 trial logs.
- `phase4`: Phase 4 (random rungs 1-2, canary pilot, partial 14B): exploratory pilots, `PHASE4_REPORT.md`.
- `p4b`: Phase 4b CPU tests (canary expansion, audit of FDR-rejected candidates, grammar census), `P4B_REPORT.md`.
- `p5`: Phase 5 regime map (naive vs 12-block LORD++), `P5_REPORT.md`, figures.
- `p5b`: Phase 5b (24-block gate, decoy-null gate), `P5B_REPORT.md` (once collected).
MD
if ! grep -q "Phase 4b" RUN_ORDER.md 2>/dev/null; then cat >> RUN_ORDER.md <<'MD'

## After Phase 3
- Phase 4 (closed early; pilots only): `scripts/run_phase4.sh`, collect with `scripts/gather_phase4_results.sh`, report with `code/af_phase4_report.py`.
- Phase 4b (CPU): `scripts/p4b_prepare.sh` -> `scripts/p4b_push.sh` -> `scripts/p4b_collect.sh`.
- Phase 5 (regime map): `scripts/p5_prepare.sh` -> `scripts/p5_run.sh push|status|collect` (analysis: `code/af_phase5_analyze.py`; calibration: `code/af_phase5_calibration.py`).
- Phase 5b (gates): `scripts/p5b_prepare.sh` -> `scripts/p5b_run.sh push|status|collect` (analysis: `code/af_phase5b_analyze.py`).
MD
fi
say; say "APPLIED. Review:  git status --short | head -60"
say "Then:  git add -A && git commit -m 'Tidy repo: scripts/, archive/, gitignore raw downloads, repo map'"
