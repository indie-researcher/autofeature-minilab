#!/usr/bin/env bash
# p5_run.sh - push / monitor / collect the three Phase 5 regime-map notebooks (they run in parallel on Kaggle).
#   bash p5_run.sh --dry | push | status | collect | collect-local
# push requires the git tag phase5-freeze (override with FORCE=1). collect downloads to out_p5/ and results/p5/<nb>/, then runs the analysis.
set -euo pipefail
cd "$(dirname "$0")"
KUSER="${KUSER:-yoka33}"; MODE="${1:-}"
NBS=(p5_a_regime:af-p5-a-regime p5_b_regime:af-p5-b-regime p5_c_regime:af-p5-c-regime)
case "$MODE" in
  status) for e in "${NBS[@]}"; do kaggle kernels status "$KUSER/${e#*:}" 2>&1 | grep -v outdated || true; done; exit 0 ;;
  collect|collect-local)
    if [ "$MODE" = "collect" ]; then
      for e in "${NBS[@]}"; do nb="${e%%:*}"; slug="${e#*:}"
        echo "== $slug =="; kaggle kernels status "$KUSER/$slug" 2>&1 | grep -v outdated || true
        rm -rf "out_p5/$slug"; mkdir -p "out_p5/$slug" "results/p5/$nb"
        kaggle kernels output "$KUSER/$slug" -p "out_p5/$slug" 2>&1 | grep -v outdated | tail -2 || echo "(download failed)"
        find "out_p5/$slug" -type f \( -name 'p5_run_*.json' -o -name 'p5_runs.csv' \) -exec cp {} "results/p5/$nb/" \;
      done
    fi
    python3 code/af_phase5_analyze.py --root results/p5 --out results/p5; exit 0 ;;
esac
echo "== pre-flight =="
for e in "${NBS[@]}"; do d="kaggle/${e%%:*}"
  [ -f "$d/run.py" ] && [ -f "$d/kernel-metadata.json" ] || { echo "ERROR: $d not built (bash p5_prepare.sh)"; exit 1; }
  python3 -m py_compile "$d/run.py"
  grep -q "\"id\": \"$KUSER/${e#*:}\"" "$d/kernel-metadata.json" || { echo "ERROR: metadata id mismatch in $d"; exit 1; }
  echo "ok  $d"
done
if ! git rev-parse -q --verify refs/tags/phase5-freeze >/dev/null 2>&1; then
  [ "${FORCE:-}" = "1" ] || { echo "ERROR: tag phase5-freeze missing (commit the plan and tag it, or FORCE=1)"; exit 1; }
  echo "WARNING: no phase5-freeze tag (FORCE=1)"
fi
[ "$MODE" = "--dry" ] && { echo "dry run: nothing pushed"; exit 0; }
[ "$MODE" = "push" ] || { echo "usage: bash p5_run.sh --dry|push|status|collect|collect-local"; exit 1; }
kaggle kernels list --mine --page-size 1 >/dev/null 2>&1 || { echo "Kaggle auth failed"; exit 1; }
FAILED=()
for e in "${NBS[@]}"; do d="kaggle/${e%%:*}"; echo "-- $d"
  kaggle kernels push -p "$d" 2>&1 | grep -v outdated || FAILED+=("$d"); done
[ ${#FAILED[@]} -eq 0 ] || { echo "FAILED pushes: ${FAILED[*]}"; exit 1; }
echo "Pushed. Estimated runtimes: a ~4 h, b ~6.5 h, c ~5.5 h (not measured). status: bash p5_run.sh status | collect: bash p5_run.sh collect"
