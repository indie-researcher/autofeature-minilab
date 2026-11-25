#!/usr/bin/env bash
# p5b_run.sh - bash p5b_run.sh --dry | push | status | collect | collect-local   (requires git tag phase5b-freeze for push; FORCE=1 overrides)
set -euo pipefail
cd "$(dirname "$0")"
KUSER="${KUSER:-yoka33}"; MODE="${1:-}"
NBS=(p5b_a_gates:af-p5b-a-gates p5b_b_gates:af-p5b-b-gates p5b_c_gates:af-p5b-c-gates)
case "$MODE" in
  status) for e in "${NBS[@]}"; do kaggle kernels status "$KUSER/${e#*:}" 2>&1 | grep -v outdated || true; done; exit 0 ;;
  collect|collect-local)
    if [ "$MODE" = "collect" ]; then
      for e in "${NBS[@]}"; do nb="${e%%:*}"; slug="${e#*:}"
        echo "== $slug =="; kaggle kernels status "$KUSER/$slug" 2>&1 | grep -v outdated || true
        rm -rf "out_p5b/$slug"; mkdir -p "out_p5b/$slug" "results/p5b/$nb"
        kaggle kernels output "$KUSER/$slug" -p "out_p5b/$slug" 2>&1 | grep -v outdated | tail -2 || echo "(download failed)"
        find "out_p5b/$slug" -type f \( -name 'p5_run_*.json' -o -name 'p5_runs.csv' \) -exec cp {} "results/p5b/$nb/" \;
      done
    fi
    python3 code/af_phase5b_analyze.py; exit 0 ;;
esac
for e in "${NBS[@]}"; do d="kaggle/${e%%:*}"
  [ -f "$d/run.py" ] && [ -f "$d/kernel-metadata.json" ] || { echo "ERROR: $d not built (bash p5b_prepare.sh)"; exit 1; }
  python3 -m py_compile "$d/run.py"; grep -q "\"id\": \"$KUSER/${e#*:}\"" "$d/kernel-metadata.json" || { echo "ERROR: id mismatch in $d"; exit 1; }
  echo "ok  $d"
done
if ! git rev-parse -q --verify refs/tags/phase5b-freeze >/dev/null 2>&1; then
  [ "${FORCE:-}" = "1" ] || { echo "ERROR: tag phase5b-freeze missing (commit the plan and tag it, or FORCE=1)"; exit 1; }
fi
[ "$MODE" = "--dry" ] && { echo "dry run: nothing pushed"; exit 0; }
[ "$MODE" = "push" ] || { echo "usage: bash p5b_run.sh --dry|push|status|collect|collect-local"; exit 1; }
kaggle kernels list --mine --page-size 1 >/dev/null 2>&1 || { echo "Kaggle auth failed"; exit 1; }
FAILED=()
for e in "${NBS[@]}"; do d="kaggle/${e%%:*}"; echo "-- $d"; kaggle kernels push -p "$d" 2>&1 | grep -v outdated || FAILED+=("$d"); done
[ ${#FAILED[@]} -eq 0 ] || { echo "FAILED pushes: ${FAILED[*]}"; exit 1; }
echo "Pushed. Estimated: a ~1.2 h, b ~2.2 h, c ~4.8 h. status: bash p5b_run.sh status | collect: bash p5b_run.sh collect"
