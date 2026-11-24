#!/usr/bin/env bash
# p4b_push.sh - push the four Phase 4b CPU notebooks; they then run in parallel on Kaggle.
#   bash p4b_push.sh --dry      checks only, pushes nothing
#   bash p4b_push.sh            push all four (requires git tag phase4b-freeze; override: FORCE=1)
#   bash p4b_push.sh --status   status of the four kernels
# Kaggle allows several concurrent CPU sessions; if a push is refused for quota, wait for a slot and push the missing one by name.
set -euo pipefail
cd "$(dirname "$0")"
KUSER="${KUSER:-yoka33}"
MODE="${1:-}"
NBS=(p4b_1_canary_naive:af-p4b-1-canary-naive p4b_2_canary_fdr:af-p4b-2-canary-fdr p4b_3_audit:af-p4b-3-audit p4b_4_census:af-p4b-4-census)

if [ "$MODE" = "--status" ]; then
  for e in "${NBS[@]}"; do kaggle kernels status "$KUSER/${e#*:}" 2>&1 | grep -v "outdated" || true; done
  exit 0
fi

echo "== pre-flight =="
for e in "${NBS[@]}"; do
  d="kaggle/${e%%:*}"; [ -f "$d/run.py" ] && [ -f "$d/kernel-metadata.json" ] || { echo "ERROR: $d not built (run p4b_prepare.sh)"; exit 1; }
  python3 -m py_compile "$d/run.py"
  grep -q "\"id\": \"$KUSER/${e#*:}\"" "$d/kernel-metadata.json" || { echo "ERROR: $d metadata id does not match $KUSER/${e#*:}"; exit 1; }
  echo "ok  $d"
done
if ! git rev-parse -q --verify refs/tags/phase4b-freeze >/dev/null 2>&1; then
  [ "${FORCE:-}" = "1" ] || { echo "ERROR: git tag phase4b-freeze missing. Commit docs/PHASE4B_PLAN.md and tag it first (or FORCE=1)."; exit 1; }
  echo "WARNING: pushing without the phase4b-freeze tag (FORCE=1)"
fi
[ "$MODE" = "--dry" ] && { echo "dry run: nothing pushed"; exit 0; }
command -v kaggle >/dev/null || { echo "kaggle CLI not found"; exit 1; }
kaggle kernels list --mine --page-size 1 >/dev/null 2>&1 || { echo "Kaggle authentication failed (check ~/.kaggle/kaggle.json or env vars)"; exit 1; }

echo "== pushing =="
FAILED=()
for e in "${NBS[@]}"; do
  d="kaggle/${e%%:*}"
  echo "-- $d"
  kaggle kernels push -p "$d" 2>&1 | grep -v "outdated" || FAILED+=("$d")
done
[ ${#FAILED[@]} -eq 0 ] || { echo "FAILED pushes: ${FAILED[*]}"; exit 1; }
echo
echo "All four pushed. Expected runtimes (estimates): canary ~7-8 h each, audit ~5.5-6.5 h, census ~3.5-4 h."
echo "status:   bash p4b_push.sh --status"
echo "collect:  bash p4b_collect.sh"
