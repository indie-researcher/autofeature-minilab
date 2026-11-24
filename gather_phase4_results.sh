#!/usr/bin/env bash
# gather_phase4_results.sh  (v2: download, then one Python step organizes + reports)
# Usage: ./gather_phase4_results.sh            (set KUSER if your Kaggle username differs)
set -euo pipefail
KUSER="${KUSER:-yoka33}"
cd "$(dirname "$0")"
OUT_ROOT="out_phase4"
mkdir -p "$OUT_ROOT"

echo "==> Kernel status"
for K in af-phase4-a-cpu af-phase4-a-gpu af-phase4-b-gpu af-phase4-c-cpu; do
  echo "--- $K ---"; kaggle kernels status "$KUSER/$K" || true
done

for SLUG in af-phase4-a-cpu af-phase4-a-gpu af-phase4-b-gpu af-phase4-c-cpu; do
  echo; echo "==> Downloading $KUSER/$SLUG"
  rm -rf "$OUT_ROOT/$SLUG"; mkdir -p "$OUT_ROOT/$SLUG"
  kaggle kernels output "$KUSER/$SLUG" -p "$OUT_ROOT/$SLUG"
  find "$OUT_ROOT/$SLUG" -type f -name 'final_*.json' | wc -l | xargs echo "final files:"
done

echo; echo "==> Organize folders and build the report"
python3 code/af_phase4_report.py --organize --raw "$OUT_ROOT" --root results/phase4
echo; echo "==> Done. Paste results/phase4/PHASE4_REPORT.md into the chat."
