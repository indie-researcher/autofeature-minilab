#!/usr/bin/env bash
# One command: pre-flight checks, then push the four Phase 4 notebooks
# (a_cpu, a_gpu, b_gpu, c_cpu).
#
#   ./run_phase4.sh --dry      build + check everything, push nothing
#   ./run_phase4.sh            push all four
#
# The one-shot seal notebook is NOT pushed here; run it last.

set -euo pipefail

export KUSER="${KUSER:-yoka33}"
DRY="${1:-}"

cd "$(dirname "$0")/.."
chmod +x scripts/push_phase4.sh

echo "== pre-flight =="

# Locate the Phase 4 source script
SRCF="af_phase4.py"
[ -f "$SRCF" ] || SRCF="code/af_phase4.py"
[ -f "$SRCF" ] || {
  echo "MISSING af_phase4.py (put it in the repo root or code/)"
  exit 1
}

[ -f "scripts/push_phase4.sh" ] || {
  echo "MISSING push_phase4.sh"
  exit 1
}

if [ "$DRY" != "--dry" ]; then
  command -v kaggle >/dev/null || {
    echo "kaggle CLI not found: pip install kaggle"
    exit 1
  }

  # Do not fail merely because ~/.kaggle/kaggle.json is absent.
  # The Kaggle CLI may use another config location or environment variables.
  # Test authentication with a lightweight API call instead.
  if kaggle kernels list --mine --page-size 1 >/dev/null 2>&1; then
    echo "kaggle authentication OK"
  else
    echo "Kaggle authentication failed."
    echo "Check one of these:"
    echo "  1. kaggle.json in ~/.kaggle/"
    echo "  2. KAGGLE_USERNAME and KAGGLE_KEY environment variables"
    echo "  3. KAGGLE_CONFIG_DIR pointing to a directory containing kaggle.json"
    echo "  4. Run: kaggle configure"
    exit 1
  fi

  # Competition access is useful but should not block a kernel push
  # if the competition rules have already been accepted in the browser.
  if kaggle competitions files talkingdata-adtracking-fraud-detection >/dev/null 2>&1; then
    echo "competition access OK"
  else
    echo "WARNING: could not list competition files."
    echo "If kernel pushes work, this is usually harmless."
    echo "If needed, accept the rules at:"
    echo "  https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/rules"
  fi

  echo "kaggle OK; user $KUSER"
fi

echo
echo "== local sanity (grammar, about 10 seconds) =="

python3 "$SRCF" --dry-run --n 60 2>&1 |
  grep -oE "dry-run rung=[0-9]: grammar_ok=[A-Za-z]+|effective_atomic': [0-9]+|total_effective': [0-9]+" |
  paste -sd' ' |
  sed 's/dry-run/\n dry-run/g'

echo
echo "== building and pushing notebooks =="

for nb in a_cpu a_gpu b_gpu c_cpu; do
  echo
  echo "== $nb =="
  ./scripts/push_phase4.sh "$nb" $DRY
done

cat <<EOF

== next ==

status:
  for n in a-cpu a-gpu b-gpu c-cpu; do
    kaggle kernels status $KUSER/af-phase4-\$n
  done

outputs:
  kaggle kernels output $KUSER/af-phase4-a-cpu -p out_phase4/af-phase4-a-cpu
  kaggle kernels output $KUSER/af-phase4-a-gpu -p out_phase4/af-phase4-a-gpu
  kaggle kernels output $KUSER/af-phase4-b-gpu -p out_phase4/af-phase4-b-gpu
  kaggle kernels output $KUSER/af-phase4-c-cpu -p out_phase4/af-phase4-c-cpu

resume:
  If a notebook stops at the time guard, attach its own previous output
  and re-push it:

  KSRC=$KUSER/af-phase4-a-cpu ./scripts/push_phase4.sh a_cpu
  KSRC=$KUSER/af-phase4-a-gpu ./scripts/push_phase4.sh a_gpu
  KSRC=$KUSER/af-phase4-b-gpu ./scripts/push_phase4.sh b_gpu
  KSRC=$KUSER/af-phase4-c-cpu ./scripts/push_phase4.sh c_cpu

LAST, once all four are done and the plan is committed:

  KSRC=$KUSER/af-phase4-a-cpu,$KUSER/af-phase4-a-gpu,$KUSER/af-phase4-b-gpu,$KUSER/af-phase4-c-cpu \
  P3DATA=$KUSER/af-phase3-results \
  ./scripts/push_phase4.sh seal
EOF
