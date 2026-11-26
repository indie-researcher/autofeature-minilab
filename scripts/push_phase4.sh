#!/usr/bin/env bash
# AutoFeature Phase 4 -> Kaggle.  Run from the repo root (finds af_phase4.py in ./ or code/).
#
#   ./scripts/push_phase4.sh <notebook> [--dry]        (KUSER defaults to yoka33)
#
# notebooks   a_cpu  random, FDR gate, rungs 1+2 x 5 runs                (CPU)  <- headline
#             a_gpu  Qwen-14B (Phase 3 prompt, operator nudge), rungs 1+2 (GPU)  <- headline
#             b_gpu  Qwen-14B WITHOUT the operator nudge, rungs 1+2        (GPU)
#             c_cpu  canary arms (all decoys, proxy bypassed), rung 2      (CPU, 30 tests x 3 runs x 2 gates)
#             seal   ONE-SHOT: opens fresh sealed day 2017-11-07 on every finished stack
# defaults in the script: 24 sign-flip blocks, max 30 full tests per run (wall-clock guard 620 min; resume via KSRC)
#
# optional env:  KSRC="user/af-phase4-a-cpu,user/af-phase4-a-gpu,..."   kernel outputs to attach
#                                                   (seal: all finished notebooks; resume: the notebook itself)
#                P3DATA="user/af-phase3-results"    dataset holding Phase 3 results/final_*.json (rung-0 bridge)
#                WCACHE="user/af-w-dev"             dataset holding W_DEV.npz (skips the ~10 min CSV scan)
set -euo pipefail
KUSER=${KUSER:-yoka33}
NB=${1:?usage: ./scripts/push_phase4.sh <a_cpu|a_gpu|b_gpu|c_cpu|seal> [--dry]}
DRY=${2:-}
if [ -z "${SRC:-}" ]; then if [ -f code/af_phase4.py ]; then SRC=code/af_phase4.py; else SRC=af_phase4.py; fi; fi
[ -f "$SRC" ] || { echo "missing $SRC"; exit 1; }

case "$NB" in
  a_cpu) GPU=false; NET=false; RUNGS="1,2"; JOBS="R-FDR:1-5";               N=60; MODE=run  ;;
  a_gpu) GPU=true;  NET=true;  RUNGS="1,2"; JOBS="L-FDR-14B:1-5";           N=60; MODE=run  ;;
  b_gpu) GPU=true;  NET=true;  RUNGS="1,2"; JOBS="LF-FDR-14B:1-5";          N=60; MODE=run  ;;
  c_cpu) GPU=false; NET=false; RUNGS="2";   JOBS="C-naive:1-3,C-FDR:1-3";   N=30; MODE=run  ;;
  seal)  GPU=false; NET=false; RUNGS="1,2"; JOBS="R-FDR:1-5";               N=60; MODE=seal ;;
  *) echo "unknown notebook $NB"; exit 1 ;;
esac

SLUG="af-phase4-${NB//_/-}"
D="kaggle/phase4_${NB}"
mkdir -p "$D"
sed -e "s|^DEFAULT_RUNGS = .*|DEFAULT_RUNGS = \"$RUNGS\"|" \
    -e "s|^DEFAULT_JOBS = .*|DEFAULT_JOBS = \"$JOBS\"|" \
    -e "s|^DEFAULT_N = .*|DEFAULT_N = $N|" \
    -e "s|^DEFAULT_MODE = .*|DEFAULT_MODE = \"$MODE\"   # run \| seal \| report|" \
    "$SRC" > "$D/run.py"
grep -q "^DEFAULT_JOBS = \"$JOBS\"" "$D/run.py" || { echo "sed failed to set DEFAULT_JOBS"; exit 1; }
python3 -m py_compile "$D/run.py"

DS="[]"; KS="[]"
[ -n "${WCACHE:-}" ] && DS="[\"$WCACHE\"]"
if [ "$NB" = "seal" ] && [ -n "${P3DATA:-}" ]; then DS="[\"${WCACHE:-}\",\"$P3DATA\"]"; DS=${DS/\"\",/}; fi
[ -n "${KSRC:-}" ] && KS="[\"${KSRC//,/\",\"}\"]"

cat > "$D/kernel-metadata.json" <<JSON
{
  "id": "$KUSER/$SLUG",
  "title": "$SLUG",
  "code_file": "run.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": "true",
  "enable_gpu": "$GPU",
  "enable_internet": "$NET",
  "dataset_sources": $DS,
  "competition_sources": ["talkingdata-adtracking-fraud-detection"],
  "kernel_sources": $KS
}
JSON
echo "built $D  (rungs=$RUNGS jobs=$JOBS n=$N mode=$MODE gpu=$GPU internet=$NET)"
if [ "$DRY" = "--dry" ]; then echo "dry run: not pushed"; exit 0; fi
kaggle kernels push -p "$D"
echo "status:  kaggle kernels status $KUSER/$SLUG"
echo "output:  kaggle kernels output $KUSER/$SLUG -p out/$NB"
