#!/usr/bin/env bash
# p4b_gather.sh - READ-ONLY discovery for the four Phase 4b CPU notebooks. Changes nothing.
# Run from the repo root:  bash p4b_gather.sh   -> writes p4b_gather.txt (paste its contents back).
# It prints no credentials (never reads kaggle.json). Output is capped, so it is safe to paste.
set +e
cd "$(dirname "$0")/.." || exit 1
OUT=p4b_gather.txt
exec > >(tee "$OUT") 2>&1
sec() { printf '\n==================== %s ====================\n' "$1"; }

sec "0. where am I / environment"
pwd; date -u +%FT%TZ
python3 -c "import sys; print(sys.version.split()[0])"
python3 -c "import numpy, pandas; print('numpy', numpy.__version__, 'pandas', pandas.__version__)" 2>&1 | tail -1
python3 -c "import lightgbm; print('lightgbm', lightgbm.__version__)" 2>&1 | tail -1
kaggle --version 2>&1 | head -1

sec "1. code/ and kaggle/ layout"
ls -la code | head -40
ls kaggle
for d in kaggle/*/; do echo "-- $d"; ls "$d" | head -8; done

sec "2. existing push / run scripts (what the Phase 4 notebooks used)"
for f in push_phase4.sh run_phase4.sh; do
  echo "----- $f ($(wc -l < "$f" 2>/dev/null) lines)"; sed -n 1,140p "$f" 2>/dev/null
done

sec "3. kernel-metadata examples (Phase 3 CPU and one Phase 4 notebook)"
for f in kaggle/phase3/kernel-metadata.json $(ls kaggle/*/kernel-metadata.json 2>/dev/null | grep -i -m2 "phase4\|a_cpu\|c_cpu"); do
  echo "----- $f"; cat "$f"; echo
done
ls -d kaggle/*phase4* kaggle/*a_cpu* kaggle/*c_cpu* 2>/dev/null
echo "--- Kaggle username / my phase 4 kernels"
kaggle config view 2>&1 | grep -i "username" | head -1
kaggle kernels list --mine -s "af-phase4" 2>&1 | head -12

sec "4. the Phase 4 runner: which file, CLI, canary logic"
P4=$(ls code/af_ph4*.py code/af_phase4*.py 2>/dev/null | head -3); echo "candidates: $P4"
P4F=$(echo "$P4" | head -1)
if [ -n "$P4F" ]; then
  echo "----- $P4F: $(wc -l < "$P4F") lines"
  echo "--- argparse flags"; grep -n "add_argument" "$P4F" | cut -c1-200 | head -40
  echo "--- --help"; timeout 60 python3 "$P4F" --help 2>&1 | head -50
  echo "--- canary / arm names"; grep -n "canary\|C-naive\|C-FDR\|CANARY" "$P4F" | cut -c1-200 | head -50
  echo "--- budget guards / env vars"; grep -n "620\|environ\|getenv\|WORK_DIR\|OUT_DIR\|/kaggle/" "$P4F" | cut -c1-200 | head -30
else
  echo "no af_ph4*.py found under code/ (try: find . -name 'af_ph4*.py')"; find . -name 'af_ph4*.py' -not -path './.git/*' | head
fi

sec "5. Phase 3 inputs for tests 3 and 4 (rejected-candidate audit, census)"
ls results/phase3 | head -50
echo "phase3_run_*.json count: $(ls results/phase3/phase3_run_*.json 2>/dev/null | wc -l)"
f=$(ls results/phase3/phase3_run_*.json 2>/dev/null | head -1)
[ -n "$f" ] && python3 -c "
import json; d=json.load(open('$f')); print('keys:', list(d.keys())); print('accepted_real_specs sample:', str(d.get('accepted_real_specs'))[:200])"
echo "embedded-data markers in the combined Phase 3 script:"
grep -n "^EMBEDDED_INPUTS\|^EMBEDDED_SPEC_FREEZE\|# ===================== PHASE 3\|^class Lab\|^class Track\|^def sample_spec\|^def get_window\|^def load_dev_window\|^F = " code/af_phase3_sealed_combined.py | cut -c1-120
echo "--- candidate counts for the audit (material but rejected, non-decoy; trial logs)"
python3 - <<'PY'
import glob, json
MAT = 0.00021865718476891195
tot = 0
for arm in ["R", "R-naive", "L-naive", "L-FDR", "L-FDR-3B", "L-FDR-14B"]:
    n = n05 = nspec = 0
    for f in glob.glob(f"results/{arm}/trials_{arm}_*.jsonl"):
        for l in open(f):
            r = json.loads(l)
            if r.get("status") == "rejected" and not r.get("is_decoy") and (r.get("d_full") or 0) >= MAT:
                n += 1; n05 += (r.get("p") or 1) <= 0.05; nspec += r.get("spec") is not None
    tot += n
    print(f"{arm}: material-but-rejected {n} | p<=0.05 {n05} | with spec {nspec}")
print("total", tot)
PY

sec "6. data paths and sizes"
grep -n "/kaggle/input" code/af_phase3_sealed_combined.py code/af_ph4*.py 2>/dev/null | cut -c1-200 | head -8
du -sh results out_phase4 2>/dev/null
du -sh code/af_phase3_sealed_combined.py code/af_ph4*.py 2>/dev/null

sec "7. git state"
git status -sb | head -8; git tag | tail -8; git log --oneline | head -4

sec "DONE - paste $OUT back (about 200-300 lines)"
