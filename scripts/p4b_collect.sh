#!/usr/bin/env bash
# p4b_collect.sh - download the four Phase 4b outputs and build one report (results/p4b/P4B_REPORT.md).
#   bash p4b_collect.sh            download + report
#   bash p4b_collect.sh --local    report only (no download)
# Canary runs are pooled with the pilot runs in results/phase4/c_cpu when present (same arms, run ids 1-3).
set -euo pipefail
cd "$(dirname "$0")/.."
KUSER="${KUSER:-yoka33}"
RAW=out_p4b; RES=results/p4b
SLUGS=(af-p4b-1-canary-naive af-p4b-2-canary-fdr af-p4b-3-audit af-p4b-4-census)
NAMES=(canary_naive canary_fdr audit census)
mkdir -p "$RAW" "$RES"
if [ "${1:-}" != "--local" ]; then
  for i in 0 1 2 3; do
    echo "== ${SLUGS[$i]} =="
    kaggle kernels status "$KUSER/${SLUGS[$i]}" 2>&1 | grep -v outdated || true
    rm -rf "$RAW/${SLUGS[$i]}"; mkdir -p "$RAW/${SLUGS[$i]}"
    kaggle kernels output "$KUSER/${SLUGS[$i]}" -p "$RAW/${SLUGS[$i]}" 2>&1 | grep -v outdated | tail -2 || echo "  (download failed)"
    mkdir -p "$RES/${NAMES[$i]}"
    find "$RAW/${SLUGS[$i]}" -type f \( -name 'final_*.json' -o -name 'trials_*.jsonl' -o -name 'audit_*' -o -name 'census_*' -o -name '*.log' \) -exec cp {} "$RES/${NAMES[$i]}/" \;
  done
fi
python3 - "$RES" <<'PY'
import csv, glob, json, math, re, sys
from pathlib import Path
res = Path(sys.argv[1])
NK = lambda k: re.sub(r"[^a-z0-9]", "", str(k).lower())

def wilson(k, n, z=1.96):
    if n == 0: return float("nan"), float("nan")
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)

def fisher_greater(a, n1, c, n2):
    """one-sided exact P(X >= a): naive accepted a of n1, fdr accepted c of n2."""
    K, N = a + c, n1 + n2
    tot = math.comb(N, K)
    return sum(math.comb(n1, x) * math.comb(n2, K - x) for x in range(a, min(K, n1) + 1)) / tot if tot else float("nan")

def load_canary(folder):
    rows = []
    for f in glob.glob(str(folder / "final_r*_C-*.json")):
        d = {NK(k): v for k, v in json.load(open(f)).items()}
        m = re.search(r"final_r(\d+)_(C-[A-Za-z]+)_(\d+)\.json", f)
        rows.append(dict(arm=d.get("arm", m.group(2)), run=int(d.get("run", m.group(3))), tested=int(d.get("decoystested", 0)),
                         acc=int(d.get("accepteddecoys", 0)), gap=float(d.get("devgapclosedpct", 0) or 0), src=Path(f).parent.name))
    return rows

rows = load_canary(res / "canary_naive") + load_canary(res / "canary_fdr")
pilot = Path("results/phase4/c_cpu")
if pilot.exists():
    for r in load_canary(pilot): r["src"] = "pilot"; rows.append(r)
seen, uniq = set(), []
for r in rows:                                   # de-duplicate (arm, run)
    if (r["arm"], r["run"]) not in seen: seen.add((r["arm"], r["run"])); uniq.append(r)
L = ["# Phase 4b report (generated)", ""]
if uniq:
    L += ["## Tests 1-2: canary (gate-only decoy stress test)", ""]
    agg = {}
    for arm in ("C-naive", "C-FDR"):
        x = [r for r in uniq if r["arm"] == arm]
        n, k = sum(r["tested"] for r in x), sum(r["acc"] for r in x)
        lo, hi = wilson(k, n)
        agg[arm] = (k, n)
        gaps = [r["gap"] for r in x if r["acc"] > 0]
        L.append(f"- **{arm}**: {k} accepted of {n} decoys over {len(x)} runs -> {k / n:.4f} (Wilson 95% {lo:.4f}-{hi:.4f})"
                 + (f"; mean DEV gap 'closed' by runs that accepted noise: {sum(gaps) / len(gaps):.2f}%" if gaps else "") if n else f"- **{arm}**: no runs")
    if all(a in agg and agg[a][1] for a in agg) and len(agg) == 2:
        (a, n1), (c, n2) = agg["C-naive"], agg["C-FDR"]
        p = fisher_greater(a, n1, c, n2)
        L += ["", f"One-sided Fisher exact (naive > FDR): p = {p:.4g}. Pre-registered rule: p < 0.05 -> 'naive passes noise that FDR rejects'; "
                  f"else 'not distinguishable at this n'. **Outcome: {'naive passes noise that FDR rejects' if p < 0.05 else 'not distinguishable at this n'}.**"]
        ids = sorted({r["run"] for r in uniq if r["arm"] == "C-naive"} & {r["run"] for r in uniq if r["arm"] == "C-FDR"})
        pairs = [(next(r for r in uniq if r["arm"] == "C-naive" and r["run"] == i)["acc"],
                  next(r for r in uniq if r["arm"] == "C-FDR" and r["run"] == i)["acc"]) for i in ids]
        L += ["", f"Paired runs (same run id = same decoys): {len(pairs)}; naive accepted noise while FDR did not in "
                  f"{sum(1 for n_, f_ in pairs if n_ > 0 and f_ == 0)}; the reverse in {sum(1 for n_, f_ in pairs if f_ > 0 and n_ == 0)}.", ""]
else:
    L += ["## Tests 1-2: no canary final files found yet", ""]
for name, title, key in (("audit", "Test 3: audit of FDR-rejected candidates (Phase 3 sealed window; post hoc, descriptive)", "audit_summary.json"),
                         ("census", "Test 4: grammar census (DEV window; descriptive)", "census_summary.json")):
    p = res / name / key
    L += [f"## {title}", ""]
    L += ["```", p.read_text(), "```", ""] if p.exists() else ["_not available yet_", ""]
(res / "P4B_REPORT.md").write_text("\n".join(L) + "\n")
print("\n".join(L)); print("wrote", res / "P4B_REPORT.md")
PY
