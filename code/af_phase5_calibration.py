#!/usr/bin/env python3
"""Phase 5 / Q1: are the gate's sign-flip p-values calibrated under decoys, and how big is the materiality floor relative to the decoy null?
Reads canary trial logs (every proposal is a decoy, proxy bypassed). Usage: python3 code/af_phase5_calibration.py [dir ...]
Default dirs: results/p4b/canary_fdr results/p4b/canary_naive results/phase4/c_cpu. No compute; post-processing only."""
import glob, json, math, sys
from pathlib import Path
import numpy as np
MAT = 0.00021865718476891195
dirs = sys.argv[1:] or ["results/p4b/canary_fdr", "results/p4b/canary_naive", "results/phase4/c_cpu"]

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)

rec = {"C-FDR": [], "C-naive": []}
seen = set()
for d in dirs:
    for f in sorted(glob.glob(str(Path(d) / "trials_r*_C-*_*.jsonl"))):
        arm, run = Path(f).stem.split("_")[2], Path(f).stem.split("_")[3]
        if (arm, run) in seen: continue
        seen.add((arm, run))
        rows = [json.loads(l) for l in open(f) if l.strip()]
        rows = [r for r in rows if r.get("is_decoy") and r.get("p") is not None and r.get("d_full") is not None]
        if arm == "C-naive":
            rows = rows[:1]            # naive ratchets the reference after an accept; only the first test per run is clean
        rec.setdefault(arm, []).extend(rows)
L = ["# Phase 5 / Q1: calibration of the gate under decoys (generated)", ""]
for arm, rows in rec.items():
    if not rows: continue
    p = np.array([r["p"] for r in rows]); d = np.array([r["d_full"] for r in rows]); n = len(p)
    L += [f"## {arm}: {n} clean decoy tests", ""]
    L += [f"- decoy d_full: mean {d.mean():+.6f}, sd {d.std():.6f}, q95 {np.quantile(d, .95):+.6f}, q99 {np.quantile(d, .99):+.6f}; "
          f"share >= materiality floor ({MAT:.6f}): {(d >= MAT).mean():.3f} ({int((d >= MAT).sum())}/{n}); share > 0: {(d > 0).mean():.3f}"]
    L += ["- ECDF of p versus nominal (calibrated or conservative => P(p <= x) <= x; anti-conservative if the Wilson 95% lower bound exceeds x):"]
    for x in (0.01, 0.05, 0.1, 0.25, 0.5):
        k = int((p <= x).sum()); lo, hi = wilson(k, n)
        L.append(f"  - P(p <= {x}) = {k / n:.3f} [{lo:.3f}, {hi:.3f}]  {'ANTI-CONSERVATIVE' if lo > x else 'ok'}")
    L += [f"- p deciles: {np.round(np.quantile(p, np.linspace(0.1, 0.9, 9)), 3).tolist()}", ""]
(Path("results") / "p5").mkdir(parents=True, exist_ok=True)
Path("results/p5/Q1_CALIBRATION.md").write_text("\n".join(L) + "\n")
print("\n".join(L)); print("wrote results/p5/Q1_CALIBRATION.md")
