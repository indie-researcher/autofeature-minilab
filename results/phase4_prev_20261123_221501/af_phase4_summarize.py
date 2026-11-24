#!/usr/bin/env python3
"""
Phase 4 summarizer. Reads results/phase4/*/final_r<rung>_<arm>_<run>.json (fallback: summary_runs.csv)
and writes results/phase4/PHASE4_SUMMARY.md + phase4_runs_all.csv.
Usage (repo root):  python3 code/af_phase4_summarize.py [--root results/phase4] [--since 2026-11-22]
--since drops files older than the date (stale pre-revision runs). Incomplete runs (trials without final) are ignored.
"""
import argparse, glob, json, math, os, time
from pathlib import Path
import numpy as np
import pandas as pd

KEYS = ["rung", "arm", "run", "n_proposals", "capped", "n_screened", "n_full_tests", "accepted_real", "accepted_decoys",
        "decoys_proposed", "decoys_screened", "decoys_tested", "n_dup_total", "needle_first_proposed_i", "needle_accepted",
        "dev_baseline_remeasured", "dev_final_auc", "dev_gap_closed_pct", "model", "minutes", "canary", "stack_sig",
        "accepted_new_axis", "gate"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load(root: Path, since):
    rows = []
    cutoff = time.mktime(time.strptime(since, "%Y-%m-%d")) if since else 0
    for f in sorted(glob.glob(str(root / "*" / "final_r*_*.json"))):
        if os.path.getmtime(f) < cutoff:
            continue
        d = json.load(open(f))
        d["notebook"] = Path(f).parent.name
        rows.append(d)
    if not rows:                                           # fallback: per-notebook CSVs
        for f in sorted(glob.glob(str(root / "*" / "summary_runs.csv"))):
            if os.path.getmtime(f) < cutoff:
                continue
            x = pd.read_csv(f)
            x["notebook"] = Path(f).parent.name
            rows += x.to_dict(orient="records")
    df = pd.DataFrame(rows)
    missing = [k for k in ("rung", "arm", "dev_gap_closed_pct") if k not in df.columns]
    if missing:
        raise SystemExit(f"final files lack expected keys {missing}; found {sorted(df.columns)}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/phase4")
    ap.add_argument("--since", default=None)
    a = ap.parse_args()
    root = Path(a.root)
    df = load(root, a.since)
    df.to_csv(root / "phase4_runs_all.csv", index=False)
    g = df[df.get("canary", False) != True]
    out = ["# Phase 4 summary (generated)", "", f"{len(df)} finished runs; incomplete runs (no final file) ignored.", ""]
    rows = []
    for (rung, arm), x in g.groupby(["rung", "arm"]):
        gap = x.dev_gap_closed_pct
        rows.append(dict(rung=rung, arm=arm, runs=len(x), proposals=x.n_proposals.mean(),
                         capped=int(x.capped.sum()) if "capped" in x else np.nan,
                         full_tests=x.n_full_tests.mean(), accepted_real=x.accepted_real.mean(),
                         gap_mean=gap.mean(), gap_median=gap.median(), gap_min=gap.min(), gap_max=gap.max(),
                         distinct_outcomes=x.dev_final_auc.round(9).nunique(), minutes=x.minutes.mean(),
                         decoys_tested=int(x.decoys_tested.sum()), decoys_accepted=int(x.accepted_decoys.sum())))
    out += ["## Search arms (DEV gap closed %, feedback = DEV so optimistic)", "",
            pd.DataFrame(rows).round(2).to_markdown(index=False), ""]
    c = df[df.get("canary", False) == True]
    if len(c):
        out += ["## Canary (gate-only decoy stress test; proxy bypassed)", ""]
        rows = []
        for (rung, arm), x in c.groupby(["rung", "arm"]):
            n, k = int(x.decoys_tested.sum()), int(x.accepted_decoys.sum())
            lo, hi = wilson(k, n)
            rows.append(dict(rung=rung, arm=arm, runs=len(x), decoys_tested=n, decoys_accepted=k,
                             fpr=k / n if n else np.nan, wilson95_low=lo, wilson95_high=hi,
                             mean_dev_gap_from_noise=x.dev_gap_closed_pct.mean()))
        out += [pd.DataFrame(rows).round(4).to_markdown(index=False), ""]
    p = root / "PHASE4_SUMMARY.md"
    p.write_text("\n".join(out))
    print("\n".join(out))
    print("wrote", p)


if __name__ == "__main__":
    main()
