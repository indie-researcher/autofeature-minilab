#!/usr/bin/env python3
"""Final write-up figures (post-processing only; reads existing results, never touches SEALED data).
Usage (repo root):  python3 code/af_final_figures.py [--root .]
Writes figures/final_net_holdout_by_gate.png, final_power_curve.png, final_canary_audit.png.
Inputs: results/p5/**/p5_run_*.json, results/p5b/**/p5_run_*.json, results/phase3/phase3_runs.csv + phase3_decoys.csv,
        results/p4b/audit/audit_candidates.csv, results/p4b/P4B_REPORT.md."""
import argparse, glob, json, math, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=".")
root = Path(ap.parse_args().root)
fig_dir = root / "figures"
fig_dir.mkdir(exist_ok=True)
COL = dict(naive="#d95f02", decoyq="#7570b3", fdr24="#1b9e77", fdr="#66a61e")
LAB = dict(naive="keep-if-better", decoyq="decoy-null gate", fdr24="LORD++ (24 blocks)", fdr="LORD++ (12 blocks)")
ORDER = ["naive", "decoyq", "fdr24", "fdr"]
CELLS = [(200, 120), (50, 120), (10, 120), (2, 60)]          # B=30 cells are in the report tables only


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def load_runs():
    rows = []
    for src in ("p5", "p5b"):
        for f in glob.glob(str(root / "results" / src / "**" / "p5_run_*.json"), recursive=True):
            r = json.load(open(f))
            r.pop("trials", None)
            r["src"] = src
            rows.append(r)
    if not rows:
        raise SystemExit("no p5_run_*.json found under results/p5 or results/p5b")
    df = pd.DataFrame(rows).sort_values("src", ascending=False)       # prefer p5b copies (naive replication is identical)
    return df.drop_duplicates(["K", "B", "gate", "run"])


def fig_net_holdout(df):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    w = 0.2
    for ci, (K, B) in enumerate(CELLS):
        for gi, g in enumerate(ORDER):
            x = df[(df.K == K) & (df.B == B) & (df.gate == g)]
            if not len(x):
                continue
            pos = ci + (gi - 1.5) * w
            ax.scatter(pos + np.random.RandomState(ci * 10 + gi).uniform(-.04, .04, len(x)), x.gain_hold, s=14, color=COL[g], alpha=.45, zorder=2)
            ax.hlines(x.gain_hold.mean(), pos - w / 2.4, pos + w / 2.4, color=COL[g], lw=3, zorder=3, label=LAB[g] if ci == 0 else None)
    ax.axhline(0, color="k", lw=.7)
    labels = []
    for K, B in CELLS:
        n = df[(df.K == K) & (df.B == B)].n_train.median()
        labels.append(f"{n / 1000:.0f}k rows\n{B} proposals")
    ax.set_xticks(range(len(CELLS)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("net holdout gain (AUC), dot = run, bar = mean")
    ax.set_title("Net holdout gain by gate (same proposals per cell; no gate beats keep-if-better)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(fig_dir / "final_net_holdout_by_gate.png", dpi=170)
    plt.close(fig)


def fig_power(df):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for g in ORDER:
        pts = []
        for K, B in CELLS:
            x = df[(df.K == K) & (df.B == B) & (df.gate == g)]
            if len(x):
                pts.append((x.n_train.median(), x.accepted_real.mean()))
        if pts:
            pts.sort()
            ax.plot(*zip(*pts), "-o", color=COL[g], label=LAB[g])
    p3 = root / "results" / "phase3" / "phase3_runs.csv"
    if p3.exists():
        r = pd.read_csv(p3)
        for arm, g in (("R", "fdr"), ("R-naive", "naive")):
            v = r[r.arm == arm].n_accepted_real
            if len(v):
                ax.scatter([1_469_995], [v.mean()], marker="*", s=170, color=COL[g], edgecolor="k", zorder=5)
        ax.annotate("Phase 3\n(full data,\nrandom proposals)", (1_469_995, 5), textcoords="offset points", xytext=(-60, 18), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("training rows (after negative subsampling), log scale")
    ax.set_ylabel("accepted real features per run")
    ax.set_title("Power: strict gates accept almost nothing below ~1M rows")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "final_power_curve.png", dpi=170)
    plt.close(fig)


def fig_canary_audit():
    rep = root / "results" / "p4b" / "P4B_REPORT.md"
    aud = root / "results" / "p4b" / "audit" / "audit_candidates.csv"
    dec = root / "results" / "phase3" / "phase3_decoys.csv"
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    if rep.exists():
        t = rep.read_text()
        got = {m.group(1): (int(m.group(2)), int(m.group(3))) for m in re.finditer(r"\*\*(C-naive|C-FDR)\*\*: (\d+) accepted of (\d+) decoys", t)}
        for i, (arm, lab, c) in enumerate((("C-naive", "keep-if-better", COL["naive"]), ("C-FDR", "LORD++ (12 blocks)", COL["fdr"]))):
            if arm in got:
                k, n = got[arm]
                lo, hi = wilson(k, n)
                axs[0].bar(i, 100 * k / n, color=c)
                axs[0].errorbar(i, 100 * k / n, yerr=[[100 * (k / n - lo)], [100 * (hi - k / n)]], color="k", capsize=4)
                axs[0].text(i, 100 * hi + 0.2, f"{k}/{n}", ha="center", fontsize=9)
        axs[0].set_xticks([0, 1])
        axs[0].set_xticklabels(["keep-if-better", "LORD++ (12 blocks)"])
    axs[0].set_ylabel("planted decoys accepted (%), Wilson 95%")
    axs[0].set_title("Canary: false accepts (all-decoy proposals, gate only)")
    if aud.exists() and dec.exists():
        a = pd.read_csv(aud).sealed_delta.values
        d = pd.read_csv(dec).delta_auc.values
        for i, (v, lab, c) in enumerate(((d, "29 decoys", "#999999"), (a, "63 FDR-rejected\ncandidates", COL["naive"]))):
            axs[1].scatter(np.random.RandomState(i).uniform(-.12, .12, len(v)) + i, v * 1e4, s=16, alpha=.7, color=c)
            axs[1].hlines(np.median(v) * 1e4, i - .25, i + .25, color="k", lw=2)
        axs[1].axhline(0, color="k", lw=.7)
        axs[1].set_xticks([0, 1])
        axs[1].set_xticklabels(["29 decoys", "63 FDR-rejected\ncandidates"])
    axs[1].set_ylabel("delta AUC on the Phase 3 sealed window (x1e-4)")
    axs[1].set_title("Audit: rejected candidates vs decoys (post hoc)")
    fig.tight_layout()
    fig.savefig(fig_dir / "final_canary_audit.png", dpi=170)
    plt.close(fig)


df = load_runs()
for fn, arg in ((fig_net_holdout, (df,)), (fig_power, (df,)), (fig_canary_audit, ())):
    try:
        fn(*arg)
        print("ok  ", fn.__name__)
    except Exception as e:
        print("FAIL", fn.__name__, repr(e))
print("figures in", fig_dir)
