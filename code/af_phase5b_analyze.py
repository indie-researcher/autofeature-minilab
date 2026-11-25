#!/usr/bin/env python3
"""Phase 5b analysis: gates versus keep-if-better, paired by (K, B, run). Reads results/p5 (naive, fdr) and results/p5b (fdr24, decoyq, naive replication).
Usage: python3 code/af_phase5b_analyze.py [--p5 results/p5] [--p5b results/p5b]. Writes results/p5b/P5B_REPORT.md, p5b_gates.csv, p5b_gates_holdout.png."""
import argparse, glob, json
from pathlib import Path
import numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--p5", default="results/p5"); ap.add_argument("--p5b", default="results/p5b")
a = ap.parse_args(); out = Path(a.p5b); out.mkdir(parents=True, exist_ok=True)
rows = []
for src, root in (("p5", a.p5), ("p5b", a.p5b)):
    for f in glob.glob(str(Path(root) / "**" / "p5_run_*.json"), recursive=True):
        r = json.load(open(f)); r.pop("trials", None); r["src"] = src; rows.append(r)
if not any(r["src"] == "p5b" for r in rows): raise SystemExit(f"no Phase 5b runs under {a.p5b}")
df = pd.DataFrame(rows)
rs = np.random.RandomState(0)
def ci(x, n=5000):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 2: return float("nan"), float("nan")
    m = rs.choice(x, (n, len(x))).mean(1); return float(np.quantile(m, .025)), float(np.quantile(m, .975))
# naive reference: the 5b replication where present, else Phase 5
nv = df[df.gate == "naive"].sort_values("src", ascending=False).drop_duplicates(["K", "B", "run"]).set_index(["K", "B", "run"])
rep = df[df.gate == "naive"].pivot_table(index=["K", "B", "run"], columns="src", values="gain_hold")
repl = None
if {"p5", "p5b"} <= set(rep.columns):
    d = (rep["p5"] - rep["p5b"]).dropna()
    repl = (len(d), float(d.abs().max()), float(d.abs().mean())) if len(d) else None
cells = []
gates = [g for g in ("naive", "fdr", "fdr24", "decoyq") if g in set(df.gate)]
for (K, B), x in df[df.src.isin(["p5", "p5b"])].groupby(["K", "B"]):
    if not (x.src == "p5b").any(): continue
    for g in gates:
        y = (nv.xs((K, B), level=(0, 1)).reset_index() if g == "naive" else x[x.gate == g].drop_duplicates("run"))
        if not len(y): continue
        y = y.set_index("run") if "run" in y.columns else y
        base = nv.xs((K, B), level=(0, 1))
        ids = sorted(set(y.index) & set(base.index))
        d_h = (y.loc[ids, "gain_hold"] - base.loc[ids, "gain_hold"]).values if g != "naive" else np.zeros(len(ids))
        d_o = (y.loc[ids, "optimism"] - base.loc[ids, "optimism"]).values if g != "naive" else np.zeros(len(ids))
        hl, hh = ci(d_h); ol, oh = ci(d_o)
        cells.append(dict(K=K, B=B, n_train=int(y.n_train.median()), gate=g, runs=len(y), acc_real=y.accepted_real.mean(),
                          acc_decoy=int(y.accepted_decoys.sum()), decoys_tested=int(y.decoys_tested.sum()), gain_dev=y.gain_dev.mean(),
                          gain_hold=y.gain_hold.mean(), optimism=y.optimism.mean(), frac_dead=y.frac_tests_dead.mean() if "frac_tests_dead" in y else np.nan,
                          d_hold_vs_naive=d_h.mean() if len(d_h) else np.nan, d_hold_lo=hl, d_hold_hi=hh, d_opt_vs_naive=d_o.mean() if len(d_o) else np.nan, d_opt_lo=ol, d_opt_hi=oh))
C = pd.DataFrame(cells).sort_values(["B", "n_train", "gate"]); C.to_csv(out / "p5b_gates.csv", index=False)
def md(d, nd=5):
    c = list(d.columns); L = ["| " + " | ".join(c) + " |", "|" + "---|" * len(c)]
    for _, r in d.iterrows(): L.append("| " + " | ".join(f"{v:.{nd}f}" if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(L)
new = C[C.gate.isin(["fdr24", "decoyq"])]
win = new[new.d_hold_lo > 0]
verdict = ("CONSTRUCTIVE: a better gate beats keep-if-better on net holdout gain (paired 95% CI > 0) in: "
           + "; ".join(f"{r.gate} @K={r.K},B={r.B}" for r in win.itertuples())) if len(win) else \
          "NO GATE TRIED beats keep-if-better on net holdout gain in this grid (no paired CI above 0)."
L = ["# Phase 5b report (generated): better gates vs keep-if-better", "",
     "Net holdout gain and optimism are in AUC units; d_* are paired differences versus naive (same proposals, same IP subsample). Holdout = 2017-11-08 (not fresh).", "",
     md(C), "", f"**Primary rule outcome: {verdict}**", "",
     "Rule (frozen in docs/PHASE5B_PLAN.md): a gate 'wins' if the paired 95% bootstrap CI of (gate - naive) holdout gain has lower bound > 0 in at least one cell."]
if repl: L += ["", f"Naive replication check (Phase 5 vs Phase 5b, same runs): {repl[0]} pairs, max |delta hold gain| = {repl[1]:.6f}, mean = {repl[2]:.6f} (near 0 confirms the pairing is deterministic)."]
(out / "P5B_REPORT.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cellkeys = sorted(set(zip(C.K, C.B))); fig, ax = plt.subplots(figsize=(1.8 * len(cellkeys) + 3, 4.2)); w = 0.8 / max(len(gates), 1)
    for i, g in enumerate(gates):
        s = [C[(C.K == K) & (C.B == B) & (C.gate == g)].gain_hold for K, B in cellkeys]
        ax.bar(np.arange(len(cellkeys)) + i * w, [v.iloc[0] if len(v) else np.nan for v in s], w, label=g)
    ax.set_xticks(np.arange(len(cellkeys)) + 0.4 - w / 2); ax.set_xticklabels([f"K={K}\nB={B}" for K, B in cellkeys]); ax.axhline(0, color="k", lw=.6)
    ax.set_ylabel("net holdout gain (AUC)"); ax.legend(); fig.tight_layout(); fig.savefig(out / "p5b_gates_holdout.png", dpi=160)
except Exception as e: print("figure skipped:", e)
print("wrote", out / "P5B_REPORT.md")
