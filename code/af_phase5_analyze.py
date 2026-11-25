#!/usr/bin/env python3
"""Phase 5 analysis: paired naive-vs-FDR regime map. Usage: python3 code/af_phase5_analyze.py [--root results/p5] [--out results/p5]
Reads **/p5_run_*.json, writes P5_REPORT.md, p5_cells.csv, p5_paired.csv, p5_optimism_vs_n.png, p5_holdout_gain_vs_n.png."""
import argparse, glob, json
from pathlib import Path
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="results/p5")
ap.add_argument("--out", default="")
a = ap.parse_args()
root, out = Path(a.root), Path(a.out or a.root)
rows = []
for f in glob.glob(str(root / "**" / "p5_run_*.json"), recursive=True):
    r = json.load(open(f)); r.pop("trials", None); rows.append(r)
if not rows:
    raise SystemExit(f"no p5_run_*.json under {root.resolve()}")
df = pd.DataFrame(rows).drop_duplicates(["K", "B", "gate", "run"])
rs = np.random.RandomState(0)

def boot_ci(x, n=5000):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 2: return (float("nan"), float("nan"))
    m = rs.choice(x, (n, len(x))).mean(1)
    return float(np.quantile(m, .025)), float(np.quantile(m, .975))

cells = []
for (K, B, g), x in df.groupby(["K", "B", "gate"]):
    lo, hi = boot_ci(x.optimism)
    cells.append(dict(K=K, B=B, gate=g, runs=len(x), n_train=int(x.n_train.median()), eval_pos=int(x.eval_pos.median()),
                      acc_real=x.accepted_real.mean(), acc_decoy=int(x.accepted_decoys.sum()), decoys_tested=int(x.decoys_tested.sum()),
                      gain_dev=x.gain_dev.mean(), gain_hold=x.gain_hold.mean(), optimism=x.optimism.mean(), opt_lo=lo, opt_hi=hi,
                      frac_dead=x.frac_tests_dead.mean(), minutes=x.minutes.mean()))
cells = pd.DataFrame(cells).sort_values(["B", "n_train", "gate"])
cells.to_csv(out / "p5_cells.csv", index=False)

paired = []
for (K, B), x in df.groupby(["K", "B"]):
    n_, f_ = x[x.gate == "naive"].set_index("run"), x[x.gate == "fdr"].set_index("run")
    ids = sorted(set(n_.index) & set(f_.index))
    if not ids: continue
    d_opt = (n_.loc[ids, "optimism"] - f_.loc[ids, "optimism"]).values
    d_hold = (n_.loc[ids, "gain_hold"] - f_.loc[ids, "gain_hold"]).values
    lo, hi = boot_ci(d_opt); hl, hh = boot_ci(d_hold)
    paired.append(dict(K=K, B=B, n_train=int(n_.loc[ids, "n_train"].median()), pairs=len(ids), d_optimism=d_opt.mean(), d_opt_lo=lo, d_opt_hi=hi,
                       d_gain_hold=d_hold.mean(), d_hold_lo=hl, d_hold_hi=hh, naive_worse_runs=int((d_opt > 0).sum())))
paired = pd.DataFrame(paired).sort_values(["B", "n_train"])
paired.to_csv(out / "p5_paired.csv", index=False)

def md(d, nd=5):
    c = list(d.columns); L = ["| " + " | ".join(c) + " |", "|" + "---|" * len(c)]
    for _, r in d.iterrows():
        L.append("| " + " | ".join(f"{v:.{nd}f}" if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(L)

rho = None
if len(paired) >= 3:
    rho = float(pd.Series(paired.d_optimism.values).corr(pd.Series(np.log(paired.n_train.values)), method="spearman"))
small = paired.sort_values("n_train").head(1)
rule = ("H5-1 supported" if rho is not None and rho <= -0.6 and len(small) and float(small.d_opt_lo.iloc[0]) > 0
        else "H5-1 not supported (no regime in this grid where the naive gate shows excess optimism)")
L = ["# Phase 5 report (generated): regime map, naive vs LORD++ gate", "",
     f"{len(df)} runs, {df.groupby(['K', 'B']).ngroups} cells. Optimism = DEV gain (fresh seeds) - holdout gain, in AUC units. "
     "Holdout = 2017-11-08 (already used in Phase 3: not a fresh holdout). Paired by (K, B, run): identical proposals and IP subsample.", "",
     "## Cells", "", md(cells), "", "## Paired difference naive - FDR (positive d_optimism = naive more optimistic)", "", md(paired), "",
     f"Spearman(d_optimism, log n_train) over cells: {rho}", f"**Rule outcome: {rule}**", "",
     "Rule: H5-1 supported if rho <= -0.6 AND the paired 95% bootstrap CI of d_optimism at the smallest n excludes 0 (lower bound > 0)."]
(out / "P5_REPORT.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    for col, lo, hi, name, yl in (("optimism", "opt_lo", "opt_hi", "p5_optimism_vs_n.png", "optimism (DEV gain - holdout gain, AUC)"),
                                  ("gain_hold", None, None, "p5_holdout_gain_vs_n.png", "holdout gain (AUC)")):
        Bs = sorted(cells.B.unique())
        fig, axs = plt.subplots(1, len(Bs), figsize=(5 * len(Bs), 4), squeeze=False)
        for ax, B in zip(axs[0], Bs):
            for g, c in (("naive", "#d95f02"), ("fdr", "#1b9e77")):
                s = cells[(cells.B == B) & (cells.gate == g)].sort_values("n_train")
                ax.plot(s.n_train, s[col], "-o", color=c, label=g)
                if lo: ax.fill_between(s.n_train, s[lo], s[hi], color=c, alpha=.15)
            ax.set_xscale("log"); ax.axhline(0, color="k", lw=.7); ax.set_xlabel("training rows (log)"); ax.set_title(f"{B} proposals per run"); ax.legend()
        axs[0][0].set_ylabel(yl); fig.tight_layout(); fig.savefig(out / name, dpi=160); plt.close(fig)
except Exception as e:
    print("figure skipped:", e)
print("wrote", out / "P5_REPORT.md")
