#!/usr/bin/env python3
"""
AutoFeature Mini-Lab - analysis: 4 pre-registered figures + exploratory diagnostics (ID1).

Run from the repo root:  python code/af_phase3_analyze.py
Reads : results/phase3/phase3_runs.csv, phase3_decoys.csv, results/<arm>/trials_<arm>_<run>.jsonl
Writes: figures/fig1..fig4*.png            (pre-registered figures)
        figures/diag_*.png                 (EXPLORATORY, post hoc)
        results/diagnostics/*.csv|json     (EXPLORATORY, post hoc)
No SEALED data is touched; this only reads Phase 3 outputs and trial logs.
"""
import argparse, json, math
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ARMS = ["R", "R-naive", "L-naive", "L-FDR", "L-FDR-3B", "L-FDR-14B"]
LABEL = {"R": "Random + FDR", "R-naive": "Random + naive", "L-naive": "7B + naive",
         "L-FDR": "7B + FDR", "L-FDR-3B": "3B + FDR", "L-FDR-14B": "14B + FDR"}
COLOR = {"R": "#1b9e77", "R-naive": "#66c2a5", "L-naive": "#e08214",
         "L-FDR": "#d95f02", "L-FDR-3B": "#7570b3", "L-FDR-14B": "#e7298a"}
SCALING_ORDER = ["R", "L-FDR-3B", "L-FDR", "L-FDR-14B", "R-naive", "L-naive"]
MATERIALITY = 0.00021865718476891195


def canon(spec):
    return json.dumps(spec, sort_keys=True, separators=(",", ":"))


def entropy_bits(counts):
    n = sum(counts.values())
    return -sum(c / n * math.log2(c / n) for c in counts.values()) if n else float("nan")


# ------------------------------------------------------------------ pre-registered figures
def fig1(path):
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.axis("off")
    boxes = ["Proposer\n(random / Qwen)", "Proxy screen\n10% IP subsample\ngain >= 0.0005",
             "Full DEV test", "Gate\n(naive | LORD++ FDR)", "Stack"]
    x = 0.02
    for i, b in enumerate(boxes):
        ax.add_patch(FancyBboxPatch((x, 0.35), 0.15, 0.42, boxstyle="round,pad=0.01", fc="#e8eef7", ec="#335"))
        ax.text(x + 0.075, 0.56, b, ha="center", va="center", fontsize=9)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x + 0.185, 0.56), xytext=(x + 0.155, 0.56), arrowprops=dict(arrowstyle="->"))
        x += 0.19
    ax.add_patch(FancyBboxPatch((0.02, 0.04), 0.92, 0.17, boxstyle="round,pad=0.01", fc="#fdecea", ec="#a33"))
    ax.text(0.48, 0.125, "SEALED day (2017-11-08), opened once at the end   |   20% of proposals are permuted-real decoys",
            ha="center", va="center", fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 0.9)
    fig.savefig(path, dpi=170, bbox_inches="tight"); plt.close(fig)


def fig2(runs, decoys, path):
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw={"width_ratios": [1.1, 1.1, 1.0]})
    xs = np.arange(len(ARMS))
    # (a) decoys by stage
    st = [("decoys_proposed", "proposed"), ("decoys_screened", "screened out"),
          ("n_decoys_tested", "gate-tested"), ("n_decoys_accepted", "accepted")]
    g = runs.groupby("arm")[[c for c, _ in st]].sum().reindex(ARMS)
    w = 0.2
    for i, (c, lab) in enumerate(st):
        axs[0].bar(xs + (i - 1.5) * w, g[c], w, label=lab)
    axs[0].set_xticks(xs); axs[0].set_xticklabels([LABEL[a] for a in ARMS], rotation=30, ha="right", fontsize=8)
    axs[0].set_title("(a) Decoys by stage (sum of 5 runs)"); axs[0].set_ylim(0, max(float(g.max().max()), 1) * 1.3); axs[0].legend(fontsize=8, ncol=2, loc="upper right")
    # (b) DEV vs SEALED gap closed, per arm
    for i, a in enumerate(ARMS):
        r = runs[runs.arm == a]
        axs[1].scatter(np.full(len(r), i) - 0.12, r.dev_gap_closed_pct, s=18, color="#888", alpha=.7,
                       label="DEV" if i == 0 else None)
        axs[1].scatter(np.full(len(r), i) + 0.12, r.sealed_gap_closed_pct, s=18, color=COLOR[a],
                       label="SEALED" if i == 0 else None)
        axs[1].hlines(r.dev_gap_closed_pct.mean(), i - .3, i - .0, color="#444")
        axs[1].hlines(r.sealed_gap_closed_pct.mean(), i + .0, i + .3, color=COLOR[a])
    axs[1].set_xticks(xs); axs[1].set_xticklabels([LABEL[a] for a in ARMS], rotation=30, ha="right", fontsize=8)
    axs[1].set_ylabel("gap closed (%)"); axs[1].set_title("(b) DEV (grey) vs SEALED (colour), dot = run, bar = mean")
    # (c) decoy delta on SEALED
    if len(decoys):
        for i, a in enumerate(ARMS):
            d = decoys[decoys.arm == a].delta_auc
            if len(d):
                axs[2].scatter(np.full(len(d), i) + np.random.RandomState(i).uniform(-.12, .12, len(d)), d * 1e4,
                               color=COLOR[a], s=22, alpha=.85)
        axs[2].axhline(0, color="k", lw=.8)
        axs[2].axhline(MATERIALITY * 1e4, color="r", ls="--", lw=1, label="materiality floor")
        axs[2].legend(fontsize=8)
    axs[2].set_xticks(xs); axs[2].set_xticklabels([LABEL[a] for a in ARMS], rotation=30, ha="right", fontsize=8)
    axs[2].set_ylabel("delta AUC on SEALED (x1e-4)")
    axs[2].set_title("(c) 29 re-permuted decoys added to final stack")
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def fig3(runs, path):
    fig, ax = plt.subplots(figsize=(10, 4.6))
    pos = {a: i + (0.6 if i >= 4 else 0) for i, a in enumerate(SCALING_ORDER)}
    for a in SCALING_ORDER:
        r = runs[runs.arm == a].sort_values("run")
        jit = np.random.RandomState(7).uniform(-.1, .1, len(r))
        ax.scatter(pos[a] + jit, r.sealed_gap_closed_pct, color=COLOR[a], s=36, alpha=.9, zorder=3)
        ax.hlines(r.sealed_gap_closed_pct.mean(), pos[a] - .28, pos[a] + .28, color="k", lw=1.6, zorder=4)
        ax.hlines(r.sealed_gap_closed_pct.median(), pos[a] - .18, pos[a] + .18, color="k", lw=1, ls=":", zorder=4)
        n_eff = r.final_set_hash.nunique()
        if n_eff < len(r):
            ax.text(pos[a], r.sealed_gap_closed_pct.min() - 4, f"{n_eff} distinct\noutcomes", ha="center", fontsize=8)
    ax.axvline(3.8, color="#bbb", lw=1)
    ax.set_xticks([pos[a] for a in SCALING_ORDER])
    ax.set_xticklabels([LABEL[a] for a in SCALING_ORDER], rotation=20, ha="right")
    ax.set_ylabel("SEALED gap closed (%)")
    ax.set_title("Proposer scaling (dot = run, solid bar = mean, dotted = median); naive arms right of the line")
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def fig4(runs, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for a in ARMS:
        r = runs[runs.arm == a]
        ax.scatter(r.minutes, r.sealed_gap_closed_pct, color=COLOR[a], s=26, alpha=.55)
        ax.scatter(r.minutes.mean(), r.sealed_gap_closed_pct.mean(), color=COLOR[a], s=120, edgecolor="k",
                   label=LABEL[a], zorder=4)
    ax.set_xlabel("minutes per run (proposal + evaluation wall-clock)")
    ax.set_ylabel("SEALED gap closed (%)")
    ax.set_title("Cost frontier (small = run, large = arm mean)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


# ------------------------------------------------------------------ exploratory diagnostics (ID1)
def load_trials(root, arm, run):
    f = root / "results" / arm / f"trials_{arm}_{run}.jsonl"
    if not f.exists():
        return []
    return [json.loads(l) for l in open(f) if l.strip()]


def find_needle(root, runs, override=None):
    """Largest-gain accepted real feature (by d_full) in the best SEALED run."""
    if override:
        return json.load(open(override))
    best = runs.sort_values("sealed_gap_closed_pct", ascending=False).iloc[0]
    rows = [r for r in load_trials(root, best.arm, int(best.run))
            if r.get("status") == "accepted" and not r.get("is_decoy") and r.get("d_full") is not None
            and r.get("spec") is not None]
    if not rows:
        return None
    top = max(rows, key=lambda r: r["d_full"])
    return dict(arm=best.arm, run=int(best.run), sealed_gap_closed_pct=float(best.sealed_gap_closed_pct),
                spec=top["spec"], d_full=top["d_full"], i=top["i"], canon=canon(top["spec"]))


def diagnostics(root, runs, outdir, figdir):
    outdir.mkdir(parents=True, exist_ok=True)
    needle = find_needle(root, runs)
    rows, curves = [], {}
    for _, rr in runs.iterrows():
        arm, run = rr.arm, int(rr.run)
        all_rows = [t for t in load_trials(root, arm, run) if not t.get("is_decoy")]
        tr = [t for t in all_rows if t.get("spec") is not None]      # rows without a spec (invalid etc.) are counted below
        if not all_rows:
            continue
        status_counts = Counter(t.get("status") for t in all_rows)
        specs = [canon(t["spec"]) for t in tr if t.get("spec") is not None]
        ops = Counter(t["spec"].get("op") for t in tr if t.get("spec") is not None)
        acc = [t for t in tr if t.get("status") == "accepted"]
        nd_i = next((t["i"] for t in tr if needle and canon(t["spec"]) == needle["canon"]), np.nan)
        nd_acc = bool(needle and any(canon(t["spec"]) == needle["canon"] for t in acc))
        rows.append(dict(arm=arm, run=run, n_real_logged=len(tr), unique_specs=len(set(specs)),
                         repeat_frac=1 - len(set(specs)) / max(len(specs), 1),
                         op_entropy_bits=entropy_bits(ops), n_ops_used=len(ops),
                         n_rows_without_spec=len(all_rows) - len(tr),
                         status_counts=json.dumps(dict(status_counts), sort_keys=True),
                         n_accepted=len(acc), first_accept_i=min((t["i"] for t in acc), default=np.nan),
                         last_accept_i=max((t["i"] for t in acc), default=np.nan),
                         best_d_full=max((t["d_full"] for t in acc), default=np.nan),
                         needle_first_proposed_i=nd_i, needle_ever_proposed=not np.isnan(nd_i),
                         needle_accepted=nd_acc, n_dup_logged=sum(t.get("n_dup", 0) or 0 for t in tr),
                         sealed_gap_closed_pct=rr.sealed_gap_closed_pct))
        order = sorted(tr, key=lambda t: t["i"])
        curves[(arm, run)] = np.cumsum([1 if t.get("status") == "accepted" else 0 for t in order]), \
            [t["i"] for t in order]
    d = pd.DataFrame(rows)
    d.to_csv(outdir / "diag_by_run.csv", index=False)
    arm = d.groupby("arm").agg(
        runs=("run", "count"), unique_specs_mean=("unique_specs", "mean"), repeat_frac_mean=("repeat_frac", "mean"),
        op_entropy_mean=("op_entropy_bits", "mean"), accepted_mean=("n_accepted", "mean"),
        first_accept_i_median=("first_accept_i", "median"), best_d_full_mean=("best_d_full", "mean"),
        needle_ever_proposed=("needle_ever_proposed", "sum"), needle_accepted=("needle_accepted", "sum")
    ).reindex(ARMS)
    arm.to_csv(outdir / "diag_by_arm.csv")
    if needle:
        json.dump({k: v for k, v in needle.items() if k != "canon"}, open(outdir / "needle.json", "w"), indent=2)

    # E1: diversity
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, col, ttl in zip(axs, ["unique_specs", "op_entropy_bits", "needle_first_proposed_i"],
                            ["Unique feature specs proposed (of ~60)", "Operation entropy (bits, max 2.58)",
                             "Proposal index where the needle was first proposed"]):
        for i, a in enumerate(ARMS):
            v = d[d.arm == a][col]
            ax.scatter(np.full(len(v), i) + np.random.RandomState(i).uniform(-.1, .1, len(v)), v,
                       color=COLOR[a], s=30, alpha=.85)
            if v.notna().any():
                ax.hlines(v.mean(), i - .25, i + .25, color="k")
        ax.set_xticks(range(len(ARMS))); ax.set_xticklabels([LABEL[a] for a in ARMS], rotation=30, ha="right", fontsize=8)
        ax.set_title(ttl, fontsize=10)
    axs[2].text(0.02, 0.02, "missing dot = needle never proposed", transform=axs[2].transAxes, fontsize=8)
    fig.suptitle("EXPLORATORY (post hoc): proposal diversity and the 'needle' feature", y=1.02, fontsize=11)
    fig.tight_layout(); fig.savefig(figdir / "diag_diversity_needle.png", dpi=170, bbox_inches="tight"); plt.close(fig)

    # E2: discovery curves (cumulative accepted real features vs proposal index)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for a in ARMS:
        ys = []
        for (ar, run), (cum, idx) in curves.items():
            if ar != a:
                continue
            y = np.zeros(61); last = 0
            for c, i in zip(cum, idx):
                y[min(int(i), 60):] = c
            ys.append(y)
        if ys:
            ax.plot(range(61), np.mean(ys, axis=0), color=COLOR[a], label=LABEL[a], lw=2)
    ax.set_xlabel("proposal index"); ax.set_ylabel("accepted real features (mean of 5 runs)")
    ax.set_title("EXPLORATORY: discovery curves", fontsize=10); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figdir / "diag_discovery_curves.png", dpi=170); plt.close(fig)
    return d, arm, needle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    figdir = root / "figures"; figdir.mkdir(exist_ok=True)
    runs = pd.read_csv(root / "results" / "phase3" / "phase3_runs.csv")
    dec_f = root / "results" / "phase3" / "phase3_decoys.csv"
    decoys = pd.read_csv(dec_f) if dec_f.exists() else pd.DataFrame()
    assert len(runs) == 30, f"expected 30 runs, got {len(runs)}"
    fig1(figdir / "fig1_design_diagram.png")
    fig2(runs, decoys, figdir / "fig2_noise_kept_and_sealed_drop.png")
    fig3(runs, figdir / "fig3_proposer_scaling.png")
    fig4(runs, figdir / "fig4_cost_frontier.png")
    print("pre-registered figures written to", figdir)
    d, arm, needle = diagnostics(root, runs, root / "results" / "diagnostics", figdir)
    print("\nNEEDLE (largest-gain accepted feature in best SEALED run):",
          None if not needle else {k: v for k, v in needle.items() if k != "canon"})
    print("\nDIAGNOSTICS BY ARM (exploratory, post hoc)\n", arm.round(2).to_string())
    tot = Counter()
    for sc in d.status_counts:
        tot.update(json.loads(sc))
    print("\nSTATUS VOCABULARY (all non-decoy logged rows):", dict(tot))
    print("rows without spec per arm:", d.groupby("arm").n_rows_without_spec.sum().reindex(ARMS).to_dict())


if __name__ == "__main__":
    main()
