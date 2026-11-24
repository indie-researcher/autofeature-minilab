#!/usr/bin/env python3
"""
af_phase4_report.py - one script for everything after the Phase 4 Kaggle runs.

  python3 code/af_phase4_report.py --organize     # tidy folders from fresh downloads, then report
  python3 code/af_phase4_report.py                # report only (reads results/phase4)

--organize : takes raw downloads out_phase4/af-phase4-<nb>/ (from `kaggle kernels output`) and rebuilds
             results/phase4/<nb>/ from them. The previous results/phase4 is MOVED (never deleted) to
             results/phase4_prev_<timestamp>/, so stale runs from older kernel versions are archived, not mixed in.
             Aborts without touching anything if the raw downloads contain no final_*.json.
Report     : results/phase4/PHASE4_REPORT.md, phase4_runs_all.csv, phase4_arms.csv, phase4_canary.csv,
             phase4_gap_by_arm.png. Key names are matched ignoring case/underscores (n_proposals == nproposals),
             and the report lists any expected key that was missing everywhere, so a schema mismatch is visible.
Incomplete runs (trials without a final file) are listed, not averaged. All numbers are DEV-feedback numbers.
"""
import argparse, ast, math, re, shutil, time
from collections import Counter
from pathlib import Path
import json
import numpy as np
import pandas as pd

CANON = ["rung", "arm", "run", "n_proposals", "budget_n", "capped", "unique_specs", "n_screened", "n_full_tests",
         "accepted_real", "accepted_decoys", "decoys_proposed", "decoys_screened", "decoys_tested",
         "decoys_tested_alive", "gate_dead_at_t", "decoy_fpr", "n_dup_total", "n_attempts_total", "repeat_frac",
         "needle_first_proposed_i", "needle_accepted", "dev_baseline_remeasured", "dev_final_auc",
         "dev_gap_closed_pct", "model", "tokens", "minutes", "canary", "stack_sig", "first_accepted_spec",
         "best_d_proxy", "proxy_pass_frac", "accepted_new_axis", "accepted_ops", "gate"]
NK = lambda k: re.sub(r"[^a-z0-9]", "", str(k).lower())
LOOKUP = {NK(c): c for c in CANON}
LOOKUP.update({"proposals": "n_proposals", "screened": "n_screened", "fulltests": "n_full_tests",
               "devgapclosed": "dev_gap_closed_pct", "acceptedreal": "accepted_real"})
REQUIRED = ["rung", "arm", "run", "n_proposals", "n_full_tests", "accepted_real", "dev_gap_closed_pct", "minutes"]
FNAME = re.compile(r"^(final|trials)_r(\d+)_(.+)_(\d+)\.(json|jsonl)$")


def canon_rec(rec):
    out = {}
    for k, v in rec.items():
        c = LOOKUP.get(NK(k))
        if c and c not in out:
            out[c] = v
    return out


def as_obj(v):
    if isinstance(v, str):
        try:
            return ast.literal_eval(v)
        except Exception:
            return v
    return v


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def md_table(df, nd=2):
    if df is None or len(df) == 0:
        return "_none_\n"
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("-" if np.isnan(v) else f"{v:.{nd}f}")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------------- organize
def organize(raw_root: Path, res_root: Path):
    slugs = sorted(p for p in raw_root.glob("af-phase4-*") if p.is_dir())
    n_final = sum(1 for s in slugs for _ in s.rglob("final_*.json"))
    if not slugs or n_final == 0:
        raise SystemExit(f"--organize: no final_*.json under {raw_root.resolve()}/af-phase4-*; nothing changed.")
    old = {f.name for f in res_root.rglob("*") if f.is_file()} if res_root.exists() else set()
    if res_root.exists() and any(res_root.iterdir()):
        arch = res_root.parent / f"{res_root.name}_prev_{time.strftime('%Y%m%d_%H%M%S')}"
        shutil.move(str(res_root), str(arch))
        print(f"archived previous results -> {arch}")
    res_root.mkdir(parents=True, exist_ok=True)
    new = set()
    for s in slugs:
        nb = s.name.replace("af-phase4-", "").replace("-", "_")
        dest = res_root / nb
        dest.mkdir(exist_ok=True)
        for pat in ("final_*.json", "trials_*.jsonl", "summary_runs.csv", "summary_arms.md"):
            for f in s.rglob(pat):
                shutil.copy2(f, dest / f.name)
                new.add(f.name)
    dropped = sorted(n for n in old - new if FNAME.match(n))
    print(f"organized {len(slugs)} kernels, {n_final} final files. "
          f"{len(dropped)} run files from the previous layout are no longer present (stale): {dropped[:12]}")


# ----------------------------------------------------------------------------- load
def load(res_root: Path):
    rows, trial_only = [], []
    finals = {}
    for nbdir in sorted(p for p in res_root.iterdir() if p.is_dir()):
        for f in sorted(nbdir.glob("final_*.json")):
            m = FNAME.match(f.name)
            try:
                rec = json.load(open(f))
            except Exception as e:
                print(f"unreadable {f}: {e}")
                continue
            c = canon_rec(rec)
            if m:
                c.setdefault("rung", int(m.group(2)))
                c.setdefault("arm", m.group(3))
                c.setdefault("run", int(m.group(4)))
            c["notebook"], c["file"] = nbdir.name, f.name
            rows.append(c)
            finals[(nbdir.name, c["rung"], c["arm"], c["run"])] = True
        for f in sorted(nbdir.glob("trials_*.jsonl")):
            m = FNAME.match(f.name)
            if not m:
                continue
            key = (nbdir.name, int(m.group(2)), m.group(3), int(m.group(4)))
            if key in finals:
                continue
            recs = [json.loads(l) for l in open(f) if l.strip()]
            st = Counter(r.get("status") for r in recs)
            trial_only.append(dict(notebook=key[0], rung=key[1], arm=key[2], run=key[3], rows_logged=len(recs),
                                   accepted_so_far=st.get("accepted", 0), statuses=dict(st)))
    df = pd.DataFrame(rows)
    if len(df):
        for c in ("accepted_ops", "first_accepted_spec", "tokens"):
            if c in df.columns:
                df[c] = df[c].map(as_obj)
        df["canary"] = [bool(c) if isinstance(c, (bool, np.bool_)) else str(a).startswith("C-")
                        for c, a in zip(df.get("canary", [None] * len(df)), df["arm"])]
    return df, pd.DataFrame(trial_only)


# ----------------------------------------------------------------------------- report
def bsum(x, col):
    """Count of True flags; NaN when the column is absent/empty (a missing flag is not a True)."""
    if col not in x or x[col].isna().all():
        return np.nan
    return int(x[col].fillna(False).astype(bool).sum())


def nuniq(x, col):
    return x[col].round(9).nunique() if col in x and x[col].notna().any() else np.nan


def agg_search(df):
    out = []
    for (rung, arm), x in df.groupby(["rung", "arm"]):
        g = x.get("dev_gap_closed_pct")
        out.append(dict(
            rung=rung, arm=arm, runs=len(x),
            proposals=x.n_proposals.mean() if "n_proposals" in x else np.nan,
            capped=bsum(x, "capped"),
            full_tests=x.n_full_tests.mean() if "n_full_tests" in x else np.nan,
            accepted_real=x.accepted_real.mean() if "accepted_real" in x else np.nan,
            gap_mean=g.mean(), gap_median=g.median(), gap_min=g.min(), gap_max=g.max(),
            distinct_outcomes=nuniq(x, "dev_final_auc"),
            needle_accepted=bsum(x, "needle_accepted"),
            new_axis=x.accepted_new_axis.mean() if "accepted_new_axis" in x else np.nan,
            dups=x.n_dup_total.mean() if "n_dup_total" in x else np.nan,
            minutes=x.minutes.mean() if "minutes" in x else np.nan,
            decoys_tested=int(x.decoys_tested.sum()) if "decoys_tested" in x else np.nan,
            decoys_accepted=int(x.accepted_decoys.sum()) if "accepted_decoys" in x else np.nan))
    return pd.DataFrame(out)


def agg_canary(df):
    out = []
    for (rung, arm), x in df.groupby(["rung", "arm"]):
        n, k = int(x.decoys_tested.sum()), int(x.accepted_decoys.sum())
        lo, hi = wilson(k, n)
        out.append(dict(rung=rung, arm=arm, runs=len(x), decoys_tested=n, decoys_accepted=k,
                        fpr=k / n if n else np.nan, wilson95_low=lo, wilson95_high=hi,
                        mean_dev_gap_from_noise=x.dev_gap_closed_pct.mean()))
    return pd.DataFrame(out)


def figure(df, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    s = df[~df.canary]
    cells = sorted(s.groupby(["rung", "arm"]).groups.keys())
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(cells)), 4.4))
    for i, (rung, arm) in enumerate(cells):
        g = s[(s.rung == rung) & (s.arm == arm)].dev_gap_closed_pct
        ax.scatter(np.full(len(g), i) + np.random.RandomState(i).uniform(-.1, .1, len(g)), g, s=34, alpha=.85)
        ax.hlines(g.mean(), i - .25, i + .25, color="k")
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([f"r{r}\n{a}" for r, a in cells], fontsize=8)
    ax.set_ylabel("DEV gap closed (%)")
    ax.set_title("Phase 4: DEV gap closed per run (DEV feedback, optimistic)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def report(res_root: Path):
    df, trial_only = load(res_root)
    if not len(df):
        raise SystemExit(f"No final_*.json under {res_root.resolve()}. Run with --organize after downloading, "
                         f"or pass --root.")
    df.to_csv(res_root / "phase4_runs_all.csv", index=False)
    search, canary = df[~df.canary], df[df.canary]
    L = ["# Phase 4 report (generated)", "",
         f"{len(df)} finished runs ({len(search)} search, {len(canary)} canary) across "
         f"{df.notebook.nunique()} notebooks; {len(trial_only)} incomplete runs listed at the end. "
         "All gap numbers use DEV feedback (optimistic); SEALED is not used in Phase 4.", ""]
    missing = [k for k in REQUIRED if k not in df.columns or df[k].isna().all()]
    if missing:
        L += [f"> WARNING: expected keys missing or empty in every final file: {missing}. "
              f"Keys actually found: {sorted(df.columns)}", ""]
    if len(search):
        a = agg_search(search)
        a.to_csv(res_root / "phase4_arms.csv", index=False)
        L += ["## Search arms by rung", "", md_table(a)]
        cols = [c for c in ["rung", "arm", "run", "n_proposals", "capped", "n_full_tests", "accepted_real",
                            "accepted_decoys", "decoys_tested", "dev_gap_closed_pct", "minutes", "needle_accepted",
                            "accepted_new_axis", "n_dup_total", "notebook"] if c in search.columns]
        L += ["## Per run (search arms)", "", md_table(search.sort_values(["rung", "arm", "run"])[cols])]
        ops = {}
        if "accepted_ops" in search.columns:
            for (rung, arm), x in search.groupby(["rung", "arm"]):
                c = Counter()
                for d in x.accepted_ops:
                    if isinstance(d, dict):
                        c.update(d)
                ops[f"r{rung} {arm}"] = dict(c)
        if ops:
            L += ["## Accepted operations (counts over runs)", ""] + [f"- {k}: {v}" for k, v in ops.items()] + [""]
        if "first_accepted_spec" in search.columns:
            L += ["## First accepted feature per run", ""]
            for _, r in search.sort_values(["rung", "arm", "run"]).iterrows():
                if isinstance(r.first_accepted_spec, dict):
                    L.append(f"- r{r.rung} {r.arm} #{r.run}: `{json.dumps(r.first_accepted_spec, sort_keys=True)}`")
            L.append("")
    if len(canary):
        c = agg_canary(canary)
        c.to_csv(res_root / "phase4_canary.csv", index=False)
        L += ["## Canary: gate-only decoy stress test (proxy bypassed)", "", md_table(c, 4),
              "_fpr = accepted decoys / decoys tested. Wilson 95% interval. mean_dev_gap_from_noise = DEV gap "
              "'closed' by accepted noise._", ""]
    if len(trial_only):
        L += ["## Incomplete runs (trials without a final file; not averaged)", "",
              md_table(trial_only[["notebook", "rung", "arm", "run", "rows_logged", "accepted_so_far"]], 0)]
    wrote = figure(df, res_root / "phase4_gap_by_arm.png")
    (res_root / "PHASE4_REPORT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"wrote {res_root / 'PHASE4_REPORT.md'}" + (", phase4_gap_by_arm.png" if wrote else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/phase4")
    ap.add_argument("--raw", default="out_phase4", help="folder with kaggle downloads (af-phase4-<nb>/)")
    ap.add_argument("--organize", action="store_true")
    a = ap.parse_args()
    res = Path(a.root)
    if a.organize:
        organize(Path(a.raw), res)
    if not res.exists():
        raise SystemExit(f"{res.resolve()} does not exist (cwd={Path.cwd()}). Run from the repo root.")
    report(res)


if __name__ == "__main__":
    main()
