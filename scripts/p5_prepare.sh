#!/usr/bin/env bash
# p5_prepare.sh - Phase 5: build the regime-map runner (Q2), 3 Kaggle CPU notebooks, the analysis + calibration (Q1) scripts, plan doc.
#   bash p5_prepare.sh [--no-smoke]        (run from the repo root; nothing is pushed or committed here)
set -euo pipefail
cd "$(dirname "$0")/.."
KUSER="${KUSER:-yoka33}"
SRC3="${SRC3:-code/af_phase3_sealed_combined.py}"
SMOKE=1; [ "${1:-}" = "--no-smoke" ] && SMOKE=0
fail() { echo "ERROR: $*" >&2; exit 1; }
[ -f "$SRC3" ] || fail "missing $SRC3"
grep -q "^EMBEDDED_INPUTS = " "$SRC3" || fail "$SRC3 has no EMBEDDED_INPUTS marker"
mkdir -p code docs results/p5 kaggle/p5_a_regime kaggle/p5_b_regime kaggle/p5_c_regime
HEADN=$(( $(grep -n "^EMBEDDED_INPUTS = " "$SRC3" | head -1 | cut -d: -f1) - 1 ))

# ============================================================================================ regime runner
{
  head -n "$HEADN" "$SRC3"
  cat <<'PY'


# ===================== PHASE 5 / Q2: REGIME MAP - when does a gate pay off? =====================
# Random proposals from the Phase 3 grammar; the SAME proposal sequence is run under the naive and the LORD++ gate (paired).
# Training size varies through an IP subsample (c0 % K == residue). No proxy screen (every proposal is a full test).
# Per run: DEV gain re-measured with fresh seeds vs the same final stack re-fitted on a holdout day (2017-11-08, already used
# for Phase 3 verdicts: NOT a fresh holdout; 2017-11-07 stays reserved). Accepted decoys are re-permuted on the holdout.
# Gate: Phase 3 gate (12 blocks, LORD++ alpha 0.10, materiality floor in absolute AUC units). With 12 blocks the gate cannot
# accept once alpha_t < 2^-12 (about test 8 without an early accept): reported as frac_tests_dead.
import argparse, json, math, os, time
from pathlib import Path

DEFAULT_CELLS = "200:120:10"          # "K:B:runs,..." (K = IP-subsample divisor, B = proposals per run); replaced per notebook
DEFAULT_GATES = "naive,fdr"
HOLDOUT_DAY, HOLDOUT_START_ROW = "2017-11-08", 68_941_878
EVAL_SEEDS = (100, 101, 102)
DEAD_P = 2.0 ** -F["blocks"]


def canon5(s):
    return json.dumps(s, sort_keys=True, separators=(",", ":"))


class Lab5:
    """One FULL track on an IP-subsampled window; no proxy screen; stack/reference state for one search run."""

    def __init__(self, W, K, residue):
        sel = (W["c0"] % K) == residue if K > 1 else np.ones(len(W["c0"]), dtype=bool)
        self.K, self.W = K, {k: v[sel] for k, v in W.items()}
        self.full = Track(self.W, F["neg_train"], F["neg_eval"])
        self._cache, self._base3 = {}, None
        self.reset()

    def reset(self):
        self.stack, self.ref, self.ref_seed = [], {}, 0
        self.refit(0)

    def refit(self, seed):
        f = self.full.fit([s["full"] for s in self.stack], seed)
        self.ref, self.ref_seed = dict(full_pred=f["pred"], full_auc=f["auc"]), seed

    def col(self, spec, is_decoy, dseed):
        k = (canon5(spec), bool(is_decoy), int(dseed) if is_decoy else 0)
        if k not in self._cache:
            if len(self._cache) > 400:
                self._cache.clear()
            af = compute(spec, self.W)
            self._cache[k] = self.full.col(make_decoy(af, dseed) if is_decoy else af)
        return self._cache[k]

    def fit3(self, extras):
        return float(np.mean([self.full.fit(extras, s)["auc"] for s in EVAL_SEEDS]))

    def base3(self):
        if self._base3 is None:
            self._base3 = self.fit3([])
        return self._base3

    def evaluate(self, spec, is_decoy, dseed, gate, lord, n_tests):
        mf = self.col(spec, is_decoy, dseed)
        rc = self.full.fit([s["full"] for s in self.stack] + [mf], self.ref_seed)
        d_full = rc["auc"] - self.ref["full_auc"]
        blk = (block_aucs(self.full.yev, rc["pred"], F["blocks"]) - block_aucs(self.full.yev, self.ref["full_pred"], F["blocks"]))
        p = signflip_p(blk)
        t = n_tests + 1
        a_t = lord.alpha_t(t)
        accept = bool(rc["auc"] > self.ref["full_auc"]) if gate == "naive" else bool(p <= a_t and d_full >= F["materiality"])
        if accept:
            self.stack.append(dict(spec=spec, is_decoy=is_decoy, dseed=dseed, full=mf))
            if gate == "naive":
                self.ref = dict(full_pred=rc["pred"], full_auc=rc["auc"])           # ratchet, as in the Phase 3 naive arm
            else:
                lord.rej.append(t)
                self.refit(self.ref_seed + 1)
        return dict(status="accepted" if accept else "rejected", d_full=round(float(d_full), 6), p=round(float(p), 6),
                    alpha_t=round(float(a_t), 6), t_test=t)


def make_proposals(run, K, B):
    rng = np.random.RandomState(1_000_003 * run + 7_919 * K + B)
    out, seen, tries = [], set(), 0
    while len(out) < B and tries < 60 * B:
        tries += 1
        ok, _, ns = normalize(sample_spec(rng))
        if not ok or canon5(ns) in seen:
            continue
        seen.add(canon5(ns))
        j = len(out)
        out.append((ns, j % 5 == 4, 5_000_000 + 10_000 * run + j))              # every 5th proposal is a permuted-real decoy
    return out


def run_one(lab, hold, props, gate, run, K, B):
    t0 = time.time()
    lab.reset()
    lord, log = Lord(F["fdr_alpha"]), []
    for n, (spec, is_decoy, dseed) in enumerate(props):
        rec = lab.evaluate(spec, is_decoy, dseed, gate, lord, n)
        rec.update(spec=spec, is_decoy=bool(is_decoy), i=n + 1)
        log.append(rec)
    acc = [r for r in log if r["status"] == "accepted"]
    dev_base = lab.base3()
    dev_final = lab.fit3([s["full"] for s in lab.stack])
    hold_final = hold.fit3([hold.col(s["spec"], s["is_decoy"], s["dseed"] + 777) for s in lab.stack])
    hold_base = hold.base3()
    gain_dev, gain_hold = dev_final - dev_base, hold_final - hold_base
    n_dec = sum(1 for r in log if r["is_decoy"])
    return dict(
        K=K, B=B, run=run, gate=gate, n_train=len(lab.full.idx[0]), n_eval=len(lab.full.idx[2]), eval_pos=int(lab.full.yev.sum()),
        n_tests=len(log), accepted_real=sum(1 for r in acc if not r["is_decoy"]), accepted_decoys=sum(1 for r in acc if r["is_decoy"]),
        decoys_tested=n_dec, frac_tests_dead=float(np.mean([r["alpha_t"] < DEAD_P for r in log])) if gate == "fdr" else float("nan"),
        dev_base=dev_base, dev_final=dev_final, gain_dev=gain_dev, hold_base=hold_base, hold_final=hold_final, gain_hold=gain_hold,
        optimism=gain_dev - gain_hold, minutes=round((time.time() - t0) / 60, 2), trials=log)


def _synthetic_window(seed, n=6000):
    rng = np.random.RandomState(seed)
    t = (np.arange(n) * 10).astype(np.int32)
    W = {f"c{i}": rng.randint(0, m, n).astype(np.int32) for i, m in enumerate((40, 30, 20, 15, 10))}
    W.update(t=t, h=(t // 3600).astype(np.int32), y=(rng.rand(n) < 0.06).astype(np.uint8))
    return W


def regime_main():
    global WINDOW_DAY, DAY_START_ROW
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=os.environ.get("AF_CELLS", DEFAULT_CELLS))
    ap.add_argument("--gates", default=os.environ.get("AF_GATES", DEFAULT_GATES))
    ap.add_argument("--max-mins", type=float, default=560.0)
    ap.add_argument("--cache-dir", default="/tmp/af_p5_cache")
    ap.add_argument("--out", default="")
    a, _ = ap.parse_known_args()
    kaggle = Path("/kaggle/working").exists()
    out = Path(a.out or os.environ.get("AF_TEST_OUT", "") or ("/kaggle/working/p5" if kaggle else "./p5_out"))
    out.mkdir(parents=True, exist_ok=True)
    cells = [tuple(int(x) for x in c.split(":")) for c in a.cells.split(",") if c]
    gates = [g for g in a.gates.split(",") if g]
    t0 = time.time()
    if TEST:
        Wd, Wh = _synthetic_window(0), _synthetic_window(1)
        cells = [(2, 10, 2)]
    else:
        Wd = get_window(Path(a.cache_dir))                                  # DEV window (Phase 1-3 defaults)
        WINDOW_DAY, DAY_START_ROW = HOLDOUT_DAY, HOLDOUT_START_ROW
        Wh = get_window(Path(a.cache_dir))                                  # holdout window (Phase 3 sealed day)
    print(f"cells {cells} gates {gates} | windows loaded in {(time.time() - t0) / 60:.1f} min", flush=True)
    rows = []
    for (K, B, nruns) in cells:
        for run in range(1, nruns + 1):
            if (time.time() - t0) / 60 > a.max_mins:
                print("time guard reached: stopping, partial results kept", flush=True)
                break
            todo = [g for g in gates if not (out / f"p5_run_K{K}_B{B}_{g}_{run}.json").exists()]
            if not todo:
                continue
            residue = run % K
            lab, hold, props = Lab5(Wd, K, residue), Lab5(Wh, K, residue), make_proposals(run, K, B)
            for g in todo:
                r = run_one(lab, hold, props, g, run, K, B)
                (out / f"p5_run_K{K}_B{B}_{g}_{run}.json").write_text(json.dumps(r, default=float))
                rows.append({k: v for k, v in r.items() if k != "trials"})
                print(f"K={K} B={B} run={run} {g}: n_train {r['n_train']:,} acc real/decoy {r['accepted_real']}/{r['accepted_decoys']} "
                      f"dev {r['gain_dev']:+.5f} hold {r['gain_hold']:+.5f} optimism {r['optimism']:+.5f} | {(time.time() - t0) / 60:.0f} min",
                      flush=True)
            del lab, hold
    import pandas as pd
    allr = [json.load(open(f)) for f in sorted(out.glob("p5_run_*.json"))]
    pd.DataFrame([{k: v for k, v in r.items() if k != "trials"} for r in allr]).to_csv(out / "p5_runs.csv", index=False)
    print("REGIME_DONE", len(allr), "runs", flush=True)


if __name__ == "__main__":
    regime_main()
PY
} > code/af_phase5_regime.py
python3 -m py_compile code/af_phase5_regime.py
echo "wrote code/af_phase5_regime.py ($(wc -l < code/af_phase5_regime.py) lines)"

# ============================================================================================ analysis (post-processing)
cat > code/af_phase5_analyze.py <<'PY'
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
PY
python3 -m py_compile code/af_phase5_analyze.py; echo "wrote code/af_phase5_analyze.py"

# ============================================================================================ Q1: p-value calibration (local, no notebook)
cat > code/af_phase5_calibration.py <<'PY'
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
PY
python3 -m py_compile code/af_phase5_calibration.py; echo "wrote code/af_phase5_calibration.py"

# ============================================================================================ plan note
cat > docs/PHASE5_PLAN.md <<'MD'
# Phase 5 plan (freeze before launching; tag `phase5-freeze`)

**Q2 - regime map.** When does a statistical gate (LORD++) pay off versus keep-if-better? Phase 3 found no benefit at 25M rows and ~10 full tests per run.
Design: random proposals from the Phase 3 grammar, same sequence under both gates (paired); 20% permuted-real decoys; no proxy screen;
training size varied by IP subsample (K = 200, 50, 10, 2 -> roughly 7k to 740k training rows); proposals per run B = 30 / 60 / 120.
Metric: optimism = DEV gain (3 fresh seeds) - holdout gain (same final stack re-fitted on 2017-11-08, 3 seeds), AUC units; also holdout gain (net benefit),
accepted decoys, fraction of tests where the gate could not accept (alpha_t < 2^-12).
- **H5-1:** the naive gate's excess optimism (naive - FDR, paired) grows as n shrinks. Supported if Spearman(d_optimism, log n) <= -0.6 AND the paired 95% bootstrap CI at the smallest n has lower bound > 0.
- **H5-2:** at the smallest n, FDR has the higher holdout gain (paired CI of naive - FDR holdout gain has upper bound < 0). Reported either way.
- Fail rule: "no regime in this grid where the gate matters; keep-if-better is harmless on this task."
- Caveats (disclosed): holdout day 2017-11-08 was already used for Phase 3 verdicts (not fresh; 2017-11-07 stays reserved); random proposer only;
  12-block gate can die after ~8 tests; materiality floor is fixed in absolute AUC units; one dataset; IP-subsample (not row-subsample) changes n.

**Q1 - calibration (post-processing, no notebook).** From the canary trial logs: ECDF of decoy p-values versus nominal levels (Wilson CIs), decoy d_full distribution and the
share above the materiality floor. Descriptive.
MD
echo "wrote docs/PHASE5_PLAN.md"

# ============================================================================================ notebooks
write_meta() { cat > "$1/kernel-metadata.json" <<JSON
{
  "id": "$KUSER/$2",
  "title": "$2",
  "code_file": "run.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": false,
  "enable_tpu": false,
  "enable_internet": false,
  "dataset_sources": [],
  "competition_sources": ["talkingdata-adtracking-fraud-detection"],
  "kernel_sources": []
}
JSON
}
mk() {   # dir slug cells
  sed "s|^DEFAULT_CELLS = .*|DEFAULT_CELLS = \"$3\"|" code/af_phase5_regime.py > "$1/run.py"
  grep -q "^DEFAULT_CELLS = \"$3\"" "$1/run.py" || fail "sed failed for $1"
  python3 -m py_compile "$1/run.py"; write_meta "$1" "$2"; echo "built $1 (cells $3)"
}
mk kaggle/p5_a_regime af-p5-a-regime "200:120:10,50:30:10,50:120:10"     # ~4 h (estimate)
mk kaggle/p5_b_regime af-p5-b-regime "10:120:8,10:30:8"                   # ~6.5 h
mk kaggle/p5_c_regime af-p5-c-regime "2:60:4"                             # ~5.5 h

# ============================================================================================ smoke tests
if [ "$SMOKE" = "1" ] && python3 -c "import lightgbm, pandas" 2>/dev/null; then
  echo "== smoke tests (synthetic windows; never read train.csv) =="
  rm -rf /tmp/p5_smoke; mkdir -p /tmp/p5_smoke
  AF_TEST=1 AF_TEST_OUT=/tmp/p5_smoke timeout 900 python3 code/af_phase5_regime.py 2>&1 | tail -4
  python3 code/af_phase5_analyze.py --root /tmp/p5_smoke --out /tmp/p5_smoke 2>&1 | tail -4
else echo "(smoke tests skipped)"; fi
echo; echo "READY. Next: git add code docs kaggle && git commit -m 'Phase 5 plan and regime-map runner' && git tag phase5-freeze"
