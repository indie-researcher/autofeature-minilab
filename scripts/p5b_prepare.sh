#!/usr/bin/env bash
# p5b_prepare.sh - Phase 5b: two better gates (24-block exact LORD++ and a decoy-empirical-null gate) vs keep-if-better.
# Derives code/af_phase5b_regime.py from code/af_phase5_regime.py (patched), writes the analysis script, plan, 3 CPU notebooks.
#   bash p5b_prepare.sh [--no-smoke]      (repo root; nothing is pushed or committed here)
set -euo pipefail
cd "$(dirname "$0")/.."
KUSER="${KUSER:-yoka33}"
SMOKE=1; [ "${1:-}" = "--no-smoke" ] && SMOKE=0
fail() { echo "ERROR: $*" >&2; exit 1; }
[ -f code/af_phase5_regime.py ] || fail "missing code/af_phase5_regime.py (run p5_prepare.sh first)"
mkdir -p code docs results/p5b kaggle/p5b_a_gates kaggle/p5b_b_gates kaggle/p5b_c_gates

# ============================================================================================ patch the runner
python3 - <<'PY'
from pathlib import Path
s = Path("code/af_phase5_regime.py").read_text()
def sub(old, new, count=1):
    global s
    assert s.count(old) >= 1, f"patch anchor not found: {old[:70]!r}"
    s = s.replace(old, new, count)

sub("# ===================== PHASE 5 / Q2: REGIME MAP - when does a gate pay off? =====================",
    "# ===================== PHASE 5b: do better gates pay off? (24-block exact LORD++, decoy-empirical-null gate) =====================\n"
    "# Same proposals / IP subsamples / holdout as Phase 5 (deterministic), so results pair with results/p5 (naive, fdr12).\n"
    "# Gates: naive | fdr (12 blocks) | fdr24 (24 blocks, exact meet-in-the-middle sign-flip) | decoyq (accept if the candidate's gain is in the\n"
    "# top 5% of a rolling null of the last 40 decoy gains, with a 20-decoy warm-up at the empty stack; the null mixes stack states: approximation).")
sub('DEFAULT_GATES = "naive,fdr"', 'DEFAULT_GATES = "fdr24,decoyq,naive"')
sub('DEAD_P = 2.0 ** -F["blocks"]', '''DEAD_P = 2.0 ** -F["blocks"]
DEAD_P24 = 2.0 ** -24
DQ_WIN, DQ_ALPHA = 40, 0.05
DQ_WARM, DQ_MIN = (6, 5) if TEST else (20, 20)


def signflip_p24(d):
    """Exact one-sided sign-flip p for mean(d) > 0 with up to ~24 blocks, meet-in-the-middle (same convention as signflip_p)."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 4:
        return 1.0
    h = n // 2

    def sums(v):
        s_ = np.zeros(1)
        for x in v:
            s_ = np.concatenate([s_ + x, s_ - x])
        return s_
    A, B = sums(d[:h]), np.sort(sums(d[h:]))
    T = d.sum() - 1e-12
    cnt = len(B) * len(A) - int(np.searchsorted(B, T - A, side="left").sum())
    return float(cnt) / float(2 ** n)
''')
sub("        self.stack, self.ref, self.ref_seed = [], {}, 0\n        self.refit(0)",
    "        self.stack, self.ref, self.ref_seed, self.null = [], {}, 0, []\n        self.refit(0)")

old = s[s.index("        blk = (block_aucs(self.full.yev, rc[\"pred\"], F[\"blocks\"])"): s.index("        return dict(status=\"accepted\" if accept else \"rejected\"")]
new = '''        t = n_tests + 1
        if gate in ("fdr", "fdr24"):
            nb = 24 if gate == "fdr24" else F["blocks"]
            blk = block_aucs(self.full.yev, rc["pred"], nb) - block_aucs(self.full.yev, self.ref["full_pred"], nb)
            p = signflip_p24(blk) if gate == "fdr24" else signflip_p(blk)
            a_t = lord.alpha_t(t)
            accept = bool(p <= a_t and d_full >= F["materiality"])
        elif gate == "decoyq":
            null = self.null[-DQ_WIN:]
            p = (1 + sum(1 for x in null if x >= d_full)) / (1 + len(null))
            a_t = DQ_ALPHA
            accept = bool(len(null) >= DQ_MIN and p <= DQ_ALPHA and d_full >= F["materiality"])
            if is_decoy:
                self.null.append(float(d_full))
        elif gate == "observe":                                     # warm-up decoys: build the null, never accept
            p, a_t, accept = 1.0, 0.0, False
            self.null.append(float(d_full))
        else:                                                       # naive keep-if-better with ratchet
            blk = block_aucs(self.full.yev, rc["pred"], F["blocks"]) - block_aucs(self.full.yev, self.ref["full_pred"], F["blocks"])
            p = signflip_p(blk)
            a_t = lord.alpha_t(t)
            accept = bool(rc["auc"] > self.ref["full_auc"])
        if accept:
            self.stack.append(dict(spec=spec, is_decoy=is_decoy, dseed=dseed, full=mf))
            if gate == "naive":
                self.ref = dict(full_pred=rc["pred"], full_auc=rc["auc"])
            else:
                if gate in ("fdr", "fdr24"):
                    lord.rej.append(t)
                self.refit(self.ref_seed + 1)
'''
s = s.replace(old, new)
sub('return dict(status="accepted" if accept else "rejected",', 'return dict(status="accepted" if accept else ("observed" if gate == "observe" else "rejected"),')

sub("def run_one(lab, hold, props, gate, run, K, B):", '''def make_warmup(run, K, n):
    rng = np.random.RandomState(2_000_003 * run + 7_919 * K + 5)
    out, seen, tries = [], set(), 0
    while len(out) < n and tries < 60 * n:
        tries += 1
        ok, _, ns = normalize(sample_spec(rng))
        if ok and canon5(ns) not in seen:
            seen.add(canon5(ns))
            out.append((ns, True, 6_000_000 + 10_000 * run + len(out)))
    return out


def run_one(lab, hold, props, gate, run, K, B):''')
sub("    lord, log = Lord(F[\"fdr_alpha\"]), []\n", '''    lord, log = Lord(F["fdr_alpha"]), []
    if gate == "decoyq":
        for spec, _, dseed in make_warmup(run, K, DQ_WARM):
            lab.evaluate(spec, True, dseed, "observe", lord, -1)
''')
sub('frac_tests_dead=float(np.mean([r["alpha_t"] < DEAD_P for r in log])) if gate == "fdr" else float("nan"),',
    'frac_tests_dead=float(np.mean([r["alpha_t"] < (DEAD_P24 if gate == "fdr24" else DEAD_P) for r in log])) if gate in ("fdr", "fdr24") else float("nan"),\n        null_n=len(lab.null), warmup=DQ_WARM if gate == "decoyq" else 0,')
sub('"/kaggle/working/p5" if kaggle else "./p5_out"', '"/kaggle/working/p5b" if kaggle else "./p5b_out"')
Path("code/af_phase5b_regime.py").write_text(s)
print("wrote code/af_phase5b_regime.py")
PY
python3 -m py_compile code/af_phase5b_regime.py

# ============================================================================================ analysis
cat > code/af_phase5b_analyze.py <<'PY'
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
PY
python3 -m py_compile code/af_phase5b_analyze.py

# ============================================================================================ plan
cat > docs/PHASE5B_PLAN.md <<'MD'
# Phase 5b plan (freeze before launching; tag `phase5b-freeze`)

Phase 5 result: keep-if-better has the higher net holdout gain than the 12-block LORD++ gate in all six cells, but the 12-block gate was dead (alpha_t < 2^-12)
for 45-94% of tests, so that comparison mixes "FDR control" with "this gate design starves". Phase 5b asks: **does any better-designed gate pay off?**

Arms (same proposals, IP subsamples and holdout as Phase 5; paired by (K, B, run)):
- **fdr24:** LORD++ alpha 0.10 with an exact 24-block sign-flip test (min p = 2^-24, gate cannot die within a run), materiality floor as before.
- **decoyq:** accept if the candidate's gain is in the top 5% of a rolling null of the last 40 decoy gains (empirical p = (1 + #{null >= d}) / (1 + n)), after a 20-decoy warm-up
  evaluated against the empty stack (warm-up tests are extra and not counted in n_tests), and gain >= materiality floor. The null mixes stack states (approximation, disclosed).
- **naive:** replicated in the cheap cells to verify that the pairing/determinism reproduces Phase 5 (max |delta hold gain| reported).
Cells: K=200/B=120, K=50/B=120 (cheap), K=10/B=120 (~144k rows), K=2/B=60 (~712k rows, the crossover region).

- **Primary rule:** a gate "wins" if the paired 95% bootstrap CI of (gate - naive) net holdout gain has lower bound > 0 in at least one cell. If none: "no gate tried beats keep-if-better below ~1M rows".
- **Secondary (descriptive):** optimism, accepted decoys (false-discovery proxy), accepted real features per run (power), frac_tests_dead.
- Caveats: random proposer; one dataset; holdout day 2017-11-08 reused (not fresh); decoyq warm-up adds ~20 tests; 4 runs in the K=2 cell.
MD

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
mk() {   # dir slug cells gates
  sed -e "s|^DEFAULT_CELLS = .*|DEFAULT_CELLS = \"$3\"|" -e "s|^DEFAULT_GATES = .*|DEFAULT_GATES = \"$4\"|" code/af_phase5b_regime.py > "$1/run.py"
  grep -q "^DEFAULT_CELLS = \"$3\"" "$1/run.py" && grep -q "^DEFAULT_GATES = \"$4\"" "$1/run.py" || fail "sed failed for $1"
  python3 -m py_compile "$1/run.py"; write_meta "$1" "$2"; echo "built $1 (cells $3, gates $4)"
}
mk kaggle/p5b_a_gates af-p5b-a-gates "50:120:10,200:120:10" "fdr24,decoyq,naive"   # ~1.2 h (estimate from Phase 5 timings)
mk kaggle/p5b_b_gates af-p5b-b-gates "10:120:8"             "fdr24,decoyq"         # ~2.2 h
mk kaggle/p5b_c_gates af-p5b-c-gates "2:60:4"               "fdr24,decoyq"         # ~4.8 h

# ============================================================================================ run script
cat > scripts/p5b_run.sh <<'RUN'
#!/usr/bin/env bash
# p5b_run.sh - bash p5b_run.sh --dry | push | status | collect | collect-local   (requires git tag phase5b-freeze for push; FORCE=1 overrides)
set -euo pipefail
cd "$(dirname "$0")/.."
KUSER="${KUSER:-yoka33}"; MODE="${1:-}"
NBS=(p5b_a_gates:af-p5b-a-gates p5b_b_gates:af-p5b-b-gates p5b_c_gates:af-p5b-c-gates)
case "$MODE" in
  status) for e in "${NBS[@]}"; do kaggle kernels status "$KUSER/${e#*:}" 2>&1 | grep -v outdated || true; done; exit 0 ;;
  collect|collect-local)
    if [ "$MODE" = "collect" ]; then
      for e in "${NBS[@]}"; do nb="${e%%:*}"; slug="${e#*:}"
        echo "== $slug =="; kaggle kernels status "$KUSER/$slug" 2>&1 | grep -v outdated || true
        rm -rf "out_p5b/$slug"; mkdir -p "out_p5b/$slug" "results/p5b/$nb"
        kaggle kernels output "$KUSER/$slug" -p "out_p5b/$slug" 2>&1 | grep -v outdated | tail -2 || echo "(download failed)"
        find "out_p5b/$slug" -type f \( -name 'p5_run_*.json' -o -name 'p5_runs.csv' \) -exec cp {} "results/p5b/$nb/" \;
      done
    fi
    python3 code/af_phase5b_analyze.py; exit 0 ;;
esac
for e in "${NBS[@]}"; do d="kaggle/${e%%:*}"
  [ -f "$d/run.py" ] && [ -f "$d/kernel-metadata.json" ] || { echo "ERROR: $d not built (bash p5b_prepare.sh)"; exit 1; }
  python3 -m py_compile "$d/run.py"; grep -q "\"id\": \"$KUSER/${e#*:}\"" "$d/kernel-metadata.json" || { echo "ERROR: id mismatch in $d"; exit 1; }
  echo "ok  $d"
done
if ! git rev-parse -q --verify refs/tags/phase5b-freeze >/dev/null 2>&1; then
  [ "${FORCE:-}" = "1" ] || { echo "ERROR: tag phase5b-freeze missing (commit the plan and tag it, or FORCE=1)"; exit 1; }
fi
[ "$MODE" = "--dry" ] && { echo "dry run: nothing pushed"; exit 0; }
[ "$MODE" = "push" ] || { echo "usage: bash p5b_run.sh --dry|push|status|collect|collect-local"; exit 1; }
kaggle kernels list --mine --page-size 1 >/dev/null 2>&1 || { echo "Kaggle auth failed"; exit 1; }
FAILED=()
for e in "${NBS[@]}"; do d="kaggle/${e%%:*}"; echo "-- $d"; kaggle kernels push -p "$d" 2>&1 | grep -v outdated || FAILED+=("$d"); done
[ ${#FAILED[@]} -eq 0 ] || { echo "FAILED pushes: ${FAILED[*]}"; exit 1; }
echo "Pushed. Estimated: a ~1.2 h, b ~2.2 h, c ~4.8 h. status: bash p5b_run.sh status | collect: bash p5b_run.sh collect"
RUN
chmod +x scripts/p5b_run.sh

# ============================================================================================ tests
if [ "$SMOKE" = "1" ] && python3 -c "import lightgbm, pandas" 2>/dev/null; then
  echo "== unit test: exact 24-block sign-flip against the 12-block routine and brute force =="
  python3 - <<'PY'
import sys, itertools, numpy as np
sys.path.insert(0, "code")
import importlib.util
spec = importlib.util.spec_from_file_location("m5b", "code/af_phase5b_regime.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
rs = np.random.RandomState(0)
for n in (12, 12, 8, 13, 16):
    d = rs.randn(n) * 0.01 + 0.003
    brute = float(np.mean([(np.array(s) @ d) >= d.sum() - 1e-12 for s in itertools.product((-1.0, 1.0), repeat=n)]))
    p24 = m.signflip_p24(d)
    ref = m.signflip_p(d) if n == 12 else brute
    assert abs(p24 - ref) < 1e-12 and abs(p24 - brute) < 1e-12, (n, p24, ref, brute)
d = np.full(24, 0.01); assert abs(m.signflip_p24(d) - 2.0 ** -24) < 1e-18
print("signflip_p24 OK (n=8,12,13,16 vs brute force; 24 equal blocks -> 2^-24)")
PY
  echo "== smoke run (synthetic windows; never reads train.csv) =="
  rm -rf /tmp/p5b_smoke; mkdir -p /tmp/p5b_smoke
  AF_TEST=1 AF_TEST_OUT=/tmp/p5b_smoke timeout 900 python3 code/af_phase5b_regime.py --gates fdr24,decoyq,naive 2>&1 | tail -5
  ls /tmp/p5b_smoke | head -4
else echo "(smoke tests skipped)"; fi
echo; echo "READY. Next: git add code docs kaggle scripts/p5b_prepare.sh scripts/p5b_run.sh && git commit -m 'Phase 5b plan, gates and runner' && git tag phase5b-freeze"
