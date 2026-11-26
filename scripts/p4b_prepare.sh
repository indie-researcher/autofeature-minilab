#!/usr/bin/env bash
# p4b_prepare.sh - build the four Phase 4b CPU notebooks (nothing is pushed here).
#   1 canary, naive gate   2 canary, FDR gate (paired decoys)   3 audit of FDR-rejected candidates   4 grammar census
# Usage (repo root):  bash p4b_prepare.sh [--no-smoke]      env: KUSER=yoka33  CAN_RUNS=4-13
set -euo pipefail
cd "$(dirname "$0")/.."
KUSER="${KUSER:-yoka33}"
CAN_RUNS="${CAN_RUNS:-4-13}"                 # pilot canary used run ids 1-3; same ids in both gates = same decoys (paired)
SRC4="${SRC4:-code/af_phase4.py}"
SRC3="${SRC3:-code/af_phase3_sealed_combined.py}"
SMOKE=1; [ "${1:-}" = "--no-smoke" ] && SMOKE=0
fail() { echo "ERROR: $*" >&2; exit 1; }

echo "== pre-flight =="
[ -f "$SRC4" ] || fail "missing $SRC4 (revised Phase 4 runner, the file push_phase4.sh uses)"
[ -f "$SRC3" ] || fail "missing $SRC3"
for k in DEFAULT_RUNGS DEFAULT_JOBS DEFAULT_N DEFAULT_MODE; do grep -q "^$k = " "$SRC4" || fail "$SRC4 has no '$k = ' line"; done
grep -q "^EMBEDDED_INPUTS = " "$SRC3" || fail "$SRC3 has no EMBEDDED_INPUTS marker"
[ -f results/phase3/phase3_decoys.csv ] || fail "missing results/phase3/phase3_decoys.csv"
[ "$(ls results/phase3/phase3_run_*.json 2>/dev/null | wc -l)" -eq 30 ] || fail "expected 30 results/phase3/phase3_run_*.json"

write_meta() {   # $1 dir  $2 slug
cat > "$1/kernel-metadata.json" <<JSON
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

mk_canary() {    # $1 dir  $2 slug  $3 jobs
  mkdir -p "$1"
  sed -e 's|^DEFAULT_RUNGS = .*|DEFAULT_RUNGS = "2"|' \
      -e "s|^DEFAULT_JOBS = .*|DEFAULT_JOBS = \"$3\"|" \
      -e 's|^DEFAULT_N = .*|DEFAULT_N = 30|' \
      -e 's|^DEFAULT_MODE = .*|DEFAULT_MODE = "run"|' "$SRC4" > "$1/run.py"
  grep -q "^DEFAULT_JOBS = \"$3\"" "$1/run.py" || fail "sed did not set DEFAULT_JOBS for $1"
  python3 -m py_compile "$1/run.py"
  write_meta "$1" "$2"
  echo "built $1  (jobs=$3, rung 2, 30 tests/run)"
}

echo "== notebooks 1 and 2: canary (gate-only stress test, proxy bypassed) =="
mk_canary kaggle/p4b_1_canary_naive af-p4b-1-canary-naive "C-naive:$CAN_RUNS"
mk_canary kaggle/p4b_2_canary_fdr   af-p4b-2-canary-fdr   "C-FDR:$CAN_RUNS"
echo "   check pairing (decoy RNG should depend on run id, not on arm):"
grep -n "RandomState" "$SRC4" | grep -i "run\|rung\| i" | head -6 || true

# ------------------------------------------------------------------------------------------------ shared pieces
HEADN=$(( $(grep -n "^EMBEDDED_INPUTS = " "$SRC3" | head -1 | cut -d: -f1) - 1 ))
echo "Phase 1 machinery = first $HEADN lines of $SRC3"

SYNTH='
def _synthetic_window(n=6000):
    rng = np.random.RandomState(0)
    t = (np.arange(n) * 10).astype(np.int32)
    W = {f"c{i}": rng.randint(0, m, n).astype(np.int32) for i, m in enumerate((40, 30, 20, 15, 10))}
    W.update(t=t, h=(t // 3600).astype(np.int32), y=(rng.rand(n) < 0.06).astype(np.uint8))
    return W
'

# ------------------------------------------------------------------------------------------------ notebook 3
D3=kaggle/p4b_3_audit; mkdir -p "$D3"
echo "== notebook 3: audit of FDR-rejected candidates (embedding data) =="
python3 - "$D3/audit_data.json" <<'PY'
import csv, glob, json, sys
MAT = 0.00021865718476891195
arms = ["R", "R-naive", "L-naive", "L-FDR", "L-FDR-3B", "L-FDR-14B"]
stacks, cands = {}, []
for arm in arms:
    for run in range(1, 6):
        stacks[f"{arm}|{run}"] = json.load(open(f"results/phase3/phase3_run_{arm}_{run}.json"))["accepted_real_specs"]
        for l in open(f"results/{arm}/trials_{arm}_{run}.jsonl"):
            r = json.loads(l)
            if r.get("status") == "rejected" and not r.get("is_decoy") and (r.get("d_full") or 0) >= MAT and r.get("spec"):
                cands.append(dict(arm=arm, run=run, i=r["i"], spec=r["spec"], dev_d_full=r["d_full"], dev_p=r["p"], alpha_t=r["alpha_t"]))
dec = [float(r["delta_auc"]) for r in csv.DictReader(open("results/phase3/phase3_decoys.csv"))]
print(f"candidates {len(cands)} (expected 63), stacks {len(stacks)}, decoy deltas {len(dec)} (expected 29)")
assert len(cands) == 63 and len(dec) == 29, "candidate/decoy counts differ from the audited Phase 3 numbers"
json.dump(dict(materiality=MAT, sealed_baseline=0.9742099952759418, stacks=stacks, candidates=cands, decoy_deltas=dec),
          open(sys.argv[1], "w"), separators=(",", ":"))
PY
{
  head -n "$HEADN" "$SRC3"
  printf "\n\nEMBEDDED_AUDIT = json.loads(r'''"; cat "$D3/audit_data.json"; printf "''')\n"
  printf "%s\n" "$SYNTH"
  cat <<'PY'
# ===================== PHASE 4b / TEST 3: AUDIT OF FDR-REJECTED CANDIDATES (post hoc, descriptive) =====================
# Each non-decoy candidate that cleared the materiality floor on DEV but was rejected by LORD++ is added to its run's FINAL
# Phase 3 stack and scored on the Phase 3 sealed window (2017-11-08), 3-seed matched deltas. NOT a fresh holdout: that day
# was already used for the Phase 3 verdicts. Compared with the 29 re-permuted decoys from Phase 3.
import hashlib as _hl
import os
AUD_SEEDS = (100, 101, 102)
AUD_MAT = EMBEDDED_AUDIT["materiality"]
AUD_BASE_FROZEN = EMBEDDED_AUDIT["sealed_baseline"]
AUD_TOL = 0.0002
AUD_MAX_MINS = 600.0


def _h(o):
    return _hl.sha256(json.dumps(o, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class AuditEval:
    def __init__(self, lab):
        self.lab = lab
        lab.reset()
        self.cols, self.stacks = {}, {}
        self.base = [float(f["auc"]) for f in self._fits([])]

    def _fits(self, extras):
        return [self.lab.full.fit(extras, s) for s in AUD_SEEDS]

    def col(self, spec):
        k = _h(spec)
        if k not in self.cols:
            self.cols[k] = self.lab.full.col(self.lab.full_cols(spec, False, 0))
        return self.cols[k]

    def stack(self, specs):
        h = _h(specs)
        if h not in self.stacks:
            self.stacks[h] = [float(f["auc"]) for f in self._fits([self.col(s) for s in specs])]
        return self.stacks[h]

    def cand(self, specs, c):
        base = self.stack(specs)
        fits = [float(f["auc"]) for f in self._fits([self.col(s) for s in specs] + [self.col(c)])]
        per = [a - b for a, b in zip(fits, base)]
        return dict(stack_auc=float(np.mean(base)), stack_cand_auc=float(np.mean(fits)), delta=float(np.mean(per)), per_seed=per)


def audit_main():
    global WINDOW_DAY, DAY_START_ROW
    t0 = time.time()
    if TEST:
        W = _synthetic_window()
        s1, s2 = {"op": "count", "keys": ["c0"]}, {"op": "nunique", "keys": ["c0", "c1"], "target": "c2"}
        D = dict(stacks={"A|1": [s1]}, candidates=[dict(arm="A", run=1, i=3, spec=s2, dev_d_full=.001, dev_p=.01, alpha_t=.001),
                                                   dict(arm="A", run=1, i=7, spec={"op": "count", "keys": ["c1"]}, dev_d_full=.001, dev_p=.02, alpha_t=.001)],
                 decoy_deltas=[-3e-4, -1e-4, 1e-4])
        work = Path(os.environ.get("AF_TEST_OUT", "/tmp/p4b_audit_test"))
    else:
        D = EMBEDDED_AUDIT
        WINDOW_DAY, DAY_START_ROW = "2017-11-08", 68_941_878
        W = get_window(Path("/tmp/af_p4b_cache"))
        work = Path("/kaggle/working/audit") if Path("/kaggle/working").exists() else Path("./audit")
    work.mkdir(parents=True, exist_ok=True)
    lab = Lab(W)
    ev = AuditEval(lab)
    measured = float(lab.full.fit([], 0)["auc"])
    print(f"SEALED baseline seed0 {measured:.7f} | frozen {AUD_BASE_FROZEN:.7f} | 3-seed mean {np.mean(ev.base):.7f}", flush=True)
    if not TEST and abs(measured - AUD_BASE_FROZEN) > AUD_TOL:
        raise RuntimeError("baseline guard failed: wrong window?")
    jl = work / "audit_candidates.jsonl"
    done = {(r["arm"], r["run"], r["i"]) for r in map(json.loads, open(jl))} if jl.exists() else set()
    rows = [json.loads(l) for l in open(jl)] if jl.exists() else []
    cands = sorted(D["candidates"], key=lambda c: (c["arm"], c["run"], c["i"]))
    for n, c in enumerate(cands, 1):
        if (c["arm"], c["run"], c["i"]) in done:
            continue
        if (time.time() - t0) / 60 > AUD_MAX_MINS:
            print("time guard reached: writing partial results", flush=True)
            break
        r = ev.cand(D["stacks"][f'{c["arm"]}|{c["run"]}'], c["spec"])
        row = dict(arm=c["arm"], run=c["run"], i=c["i"], spec=json.dumps(c["spec"], sort_keys=True), dev_d_full=c["dev_d_full"],
                   dev_p=c["dev_p"], alpha_t=c["alpha_t"], sealed_delta=r["delta"], per_seed=r["per_seed"],
                   positive=bool(r["delta"] > 0), material=bool(r["delta"] >= AUD_MAT))
        rows.append(row)
        with open(jl, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[{n}/{len(cands)}] {c['arm']}#{c['run']} i={c['i']} sealed_delta={r['delta']:+.6f} | {(time.time() - t0) / 60:.0f} min", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(work / "audit_candidates.csv", index=False)
    dec = np.array(D["decoy_deltas"])
    d = df.sealed_delta.values
    summ = dict(n_candidates=int(len(df)), n_stacks=len(ev.stacks), sealed_baseline_3seed=float(np.mean(ev.base)),
                share_positive=float((d > 0).mean()), share_material=float((d >= AUD_MAT).mean()),
                median_delta=float(np.median(d)), mean_delta=float(d.mean()),
                decoy_n=int(len(dec)), decoy_median=float(np.median(dec)), decoy_p95=float(np.quantile(dec, .95)),
                decoy_share_positive=float((dec > 0).mean()),
                share_above_decoy_p95=float((d > np.quantile(dec, .95)).mean()),
                prob_cand_gt_decoy=float(np.mean([(x > y) + 0.5 * (x == y) for x in d for y in dec])),
                by_arm={a: dict(n=int(len(g)), median=float(g.sealed_delta.median()), n_positive=int((g.sealed_delta > 0).sum()),
                                n_material=int((g.sealed_delta >= AUD_MAT).sum())) for a, g in df.groupby("arm")},
                rule="gate discarded real signal if >=50% of candidates have delta>0 AND median candidate delta > decoy p95; "
                     "else rejections are consistent with noise",
                rule_outcome=("discarded real signal" if (d > 0).mean() >= .5 and np.median(d) > np.quantile(dec, .95)
                              else "consistent with noise / inconclusive"))
    (work / "audit_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2), flush=True)
    print("AUDIT_DONE", flush=True)


if __name__ == "__main__":
    audit_main()
PY
} > "$D3/run.py"
python3 -m py_compile "$D3/run.py"; write_meta "$D3" af-p4b-3-audit
echo "built $D3 ($(wc -l < "$D3/run.py") lines)"

# ------------------------------------------------------------------------------------------------ notebook 4
D4=kaggle/p4b_4_census; mkdir -p "$D4"
echo "== notebook 4: grammar census =="
{
  head -n "$HEADN" "$SRC3"
  printf "%s\n" "$SYNTH"
  cat <<'PY'
# ===================== PHASE 4b / TEST 4: GRAMMAR CENSUS (DEV window, Phase 3 grammar; descriptive) =====================
# Exhaustive single-feature proxy gain for every non-ratio spec, a sampler-distributed sample of ratio specs, full-track gain
# for the top specs and for the Phase 3 "needle". Sampler probabilities come from Monte Carlo draws of sample_spec.
import itertools, math, os
from collections import Counter

NEEDLE = {"op": "lag", "keys": ["c0", "c1", "c3"], "k": 1, "dir": "next"}
N_RATIO, MC_N, FULL_TOP, CEN_MAX_MINS = 300, 300_000, 40, 600.0
if TEST:
    N_RATIO, MC_N, FULL_TOP = 4, 2000, 2


def canon(s):
    return json.dumps(s, sort_keys=True, separators=(",", ":"))


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def enum_nonratio():
    out, seen = [], set()
    for r in range(1, 5):
        for ks in itertools.combinations(KEYS, r):
            keys = list(ks)
            cand = [{"op": "count", "keys": keys}, {"op": "cumcount", "keys": keys}]
            cand += [{"op": "roll", "keys": keys, "w": w} for w in ROLL_W]
            cand += [{"op": "lag", "keys": keys, "k": k, "dir": d} for k in LAGS for d in ("prev", "next")]
            cand += [{"op": "nunique", "keys": keys, "target": t} for t in KEYS if t not in keys]
            for s in cand:
                ok, _, ns = normalize(s)
                if ok and canon(ns) not in seen:
                    seen.add(canon(ns))
                    out.append(ns)
    return out


def census_main():
    t0 = time.time()
    global WINDOW_DAY, DAY_START_ROW
    if TEST:
        W = _synthetic_window()
        work = Path(os.environ.get("AF_TEST_OUT", "/tmp/p4b_census_test"))
    else:
        W = get_window(Path("/tmp/af_p4b_cache"))                      # DEV window (default constants)
        work = Path("/kaggle/working/census") if Path("/kaggle/working").exists() else Path("./census")
    work.mkdir(parents=True, exist_ok=True)
    lab = Lab(W)
    lab.reset()
    lab.refit(0)
    base_prox, base_full = float(lab.ref["prox_auc"]), float(lab.ref["full_auc"])
    print(f"baseline proxy {base_prox:.5f} full {base_full:.5f}", flush=True)
    nonratio = enum_nonratio()
    if TEST:
        nonratio = nonratio[:24] + [NEEDLE]
    # sampler distribution
    rng, cnt, n_ratio, n_ok = np.random.RandomState(0), Counter(), 0, 0
    for _ in range(MC_N):
        s = sample_spec(rng)
        if s.get("op") == "ratio":
            n_ratio += 1
            continue
        ok, _, ns = normalize(s)
        if ok:
            cnt[canon(ns)] += 1
    p_spec = {k: v / MC_N for k, v in cnt.items()}
    # sampler-distributed ratio sample
    rr, ratio, seen, tries = np.random.RandomState(1), [], set(), 0
    while len(ratio) < N_RATIO and tries < 300000:
        tries += 1
        s = sample_spec(rr)
        if s.get("op") == "ratio":
            ok, _, ns = normalize(s)
            if ok and canon(ns) not in seen:
                seen.add(canon(ns))
                ratio.append(ns)
    jobs = [("nonratio", s) for s in nonratio] + [("ratio", s) for s in ratio]
    print(f"non-ratio specs {len(nonratio)} | ratio sample {len(ratio)} | sampler P(ratio) {n_ratio / MC_N:.3f}", flush=True)
    jl = work / "census_proxy.jsonl"
    rows = [json.loads(l) for l in open(jl)] if jl.exists() else []
    have = {r["canon"] for r in rows}
    for n, (grp, s) in enumerate(jobs, 1):
        if canon(s) in have:
            continue
        if (time.time() - t0) / 60 > CEN_MAX_MINS * 0.6:
            print("time guard (proxy stage): stopping early", flush=True)
            break
        ap = compute(s, lab.Wp)
        fin = float(np.isfinite(ap).mean())
        d = float(lab.prox.fit([lab.prox.col(ap)], 0)["auc"] - base_prox)
        row = dict(canon=canon(s), spec=s, op=s["op"], group=grp, d_proxy=d, finite_frac=fin, sampler_p=p_spec.get(canon(s), 0.0))
        rows.append(row)
        with open(jl, "a") as f:
            f.write(json.dumps(row) + "\n")
        if n % 50 == 0:
            print(f"[{n}/{len(jobs)}] {(time.time() - t0) / 60:.0f} min", flush=True)
    df = pd.DataFrame(rows).sort_values("d_proxy", ascending=False).reset_index(drop=True)
    theta = F["proxy_theta"]
    df["good"] = df.d_proxy >= theta
    nr, ra = df[df.group == "nonratio"], df[df.group == "ratio"]
    p_good_nr = float(nr[nr.good].sampler_p.sum())
    p_ratio = n_ratio / MC_N
    k_r, n_r = int(ra.good.sum()), int(len(ra))
    lo, hi = wilson(k_r, n_r)
    p_good = p_good_nr + p_ratio * (k_r / n_r if n_r else 0.0)
    needle_c = canon(normalize(NEEDLE)[2] or NEEDLE)
    nrow = df[df.canon == needle_c]
    # full-track gain for the top specs and the needle
    top = list(df.head(FULL_TOP).spec) + ([normalize(NEEDLE)[2]] if len(nrow) and needle_c not in set(df.head(FULL_TOP).canon) else [])
    full_rows = []
    for s in top:
        if (time.time() - t0) / 60 > CEN_MAX_MINS:
            break
        mf = lab.full.col(lab.full_cols(s, False, 0))
        full_rows.append(dict(spec=json.dumps(s, sort_keys=True), d_full_single=float(lab.full.fit([mf], 0)["auc"] - base_full),
                              d_proxy=float(df[df.canon == canon(s)].d_proxy.iloc[0])))
    ff = pd.DataFrame(full_rows).sort_values("d_full_single", ascending=False).reset_index(drop=True) if full_rows else pd.DataFrame()
    df.drop(columns=["spec"]).to_csv(work / "census_proxy.csv", index=False)
    ff.to_csv(work / "census_top_full.csv", index=False)
    summ = dict(n_nonratio=int(len(nr)), n_ratio_sampled=n_r, proxy_theta=theta,
                nonratio_good=int(nr.good.sum()), nonratio_good_share=float(nr.good.mean()) if len(nr) else None,
                ratio_good=k_r, ratio_good_share=(k_r / n_r if n_r else None), ratio_good_wilson95=[lo, hi],
                sampler_p_ratio=p_ratio, sampler_p_good_nonratio=p_good_nr, sampler_p_good_total=p_good,
                expected_good_in_48_random_proposals=48 * p_good,
                needle=dict(found_in_census=bool(len(nrow)),
                            d_proxy=float(nrow.d_proxy.iloc[0]) if len(nrow) else None,
                            rank_proxy_nonratio=int((nr.d_proxy > float(nrow.d_proxy.iloc[0])).sum() + 1) if len(nrow) else None,
                            sampler_p=float(nrow.sampler_p.iloc[0]) if len(nrow) else None,
                            mc_n=MC_N, d_full_single=(float(ff[ff.spec == json.dumps(normalize(NEEDLE)[2], sort_keys=True)].d_full_single.iloc[0])
                                                      if len(ff) and (ff.spec == json.dumps(normalize(NEEDLE)[2], sort_keys=True)).any() else None)),
                good_share_by_op={o: float(g.good.mean()) for o, g in df.groupby("op")}, minutes=round((time.time() - t0) / 60, 1))
    (work / "census_summary.json").write_text(json.dumps(summ, indent=2, default=float))
    print(json.dumps(summ, indent=2, default=float), flush=True)
    print("CENSUS_DONE", flush=True)


if __name__ == "__main__":
    census_main()
PY
} > "$D4/run.py"
python3 -m py_compile "$D4/run.py"; write_meta "$D4" af-p4b-4-census
echo "built $D4 ($(wc -l < "$D4/run.py") lines)"

# ------------------------------------------------------------------------------------------------ plan note
mkdir -p docs
if [ ! -f docs/PHASE4B_PLAN.md ]; then
cat > docs/PHASE4B_PLAN.md <<'MD'
# Phase 4b plan (freeze before launching; tag `phase4b-freeze`)

**Closure of Phase 4.** The GPU arms (14B, no-nudge) and the fresh-sealed-day headline H4-1 are NOT run. Reason: cost versus expected
information after Phase 3 (LLM prior added nothing under the fixed grammar). Existing Phase 4 results (random rungs 1-2, canary pilot,
1-2 partial 14B runs) are reported as exploratory pilots.

All four tests are CPU-only, descriptive, and use data already seen (DEV, or the Phase 3 sealed day, which is NOT a fresh holdout).

1. **Canary, naive gate** (rung 2; run ids 4-13, 30 decoys per run, proxy bypassed) and
2. **Canary, FDR gate** (same run ids = same decoys, paired). Pooled with the 3+3 pilot runs. Report per-gate acceptance rate with a
   Wilson 95% interval and a one-sided Fisher exact test (naive > FDR). Verdict rule: "naive passes noise that FDR rejects" if
   Fisher p < 0.05; otherwise "not distinguishable at this n". Also report the DEV gap "closed" by accepted noise.
3. **Audit of FDR-rejected candidates** (63 non-decoy candidates that cleared the materiality floor but were rejected by LORD++ in
   Phase 3), scored on the Phase 3 sealed window (3-seed matched deltas, added to the run's final stack), compared with the 29 decoys.
   Verdict rule: "gate discarded real signal" if >= 50% have delta > 0 AND the median delta exceeds the decoy 95th percentile; otherwise
   "rejections consistent with noise / inconclusive".
4. **Grammar census** (Phase 3 grammar, DEV window): proxy gain of every non-ratio spec and a sampler-distributed sample of ratio specs;
   sampler probability of hitting a useful spec; rank and prior probability of the Phase 3 needle. Descriptive only.

Not claimed: any novelty on online FDR; any LLM-vs-random statement beyond Phase 3.
MD
echo "wrote docs/PHASE4B_PLAN.md (commit and tag phase4b-freeze before pushing)"
fi

# ------------------------------------------------------------------------------------------------ smoke test
if [ "$SMOKE" = "1" ] && python3 -c "import lightgbm" 2>/dev/null; then
  echo "== local smoke tests (synthetic window; never reads train.csv) =="
  for d in kaggle/p4b_3_audit kaggle/p4b_4_census; do
    AF_TEST=1 timeout 900 python3 "$d/run.py" 2>&1 | tail -3
  done
else
  echo "(smoke tests skipped)"
fi
echo; echo "READY. Next: git add docs kaggle && git commit -m 'Phase 4b plan' && git tag phase4b-freeze ; then bash p4b_push.sh"
