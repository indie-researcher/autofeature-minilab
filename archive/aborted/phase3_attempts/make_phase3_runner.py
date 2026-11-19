from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"

phase1 = (CODE / "af_phase1_R_naive.py").read_text()
phase3 = (CODE / "af_phase3_sealed.py").read_text()

# Remove Phase 1's executable entry point; its functions remain available.
phase1 = re.sub(
    r'\nif __name__ == "__main__":\s*\n\s*main\(\)\s*$',
    "\n",
    phase1,
)

m_inputs = re.search(
    r"EMBEDDED_INPUTS\s*=\s*json\.loads\(r'''(.*?)'''\)",
    phase3,
    re.S,
)
m_spec = re.search(
    r"EMBEDDED_SPEC_FREEZE\s*=\s*json\.loads\(r'''(.*?)'''\)",
    phase3,
    re.S,
)
if not m_inputs or not m_spec:
    raise SystemExit("could not extract embedded Phase 3 inputs/spec")

phase3_code = '''
# ===================== PHASE 3: SEALED EVALUATION =====================
# Frozen SEALED window parameters are reused exactly from af_phase0.py:
# SEALED day = 2017-11-08; DAY_START_ROW = 68_941_878; SKIP_MARGIN = 6_000_000.

PHASE3_SEEDS = (100, 101, 102)
SEALED_CEILING = EMBEDDED_SPEC_FREEZE["results"]["SEALED"]["ceiling"]
SEALED_BASELINE_FROZEN = EMBEDDED_SPEC_FREEZE["results"]["SEALED"]["baseline"]
PHASE3_BASE_TOL = 0.003

def phase3_write_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)

def phase3_evaluate_real_stack(lab, specs):
    extras = []
    for spec in specs:
        af = lab.fullcols(spec, False, 0)
        extras.append(lab.full.col(af))

    base_fits = [lab.full.fit([], seed) for seed in PHASE3_SEEDS]
    final_fits = [lab.full.fit(extras, seed) for seed in PHASE3_SEEDS]

    base = float(np.mean([fit["auc"] for fit in base_fits]))
    final = float(np.mean([fit["auc"] for fit in final_fits]))
    gap = 100.0 * (final - base) / (SEALED_CEILING - base)

    return {
        "sealed_baseline": base,
        "sealed_final_auc": final,
        "sealed_gap_closed_pct": gap,
        "base_fits": [{"seed": s, "auc": f["auc"], "iters": f["iters"]}
                      for s, f in zip(PHASE3_SEEDS, base_fits)],
        "final_fits": [{"seed": s, "auc": f["auc"], "iters": f["iters"]}
                       for s, f in zip(PHASE3_SEEDS, final_fits)],
        "baseline_pred": base_fits[0]["pred"],
    }

def phase3_evaluate_decoy(lab, decoy, run_index, decoy_index, baseline_pred, arm):
    dseed = 7_000_000 + 1000 * run_index + decoy_index
    af = lab.fullcols(decoy["spec"], True, dseed)
    fit = lab.full.fit([lab.full.col(af)], 100)
    delta = float(fit["auc"] - lab.full.ref["fullauc"])

    if arm in ("L-naive", "R-naive"):
        accepted = bool(delta > 0.0)
        p_value = None
        alpha_t = None
    else:
        blocks = (
            block_aucs(lab.full.yev, fit["pred"], F["blocks"])
            - block_aucs(lab.full.yev, baseline_pred, F["blocks"])
        )
        p_value = signflip_p(blocks)
        lord = Lord(F["fdralpha"])
        alpha_t = lord.alphat(1)
        accepted = bool(p_value < alpha_t and delta >= F["materiality"])

    return {
        "idx": decoy.get("idx"),
        "spec": decoy["spec"],
        "repermute_seed": dseed,
        "sealed_auc": float(fit["auc"]),
        "sealed_delta_auc": delta,
        "accepted": accepted,
        "p_value": p_value,
        "alpha_t": alpha_t,
    }

def phase3_main():
    work = Path("/kaggle/working/phase3") if Path("/kaggle/working").exists() else Path("./phase3_results")
    work.mkdir(parents=True, exist_ok=True)

    if TEST:
        rng = np.random.RandomState(0)
        n = 4000
        W = {
            "c0": rng.randint(0, 40, n).astype(np.int32),
            "c1": rng.randint(0, 30, n).astype(np.int32),
            "c2": rng.randint(0, 20, n).astype(np.int32),
            "c3": rng.randint(0, 15, n).astype(np.int32),
            "c4": rng.randint(0, 10, n).astype(np.int32),
            "t": (np.arange(n) * 15).astype(np.int32),
            "h": ((np.arange(n) * 15) // 3600).astype(np.int32),
            "y": (rng.rand(n) < 0.02).astype(np.uint8),
        }
        lab = Lab(W)
        lab.reset()
        lab.refit(0)
        result = phase3_evaluate_real_stack(lab, [sample_spec(np.random.RandomState(1))])
        print("SMOKE_TEST_OK", result["sealed_baseline"], result["sealed_final_auc"])
        return

    # Use the exact frozen SEALED window: 2017-11-08, start row 68,941,878.
    global WINDOW_DAY, DAY_START_ROW
    WINDOW_DAY = "2017-11-08"
    DAY_START_ROW = 68_941_878

    cache = Path("/tmp/af_phase3_cache")
    W = get_window(cache)
    lab = Lab(W)
    lab.reset()
    lab.refit(0)

    measured_baseline = float(lab.full.ref["fullauc"])
    print(f"SEALED measured baseline: {measured_baseline:.7f}")
    print(f"SEALED frozen baseline:  {SEALED_BASELINE_FROZEN:.7f}")

    if abs(measured_baseline - SEALED_BASELINE_FROZEN) > PHASE3_BASE_TOL:
        raise RuntimeError(
            "SEALED baseline mismatch: "
            f"measured={measured_baseline:.7f}, frozen={SEALED_BASELINE_FROZEN:.7f}, "
            f"tolerance={PHASE3_BASE_TOL}. Aborting before evaluating runs."
        )

    run_rows = []
    decoy_rows = []

    for run_index, run in enumerate(EMBEDDED_INPUTS["runs"]):
        arm = run["arm"]
        run_number = run["run"]
        tag = f"{arm}_{run_number}"
        print(f"===== {tag}: evaluating {len(run['accepted'])} accepted real features =====")

        lab.reset()
        lab.refit(0)

        specs = [
            item["spec"]
            for item in run["accepted"]
            if not item.get("is_decoy", False)
        ]
        result = phase3_evaluate_real_stack(lab, specs)

        decoys = []
        for decoy_index, decoy in enumerate(run.get("decoys", [])):
            decoy_result = phase3_evaluate_decoy(
                lab, decoy, run_index, decoy_index,
                result["baseline_pred"], arm,
            )
            decoys.append(decoy_result)
            decoy_rows.append({
                "arm": arm,
                "run": run_number,
                **{k: v for k, v in decoy_result.items() if k != "spec"},
                "spec": json.dumps(decoy_result["spec"], sort_keys=True),
            })

        row = {
            "arm": arm,
            "run": run_number,
            "n_accepted_real": len(specs),
            "n_decoys_tested": len(decoys),
            "n_decoys_accepted": sum(int(d["accepted"]) for d in decoys),
            "decoy_fpr": (
                sum(int(d["accepted"]) for d in decoys) / len(decoys)
                if decoys else None
            ),
            "sealed_baseline": result["sealed_baseline"],
            "sealed_final_auc": result["sealed_final_auc"],
            "sealed_gap_closed_pct": result["sealed_gap_closed_pct"],
            "dev_baseline": run.get("dev_baseline"),
            "dev_final_auc": run.get("dev_final_auc"),
            "model": run.get("model"),
            "minutes": run.get("minutes"),
        }
        run_rows.append(row)
        print(
            f"{tag}: baseline={row['sealed_baseline']:.5f} "
            f"final={row['sealed_final_auc']:.5f} "
            f"gap_closed={row['sealed_gap_closed_pct']:.1f}% "
            f"decoys={row['n_decoys_accepted']}/{row['n_decoys_tested']}"
        )

        phase3_write_json(work / f"phase3_run_{tag}.json", {
            "run": row,
            "accepted_real_specs": specs,
            "decoys": decoys,
            "base_fits": result["base_fits"],
            "final_fits": result["final_fits"],
        })

    pd.DataFrame(run_rows).to_csv(work / "phase3_runs.csv", index=False)
    pd.DataFrame(decoy_rows).to_csv(work / "phase3_decoys.csv", index=False)

    grouped = (
        pd.DataFrame(run_rows)
        .groupby("arm", sort=False)
        .agg(
            runs=("run", "count"),
            mean_gap_closed=("sealed_gap_closed_pct", "mean"),
            min_gap_closed=("sealed_gap_closed_pct", "min"),
            max_gap_closed=("sealed_gap_closed_pct", "max"),
            mean_final_auc=("sealed_final_auc", "mean"),
            decoys_tested=("n_decoys_tested", "sum"),
            decoys_accepted=("n_decoys_accepted", "sum"),
        )
        .reset_index()
    )
    grouped["decoy_fpr"] = grouped["decoys_accepted"] / grouped["decoys_tested"].replace(0, np.nan)
    grouped.to_csv(work / "phase3_arms.csv", index=False)
    try:
        grouped.round(4).to_markdown(work / "phase3_arms.md", index=False)
    except Exception:
        grouped.round(4).to_csv(work / "phase3_arms.md", index=False)

    phase3_write_json(work / "phase3_summary.json", {
        "status": "complete",
        "sealed_baseline_frozen": SEALED_BASELINE_FROZEN,
        "sealed_ceiling_frozen": SEALED_CEILING,
        "runs": run_rows,
        "arms": grouped.to_dict(orient="records"),
    })

    lock = work / "phase3_lock.json"
    phase3_write_json(lock, {
        "status": "complete",
        "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_runs": len(run_rows),
        "n_arms": grouped.shape[0],
        "sealed_baseline_frozen": SEALED_BASELINE_FROZEN,
        "sealed_ceiling_frozen": SEALED_CEILING,
    })
    print("PHASE3_COMPLETE")
    print(grouped.round(4).to_string(index=False))

if __name__ == "__main__":
    phase3_main()
'''

combined = (
    phase1
    + "\n\n"
    + 'EMBEDDED_INPUTS = json.loads(r\'\'\'\n'
    + m_inputs.group(1)
    + "''')\n\n"
    + 'EMBEDDED_SPEC_FREEZE = json.loads(r\'\'\'\n'
    + m_spec.group(1)
    + "''')\n\n"
    + phase3_code
)

out = CODE / "af_phase3_sealed_combined.py"
out.write_text(combined)
print("WROTE", out, out.stat().st_size, "bytes")
