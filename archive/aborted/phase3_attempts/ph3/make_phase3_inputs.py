#!/usr/bin/env python3
"""Build phase3_inputs.json from local Phase 1+2 results. No p12 needed."""
import json
from pathlib import Path

ARMS = ["R", "R-naive", "L-naive", "L-FDR", "L-FDR-3B", "L-FDR-14B"]
RUNS = [1, 2, 3, 4, 5]
REPO = Path(__file__).resolve().parents[2]
out = {"runs": {}}

for arm in ARMS:
    for run in RUNS:
        f = REPO / "results" / arm / f"final_{arm}_{run}.json"
        fj = json.load(open(f))
        recs = [json.loads(l) for l in open(REPO / "results" / arm / f"trials_{arm}_{run}.jsonl") if l.strip()]
        decoys = []
        for i, r in enumerate(recs):
            if not (r.get("is_decoy") or r.get("decoy")):
                continue
            gated = (r.get("d_full") is not None) or r.get("stage") in ("full", "gate")
            if not gated:
                continue
            spec = r.get("spec") or r.get("base_spec")
            if spec is None:
                continue
            decoys.append({"idx": i, "spec": spec,
                           "dev_accepted": bool(r.get("accepted", False))})
        out["runs"][f"{arm}_{run}"] = {
            "arm": arm, "run": run,
            "accepted": [a["spec"] if isinstance(a, dict) and "spec" in a else a
                         for a in fj.get("accepted", [])],
            "decoys": decoys,
            "dev_baseline": fj.get("dev_baseline_remeasured"),
            "dev_final_auc": fj.get("dev_final_auc"),
            "model": fj.get("model", ""),
            "minutes": fj.get("minutes"),
            "tokens": fj.get("tokens"),
            "n_proposals": fj.get("n_proposals"),
            "n_full_tests": fj.get("n_full_tests"),
            "n_dup_total": fj.get("n_dup_total"),
            "decoys_proposed": fj.get("decoys_proposed", 0),
            "decoys_screened": fj.get("decoys_screened", 0),
            "decoys_tested": fj.get("decoys_tested", 0),
            "accepted_decoys": fj.get("accepted_decoys", 0),
        }

out_path = REPO / "phase3_inputs.json"
json.dump(out, open(out_path, "w"), indent=1)
print("wrote", out_path)
