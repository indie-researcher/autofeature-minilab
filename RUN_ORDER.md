# AutoFeature Mini-Lab Run Order & Execution Checklist

1. Run R-naive arm on CPU (5 runs):
   - python code/af_phase1_R_naive.py --jobs "R-naive:1-5"
2. Optional Tier 2 model scale experiments (3B then 14B on GPU):
   - python code/af_phase2_L-FDR-3B.py --jobs "L-FDR:1-5"
   - python code/af_phase2_L-FDR-14B.py --jobs "L-FDR:1-5"
3. Freeze results, verify summaries, tag run artifacts.
4. Run Phase 3 unsealed evaluation exactly once:
   - python code/af_phase3_unseal.py
5. Write final README report card and publish results.

## Stop Rule
Stop immediately and verify results if any arm exceeds expected decoy false positive rate or if validation diverges by >0.003 from baseline.
