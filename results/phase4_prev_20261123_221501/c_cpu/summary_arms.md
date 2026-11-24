                       runs  zero_acc  distinct_stacks  proxy_pass  dev_gap_mean  dev_gap_med  dev_gap_min  dev_gap_max  real_acc  new_axis_acc  needle_hits  repeat_frac  mins_mean
rung arm     budget_n                                                                                                                                                               
2    C-FDR   30           3         3                1         1.0          0.00         0.00        0.000        0.000       0.0           0.0            0          0.0     43.630
     C-naive 30           3         3                1         1.0          2.32         2.12        1.958        2.881       0.0           0.0            0          0.0     47.517

CANARY AUDIT (decoys tested while the gate could still accept; canary arms shown separately)
  gate=fdr   canary_arm=True  tested=  90 accepted=  0 FPR=0.000 upper95=0.033
  gate=naive canary_arm=True  tested=  90 accepted=  3 FPR=0.033 upper95=0.084