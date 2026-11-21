#!/usr/bin/env python3
"""
AutoFeature Mini-Lab - PHASE 4: Scale the Search Space, Not the Freedom (Kaggle-ready).

Rungs (--rung 0|1|2):
    0  Frozen Phase 3 grammar (6 ops, 6 keys up to order-4, 3 roll windows, 3 lags)
    1  Key-space expansion (~12x atomic): adds derived time-bucket keys (m15, m5, m1, s10) up to order-5; same 6 ops & grids
    2  Parameter grids + 2 extra ops (~70x atomic): 5 windows, 5 lags, 5 quantile levels + `quantile` and `burst` ops

Arms:
    R-naive / R-FDR (alias R)           Random grammar sampler + naive or LORD++ online-FDR gate
    L-naive / L-FDR                     Qwen2.5-7B + naive or LORD++ gate
    L-naive-14B / L-FDR-14B             Qwen2.5-14B + naive or LORD++ gate
    L-naive-3B / L-FDR-3B               Qwen2.5-3B + naive or LORD++ gate

Usage:
    python af_phase4.py --dry-run --rung 1
    python af_phase4.py --rung 1 --jobs "R-naive:1-5"
    python af_phase4.py --rung 1 --jobs "L-naive:1-5"
    python af_phase4.py --rung 2 --jobs "L-naive-14B:1-5"
    python af_phase4.py --rung 2 --jobs "L-FDR-14B:1-5"
    python af_phase4.py --rung 2 --report
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
TEST = os.environ.get("AF_TEST") == "1"
SEED = 42

# =============================================================================
# 1. FROZEN PHASE 0 CONSTANTS
# =============================================================================
F = dict(
    hours_train=5, hours_eval=2, es_frac=0.12,
    neg_train=0.1, neg_eval=0.3, rounds=500, early=40, spw_full=10.0,
    proxy_mod=10, proxy_neg_train=0.3,
    materiality=0.00021865718476891195,
    dev_baseline=0.970396967505926, dev_ceiling=0.9847143092101822,
    proxy_theta=0.0005,
    blocks=12, fdr_alpha=0.10, decoy_frac=0.20,
)
if TEST:
    F.update(neg_train=0.5, neg_eval=1.0, rounds=25, early=5, proxy_mod=3, proxy_neg_train=0.8,
             materiality=0.002, proxy_theta=0.0)

WINDOW_DAY = "2017-11-09"
DAY_START_ROW = 131_886_953
SKIP_MARGIN = 6_000_000
PARAMS = dict(
    objective="binary", metric="auc", learning_rate=0.1, num_leaves=31, max_depth=-1,
    min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
    reg_lambda=5.0, max_bin=63, verbose=-1,
)

ARM_SPECS: Dict[str, Tuple[str, Optional[str]]] = {
    "R": ("fdr", None),
    "R-FDR": ("fdr", None),
    "R-naive": ("naive", None),
    "L-naive": ("naive", "qwen2.5:7b"),
    "L-FDR": ("fdr", "qwen2.5:7b"),
    "L-naive-14B": ("naive", "qwen2.5:14b"),
    "L-FDR-14B": ("fdr", "qwen2.5:14b"),
    "L-naive-3B": ("naive", "qwen2.5:3b"),
    "L-FDR-3B": ("fdr", "qwen2.5:3b"),
}
ARMS = tuple(ARM_SPECS.keys())
MAX_ATTEMPTS = 6
FAIL_LIMIT = 30
TEMP_CAP = 1.2

# ---- EDIT PER KAGGLE NOTEBOOK ---------------------------------------------------------------------------------------
DEFAULT_RUNG = 2
DEFAULT_JOBS = "L-naive-14B:1-5"      # e.g. "R-naive:1-5", "L-naive:1-5", "L-naive-14B:1-5", "L-FDR-14B:1-5"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def auc(y, s) -> float:
    return float(roc_auc_score(y, s))


# =============================================================================
# 2. DATA LOADING + DERIVED TEMPORAL KEYS (W_DEV.npz format preserved)
# =============================================================================
def find_file(name: str) -> Path:
    roots = [Path(os.environ.get("TD_INPUT", "/kaggle/input")), Path("."), Path("data")]
    for root in roots:
        if root.exists():
            hits = sorted(root.rglob(name), key=lambda h: ("talkingdata" not in str(h).lower(), len(h.parts)))
            if hits:
                return hits[0]
    raise FileNotFoundError(f"{name} not found (attach TalkingData competition data or set TD_INPUT)")


def to_seconds(s: pd.Series) -> np.ndarray:
    return pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S").values.astype("datetime64[s]").astype(np.int64)


def load_dev_window() -> pd.DataFrame:
    f = find_file("train.csv")
    lo, hi = f"{WINDOW_DAY} 08:00:00", f"{WINDOW_DAY} 16:00:00"
    skip = 0 if TEST else max(0, DAY_START_ROW - SKIP_MARGIN)
    log(f"DEV window: scanning {f} from row {skip:,}")
    dt = dict(ip="uint32", app="uint16", device="uint16", os="uint16", channel="uint16", is_attributed="uint8")
    rd = pd.read_csv(
        f, skiprows=range(1, skip + 1) if skip else None, dtype=dt,
        chunksize=200_000 if TEST else 4_000_000,
        usecols=["ip", "app", "device", "os", "channel", "click_time", "is_attributed"],
    )
    parts, first = [], True
    for ch in rd:
        ct = ch["click_time"]
        if first:
            first = False
            if skip and ct.iloc[0] >= lo:
                raise RuntimeError(f"scan started at {ct.iloc[0]} >= {lo}; raise SKIP_MARGIN")
        m = (ct >= lo) & (ct < hi)
        if m.any():
            parts.append(ch[m])
        if ct.iloc[-1] >= hi:
            break
    df = pd.concat(parts, ignore_index=True)
    t0 = int(to_seconds(pd.Series([lo]))[0])
    df["t"] = (to_seconds(df.pop("click_time")) - t0).astype(np.int32)
    if not np.all(df["t"].to_numpy()[1:] >= df["t"].to_numpy()[:-1]):
        df = df.iloc[np.argsort(df["t"].to_numpy(), kind="stable")].reset_index(drop=True)
    log(f"DEV window: {len(df):,} rows, positives {df['is_attributed'].mean():.4%}")
    return df


def window_arrays(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    W: Dict[str, np.ndarray] = {}
    for i, c in enumerate(["ip", "app", "device", "os", "channel"]):
        rs = np.random.RandomState(SEED + 100 + i)
        mult = np.uint64(int(rs.randint(1, 2 ** 31)) * 2 + 1)
        salt = np.uint64(int(rs.randint(0, 2 ** 31)))
        scr = (df[c].to_numpy().astype(np.uint64) * mult + salt) & np.uint64(0xFFFFFFFF)
        W[f"c{i}"] = np.unique(scr, return_inverse=True)[1].astype(np.int32).reshape(-1)
    W["t"] = df["t"].to_numpy().astype(np.int32)
    W["h"] = (W["t"] // 3600).astype(np.int32)
    W["y"] = df["is_attributed"].to_numpy().astype(np.uint8)
    return W


def enrich_temporal_keys(W: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Deterministically derive sub-hour bucket keys from t (zero leakage, compatible with cached W_DEV.npz)."""
    t = W["t"]
    W["m15"] = (t // 900).astype(np.int32)
    W["m5"] = (t // 300).astype(np.int32)
    W["m1"] = (t // 60).astype(np.int32)
    W["s10"] = (t // 10).astype(np.int32)
    return W


def get_window(cache_dir: Path) -> Dict[str, np.ndarray]:
    p = cache_dir / "W_DEV.npz"
    if p.exists():
        log(f"DEV window: cache {p}")
        z = np.load(p)
        return enrich_temporal_keys({k: z[k] for k in z.files})
    W = window_arrays(load_dev_window())
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache_dir / "W_DEV.tmp.npz"
    np.savez(tmp, **W)
    os.replace(tmp, p)
    return enrich_temporal_keys(W)


# =============================================================================
# 3. STEPPED GRAMMAR RUNGS (Rung 0 -> Rung 1 -> Rung 2)
# =============================================================================
RUNG_CFG: Dict[int, Dict[str, Any]] = {
    0: dict(
        keys=("c0", "c1", "c2", "c3", "c4", "h"),
        max_keys=4,
        lags=(1, 2, 3),
        roll_w=(60, 600, 3600),
        quantiles=(),
        basic_ops=("count", "nunique", "cumcount", "roll"),
        all_ops=("count", "nunique", "cumcount", "lag", "roll", "ratio"),
    ),
    1: dict(
        # Axis 1: Key-space expansion ONLY (~12x atomic specs). Ops & parameter grids identical to Rung 0.
        keys=("c0", "c1", "c2", "c3", "c4", "h", "m15", "m5", "m1", "s10"),
        max_keys=5,
        lags=(1, 2, 3),
        roll_w=(60, 600, 3600),
        quantiles=(),
        basic_ops=("count", "nunique", "cumcount", "roll"),
        all_ops=("count", "nunique", "cumcount", "lag", "roll", "ratio"),
    ),
    2: dict(
        # Axis 2 & 3: Dense parameter grids (5 windows, 5 lags, 5 quantiles) + 2 extra ops (`quantile`, `burst`).
        keys=("c0", "c1", "c2", "c3", "c4", "h", "m15", "m5", "m1", "s10"),
        max_keys=5,
        lags=(1, 2, 3, 5, 10),
        roll_w=(10, 60, 300, 600, 3600),
        quantiles=(0.1, 0.25, 0.5, 0.75, 0.9),
        basic_ops=("count", "nunique", "cumcount", "roll", "quantile", "burst"),
        all_ops=("count", "nunique", "cumcount", "lag", "roll", "quantile", "burst", "ratio"),
    ),
}


def grammar_stats(rung: int, budget: int = 60) -> Dict[str, Any]:
    cfg = RUNG_CFG[rung]
    n_k = len(cfg["keys"])
    max_k = cfg["max_keys"]
    key_tuples = sum(math.comb(n_k, r) for r in range(1, max_k + 1))
    nunique_cnt = sum(math.comb(n_k, r) * (n_k - r) for r in range(1, max_k + 1))
    lag_cnt = key_tuples * len(cfg["lags"]) * 2
    roll_cnt = key_tuples * len(cfg["roll_w"])
    quant_cnt = nunique_cnt * len(cfg["quantiles"]) if "quantile" in cfg["all_ops"] else 0
    burst_pairs = sum(1 for a in cfg["roll_w"] for b in cfg["roll_w"] if a < b)
    burst_cnt = key_tuples * burst_pairs if "burst" in cfg["all_ops"] else 0
    leaf_cnt = key_tuples + key_tuples + nunique_cnt + roll_cnt + quant_cnt + burst_cnt
    atomic_cnt = leaf_cnt + lag_cnt
    ratio_cnt = leaf_cnt * (leaf_cnt - 1)
    p1_atomic = 1.0 - (1.0 - 1.0 / atomic_cnt) ** budget
    p5_atomic = 1.0 - (1.0 - p1_atomic) ** 5
    return dict(
        rung=rung, n_keys=n_k, max_keys=max_k, key_tuples=key_tuples,
        atomic_specs=atomic_cnt, ratio_specs=ratio_cnt, total_specs=atomic_cnt + ratio_cnt,
        needle_hit_1run_atomic_pct=round(100.0 * p1_atomic, 2),
        needle_hit_5runs_atomic_pct=round(100.0 * p5_atomic, 2),
    )


def build_grammar_prompt(rung: int) -> str:
    cfg = RUNG_CFG[rung]
    keys_str = ",".join(cfg["keys"])
    lags_str = "|".join(str(x) for x in cfg["lags"])
    roll_str = "|".join(str(x) for x in cfg["roll_w"])
    leaf_names = "|".join(cfg["basic_ops"])
    key_desc = "c0..c4 (anonymous categorical IDs) and h (hour index 0..7)"
    if rung >= 1:
        key_desc += ", plus sub-hour time buckets m15 (15m), m5 (5m), m1 (1m), s10 (10s)"
    lines = [
        f"Propose ONE feature as a JSON object. Columns: {key_desc}.",
        "Click rows are time-ordered events; the target is whether the click converted. Allowed specs:",
        ' {"op":"count","keys":[...]}                      rows sharing the same key values (whole window)',
        ' {"op":"nunique","keys":[...],"target":"cX"}      distinct target values within the key group',
        ' {"op":"cumcount","keys":[...]}                   running index of the row inside its key group',
        f' {{"op":"lag","keys":[...],"k":{lags_str},"dir":"next"|"prev"}}   seconds to the k-th next/previous row of the group',
        f' {{"op":"roll","keys":[...],"w":{roll_str}}}       rows of the group in the previous w seconds',
    ]
    if rung >= 2:
        q_str = "|".join(str(x) for x in cfg["quantiles"])
        lines.append(f' {{"op":"quantile","keys":[...],"target":"cX","q":{q_str}}}   q-quantile of target inside the key group')
        lines.append(f' {{"op":"burst","keys":[...],"w_short":{roll_str},"w_long":{roll_str}}}   short/long window count velocity ratio (w_short < w_long)')
    lines.append(f' {{"op":"ratio","num":<{leaf_names} spec>,"den":<same>}}   num / (den + 1)')
    lines.append(f"keys: 1-{cfg['max_keys']} distinct names from [{keys_str}]. Return JSON only.")
    return "\n".join(lines)


def build_spec_schema(rung: int) -> dict:
    cfg = RUNG_CFG[rung]
    keys = list(cfg["keys"])
    max_k = cfg["max_keys"]
    leaf_props: Dict[str, Any] = {
        "op": {"type": "string", "enum": list(cfg["basic_ops"])},
        "keys": {"type": "array", "items": {"type": "string", "enum": keys}, "minItems": 1, "maxItems": max_k},
        "target": {"type": "string", "enum": keys},
        "w": {"type": "integer", "enum": list(cfg["roll_w"])},
    }
    if rung >= 2:
        leaf_props["q"] = {"type": "number", "enum": list(cfg["quantiles"])}
        leaf_props["w_short"] = {"type": "integer", "enum": list(cfg["roll_w"])}
        leaf_props["w_long"] = {"type": "integer", "enum": list(cfg["roll_w"])}
    leaf = {"type": "object", "properties": leaf_props, "required": ["op", "keys"]}
    top_props = {
        **leaf_props,
        "op": {"type": "string", "enum": list(cfg["all_ops"])},
        "k": {"type": "integer", "enum": list(cfg["lags"])},
        "dir": {"type": "string", "enum": ["next", "prev"]},
        "num": leaf,
        "den": leaf,
    }
    return {"type": "object", "properties": top_props, "required": ["op"]}


def spec_id(spec: dict) -> str:
    return hashlib.md5(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:10]


def is_needle_spec(spec: dict) -> bool:
    """Checks for the known TalkingData needle `next_c0c1c3_anyk` (lag next on ip+app+os)."""
    return (
        spec.get("op") == "lag"
        and spec.get("dir") == "next"
        and spec.get("keys") == ["c0", "c1", "c3"]
    )


def normalize(spec: Any, rung: int, nested: bool = False) -> Tuple[bool, str, Optional[dict]]:
    """Strict validation against the active rung; returns a canonical spec."""
    cfg = RUNG_CFG[rung]
    try:
        if not isinstance(spec, dict):
            return False, "not an object", None
        op = spec.get("op")
        if op not in cfg["all_ops"] or (nested and op not in cfg["basic_ops"]):
            return False, f"bad op {op!r}", None
        if op == "ratio":
            if set(spec) != {"op", "num", "den"}:
                return False, "ratio needs exactly op/num/den", None
            ok1, m1, n = normalize(spec["num"], rung, True)
            ok2, m2, d = normalize(spec["den"], rung, True)
            if not (ok1 and ok2):
                return False, m1 or m2, None
            if n == d:
                return False, "trivial ratio", None
            return True, "", {"op": "ratio", "num": n, "den": d}
        keys = spec.get("keys")
        if (
            not isinstance(keys, list)
            or not 1 <= len(keys) <= cfg["max_keys"]
            or len(set(keys)) != len(keys)
            or any(k not in cfg["keys"] for k in keys)
        ):
            return False, "bad keys", None
        out: Dict[str, Any] = {"op": op, "keys": sorted(keys)}
        allowed = {"op", "keys"}
        if op == "nunique":
            if spec.get("target") not in cfg["keys"] or spec["target"] in keys:
                return False, "bad target", None
            out["target"] = spec["target"]
            allowed.add("target")
        elif op == "lag":
            if (
                spec.get("k") not in cfg["lags"]
                or isinstance(spec.get("k"), bool)
                or spec.get("dir") not in ("next", "prev")
            ):
                return False, "bad lag", None
            out["k"], out["dir"] = int(spec["k"]), spec["dir"]
            allowed |= {"k", "dir"}
        elif op == "roll":
            if spec.get("w") not in cfg["roll_w"] or isinstance(spec.get("w"), bool):
                return False, "bad window", None
            out["w"] = int(spec["w"])
            allowed.add("w")
        elif op == "quantile":
            q_raw = spec.get("q")
            if spec.get("target") not in cfg["keys"] or spec["target"] in keys:
                return False, "bad quantile target", None
            if isinstance(q_raw, bool) or not isinstance(q_raw, (int, float)):
                return False, "bad quantile q", None
            matched_q = [q for q in cfg["quantiles"] if abs(float(q_raw) - q) < 1e-6]
            if not matched_q:
                return False, "bad quantile q", None
            out["target"] = spec["target"]
            out["q"] = float(matched_q[0])
            allowed |= {"target", "q"}
        elif op == "burst":
            ws, wl = spec.get("w_short"), spec.get("w_long")
            if (
                ws not in cfg["roll_w"] or isinstance(ws, bool)
                or wl not in cfg["roll_w"] or isinstance(wl, bool)
                or int(ws) >= int(wl)
            ):
                return False, "bad burst windows (need w_short < w_long)", None
            out["w_short"], out["w_long"] = int(ws), int(wl)
            allowed |= {"w_short", "w_long"}
        if set(spec) - allowed:
            return False, f"unexpected fields {sorted(set(spec) - allowed)}", None
        return True, "", out
    except Exception as e:
        return False, repr(e), None


# =============================================================================
# VECTORIZED FEATURE OPERATORS (25.7M rows safe)
# =============================================================================
def _codes(cols):
    key = cols[0].astype(np.int64)
    for v in cols[1:]:
        key = key * (int(v.max()) + 1) + v.astype(np.int64)
    return pd.factorize(key)[0].astype(np.int64)


def _cumcount(c):
    order = np.argsort(c, kind="stable")
    cs = c[order]
    start = np.r_[True, cs[1:] != cs[:-1]]
    idx = np.arange(len(c))
    first = np.maximum.accumulate(np.where(start, idx, 0))
    out = np.empty(len(c), np.int32)
    out[order] = idx - first
    return out


def _deltas(c, t, steps=(1,)):
    n = len(c)
    order = np.lexsort((t, c))
    cs, ts = c[order], t[order]
    out = {}
    for k in steps:
        nxt = np.full(n, np.nan, np.float32)
        prv = np.full(n, np.nan, np.float32)
        same = cs[k:] == cs[:-k]
        d = (ts[k:] - ts[:-k]).astype(np.float32, copy=False)
        d[~same] = np.nan
        nxt[: n - k] = d
        prv[k:] = d
        a, b = np.empty(n, np.float32), np.empty(n, np.float32)
        a[order], b[order] = nxt, prv
        out[f"next{k}"], out[f"prev{k}"] = a, b
    return out


def _nunique(a, b):
    bm = int(b.max()) + 1
    u = np.unique(a * bm + b)
    cnt = np.bincount(u // bm, minlength=int(a.max()) + 1)
    return cnt[a].astype(np.float32, copy=False)


def _roll(code, t, w):
    n = len(code)
    order = np.lexsort((t, code))
    cs, ts = code[order].astype(np.int64), t[order].astype(np.int64)
    key = cs * 65536 + ts
    lo = np.searchsorted(key, cs * 65536 + np.maximum(ts - w, 0), side="left")
    out = np.empty(n, np.float32)
    out[order] = (np.arange(n) - lo).astype(np.float32)
    return out


def _burst(code, t, w_short: int, w_long: int):
    n = len(code)
    order = np.lexsort((t, code))
    cs, ts = code[order].astype(np.int64), t[order].astype(np.int64)
    key = cs * 65536 + ts
    idx = np.arange(n)
    lo_s = np.searchsorted(key, cs * 65536 + np.maximum(ts - w_short, 0), side="left")
    lo_l = np.searchsorted(key, cs * 65536 + np.maximum(ts - w_long, 0), side="left")
    out = np.empty(n, np.float32)
    out[order] = ((idx - lo_s) / ((idx - lo_l) + 1.0)).astype(np.float32)
    return out


def _quantile(code: np.ndarray, target: np.ndarray, q: float) -> np.ndarray:
    """Exact linear-interpolated within-group quantile in O(N log N) time."""
    order = np.lexsort((target, code))
    cs = code[order]
    ts = target[order].astype(np.float32, copy=False)
    start = np.r_[True, cs[1:] != cs[:-1]]
    g_starts = np.flatnonzero(start)
    g_sizes = np.diff(np.r_[g_starts, len(code)])
    pos = q * (g_sizes - 1).astype(np.float64)
    lo_off = np.floor(pos).astype(np.int64)
    hi_off = np.ceil(pos).astype(np.int64)
    frac = (pos - lo_off).astype(np.float32)
    lo_val = ts[g_starts + lo_off]
    hi_val = ts[g_starts + hi_off]
    g_vals = lo_val + frac * (hi_val - lo_val)
    return g_vals[code].astype(np.float32, copy=False)


def _basic(spec, W, cache):
    k = tuple(spec["keys"])
    if k not in cache:
        cache[k] = _codes([W[x] for x in k])
    code, op = cache[k], spec["op"]
    if op == "count":
        return np.bincount(code)[code].astype(np.float32)
    if op == "cumcount":
        return _cumcount(code).astype(np.float32)
    if op == "nunique":
        return _nunique(code, W[spec["target"]].astype(np.int64))
    if op == "lag":
        return _deltas(code, W["t"], steps=(spec["k"],))[f"{spec['dir']}{spec['k']}"]
    if op == "roll":
        return _roll(code, W["t"], spec["w"])
    if op == "quantile":
        return _quantile(code, W[spec["target"]], spec["q"])
    if op == "burst":
        return _burst(code, W["t"], spec["w_short"], spec["w_long"])
    raise ValueError(f"unsupported op {op!r}")


def compute(spec: dict, W: Dict[str, np.ndarray]) -> np.ndarray:
    cache: dict = {}
    if spec["op"] == "ratio":
        return (_basic(spec["num"], W, cache) / (_basic(spec["den"], W, cache) + 1.0)).astype(np.float32)
    return _basic(spec, W, cache)


def make_decoy(arr: np.ndarray, seed: int) -> np.ndarray:
    return arr[np.random.RandomState(seed).permutation(len(arr))]


def sample_spec(rng: np.random.RandomState, rung: int) -> dict:
    cfg = RUNG_CFG[rung]
    keys_pool = list(cfg["keys"])
    max_k = cfg["max_keys"]
    len_probs = [0.25, 0.35, 0.30, 0.10] if max_k == 4 else [0.15, 0.30, 0.30, 0.15, 0.10]

    def keys():
        n = int(rng.choice(range(1, max_k + 1), p=len_probs))
        return sorted(str(k) for k in rng.choice(keys_pool, n, replace=False))

    def basic(op=None):
        op = op or str(rng.choice(cfg["basic_ops"]))
        s: Dict[str, Any] = dict(op=op, keys=keys())
        if op == "nunique":
            s["target"] = str(rng.choice([k for k in keys_pool if k not in s["keys"]]))
        elif op == "roll":
            s["w"] = int(rng.choice(cfg["roll_w"]))
        elif op == "quantile":
            s["target"] = str(rng.choice([k for k in keys_pool if k not in s["keys"]]))
            s["q"] = float(rng.choice(cfg["quantiles"]))
        elif op == "burst":
            ws, wl = sorted(int(x) for x in rng.choice(cfg["roll_w"], 2, replace=False))
            s["w_short"], s["w_long"] = ws, wl
        return s

    op_probs = (
        [0.20, 0.20, 0.10, 0.20, 0.15, 0.15]
        if rung < 2
        else [0.14, 0.14, 0.08, 0.16, 0.12, 0.12, 0.12, 0.12]
    )
    while True:
        op = str(rng.choice(cfg["all_ops"], p=op_probs))
        if op == "ratio":
            s = dict(op="ratio", num=basic(), den=basic())
        elif op == "lag":
            s = dict(op="lag", keys=keys(), k=int(rng.choice(cfg["lags"])), dir=str(rng.choice(["next", "prev"])))
        else:
            s = basic(op)
        ok, _, clean = normalize(s, rung)
        if ok:
            return clean


# =============================================================================
# 4. FOLD + MODEL
# =============================================================================
def _sub(y, lo, hi, frac, rng):
    yy = y[lo:hi]
    pos, neg = np.flatnonzero(yy == 1), np.flatnonzero(yy == 0)
    if frac < 1.0:
        neg = rng.choice(neg, int(round(frac * len(neg))), replace=False)
    return np.sort(np.r_[pos, neg]) + lo


class Track:
    def __init__(self, W, neg_train, neg_eval):
        y, t = W["y"], W["t"]
        a = int(np.searchsorted(t, 3600 * F["hours_train"], "left"))
        b = int(np.searchsorted(t, 3600 * (F["hours_train"] + F["hours_eval"]), "left"))
        es0 = a - int(F["es_frac"] * a)
        self.neg_train = neg_train
        self.idx = (
            _sub(y, 0, es0, neg_train, np.random.RandomState(SEED + 13)),
            _sub(y, es0, a, neg_eval, np.random.RandomState(SEED + 7)),
            _sub(y, a, b, neg_eval, np.random.RandomState(SEED + 8)),
        )
        self.y = tuple(y[i] for i in self.idx)
        self.base = tuple(
            np.column_stack([W[n][i] for n in ("c1", "c2", "c3", "c4")]).astype(np.float32) for i in self.idx
        )
        self.yev = self.y[2]

    def col(self, arr: np.ndarray):
        return tuple(arr[i].astype(np.float32)[:, None] for i in self.idx)

    def fit(self, extras: list, seed: int) -> dict:
        mats = [
            np.ascontiguousarray(np.column_stack([self.base[k]] + [e[k] for e in extras]), dtype=np.float32)
            for k in range(3)
        ]
        p = {**PARAMS, "scale_pos_weight": F["spw_full"] * self.neg_train, "seed": SEED + int(seed)}
        dp = dict(max_bin=p["max_bin"], verbose=-1, feature_pre_filter=False)
        d_tr = lgb.Dataset(mats[0], self.y[0], params=dp, free_raw_data=True).construct()
        d_es = lgb.Dataset(mats[1], self.y[1], reference=d_tr, params=dp, free_raw_data=True).construct()
        m = lgb.train(
            p, d_tr, num_boost_round=F["rounds"], valid_sets=[d_es], valid_names=["es"],
            callbacks=[lgb.early_stopping(F["early"], verbose=False)],
        )
        bi = int(m.best_iteration or F["rounds"])
        pe = m.predict(mats[2], num_iteration=bi).astype(np.float32)
        return dict(pred=pe, auc=auc(self.yev, pe), iters=bi)


def block_aucs(y, p, nb):
    e = np.linspace(0, len(y), nb + 1).astype(int)
    return np.array([auc(y[a:b], p[a:b]) if 0 < y[a:b].sum() < b - a else np.nan for a, b in zip(e[:-1], e[1:])])


_SIGNS: Dict[int, np.ndarray] = {}


def signflip_p(d: np.ndarray) -> float:
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 4:
        return 1.0
    if n not in _SIGNS:
        _SIGNS[n] = np.array(list(itertools.product((-1.0, 1.0), repeat=n)))
    return float(np.mean((_SIGNS[n] @ d) / n >= d.mean() - 1e-15))


# =============================================================================
# 5. ONLINE FDR (LORD++)
# =============================================================================
def _gamma(j: int) -> float:
    j = max(int(j), 1)
    return 0.07720838 * np.log(max(j, 2)) / (j * np.exp(np.sqrt(np.log(j))))


class Lord:
    def __init__(self, alpha: float):
        self.alpha, self.w0, self.rej = alpha, alpha / 2.0, []

    def alpha_t(self, t: int) -> float:
        a = _gamma(t) * self.w0
        if self.rej:
            a += (self.alpha - self.w0) * _gamma(t - self.rej[0])
            a += sum(self.alpha * _gamma(t - r) for r in self.rej[1:])
        return float(a)


# =============================================================================
# 6. PROPOSERS: Ollama client + quantized feedback prompt
# =============================================================================
class Ollama:
    def __init__(self, url: str, model: str, timeout: int = 300):
        self.url, self.model, self.timeout = url.rstrip("/"), model, timeout
        self.tokens = {"prompt": 0, "completion": 0}

    def _get(self, path):
        with urllib.request.urlopen(urllib.request.Request(self.url + path), timeout=5) as r:
            return json.loads(r.read().decode())

    def alive(self) -> bool:
        try:
            self._get("/api/tags")
            return True
        except Exception:
            return False

    def ensure_ready(self, start: bool, install: bool) -> None:
        if not self.alive():
            if not start:
                raise RuntimeError(f"Ollama unreachable at {self.url} (start it or pass --start-ollama)")
            if shutil.which("ollama") is None:
                if not install:
                    raise RuntimeError("ollama binary missing (pass --install-ollama)")
                log("installing Ollama ...")
                subprocess.run(
                    "apt-get update -qq && apt-get install -y -qq zstd curl && "
                    "curl -fsSL https://ollama.com/install.sh | sh",
                    shell=True, check=True,
                )
            env = os.environ.copy()
            Path("/tmp/ollama_models").mkdir(parents=True, exist_ok=True)
            env.setdefault("OLLAMA_MODELS", "/tmp/ollama_models")
            subprocess.Popen(
                ["ollama", "serve"], stdout=open("/tmp/ollama_serve.log", "a"),
                stderr=subprocess.STDOUT, env=env, start_new_session=True,
            )
            for _ in range(60):
                if self.alive():
                    break
                time.sleep(1)
            else:
                raise RuntimeError("ollama serve did not start in 60 s (see /tmp/ollama_serve.log)")
        have = [m.get("name", "") for m in self._get("/api/tags").get("models", [])]
        if not any(m == self.model or m.startswith(self.model + ":") for m in have):
            log(f"pulling {self.model} ...")
            subprocess.run(["ollama", "pull", self.model], check=True)
        try:
            gpu = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip().replace("\n", " | ")
            if gpu:
                log("GPU: " + gpu)
        except Exception:
            pass

    def chat(self, messages, temperature=0.6, schema=None, num_predict=200) -> str:
        payload: Dict[str, Any] = {
            "model": self.model, "messages": messages, "stream": False,
            "options": {"temperature": temperature, "num_predict": num_predict, "num_ctx": 4096},
        }
        if schema is not None:
            payload["format"] = schema

        def post(pl):
            req = urllib.request.Request(
                self.url + "/api/chat", data=json.dumps(pl).encode(),
                headers={"Content-Type": "application/json"}, method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                d = json.loads(r.read().decode())
                self.tokens["prompt"] += d.get("prompt_eval_count", 0)
                self.tokens["completion"] += d.get("eval_count", 0)
                return d.get("message", {}).get("content", "").strip()

        try:
            return post(payload)
        except urllib.error.HTTPError as e:
            if schema is not None and e.code in (400, 422):
                payload["format"] = "json"
                return post(payload)
            raise


def parse_json(text: str) -> Any:
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        a, b = s.find("{"), s.rfind("}")
        if a != -1 and b > a:
            return json.loads(s[a:b + 1])
        raise


def fmt_delta(rec: dict) -> str:
    if rec["status"] == "accepted":
        return f"ACCEPTED ({round(rec['d_full'], 3):+.3f})"
    if rec["status"] == "rejected":
        return f"rejected ({round(rec['d_full'], 3):+.3f})"
    return "screened out (no gain)"


def build_messages(i: int, n: int, history: List[dict], rung: int, grammar_prompt: str) -> List[dict]:
    """Same quantized feedback interface across all rungs (scales search space, not freedom)."""
    cfg = RUNG_CFG[rung]
    ops = {o: 0 for o in cfg["all_ops"]}
    for h in history:
        ops[h["spec"]["op"]] = ops.get(h["spec"]["op"], 0) + 1
    least = min(ops, key=lambda k: ops[k])
    acc = [h for h in history if h["status"] == "accepted"]
    lines = [
        f"Proposal {i} of {n}.",
        "Accepted features so far (do not repeat): "
        + (", ".join(json.dumps(h["spec"], sort_keys=True) for h in acc[-8:]) if acc else "none yet"),
    ]
    if history:
        lines.append("Recent results:")
        for h in history[-10:]:
            lines.append(f"  {json.dumps(h['spec'], sort_keys=True)} -> {fmt_delta(h)}")
    lines.append(f"Operator '{least}' is the least explored so far. Combine different key sets. Output ONE new JSON spec.")
    return [{"role": "system", "content": grammar_prompt}, {"role": "user", "content": "\n".join(lines)}]


# =============================================================================
# 7. THE LAB: one (rung, arm, run) job
# =============================================================================
class Lab:
    def __init__(self, W: Dict[str, np.ndarray]):
        t0 = time.time()
        self.W = W
        self.Wp = {k: v[(W["c0"] % F["proxy_mod"]) == 0] for k, v in W.items()}
        self.full = Track(W, F["neg_train"], F["neg_eval"])
        self.prox = Track(self.Wp, F["proxy_neg_train"], 1.0)
        log(
            f"tracks ready in {time.time() - t0:.0f}s | full train/ES/eval {[len(i) for i in self.full.idx]} "
            f"(eval pos {int(self.full.yev.sum()):,}) | proxy {[len(i) for i in self.prox.idx]} (eval pos {int(self.prox.yev.sum()):,})"
        )

    def reset(self):
        self.stack: List[dict] = []
        self.ref_seed = 0
        self.ref: Dict[str, Any] = {}

    def refit(self, seed: int):
        self.ref_seed = seed
        f = self.full.fit([s["full"] for s in self.stack], seed)
        p = self.prox.fit([s["prox"] for s in self.stack], seed)
        self.ref = dict(full_pred=f["pred"], full_auc=f["auc"], prox_auc=p["auc"])
        return f["auc"], p["auc"]

    def columns(self, spec: dict, is_decoy: bool, dseed: int):
        ap = compute(spec, self.Wp)
        fin = float(np.isfinite(ap).mean())
        if is_decoy:
            ap = make_decoy(ap, dseed)
        return ap, fin

    def full_cols(self, spec: dict, is_decoy: bool, dseed: int):
        af = compute(spec, self.W)
        return make_decoy(af, dseed) if is_decoy else af

    def add_to_stack(self, spec, is_decoy, dseed, mats_full, mats_prox):
        self.stack.append(dict(spec=spec, is_decoy=is_decoy, dseed=dseed, full=mats_full, prox=mats_prox))

    def evaluate(self, spec: dict, is_decoy: bool, dseed: int, gate: str, lord: Lord, n_tests: int) -> dict:
        t0 = time.time()
        ap, fin = self.columns(spec, is_decoy, dseed)
        mp = self.prox.col(ap)
        del ap
        rp = self.prox.fit([s["prox"] for s in self.stack] + [mp], self.ref_seed)
        d_proxy = rp["auc"] - self.ref["prox_auc"]
        rec = dict(finite_frac=round(fin, 4), d_proxy=round(d_proxy, 6))
        if d_proxy < F["proxy_theta"]:
            rec.update(status="screened", d_full=None, p=None, alpha_t=None, eval_secs=round(time.time() - t0, 2))
            return rec
        af = self.full_cols(spec, is_decoy, dseed)
        mf = self.full.col(af)
        del af
        rc = self.full.fit([s["full"] for s in self.stack] + [mf], self.ref_seed)
        d_full = rc["auc"] - self.ref["full_auc"]
        blk = block_aucs(self.full.yev, rc["pred"], F["blocks"]) - block_aucs(self.full.yev, self.ref["full_pred"], F["blocks"])
        p = signflip_p(blk)
        t_idx = n_tests + 1
        alpha_t = lord.alpha_t(t_idx)
        if gate == "naive":
            accept = rc["auc"] > self.ref["full_auc"]
        else:
            accept = bool(p <= alpha_t and d_full >= F["materiality"])
        rec.update(
            status="accepted" if accept else "rejected",
            d_full=round(d_full, 6), p=round(p, 6), alpha_t=round(alpha_t, 6),
            t_test=t_idx, cand_seed=self.ref_seed,
            blocks_pos=int(np.sum(blk[np.isfinite(blk)] > 0)),
            cand_auc=round(rc["auc"], 6),
        )
        if accept:
            self.add_to_stack(spec, is_decoy, dseed, mf, mp)
            if gate == "naive":
                self.ref.update(full_pred=rc["pred"], full_auc=rc["auc"], prox_auc=rp["auc"])
            else:
                self.refit(self.ref_seed + 1)
        rec["eval_secs"] = round(time.time() - t0, 2)
        return rec


def replay(lab: Lab, recs: List[dict], gate: str, lord: Lord) -> int:
    lab.reset()
    lab.refit(0)
    n_tests = 0
    for r in recs:
        if r.get("t_test"):
            n_tests = max(n_tests, r["t_test"])
        if r["status"] != "accepted":
            continue
        spec = r["spec"]
        mf = lab.full.col(lab.full_cols(spec, r["is_decoy"], r["dseed"]))
        ap, _ = lab.columns(spec, r["is_decoy"], r["dseed"])
        mp = lab.prox.col(ap)
        lab.add_to_stack(spec, r["is_decoy"], r["dseed"], mf, mp)
        if gate == "fdr":
            lord.rej.append(r["t_test"])
            lab.refit(lab.ref_seed + 1)
        else:
            lab.ref_seed = r["cand_seed"]
            f = lab.full.fit([s["full"] for s in lab.stack], r["cand_seed"])
            p = lab.prox.fit([s["prox"] for s in lab.stack], r["cand_seed"])
            lab.ref.update(full_pred=f["pred"], full_auc=f["auc"], prox_auc=p["auc"])
    return n_tests


def parse_jobs(s: str) -> List[Tuple[str, int]]:
    jobs = []
    for part in s.split(","):
        arm, _, runs = part.strip().rpartition(":")
        if arm not in ARMS:
            raise SystemExit(f"unknown arm {arm!r}; choose from {ARMS}")
        for r in runs.split("+"):
            if "-" in r:
                a, b = r.split("-")
                jobs += [(arm, k) for k in range(int(a), int(b) + 1)]
            else:
                jobs.append((arm, int(r)))
    return jobs


def run_job(
    lab: Lab, rung: int, arm: str, run: int, n: int, work: Path,
    llm: Optional[Ollama], mock: bool, deadline: float,
) -> bool:
    gate, _ = ARM_SPECS[arm]
    tag = f"r{rung}_{arm}_{run}"
    path = work / f"trials_{tag}.jsonl"
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []
    valid = [r for r in recs if r.get("status") in ("screened", "accepted", "rejected")]
    done_flag = work / f"final_{tag}.json"
    if done_flag.exists():
        log(f"{tag}: already finished")
        return True

    grammar_prompt = build_grammar_prompt(rung)
    spec_schema = build_spec_schema(rung)
    lord = Lord(F["fdr_alpha"])
    n_tests = replay(lab, valid, gate, lord)
    log(f"=== {tag} | rung={rung} | gate={gate} | resumed {len(valid)}/{n} proposals, stack {len(lab.stack)} ===")

    rng_prop = np.random.RandomState(7919 * run + ARMS.index(arm) + 101 * rung)
    for _ in range(len(valid)):
        sample_spec(rng_prop, rung)
    decoy_mask = np.random.RandomState(104729 * run + 13).rand(n) < F["decoy_frac"]
    seen = {(r["id"], r["is_decoy"]) for r in valid}
    fail_streak = 0
    t_job = time.time()

    while len(valid) < n:
        if time.time() > deadline:
            log(f"{tag}: wall-clock guard reached; progress saved ({len(valid)}/{n}). Re-run to resume.")
            return False
        i = len(valid) + 1
        is_decoy = bool(decoy_mask[i - 1])
        n_dup = 0
        n_attempts = 0
        spec, why = None, ""
        t_prop0 = time.time()

        if arm.startswith("R") or mock:
            for _ in range(50):
                n_attempts += 1
                cand = sample_spec(rng_prop, rung)
                if (spec_id(cand), is_decoy) not in seen:
                    spec = cand
                    break
                n_dup += 1
        else:
            msgs = build_messages(i, n, valid, rung, grammar_prompt)
            for attempt in range(MAX_ATTEMPTS):
                n_attempts += 1
                try:
                    raw = llm.chat(msgs, temperature=min(TEMP_CAP, 0.6 + 0.15 * attempt), schema=spec_schema)
                    ok, why, clean = normalize(parse_json(raw), rung)
                    hint = None
                    if ok and (spec_id(clean), is_decoy) in seen:
                        ok, why, hint = False, "duplicate", clean
                        n_dup += 1
                    if ok:
                        spec = clean
                        break
                    extra = ""
                    if hint:
                        tried = [h["spec"] for h in valid if h["spec"]["op"] == hint["op"]][-15:]
                        extra = (
                            " " + json.dumps(hint, sort_keys=True)
                            + " was already tried. Choose a different key set or a different operator. Already tried with this operator: "
                            + "; ".join(json.dumps(t, sort_keys=True) for t in tried) + "."
                        )
                    msgs = msgs + [
                        {"role": "assistant", "content": raw},
                        {"role": "user", "content": f"Rejected ({why}).{extra} Output ONE valid, NEW JSON spec."},
                    ]
                except Exception as e:
                    why = repr(e)[:120]

        prop_secs = round(time.time() - t_prop0, 2)
        if spec is None:
            fail_streak += 1
            with path.open("a") as fh:
                fh.write(json.dumps(dict(status="proposal_failed", rung=rung, i=i, reason=why, n_dup=n_dup, n_attempts=n_attempts)) + "\n")
            log(f"{tag} #{i}: no valid proposal ({why})")
            if fail_streak >= FAIL_LIMIT:
                log(f"{tag}: {FAIL_LIMIT} consecutive proposal failures; stopping this job")
                return False
            continue

        fail_streak = 0
        sid = spec_id(spec)
        dseed = 1000003 * run + i
        rec = lab.evaluate(spec, is_decoy, dseed, gate, lord, n_tests)
        if rec.get("t_test"):
            n_tests = rec["t_test"]
            if rec["status"] == "accepted" and gate == "fdr":
                lord.rej.append(rec["t_test"])
        total_secs = round(prop_secs + rec["eval_secs"], 2)
        rec.update(
            rung=rung, arm=arm, run=run, i=i, id=sid, spec=spec,
            is_decoy=is_decoy, dseed=dseed, stack_size=len(lab.stack),
            n_dup=n_dup, n_attempts=n_attempts,
            repeat_frac=round(n_dup / max(n_attempts, 1), 4),
            prop_secs=prop_secs, secs=total_secs,
            is_needle=(not is_decoy and is_needle_spec(spec)),
        )
        with path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        valid.append(rec)
        seen.add((sid, is_decoy))
        log(
            f"{tag} #{i}/{n} {'DECOY ' if is_decoy else ''}{rec['status'].upper():9s} d_proxy={rec['d_proxy']:+.5f} "
            f"d_full={rec['d_full'] if rec['d_full'] is not None else float('nan'):+.5f} dup={n_dup}/{n_attempts} "
            f"| stack {len(lab.stack)} | {total_secs}s | {json.dumps(spec, sort_keys=True)[:85]}"
        )
        gc.collect()

    # -- selection-free re-measurement of final DEV stack (3 fresh seeds) --------------------------------------------
    fresh = [lab.full.fit([s["full"] for s in lab.stack], 100 + k)["auc"] for k in range(3)]
    b0 = [lab.full.fit([], 100 + k)["auc"] for k in range(3)]
    base_m, final = float(np.mean(b0)), float(np.mean(fresh))
    tested = [r for r in valid if r["status"] in ("accepted", "rejected")]
    acc = [r for r in valid if r["status"] == "accepted"]
    dec_t = [r for r in tested if r["is_decoy"]]
    needle_hits = [r for r in valid if r.get("is_needle")]
    n_dup_total = sum(r.get("n_dup", 0) for r in valid)
    n_att_total = sum(r.get("n_attempts", 1) for r in valid)
    wall_secs = round(sum(r.get("secs", 0.0) for r in valid), 1)
    prop_secs_sum = round(sum(r.get("prop_secs", 0.0) for r in valid), 1)
    eval_secs_sum = round(sum(r.get("eval_secs", 0.0) for r in valid), 1)

    out = dict(
        rung=rung, arm=arm, run=run,
        n_proposals=len(valid),
        unique_specs=len({r["id"] for r in valid}),
        n_screened=sum(r["status"] == "screened" for r in valid),
        n_full_tests=len(tested),
        accepted_real=sum(not r["is_decoy"] for r in acc),
        accepted_decoys=sum(r["is_decoy"] for r in acc),
        decoys_proposed=sum(r["is_decoy"] for r in valid),
        decoys_screened=sum(bool(r["is_decoy"]) and r["status"] == "screened" for r in valid),
        decoys_tested=len(dec_t),
        decoy_fpr=(sum(r["status"] == "accepted" for r in dec_t) / len(dec_t)) if dec_t else None,
        n_dup_total=n_dup_total,
        n_attempts_total=n_att_total,
        repeat_frac=round(n_dup_total / max(n_att_total, 1), 4),
        needle_first_proposed_i=(needle_hits[0]["i"] if needle_hits else None),
        needle_accepted=any(r["status"] == "accepted" for r in needle_hits),
        dev_baseline_remeasured=base_m,
        dev_final_auc=final,
        dev_gap_closed_pct=100.0 * (final - base_m) / (F["dev_ceiling"] - base_m),
        model=(llm.model if llm else "random"),
        tokens=(dict(llm.tokens) if llm else None),
        prop_secs_total=prop_secs_sum,
        eval_secs_total=eval_secs_sum,
        wall_clock_secs=wall_secs,
        minutes=round(wall_secs / 60.0, 2),
        accepted=[dict(spec=r["spec"], is_decoy=r["is_decoy"], dseed=r["dseed"], iter=r["i"]) for r in acc],
    )
    done_flag.write_text(json.dumps(out, indent=2))
    log(
        f"{tag} FINAL: real={out['accepted_real']} decoys={out['accepted_decoys']}/{out['decoys_tested']} "
        f"| repeat_frac={out['repeat_frac']:.3f} | needle_i={out['needle_first_proposed_i']} "
        f"| DEV {base_m:.5f}->{final:.5f} ({out['dev_gap_closed_pct']:.1f}%) | {out['minutes']}m"
    )
    return True


# =============================================================================
# 8. REPORT + CLI
# =============================================================================
def report(work: Path) -> None:
    rows = [json.loads(p.read_text()) for p in sorted(work.glob("final_*.json"))]
    if not rows:
        print("no finished runs yet")
        return
    df = pd.DataFrame(rows)
    if "rung" not in df:
        df["rung"] = 0
    df.drop(columns=["accepted"]).to_csv(work / "summary_runs.csv", index=False)
    g = df.groupby(["rung", "arm"]).agg(
        runs=("run", "count"),
        gap_mean=("dev_gap_closed_pct", "mean"),
        gap_med=("dev_gap_closed_pct", "median"),
        gap_min=("dev_gap_closed_pct", "min"),
        gap_max=("dev_gap_closed_pct", "max"),
        real_acc=("accepted_real", "mean"),
        decoy_acc=("accepted_decoys", "mean"),
        decoys_tested=("decoys_tested", "sum"),
        repeat_frac=("repeat_frac", "mean"),
        needle_hits=("needle_accepted", "sum"),
        mins_mean=("minutes", "mean"),
    )
    g["decoy_fpr"] = df.groupby(["rung", "arm"]).apply(
        lambda d: d.accepted_decoys.sum() / max(d.decoys_tested.sum(), 1)
    )
    try:
        txt = g.round(3).to_markdown()
    except Exception:
        txt = g.round(3).to_string()
    (work / "summary_arms.md").write_text(txt)
    print(g.round(3).to_string())


def main(argv=None) -> None:
    kaggle = Path("/kaggle/working").exists()
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", type=int, default=DEFAULT_RUNG, choices=[0, 1, 2], help="grammar scale rung (0, 1, 2)")
    ap.add_argument("--jobs", default="", help='e.g. "R-naive:1-5" or "L-naive-14B:1-10"')
    ap.add_argument("--n", type=int, default=60, help="evaluated proposals per run (default: 60)")
    ap.add_argument("--work-dir", default="", help="default: /kaggle/working/af_lab_r{rung}")
    ap.add_argument("--cache-dir", default="/tmp/af_cache" if kaggle else "./af_cache")
    ap.add_argument("--model", default="", help="override Ollama model (default: inferred per arm)")
    ap.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    ap.add_argument("--start-ollama", action="store_true", default=kaggle)
    ap.add_argument("--install-ollama", action="store_true", default=kaggle)
    ap.add_argument("--mock-llm", action="store_true", help="LLM arms use random sampler (smoke tests only)")
    ap.add_argument("--max-mins", type=float, default=700.0)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a, _ = ap.parse_known_args(argv)

    default_work = f"/kaggle/working/af_lab_r{a.rung}" if kaggle else f"./af_lab_r{a.rung}"
    work = Path(a.work_dir or default_work)
    work.mkdir(parents=True, exist_ok=True)

    if a.report:
        return report(work)

    if a.dry_run:
        for r in (0, 1, 2):
            rng = np.random.RandomState(r)
            bad = normalize({"op": "count", "keys": ["zz"]}, r)[0]
            ok = all(normalize(sample_spec(rng, r), r)[0] for _ in range(300))
            st = grammar_stats(r, a.n)
            log(f"dry-run rung={r}: grammar_ok={ok} invalid_rejected={not bad} stats={st}")
        return

    jobs_str = a.jobs or os.environ.get("AF_JOBS") or DEFAULT_JOBS
    jobs = parse_jobs(jobs_str) if jobs_str else []
    if not jobs:
        raise SystemExit('nothing to do: set DEFAULT_JOBS or pass --jobs "R-naive:1-5"')

    st = grammar_stats(a.rung, a.n)
    log(f"rung={a.rung} | jobs={jobs_str} | n={a.n} | space={st}")

    llms: Dict[str, Ollama] = {}
    if not a.mock_llm:
        for arm, _ in jobs:
            _, default_model = ARM_SPECS[arm]
            model_name = a.model or default_model
            if model_name and model_name not in llms:
                client = Ollama(a.ollama_url, model_name)
                client.ensure_ready(a.start_ollama, a.install_ollama)
                llms[model_name] = client
                probe = work / f"contamination_probe_{model_name.replace(':', '_')}.json"
                if not probe.exists():
                    ans = client.chat(
                        [{"role": "user", "content": "Here is an anonymized click log with columns c0..c4 (categorical IDs) and t "
                          "(seconds), target = click converted. Which public dataset or competition is this? Answer briefly."}],
                        temperature=0.0, num_predict=200,
                    )
                    probe.write_text(json.dumps(dict(model=model_name, answer=ans), indent=2))

    W = get_window(Path(a.cache_dir))
    lab = Lab(W)
    lab.reset()
    t0 = time.time()
    lab.refit(0)
    log(f"baseline (4 raw IDs) full-track AUC {lab.ref['full_auc']:.5f} (frozen Phase 0: {F['dev_baseline']:.5f})")
    if not TEST and abs(lab.ref["full_auc"] - F["dev_baseline"]) > 0.003:
        log("WARNING: baseline differs from frozen Phase 0 by > 0.003 - check window/obfuscation")

    deadline = t0 + 60 * a.max_mins
    for arm, run in jobs:
        _, default_model = ARM_SPECS[arm]
        model_name = a.model or default_model
        llm = llms.get(model_name) if (model_name and not a.mock_llm) else None
        if not run_job(lab, a.rung, arm, run, a.n, work, llm, a.mock_llm, deadline):
            break
    report(work)


if __name__ == "__main__":
    main()
