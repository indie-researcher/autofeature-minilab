#!/usr/bin/env python
"""
AutoFeature Mini-Lab - PHASE 0 runner (single-file Kaggle bundle).

Bundles:
  1. td_ablation_v2 (feature builders, atomic memmap cache, LightGBM utilities)
  2. af_core        (window loader, ID obfuscation, forward-chained Fold, feature GRAMMAR)
  3. af_phase0      (DEV/SEALED references, proxy Spearman check, grammar self-test, spec freeze)
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# =============================================================================
# PART 1: td_ablation_v2 (shared feature engineering, caching, and utilities)
# =============================================================================
TESTING = os.environ.get("TD_TESTING") == "1"            # tiny smoke-test mode (never set on Kaggle)
QUICK = os.environ.get("TD_ABL_QUICK") == "1"
TEST_ROWS = int(os.environ.get("TD_TEST_ROWS", 300_000))
INPUT_DIR = os.environ.get("TD_INPUT", "/kaggle/input")
WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
OUT_DIR = WORK / "outputs"
ADIR = OUT_DIR / "ablation_v2"
PRED_DIR, EXTRA_DIR = ADIR / "preds", ADIR / "extras"
JOBS_FILE = ADIR / "jobs.jsonl"
CACHE_DIR = OUT_DIR / "cache"

TOTAL_TRAIN_ROWS = 184_903_890
TRAIN_ROWS = TEST_ROWS if TESTING else int(os.environ.get("TD_ABL_ROWS", 40_000_000))   # = main-run window
SEED = 42
TEST_HOURS = (4, 5, 9, 10, 13, 14)
EVAL_BLOCKS = ((9, 10), (13, 14))                        # test-hour blocks used as forward-chained evaluation folds
DROPPED = ["r_iph_ip"]                                   # what the drift scan dropped in the main run
RAW_FEATS = ["app", "device", "os", "channel", "hour"]

ABL = dict(
    neg_frac_train=0.5 if TESTING else 0.10,             # negatives kept for TRAIN (all positives kept)
    neg_frac_eval=1.0 if TESTING else 0.30,              # negatives kept for ES + EVAL sets (AUC stays unbiased)
    es_frac=0.12,                                        # last share of the pre-eval period = early-stopping set
    min_pre_rows=40_000 if TESTING else 3_000_000,
    min_eval_pos=30 if TESTING else 500,
    blocks=4 if TESTING else 12,                         # time blocks per fold for paired delta-AUC
    rounds=25 if TESTING else 500, early=5 if TESTING else 40,
    seeds=(0, 1) if TESTING else (0, 1, 2),
    healthy_iter=3 if TESTING else 25,                   # best_iter below this => flagged UNSTABLE
    topk_drop=2 if TESTING else 5,
    n_boot=500 if TESTING else 5000,
    scale_negfracs=(0.2, 0.8) if TESTING else (0.03, 0.30),
    perm_rows=30_000 if TESTING else 300_000,            # negatives used for permutation importance
    budget_min=float(os.environ.get("TD_ABL_BUDGET_MIN", 0)),   # 0 = unlimited; else stop launching new models
)
if QUICK:
    ABL.update(topk_drop=0, seeds=ABL["seeds"][:2], scale_negfracs=())

PARAMS_BASE = dict(objective="binary", metric="auc", learning_rate=0.1, num_leaves=31, max_depth=-1,
                   min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                   reg_lambda=5.0, max_bin=63, verbose=-1)
MAIN_RUN_OVER = dict(learning_rate=0.08, min_child_samples=200, max_bin=255, spw_full=50.0)

BASE: dict = {}                                          # filled by calibration: feats, spw_full, use_hour
COLS: list = []                                          # cached feature columns (canonical order)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def auc(y, s) -> float:
    return float(roc_auc_score(y, s))


def mem_limit_gb() -> float:
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = Path(p).read_text().strip()
            if v.isdigit() and int(v) < (1 << 50):
                return int(v) / 1e9
        except Exception:
            pass
    try:
        import psutil
        return psutil.virtual_memory().total / 1e9
    except Exception:
        return float("nan")


def rss_gb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e9
    except Exception:
        return float("nan")


def log_mem(tag: str = "") -> None:
    log(f"mem {tag}: process {rss_gb():.1f} GB (limit ~{mem_limit_gb():.1f} GB)")


def save_npy_atomic(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_name(path.stem + ".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def save_json_atomic(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)
    os.replace(tmp, path)


def sig_of(*objs) -> str:
    return hashlib.md5(json.dumps(objs, sort_keys=True, default=str).encode()).hexdigest()[:10]


def find_file(name: str) -> Path:
    for root in (Path(INPUT_DIR), Path("."), Path("data")):
        if root.exists():
            hits = sorted(root.rglob(name), key=lambda h: ("talkingdata" not in str(h).lower(), len(h.parts)))
            if hits:
                return hits[0]
    raise FileNotFoundError(f"{name} not found under {INPUT_DIR}")


def to_seconds(s: pd.Series) -> np.ndarray:
    return pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S").values.astype("datetime64[s]").astype(np.int64)


def pick_cache_dir() -> Path:
    """Scratch disk with the most free space; the 20 GB /kaggle/working area is only a last resort."""
    if TESTING:
        return OUT_DIR / "cache"
    best, best_free = None, -1.0
    for c in (Path("/kaggle/temp"), Path("/tmp")):
        try:
            c.mkdir(parents=True, exist_ok=True)
            if not os.access(c, os.W_OK):
                continue
            free = shutil.disk_usage(c).free / 1e9
            log(f"disk: {c} free {free:.1f} GB")
            if free > best_free:
                best, best_free = c / "td_cache", free
        except Exception:
            pass
    if best is None:
        best = OUT_DIR / "cache"
        log(f"disk: falling back to {best} (inside the 20 GB output area)")
    return best


def clean_tmp(d: Path) -> None:
    for f in list(d.glob("*.tmp*")):
        try:
            f.unlink()
            log(f"removed stale temp file {f.name}")
        except Exception:
            pass


def disk_preflight(n_cols_est: int = 50) -> None:
    need = TRAIN_ROWS * (n_cols_est * 4 + 26) / 1e9 * 1.15
    free = shutil.disk_usage(CACHE_DIR).free / 1e9
    log(f"disk preflight: need ~{need:.1f} GB, free {free:.1f} GB at {CACHE_DIR}")
    if free < need:
        raise RuntimeError(f"Not enough disk for the feature cache: need ~{need:.1f} GB, free {free:.1f} GB.")


def load_train() -> pd.DataFrame:
    f = find_file("train.csv")
    skip = 0 if TESTING else TOTAL_TRAIN_ROWS - TRAIN_ROWS
    log(f"reading {TRAIN_ROWS:,} train rows (skipping first {skip:,})")
    dt = dict(ip="uint32", app="uint16", device="uint16", os="uint16", channel="uint16", is_attributed="uint8")
    df = pd.read_csv(f, skiprows=range(1, skip + 1) if skip else None, nrows=TRAIN_ROWS,
                     usecols=["ip", "app", "device", "os", "channel", "click_time", "is_attributed"], dtype=dt)
    df["t"] = to_seconds(df.pop("click_time"))
    if not np.all(df["t"].to_numpy()[1:] >= df["t"].to_numpy()[:-1]):
        df = df.iloc[np.argsort(df["t"].to_numpy(), kind="stable")].reset_index(drop=True)
    return df


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


def _deltas(c, t, steps=(1, 2, 3)):
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


GROUPS = {
    "ip":            (["ip"],                         dict(cnt=1, nu=["channel", "app", "device", "os"])),
    "app":           (["app"],                        dict(cnt=1, nu=["channel"])),
    "ip_app":        (["ip", "app"],                  dict(cnt=1, new=1)),
    "ip_app_os":     (["ip", "app", "os"],            dict(cnt=1, new=1)),
    "ip_dev_os":     (["ip", "device", "os"],         dict(cnt=1)),
    "ip_app_dev_os": (["ip", "app", "device", "os"],  dict(cnt=1, new=1, delta=1)),
    "ip_chan":       (["ip", "channel"],              dict(cnt=1, new=1, delta=1)),
    "ip_app_chan":   (["ip", "app", "channel"],       dict(cnt=1, delta=1)),
    "app_chan":      (["app", "channel"],             dict(cnt=1)),
    "app_os":        (["app", "os"],                  dict(cnt=1)),
    "ip_dh":         (["ip", "dh"],                   dict(cnt=1, nu=["channel", "app"])),
    "ip_app_dh":     (["ip", "app", "dh"],            dict(cnt=1)),
}


def build_features(df: pd.DataFrame, keep: np.ndarray | None = None) -> dict:
    """Features over the WHOLE frame; only rows in `keep` (boolean mask) are materialised."""
    t = df["t"].to_numpy()
    base = {c: df[c].to_numpy().astype(np.int32) for c in ["ip", "app", "device", "os", "channel"]}
    base["dh"] = (t // 3600).astype(np.int32)
    base["hour"] = ((t % 86400) // 3600).astype(np.int32)
    out: dict = {}

    def put(name, arr):
        out[name] = (arr if keep is None else arr[keep]).astype(np.float32, copy=False)

    for c in RAW_FEATS:
        put(c, base[c])
    for g, (cols, fl) in GROUPS.items():
        code = _codes([base[c] for c in cols])
        if fl.get("cnt"):
            put(f"cnt_{g}", np.bincount(code)[code])
        if fl.get("new"):
            put(f"new_{g}", _cumcount(code))
        if fl.get("delta"):
            for k, v in _deltas(code, t).items():
                put(f"dt_{g}_{k}", v)
        for tgt in fl.get("nu", []):
            put(f"nu_{g}_{tgt}", _nunique(code, base[tgt]))
        del code
        gc.collect()
    eps = 1.0
    out["r_ipapp_ip"] = out["cnt_ip_app"] / (out["cnt_ip"] + eps)
    out["r_newness_usr"] = out["new_ip_app_dev_os"] / (out["cnt_ip_app_dev_os"] + eps)
    out["r_ipchan_ip"] = out["cnt_ip_chan"] / (out["cnt_ip"] + eps)
    out["r_iph_ip"] = out["cnt_ip_dh"] / (out["cnt_ip"] + eps)
    return out


def expected_cols() -> list[str]:
    out = list(RAW_FEATS)
    for g, (_, fl) in GROUPS.items():
        if fl.get("cnt"):
            out.append(f"cnt_{g}")
        if fl.get("new"):
            out.append(f"new_{g}")
        if fl.get("delta"):
            for k in (1, 2, 3):
                out += [f"dt_{g}_next{k}", f"dt_{g}_prev{k}"]
        for tgt in fl.get("nu", []):
            out.append(f"nu_{g}_{tgt}")
    return out + ["r_ipapp_ip", "r_newness_usr", "r_ipchan_ip", "r_iph_ip"]


def group_of(c: str) -> str:
    if c in RAW_FEATS:
        return "raw"
    for p, g in (("cnt_", "counts"), ("new_", "newness"), ("dt_", "deltas"), ("nu_", "uniques"), ("r_", "ratios")):
        if c.startswith(p):
            return g
    return "other"


def core_cols(cols):
    return [c for c in cols if not c.startswith(("dt_", "nu_"))]


def _writemm(path: Path, cols, out: dict, order=None):
    tmp = path.with_name(path.stem + ".tmp.npy")
    n = len(out[cols[0]])
    mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(n, len(cols)))
    for j, c in enumerate(cols):
        col = out.pop(c)
        mm[:, j] = col if order is None else col[order]
    mm.flush()
    del mm
    os.replace(tmp, path)


# =============================================================================
# PART 2: af_core (windows, ID obfuscation, forward-chained Fold, GRAMMAR)
# =============================================================================
AF = dict(
    hours_train=5, hours_eval=2,                 # window hours 8-12 = train/ES (offset 0-5h), 13-14 = eval (5h-7h)
    es_frac=0.12,                                # last share of the pre-eval rows = early-stop set
    neg_train=ABL["neg_frac_train"], neg_eval=ABL["neg_frac_eval"],
    rounds=ABL["rounds"], early=ABL["early"], spw_full=10.0,
    proxy_mod=3 if TESTING else 10,              # proxy = rows whose obfuscated IP code % proxy_mod == 0
    proxy_neg_train=0.8 if TESTING else 0.3,
    proxy_rho_min=0.7, seeds=ABL["seeds"], noise_min=2e-4,
)
WINDOWS = {"DEV": "2017-11-09", "SEALED": "2017-11-08"}
DAY_START_ROW = {"2017-11-08": 68_941_878, "2017-11-09": 131_886_953}
SKIP_MARGIN = 6_000_000
RAW_MAP = {"c0": "ip", "c1": "app", "c2": "device", "c3": "os", "c4": "channel"}
BASELINE = ["app", "device", "os", "channel"]    # no 'hour': unseen under forward-chaining
HUMAN_EXCLUDE = {"hour", "r_iph_ip"}


def load_window(day: str) -> pd.DataFrame:
    """Rows of `day` with 08:00 <= time < 16:00, found by timestamp (string compare is exact for this format)."""
    f = find_file("train.csv")
    lo, hi = f"{day} 08:00:00", f"{day} 16:00:00"
    skip = 0 if TESTING else max(0, DAY_START_ROW[day] - SKIP_MARGIN)
    log(f"window {day} 08-16h: scanning train.csv from row {skip:,}")
    dt = dict(ip="uint32", app="uint16", device="uint16", os="uint16", channel="uint16", is_attributed="uint8")
    rd = pd.read_csv(f, skiprows=range(1, skip + 1) if skip else None, dtype=dt, chunksize=200_000 if TESTING else 4_000_000,
                     usecols=["ip", "app", "device", "os", "channel", "click_time", "is_attributed"])
    parts, first = [], True
    for ch in rd:
        ct = ch["click_time"]
        if first:
            first = False
            if skip and ct.iloc[0] >= lo:
                raise RuntimeError(f"scan started at {ct.iloc[0]} >= window start {lo}: lower DAY_START_ROW / raise SKIP_MARGIN")
        m = (ct >= lo) & (ct < hi)
        if m.any():
            parts.append(ch[m])
        if ct.iloc[-1] >= hi:
            break
    if not parts:
        raise RuntimeError(f"no rows found for window {day}")
    df = pd.concat(parts, ignore_index=True)
    t0 = int(to_seconds(pd.Series([lo]))[0])
    df["t"] = (to_seconds(df.pop("click_time")) - t0).astype(np.int32)
    if not np.all(df["t"].to_numpy()[1:] >= df["t"].to_numpy()[:-1]):
        df = df.iloc[np.argsort(df["t"].to_numpy(), kind="stable")].reset_index(drop=True)
    log(f"window {day}: {len(df):,} rows, positives {df['is_attributed'].mean():.4%}")
    return df


def obfuscate(df: pd.DataFrame) -> dict:
    """Keyed scrambling of every ID column, then dense re-coding: equality structure kept, original values/ordering lost."""
    out = {}
    for i, c in enumerate(["ip", "app", "device", "os", "channel"]):
        rs = np.random.RandomState(SEED + 100 + i)
        mult = np.uint64(int(rs.randint(1, 2 ** 31)) * 2 + 1)
        salt = np.uint64(int(rs.randint(0, 2 ** 31)))
        scr = (df[c].to_numpy().astype(np.uint64) * mult + salt) & np.uint64(0xFFFFFFFF)
        out[f"c{i}"] = np.unique(scr, return_inverse=True)[1].astype(np.int32).reshape(-1)
    return out


def window_arrays(df: pd.DataFrame) -> dict:
    W = obfuscate(df)
    W["t"] = df["t"].to_numpy().astype(np.int32)
    W["h"] = (W["t"] // 3600).astype(np.int32)
    W["y"] = df["is_attributed"].to_numpy().astype(np.uint8)
    return W


def human_frame(W: dict) -> pd.DataFrame:
    return pd.DataFrame({"ip": W["c0"], "app": W["c1"], "device": W["c2"], "os": W["c3"], "channel": W["c4"], "t": W["t"]})


def subsample_window(W: dict, mod: int) -> dict:
    keep = (W["c0"] % mod) == 0
    return {k: v[keep] for k, v in W.items()}


def _sub(y, lo, hi, frac, rng):
    yy = y[lo:hi]
    pos, neg = np.flatnonzero(yy == 1), np.flatnonzero(yy == 0)
    if frac < 1.0:
        neg = rng.choice(neg, int(round(frac * len(neg))), replace=False)
    return np.sort(np.r_[pos, neg]) + lo


class Fold:
    """[train | early-stop (last es_frac) | eval = hours 13-14]; negatives subsampled, positives always kept."""

    def __init__(self, y, t, neg_train, neg_eval):
        a = int(np.searchsorted(t, 3600 * AF["hours_train"], "left"))
        b = int(np.searchsorted(t, 3600 * (AF["hours_train"] + AF["hours_eval"]), "left"))
        es0 = a - int(AF["es_frac"] * a)
        self.a, self.b, self.es0, self.neg_train = a, b, es0, neg_train
        self.idx_tr = _sub(y, 0, es0, neg_train, np.random.RandomState(SEED + 13))
        self.idx_es = _sub(y, es0, a, neg_eval, np.random.RandomState(SEED + 7))
        self.idx_ev = _sub(y, a, b, neg_eval, np.random.RandomState(SEED + 8))
        self.ytr, self.yes, self.yev = y[self.idx_tr], y[self.idx_es], y[self.idx_ev]

    def describe(self):
        return dict(train=len(self.idx_tr), es=len(self.idx_es), eval=len(self.idx_ev), eval_pos=int(self.yev.sum()))


def fold_mats(fold: Fold, getter):
    return getter(fold.idx_tr), getter(fold.idx_es), getter(fold.idx_ev)


def fit_eval(fold: Fold, mats, ix, names, seed=0, spw_full=None, over=None) -> dict:
    """Train on mats[0][:, ix], early-stop on mats[1], score mats[2]. Returns eval preds + AUCs."""
    Xtr, Xes, Xev = (np.ascontiguousarray(m[:, ix], dtype=np.float32) for m in mats)
    p = {**PARAMS_BASE, **(over or {})}
    p["scale_pos_weight"] = float(spw_full or AF["spw_full"]) * float(fold.neg_train)
    p["seed"] = SEED + int(seed)
    dp = dict(max_bin=p["max_bin"], verbose=-1, feature_pre_filter=False)
    d_tr = lgb.Dataset(Xtr, fold.ytr, feature_name=list(names), params=dp, free_raw_data=True).construct()
    d_es = lgb.Dataset(Xes, fold.yes, reference=d_tr, params=dp, free_raw_data=True).construct()
    m = lgb.train(p, d_tr, num_boost_round=AF["rounds"], valid_sets=[d_es], valid_names=["es"],
                  callbacks=[lgb.early_stopping(AF["early"], verbose=False)])
    bi = int(m.best_iteration or AF["rounds"])
    pe = m.predict(Xev, num_iteration=bi)
    return dict(pred=pe.astype(np.float32), es_auc=float(m.best_score["es"]["auc"]), eval_auc=auc(fold.yev, pe), best_iter=bi)


def raw_getter(W, names=("c1", "c2", "c3", "c4")):
    """Baseline matrix builder: obfuscated raw IDs (app, device, os, channel) at given row indices."""
    return lambda idx: np.column_stack([W[n][idx] for n in names]).astype(np.float32)


KEYS = ["c0", "c1", "c2", "c3", "c4", "h"]        # c0..c4 obfuscated ID columns, h = hour index of the window
LAGS, ROLL_W = (1, 2, 3), (60, 600, 3600)
BASIC_OPS = ("count", "nunique", "cumcount", "roll")
ALL_OPS = ("count", "nunique", "cumcount", "lag", "roll", "ratio")

GRAMMAR_DOC = """Propose ONE feature as a JSON object. Columns: c0..c4 (anonymous categorical IDs) and h (hour index 0..7).
Click rows are time-ordered events; the target is whether the click converted. Allowed specs:
 {"op":"count","keys":[...]}                      rows sharing the same key values (whole window)
 {"op":"nunique","keys":[...],"target":"cX"}      distinct target values within the key group
 {"op":"cumcount","keys":[...]}                   running index of the row inside its key group
 {"op":"lag","keys":[...],"k":1|2|3,"dir":"next"|"prev"}   seconds to the k-th next/previous row of the group
 {"op":"roll","keys":[...],"w":60|600|3600}       rows of the group in the previous w seconds
 {"op":"ratio","num":<count|nunique|cumcount|roll spec>,"den":<same>}   num / (den + 1)
keys: 1-4 distinct names from [c0,c1,c2,c3,c4,h]. Return JSON only."""


def canon_spec(spec):
    s = json.loads(json.dumps(spec, sort_keys=True))
    if "keys" in s:
        s["keys"] = sorted(s["keys"])
    for k in ("num", "den"):
        if k in s:
            s[k] = canon_spec(s[k])
    return s


def spec_id(spec) -> str:
    return hashlib.md5(json.dumps(canon_spec(spec), sort_keys=True).encode()).hexdigest()[:10]


def validate(spec, depth=0):
    try:
        op = spec["op"]
        if op == "ratio":
            if depth:
                return False, "nested ratio"
            for k in ("num", "den"):
                ok, msg = validate(spec[k], depth + 1)
                if not ok or spec[k]["op"] not in BASIC_OPS:
                    return False, f"{k}: {msg or 'op not allowed inside ratio'}"
            return True, ""
        keys = spec["keys"]
        if not (1 <= len(keys) <= 4) or len(set(keys)) != len(keys) or any(k not in KEYS for k in keys):
            return False, "bad keys"
        if op in ("count", "cumcount"):
            return True, ""
        if op == "nunique":
            return (spec["target"] in KEYS and spec["target"] not in keys), "bad target"
        if op == "lag":
            return (spec["k"] in LAGS and spec["dir"] in ("next", "prev")), "bad lag"
        if op == "roll":
            return spec["w"] in ROLL_W, "bad window"
        return False, "unknown op"
    except Exception as e:                                         # malformed LLM output never crashes the lab
        return False, repr(e)


def _code(keys, W, cache):
    k = tuple(sorted(keys))
    if k not in cache:
        cache[k] = _codes([W[x] for x in k])
    return cache[k]


def _roll(code, t, w):
    n = len(code)
    order = np.lexsort((t, code))
    cs, ts = code[order].astype(np.int64), t[order].astype(np.int64)
    assert ts.max() < 65536
    key = cs * 65536 + ts
    lo = np.searchsorted(key, cs * 65536 + np.maximum(ts - w, 0), side="left")
    out = np.empty(n, np.float32)
    out[order] = (np.arange(n) - lo).astype(np.float32)
    return out


def _basic(spec, W, cache):
    op, code = spec["op"], _code(spec["keys"], W, cache)
    if op == "count":
        return np.bincount(code)[code].astype(np.float32)
    if op == "cumcount":
        return _cumcount(code).astype(np.float32)
    if op == "nunique":
        return _nunique(code, W[spec["target"]].astype(np.int64))
    if op == "lag":
        return _deltas(code, W["t"], steps=(spec["k"],))[f"{spec['dir']}{spec['k']}"]
    return _roll(code, W["t"], spec["w"])


def compute(spec, W, cache=None) -> np.ndarray:
    """Execute a validated spec over a window; returns float32 (NaN allowed)."""
    ok, msg = validate(spec)
    if not ok:
        raise ValueError(f"invalid spec: {msg}")
    cache = {} if cache is None else cache
    if spec["op"] == "ratio":
        return (_basic(spec["num"], W, cache) / (_basic(spec["den"], W, cache) + 1.0)).astype(np.float32)
    return _basic(spec, W, cache)


def make_decoy(arr: np.ndarray, seed: int) -> np.ndarray:
    """Permuted-real decoy: same marginal distribution, zero relation to the row's label."""
    return arr[np.random.RandomState(seed).permutation(len(arr))]


def sample_spec(rng) -> dict:
    """Uniform-ish random spec from the grammar (used by the random arm and the decoy generator)."""
    def keys():
        n = int(rng.choice([1, 2, 3, 4], p=[0.25, 0.35, 0.30, 0.10]))
        return sorted(str(k) for k in rng.choice(KEYS, n, replace=False))

    def basic(op=None):
        op = op or str(rng.choice(BASIC_OPS))
        s = dict(op=op, keys=keys())
        if op == "nunique":
            s["target"] = str(rng.choice([k for k in KEYS if k not in s["keys"]]))
        if op == "roll":
            s["w"] = int(rng.choice(ROLL_W))
        return s

    while True:
        op = str(rng.choice(ALL_OPS, p=[0.20, 0.20, 0.10, 0.20, 0.15, 0.15]))
        if op == "ratio":
            s = dict(op="ratio", num=basic(), den=basic())
        elif op == "lag":
            s = dict(op="lag", keys=keys(), k=int(rng.choice(LAGS)), dir=str(rng.choice(["next", "prev"])))
        else:
            s = basic(op)
        if validate(s)[0]:
            return canon_spec(s)


# =============================================================================
# PART 3: af_phase0 (original Phase 0 runner)
# =============================================================================
OUT = OUT_DIR / "af_phase0"
AFC: Path = Path(".")


class JsonCache:
    def __init__(self, path: Path):
        self.path, self.d = path, (json.load(open(path)) if path.exists() else {})

    def get(self, k):
        return self.d.get(k)

    def set(self, k, v):
        self.d[k] = v
        save_json_atomic(self.path, self.d)


def init_dirs():
    global AFC
    AFC = pick_cache_dir().parent / "af_cache"
    AFC.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"features/windows cache: {AFC} | outputs: {OUT}")


def build_window(name: str) -> dict:
    P = dict(npz=AFC / f"W_{name}.npz", X=AFC / f"X_{name}.npy", cols=AFC / f"cols_{name}.json")
    if all(p.exists() for p in P.values()):
        log(f"window {name}: cache found")
        return P
    W = window_arrays(load_window(WINDOWS[name]))
    log(f"window {name}: building human features")
    out = build_features(human_frame(W))
    cols = list(out)
    _writemm(P["X"], cols, out)
    tmp = AFC / f"W_{name}.tmp.npz"
    np.savez(tmp, **W)
    os.replace(tmp, P["npz"])
    save_json_atomic(P["cols"], cols)                                  # completion marker
    return P


def load_window_cache(P):
    z = np.load(P["npz"])
    return {k: z[k] for k in z.files}, np.load(P["X"], mmap_mode="r"), json.load(open(P["cols"]))


def rows(mm, idx):
    return np.asarray(mm[idx])


def human_idx(cols):
    base = [cols.index(c) for c in BASELINE]
    ceil = [i for i, c in enumerate(cols) if c not in HUMAN_EXCLUDE]
    return base, ceil


def main():
    t0 = time.time()
    init_dirs()
    cache = JsonCache(OUT / "phase0_cache.json")
    res: dict = {}

    # ---- 1-2. windows + references ----------------------------------------------------------------------------
    data = {}
    for name in ("DEV", "SEALED"):
        P = build_window(name)
        W, mm, cols = load_window_cache(P)
        fold = Fold(W["y"], W["t"], AF["neg_train"], AF["neg_eval"])
        log(f"{name}: {len(W['y']):,} rows | fold {fold.describe()}")
        data[name] = (W, mm, cols, fold)
        ib, ic = human_idx(cols)
        mats = tuple(rows(mm, i) for i in (fold.idx_tr, fold.idx_es, fold.idx_ev))
        seeds = AF["seeds"] if name == "DEV" else (0,)
        for tag, ix in (("baseline", ib), ("ceiling", ic)):
            for s in seeds:
                k = f"ref:{name}:{tag}:{s}"
                if cache.get(k) is None:
                    r = fit_eval(fold, mats, ix, [cols[i] for i in ix], seed=s)
                    cache.set(k, dict(eval_auc=r["eval_auc"], es_auc=r["es_auc"], best_iter=r["best_iter"]))
                    log(f"{k}: eval AUC {r['eval_auc']:.5f} (iters {r['best_iter']}, {len(ix)} features)")
        del mats
        res[name] = dict(
            baseline=cache.get(f"ref:{name}:baseline:0")["eval_auc"],
            ceiling=cache.get(f"ref:{name}:ceiling:0")["eval_auc"],
            fold=fold.describe(),
        )
        res[name]["gap"] = res[name]["ceiling"] - res[name]["baseline"]

    cs = [cache.get(f"ref:DEV:ceiling:{s}")["eval_auc"] for s in AF["seeds"]]
    bs = [cache.get(f"ref:DEV:baseline:{s}")["eval_auc"] for s in AF["seeds"]]
    sd = float(np.std(cs, ddof=1)) if len(cs) > 1 else float("nan")
    thr = max(2 * sd, AF["noise_min"]) if sd == sd else AF["noise_min"]
    res["noise"] = dict(
        ceiling_seed_sd=sd,
        baseline_seed_sd=float(np.std(bs, ddof=1)) if len(bs) > 1 else float("nan"),
        materiality_threshold=thr,
    )
    log(f"noise: ceiling seed sd {sd:.5f} -> threshold {thr:.5f}")

    # ---- 3. proxy check on DEV ---------------------------------------------------------------------------------
    W, mm, cols, fold = data["DEV"]
    ib, ic = human_idx(cols)
    feats = [c for c in cols if c not in HUMAN_EXCLUDE and c not in BASELINE]
    mats = tuple(rows(mm, i) for i in (fold.idx_tr, fold.idx_es, fold.idx_ev))
    base_full = cache.get("ref:DEV:baseline:0")["eval_auc"]
    for f in feats:
        k = f"full:{f}"
        if cache.get(k) is None:
            r = fit_eval(fold, mats, ib + [cols.index(f)], BASELINE + [f], seed=0)
            cache.set(k, r["eval_auc"] - base_full)
            log(f"proxy-check FULL  {f}: dAUC {cache.get(k):+.5f}")
    del mats
    Wp = subsample_window(W, AF["proxy_mod"])
    log(f"proxy window: {len(Wp['y']):,} rows ({1 / AF['proxy_mod']:.0%} of IPs)")
    outp = build_features(human_frame(Wp))
    Xp = np.column_stack([outp[c] for c in cols]).astype(np.float32)
    del outp
    fold_p = Fold(Wp["y"], Wp["t"], AF["proxy_neg_train"], 1.0)
    log(f"proxy fold {fold_p.describe()}")
    mats_p = tuple(rows(Xp, i) for i in (fold_p.idx_tr, fold_p.idx_es, fold_p.idx_ev))
    if cache.get("proxy_base") is None:
        cache.set("proxy_base", fit_eval(fold_p, mats_p, ib, BASELINE, seed=0)["eval_auc"])
    for f in feats:
        k = f"proxy:{f}"
        if cache.get(k) is None:
            r = fit_eval(fold_p, mats_p, ib + [cols.index(f)], BASELINE + [f], seed=0)
            cache.set(k, r["eval_auc"] - cache.get("proxy_base"))
    pc = pd.DataFrame(dict(feature=feats, d_full=[cache.get(f"full:{f}") for f in feats],
                           d_proxy=[cache.get(f"proxy:{f}") for f in feats]))
    rho = float(spearmanr(pc.d_full, pc.d_proxy)[0])
    pc.to_csv(OUT / "proxy_check.csv", index=False)
    passed = bool(rho >= AF["proxy_rho_min"])
    res["proxy"] = dict(spearman=rho, n_features=len(feats), passed=passed, min_required=AF["proxy_rho_min"])
    log(f"PROXY CHECK: Spearman {rho:.3f} over {len(feats)} features -> {'PASS' if passed else 'FAIL (drop the proxy)'}")

    # ---- 4. grammar self-test ----------------------------------------------------------------------------------
    rng = np.random.RandomState(SEED)
    rows_t, seen = [], set()
    ops_seen = set()
    for i in range(60):
        s = sample_spec(rng)
        if spec_id(s) in seen:
            continue
        seen.add(spec_id(s))
        t1 = time.time()
        a = compute(s, Wp)
        ops_seen.add(s["op"])
        rows_t.append(dict(id=spec_id(s), op=s["op"], spec=json.dumps(s), secs=round(time.time() - t1, 2),
                           finite_frac=float(np.isfinite(a).mean()), std=float(np.nanstd(a))))
    gt = pd.DataFrame(rows_t)
    gt.to_csv(OUT / "grammar_selftest.csv", index=False)
    spec0 = canon_spec(dict(op="lag", keys=["c0", "c1", "c2", "c3"], k=1, dir="next"))
    real, decoy = compute(spec0, Wp), make_decoy(compute(spec0, Wp), 1)
    bg = raw_getter(Wp)
    for tag, arr in (("real(lag next1, ip-app-dev-os)", real), ("decoy(permuted)", decoy)):
        getter = lambda idx, a=arr: np.column_stack([bg(idx), a[idx]]).astype(np.float32)
        r = fit_eval(fold_p, fold_mats(fold_p, getter), list(range(5)), BASELINE + ["cand"], seed=0)
        res.setdefault("grammar_test", {})[tag] = r["eval_auc"] - cache.get("proxy_base")
        log(f"grammar end-to-end: {tag}: dAUC {res['grammar_test'][tag]:+.5f}")
    res["grammar"] = dict(
        specs_tested=len(gt),
        ops_covered=sorted(ops_seen),
        all_finite_mean=float(gt.finite_frac.mean()),
        invalid_rejected=not validate(dict(op="count", keys=["zz"]))[0],
    )
    del Xp, mats_p

    # ---- 5. probe prompt, schema, freeze, report ---------------------------------------------------------------
    head = pd.DataFrame({k: W[k][:12] for k in ("c0", "c1", "c2", "c3", "c4", "t")})
    (OUT / "contamination_probe.txt").write_text(
        "Paste this into the proposer LLM BEFORE the runs and log its answer (recall vs discovery):\n\n"
        "Here are the first rows of an anonymized click log (c0..c4 are categorical IDs, t = seconds since window start).\n"
        "Which public dataset or competition is this, and what feature ideas does it suggest?\n\n" + head.to_string(index=False)
    )
    (OUT / "grammar_prompt.txt").write_text(GRAMMAR_DOC)
    self_hash = hashlib.md5(Path(__file__).read_bytes()).hexdigest()[:10]
    h = lambda p: hashlib.md5(Path(p).read_bytes()).hexdigest()[:10] if Path(p).exists() else self_hash
    freeze = dict(
        frozen_at=time.strftime("%Y-%m-%d %H:%M"),
        config=AF,
        results=res,
        files=dict(
            bundle=self_hash,
            af_core=h("af_core.py"),
            td_ablation_v2=h("td_ablation_v2.py"),
            af_phase0=h("af_phase0.py"),
        ),
    )
    save_json_atomic(OUT / "spec_freeze.json", freeze)
    L = [
        "# AutoFeature Mini-Lab - Phase 0 report",
        "",
        f"Windows: DEV {WINDOWS['DEV']} / SEALED {WINDOWS['SEALED']}, hours 8-15, obfuscated. Eval = hours 13-14.",
        "",
        "| window | baseline (4 raw) | ceiling (human set) | gap | train/ES/eval rows | eval pos |",
        "|---|---|---|---|---|---|",
    ]
    for n in ("DEV", "SEALED"):
        r = res[n]
        L.append(
            f"| {n} | {r['baseline']:.5f} | {r['ceiling']:.5f} | {r['gap']:.5f} | "
            f"{r['fold']['train']:,}/{r['fold']['es']:,}/{r['fold']['eval']:,} | {r['fold']['eval_pos']:,} |"
        )
    L += [
        "",
        f"**Noise:** ceiling seed sd {sd:.5f} -> materiality threshold **{thr:.5f}**.",
        "",
        f"**Proxy check:** Spearman {rho:.3f} over {len(feats)} features (need >= {AF['proxy_rho_min']}) -> "
        f"**{'PASS: use the proxy' if passed else 'FAIL: drop the proxy and cut to 30 proposals per run'}**.",
        "",
        f"**Grammar:** {res['grammar']['specs_tested']} random specs executed, ops covered {res['grammar']['ops_covered']}, "
        f"mean finite fraction {res['grammar']['all_finite_mean']:.3f}; invalid spec rejected: {res['grammar']['invalid_rejected']}.",
        f"End-to-end (proxy fold): real lag feature {res['grammar_test']['real(lag next1, ip-app-dev-os)']:+.5f} dAUC vs "
        f"permuted decoy {res['grammar_test']['decoy(permuted)']:+.5f}.",
        "",
        "## Before Phase 1",
        "1. Run the contamination probe and log the answer.",
        "2. `git tag phase0-freeze` (spec, grammar, thresholds are now frozen).",
        "3. Do not look at SEALED again until Phase 3.",
    ]
    (OUT / "phase0_report.md").write_text("\n".join(L), encoding="utf-8")
    log(f"PHASE 0 done in {(time.time() - t0) / 60:.1f} min -> {OUT}")


if __name__ == "__main__":
    main()
