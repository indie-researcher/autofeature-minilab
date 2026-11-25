#!/usr/bin/env python3
"""
AutoFeature Mini-Lab - PHASE 1+2 in ONE file (Kaggle-ready).   af_lab.py

Arms (pre-registered, see autofeature_minilab_spec.md):
    R        random proposals from the grammar          + online-FDR gate
    L-naive  LLM (Ollama, Qwen2.5-7B) proposals         + naive gate (keep if DEV AUC goes up)
    L-FDR    LLM proposals                              + online-FDR gate (LORD++), materiality, sign-flip p-value
Every arm: 60 evaluated proposals per run, 20% of them permuted-real DECOYS (same schedule across arms for a given run),
proxy screen (10% IP subsample) -> full DEV test. SEALED is never loaded here.

Usage (examples)
    python af_lab.py --dry-run
    python af_lab.py --jobs "R:1-5"                      # no LLM needed
    python af_lab.py --jobs "L-FDR:1-5"                  # starts Ollama, pulls qwen2.5:7b
    python af_lab.py --jobs "L-naive:1-5"
    python af_lab.py --report                            # aggregate all finished runs
Resumable: re-run the same command after a crash/timeout; finished proposals are replayed, not recomputed.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
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
TEST = os.environ.get("AF_TEST") == "1"                      # tiny smoke-test mode (synthetic data); never on Kaggle
SEED = 42

# =============================================================================
# 1. FROZEN PHASE 0 CONSTANTS (from spec_freeze.json / proxy_check.csv). No labels-derived priors are embedded.
# =============================================================================
F = dict(
    hours_train=5, hours_eval=2, es_frac=0.12,
    neg_train=0.1, neg_eval=0.3, rounds=500, early=40, spw_full=10.0,
    proxy_mod=10, proxy_neg_train=0.3,
    materiality=0.00021865718476891195,                       # 2 x ceiling seed sd (Phase 0)
    dev_baseline=0.970396967505926, dev_ceiling=0.9847143092101822,
    proxy_theta=0.0005,                                       # promote to full test if proxy dAUC >= theta (recall .97 on Phase 0)
    blocks=12, fdr_alpha=0.10, decoy_frac=0.20,
)
if TEST:
    F.update(neg_train=0.5, neg_eval=1.0, rounds=25, early=5, proxy_mod=3, proxy_neg_train=0.8,
             materiality=0.002, proxy_theta=0.0)
WINDOW_DAY = "2017-11-09"                                     # DEV only
DAY_START_ROW = 131_886_953
SKIP_MARGIN = 6_000_000
PARAMS = dict(objective="binary", metric="auc", learning_rate=0.1, num_leaves=31, max_depth=-1, min_child_samples=100,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=5.0, max_bin=63, verbose=-1)
ARMS = ("R", "L-naive", "L-FDR", "R-naive")
MAX_ATTEMPTS = 6
FAIL_LIMIT = 30
TEMP_CAP = 1.2

# ---- EDIT THIS PER NOTEBOOK (used when no --jobs argument is given, e.g. Kaggle "Save & Run All") -------------------
DEFAULT_JOBS = "R-naive:1-5"        # notebook A: "R:1-5"   notebook B: "L-FDR:1-5"   notebook C: "L-naive:1-5"   notebook D: "R-naive:1-5"
# (env var AF_JOBS or the --jobs argument override it)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def auc(y, s) -> float:
    return float(roc_auc_score(y, s))


# =============================================================================
# 2. DATA: window by timestamp + keyed obfuscation (identical to Phase 0, so the baseline reproduces)
# =============================================================================
def find_file(name: str) -> Path:
    roots = [Path(os.environ.get("TD_INPUT", "/kaggle/input")), Path("."), Path("data")]
    for root in roots:
        if root.exists():
            hits = sorted(root.rglob(name), key=lambda h: ("talkingdata" not in str(h).lower(), len(h.parts)))
            if hits:
                return hits[0]
    raise FileNotFoundError(f"{name} not found (attach the TalkingData competition data or set TD_INPUT)")


def to_seconds(s: pd.Series) -> np.ndarray:
    return pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S").values.astype("datetime64[s]").astype(np.int64)


def load_dev_window() -> pd.DataFrame:
    f = find_file("train.csv")
    lo, hi = f"{WINDOW_DAY} 08:00:00", f"{WINDOW_DAY} 16:00:00"
    skip = 0 if TEST else max(0, DAY_START_ROW - SKIP_MARGIN)
    log(f"window: scanning {f} from row {skip:,}")
    dt = dict(ip="uint32", app="uint16", device="uint16", os="uint16", channel="uint16", is_attributed="uint8")
    rd = pd.read_csv(f, skiprows=range(1, skip + 1) if skip else None, dtype=dt, chunksize=200_000 if TEST else 4_000_000,
                     usecols=["ip", "app", "device", "os", "channel", "click_time", "is_attributed"])
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
    log(f"window: {len(df):,} rows, positives {df['is_attributed'].mean():.4%}")
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


def get_window(cache_dir: Path) -> Dict[str, np.ndarray]:
    p = cache_dir / f"W_{WINDOW_DAY}.npz"
    if p.exists():
        log(f"window: cache {p}")
        z = np.load(p)
        return {k: z[k] for k in z.files}
    W = window_arrays(load_dev_window())
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache_dir / f"W_{WINDOW_DAY}.tmp.npz"
    np.savez(tmp, **W)
    os.replace(tmp, p)
    return W


# =============================================================================
# 3. GRAMMAR (same ops as Phase 0; the proposer can only emit these JSON specs)
# =============================================================================
KEYS = ["c0", "c1", "c2", "c3", "c4", "h"]
LAGS, ROLL_W = (1, 2, 3), (60, 600, 3600)
BASIC_OPS = ("count", "nunique", "cumcount", "roll")
ALL_OPS = ("count", "nunique", "cumcount", "lag", "roll", "ratio")

GRAMMAR_PROMPT = """Propose ONE feature as a JSON object. Columns: c0..c4 (anonymous categorical IDs) and h (hour index 0..7).
Click rows are time-ordered events; the target is whether the click converted. Allowed specs:
 {"op":"count","keys":[...]}                      rows sharing the same key values (whole window)
 {"op":"nunique","keys":[...],"target":"cX"}      distinct target values within the key group
 {"op":"cumcount","keys":[...]}                   running index of the row inside its key group
 {"op":"lag","keys":[...],"k":1|2|3,"dir":"next"|"prev"}   seconds to the k-th next/previous row of the group
 {"op":"roll","keys":[...],"w":60|600|3600}       rows of the group in the previous w seconds
 {"op":"ratio","num":<count|nunique|cumcount|roll spec>,"den":<same>}   num / (den + 1)
keys: 1-4 distinct names from [c0,c1,c2,c3,c4,h]. Return JSON only."""

_LEAF = {"type": "object", "properties": {
    "op": {"type": "string", "enum": list(BASIC_OPS)},
    "keys": {"type": "array", "items": {"type": "string", "enum": KEYS}, "minItems": 1, "maxItems": 4},
    "target": {"type": "string", "enum": KEYS}, "w": {"type": "integer", "enum": list(ROLL_W)}}, "required": ["op", "keys"]}
SPEC_SCHEMA = {"type": "object", "properties": {
    "op": {"type": "string", "enum": list(ALL_OPS)},
    "keys": {"type": "array", "items": {"type": "string", "enum": KEYS}, "minItems": 1, "maxItems": 4},
    "target": {"type": "string", "enum": KEYS}, "k": {"type": "integer", "enum": list(LAGS)},
    "dir": {"type": "string", "enum": ["next", "prev"]}, "w": {"type": "integer", "enum": list(ROLL_W)},
    "num": _LEAF, "den": _LEAF}, "required": ["op"]}


def spec_id(spec: dict) -> str:
    return hashlib.md5(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:10]


def normalize(spec: Any, nested: bool = False) -> Tuple[bool, str, Optional[dict]]:
    """Strict validation; returns a clean canonical spec (extra fields rejected, keys sorted)."""
    try:
        if not isinstance(spec, dict):
            return False, "not an object", None
        op = spec.get("op")
        if op not in ALL_OPS or (nested and op not in BASIC_OPS):
            return False, f"bad op {op!r}", None
        if op == "ratio":
            if set(spec) != {"op", "num", "den"}:
                return False, "ratio needs exactly op/num/den", None
            ok1, m1, n = normalize(spec["num"], True)
            ok2, m2, d = normalize(spec["den"], True)
            if not (ok1 and ok2):
                return False, m1 or m2, None
            if n == d:
                return False, "trivial ratio", None
            return True, "", {"op": "ratio", "num": n, "den": d}
        keys = spec.get("keys")
        if (not isinstance(keys, list) or not 1 <= len(keys) <= 4 or len(set(keys)) != len(keys)
                or any(k not in KEYS for k in keys)):
            return False, "bad keys", None
        out: Dict[str, Any] = {"op": op, "keys": sorted(keys)}
        allowed = {"op", "keys"}
        if op == "nunique":
            if spec.get("target") not in KEYS or spec["target"] in keys:
                return False, "bad target", None
            out["target"] = spec["target"]
            allowed.add("target")
        elif op == "lag":
            if spec.get("k") not in LAGS or isinstance(spec.get("k"), bool) or spec.get("dir") not in ("next", "prev"):
                return False, "bad lag", None
            out["k"], out["dir"] = int(spec["k"]), spec["dir"]
            allowed |= {"k", "dir"}
        elif op == "roll":
            if spec.get("w") not in ROLL_W or isinstance(spec.get("w"), bool):
                return False, "bad window", None
            out["w"] = int(spec["w"])
            allowed.add("w")
        if set(spec) - allowed:
            return False, f"unexpected fields {sorted(set(spec) - allowed)}", None
        return True, "", out
    except Exception as e:
        return False, repr(e), None


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


def _roll(code, t, w):
    n = len(code)
    order = np.lexsort((t, code))
    cs, ts = code[order].astype(np.int64), t[order].astype(np.int64)
    key = cs * 65536 + ts
    lo = np.searchsorted(key, cs * 65536 + np.maximum(ts - w, 0), side="left")
    out = np.empty(n, np.float32)
    out[order] = (np.arange(n) - lo).astype(np.float32)
    return out


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
    return _roll(code, W["t"], spec["w"])


def compute(spec: dict, W: Dict[str, np.ndarray]) -> np.ndarray:
    cache: dict = {}
    if spec["op"] == "ratio":
        return (_basic(spec["num"], W, cache) / (_basic(spec["den"], W, cache) + 1.0)).astype(np.float32)
    return _basic(spec, W, cache)


def make_decoy(arr: np.ndarray, seed: int) -> np.ndarray:
    return arr[np.random.RandomState(seed).permutation(len(arr))]


def sample_spec(rng) -> dict:
    def keys():
        n = int(rng.choice([1, 2, 3, 4], p=[0.25, 0.35, 0.30, 0.10]))
        return sorted(str(k) for k in rng.choice(KEYS, n, replace=False))

    def basic(op=None):
        op = op or str(rng.choice(BASIC_OPS))
        s: Dict[str, Any] = dict(op=op, keys=keys())
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
        ok, _, clean = normalize(s)
        if ok:
            return clean


# =============================================================================
# 4. FOLD + MODEL (one fold per window: train hours 8-12, ES = last 12%, eval = hours 13-14)
# =============================================================================
def _sub(y, lo, hi, frac, rng):
    yy = y[lo:hi]
    pos, neg = np.flatnonzero(yy == 1), np.flatnonzero(yy == 0)
    if frac < 1.0:
        neg = rng.choice(neg, int(round(frac * len(neg))), replace=False)
    return np.sort(np.r_[pos, neg]) + lo


class Track:
    """One evaluation track (FULL window or 10%-IP PROXY window): fold indices + raw baseline matrices."""

    def __init__(self, W, neg_train, neg_eval):
        y, t = W["y"], W["t"]
        a = int(np.searchsorted(t, 3600 * F["hours_train"], "left"))
        b = int(np.searchsorted(t, 3600 * (F["hours_train"] + F["hours_eval"]), "left"))
        es0 = a - int(F["es_frac"] * a)
        self.neg_train = neg_train
        self.idx = (_sub(y, 0, es0, neg_train, np.random.RandomState(SEED + 13)),
                    _sub(y, es0, a, neg_eval, np.random.RandomState(SEED + 7)),
                    _sub(y, a, b, neg_eval, np.random.RandomState(SEED + 8)))
        self.y = tuple(y[i] for i in self.idx)
        self.base = tuple(np.column_stack([W[n][i] for n in ("c1", "c2", "c3", "c4")]).astype(np.float32) for i in self.idx)
        self.yev = self.y[2]

    def col(self, arr: np.ndarray):
        return tuple(arr[i].astype(np.float32)[:, None] for i in self.idx)

    def fit(self, extras: list, seed: int) -> dict:
        mats = [np.ascontiguousarray(np.column_stack([self.base[k]] + [e[k] for e in extras]), dtype=np.float32)
                for k in range(3)]
        p = {**PARAMS, "scale_pos_weight": F["spw_full"] * self.neg_train, "seed": SEED + int(seed)}
        dp = dict(max_bin=p["max_bin"], verbose=-1, feature_pre_filter=False)
        d_tr = lgb.Dataset(mats[0], self.y[0], params=dp, free_raw_data=True).construct()
        d_es = lgb.Dataset(mats[1], self.y[1], reference=d_tr, params=dp, free_raw_data=True).construct()
        m = lgb.train(p, d_tr, num_boost_round=F["rounds"], valid_sets=[d_es], valid_names=["es"],
                      callbacks=[lgb.early_stopping(F["early"], verbose=False)])
        bi = int(m.best_iteration or F["rounds"])
        pe = m.predict(mats[2], num_iteration=bi).astype(np.float32)
        return dict(pred=pe, auc=auc(self.yev, pe), iters=bi)


def block_aucs(y, p, nb):
    e = np.linspace(0, len(y), nb + 1).astype(int)
    return np.array([auc(y[a:b], p[a:b]) if 0 < y[a:b].sum() < b - a else np.nan for a, b in zip(e[:-1], e[1:])])


_SIGNS: Dict[int, np.ndarray] = {}


def signflip_p(d: np.ndarray) -> float:
    """Exact one-sided sign-flip test of mean(d) > 0 over per-block deltas (12 blocks -> 4096 patterns)."""
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 4:
        return 1.0
    if n not in _SIGNS:
        _SIGNS[n] = np.array(list(itertools.product((-1.0, 1.0), repeat=n)))
    return float(np.mean((_SIGNS[n] @ d) / n >= d.mean() - 1e-15))


# =============================================================================
# 5. ONLINE FDR (LORD++, Ramdas et al. 2017)
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
# 6. PROPOSERS: random sampler, mock, Ollama (Qwen2.5-7B)
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
                subprocess.run("apt-get update -qq && apt-get install -y -qq zstd curl && "
                               "curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)
            env = os.environ.copy()
            Path("/tmp/ollama_models").mkdir(parents=True, exist_ok=True)
            env.setdefault("OLLAMA_MODELS", "/tmp/ollama_models")
            subprocess.Popen(["ollama", "serve"], stdout=open("/tmp/ollama_serve.log", "a"), stderr=subprocess.STDOUT,
                             env=env, start_new_session=True)
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
            log("GPU: " + subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip().replace("\n", " | "))
        except Exception:
            pass

    def chat(self, messages, temperature=0.6, schema=None, num_predict=200) -> str:
        payload: Dict[str, Any] = {"model": self.model, "messages": messages, "stream": False,
                                   "options": {"temperature": temperature, "num_predict": num_predict, "num_ctx": 4096}}
        if schema is not None:
            payload["format"] = schema
        def post(pl):
            req = urllib.request.Request(self.url + "/api/chat", data=json.dumps(pl).encode(),
                                         headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                d = json.loads(r.read().decode())
                self.tokens["prompt"] += d.get("prompt_eval_count", 0)
                self.tokens["completion"] += d.get("eval_count", 0)
                return d.get("message", {}).get("content", "").strip()
        try:
            return post(payload)
        except urllib.error.HTTPError as e:
            if schema is not None and e.code in (400, 422):          # older Ollama: plain JSON mode
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


def build_messages(i: int, n: int, history: List[dict]) -> List[dict]:
    """Quantized feedback only: outcome + dAUC rounded to 3 decimals. No absolute AUCs, no ceiling, no priors."""
    ops = {o: 0 for o in ALL_OPS}
    for h in history:
        ops[h["spec"]["op"]] = ops.get(h["spec"]["op"], 0) + 1
    least = min(ops, key=lambda k: ops[k])
    acc = [h for h in history if h["status"] == "accepted"]
    lines = [f"Proposal {i} of {n}.", "Accepted features so far (do not repeat): "
             + (", ".join(json.dumps(h["spec"], sort_keys=True) for h in acc[-8:]) if acc else "none yet")]
    if history:
        lines.append("Recent results:")
        for h in history[-10:]:
            lines.append(f"  {json.dumps(h['spec'], sort_keys=True)} -> {fmt_delta(h)}")
    lines.append(f"Operator '{least}' is the least explored so far. Combine different key sets. Output ONE new JSON spec.")
    return [{"role": "system", "content": GRAMMAR_PROMPT}, {"role": "user", "content": "\n".join(lines)}]


# =============================================================================
# 7. THE LAB: one (arm, run) job
# =============================================================================
class Lab:
    def __init__(self, W: Dict[str, np.ndarray]):
        t0 = time.time()
        self.W = W
        self.Wp = {k: v[(W["c0"] % F["proxy_mod"]) == 0] for k, v in W.items()}
        self.full = Track(W, F["neg_train"], F["neg_eval"])
        self.prox = Track(self.Wp, F["proxy_neg_train"], 1.0)
        log(f"tracks ready in {time.time() - t0:.0f}s | full train/ES/eval {[len(i) for i in self.full.idx]} "
            f"(eval pos {int(self.full.yev.sum()):,}) | proxy {[len(i) for i in self.prox.idx]} (eval pos {int(self.prox.yev.sum()):,})")

    # -- stack state ------------------------------------------------------------------------------------------------
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

    # -- one proposal ------------------------------------------------------------------------------------------------
    def evaluate(self, spec: dict, is_decoy: bool, dseed: int, gate: str, lord: Lord, n_tests: int) -> dict:
        t0 = time.time()
        ap, fin = self.columns(spec, is_decoy, dseed)
        mp = self.prox.col(ap)
        del ap
        rp = self.prox.fit([s["prox"] for s in self.stack] + [mp], self.ref_seed)
        d_proxy = rp["auc"] - self.ref["prox_auc"]
        rec = dict(finite_frac=round(fin, 4), d_proxy=round(d_proxy, 6))
        if d_proxy < F["proxy_theta"]:
            rec.update(status="screened", d_full=None, p=None, alpha_t=None, secs=round(time.time() - t0, 1))
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
        rec.update(status="accepted" if accept else "rejected", d_full=round(d_full, 6), p=round(p, 6),
                   alpha_t=round(alpha_t, 6), t_test=t_idx, cand_seed=self.ref_seed,
                   blocks_pos=int(np.sum(blk[np.isfinite(blk)] > 0)), cand_auc=round(rc["auc"], 6))
        if accept:
            self.add_to_stack(spec, is_decoy, dseed, mf, mp)
            if gate == "naive":                                       # ratchet: keep the (lucky) candidate as the new reference
                self.ref.update(full_pred=rc["pred"], full_auc=rc["auc"], prox_auc=rp["auc"])
            else:
                self.refit(self.ref_seed + 1)                         # fresh seed: no winner's-curse reference
        rec["secs"] = round(time.time() - t0, 1)
        return rec


def replay(lab: Lab, recs: List[dict], gate: str, lord: Lord) -> int:
    """Rebuild stack/reference/FDR state from the trial log after a restart. Returns the number of full tests."""
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


def run_job(lab: Lab, arm: str, run: int, n: int, work: Path, llm: Optional[Ollama], mock: bool, deadline: float) -> bool:
    gate = "naive" if arm in ("L-naive", "R-naive") else "fdr"
    tag = f"{arm}_{run}"
    path = work / f"trials_{tag}.jsonl"
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []
    valid = [r for r in recs if r.get("status") in ("screened", "accepted", "rejected")]
    done_flag = work / f"final_{tag}.json"
    if done_flag.exists():
        log(f"{tag}: already finished")
        return True
    lord = Lord(F["fdr_alpha"])
    n_tests = replay(lab, valid, gate, lord)
    log(f"=== {tag} | gate={gate} | resumed {len(valid)}/{n} proposals, stack {len(lab.stack)} ===")
    rng_prop = np.random.RandomState(7919 * run + ARMS.index("R" if arm == "R-naive" else arm))
    for _ in range(len(valid)):                                       # advance proposer RNG consistently on resume
        sample_spec(rng_prop)
    decoy_mask = np.random.RandomState(104729 * run + 13).rand(n) < F["decoy_frac"]
    seen = {(r["id"], r["is_decoy"]) for r in valid}
    fail_streak = 0
    t_job = time.time()
    n_dup = 0
    while len(valid) < n:
        if time.time() > deadline:
            log(f"{tag}: wall-clock guard reached; progress saved ({len(valid)}/{n}). Re-run to resume.")
            return False
        i = len(valid) + 1
        is_decoy = bool(decoy_mask[i - 1])
        n_dup = 0
        spec, why = None, ""
        if arm in ("R", "R-naive") or mock:
            for _ in range(50):
                cand = sample_spec(rng_prop)
                if (spec_id(cand), is_decoy) not in seen:
                    spec = cand
                    break
        else:
            msgs = build_messages(i, n, valid)
            for attempt in range(MAX_ATTEMPTS):
                try:
                    raw = llm.chat(msgs, temperature=min(TEMP_CAP, 0.6 + 0.15 * attempt), schema=SPEC_SCHEMA)
                    ok, why, clean = normalize(parse_json(raw))
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
                        extra = (" " + json.dumps(hint, sort_keys=True) + " was already tried. Choose a different key set or a different operator. Already tried with this operator: "
                                 + "; ".join(json.dumps(t, sort_keys=True) for t in tried) + ".")
                    msgs = msgs + [{"role": "assistant", "content": raw},
                                   {"role": "user", "content": f"Rejected ({why}).{extra} Output ONE valid, NEW JSON spec."}]
                except Exception as e:
                    why = repr(e)[:120]
        if spec is None:
            fail_streak += 1
            with path.open("a") as fh:
                fh.write(json.dumps(dict(status="proposal_failed", i=i, reason=why)) + "\n")
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
        rec.update(arm=arm, run=run, i=i, id=sid, spec=spec, is_decoy=is_decoy, dseed=dseed, stack_size=len(lab.stack), n_dup=n_dup)
        with path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        valid.append(rec)
        seen.add((sid, is_decoy))
        log(f"{tag} #{i}/{n} {'DECOY ' if is_decoy else ''}{rec['status'].upper():9s} d_proxy={rec['d_proxy']:+.5f} "
            f"d_full={rec['d_full'] if rec['d_full'] is not None else float('nan'):+.5f} p={rec['p']} "
            f"a_t={rec['alpha_t']} | stack {len(lab.stack)} | {rec['secs']}s | {json.dumps(spec, sort_keys=True)[:90]}")
        gc.collect()

    # -- final, selection-free re-measurement of the DEV stack (fresh seeds) -----------------------------------------
    fresh = [lab.full.fit([s["full"] for s in lab.stack], 100 + k)["auc"] for k in range(3)]
    b0 = [lab.full.fit([], 100 + k)["auc"] for k in range(3)]
    base_m, final = float(np.mean(b0)), float(np.mean(fresh))
    tested = [r for r in valid if r["status"] in ("accepted", "rejected")]
    acc = [r for r in valid if r["status"] == "accepted"]
    dec_t = [r for r in tested if r["is_decoy"]]
    out = dict(arm=arm, run=run, n_proposals=len(valid), n_screened=sum(r["status"] == "screened" for r in valid),
               n_full_tests=len(tested), accepted_real=sum(not r["is_decoy"] for r in acc),
               accepted_decoys=sum(r["is_decoy"] for r in acc), decoys_tested=len(dec_t),
               decoy_fpr=(sum(r["status"] == "accepted" for r in dec_t) / len(dec_t)) if dec_t else None,
               dev_baseline_remeasured=base_m, dev_final_auc=final,
               dev_gap_closed_pct=100.0 * (final - base_m) / (F["dev_ceiling"] - base_m),
               model=(llm.model if llm else "random"),
               decoys_proposed=sum(r["is_decoy"] for r in valid),
               decoys_screened=sum(bool(r["is_decoy"]) and r["status"] == "screened" for r in valid),
               n_dup_total=sum(r.get("n_dup", 0) for r in valid),
               tokens=(dict(llm.tokens) if llm else None),
               accepted=[dict(spec=r["spec"], is_decoy=r["is_decoy"], dseed=r["dseed"], iter=r["i"]) for r in acc],
               minutes=round((time.time() - t_job) / 60, 1))
    done_flag.write_text(json.dumps(out, indent=2))
    log(f"{tag} FINAL: accepted real {out['accepted_real']} / decoys {out['accepted_decoys']} (of {out['decoys_tested']} tested) "
        f"| DEV AUC {base_m:.5f} -> {final:.5f} | {out['dev_gap_closed_pct']:.1f}% of gap")
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
    for c in ("decoys_proposed", "n_dup_total"):
        if c not in df:
            df[c] = 0
    df.drop(columns=["accepted"]).to_csv(work / "summary_runs.csv", index=False)
    g = df.groupby("arm").agg(runs=("run", "count"), gap_mean=("dev_gap_closed_pct", "mean"), gap_min=("dev_gap_closed_pct", "min"),
                              gap_max=("dev_gap_closed_pct", "max"), real_acc=("accepted_real", "mean"),
                              decoy_acc=("accepted_decoys", "mean"), decoys_tested=("decoys_tested", "sum"),
                              full_tests=("n_full_tests", "mean"),
                              decoys_proposed=("decoys_proposed", "sum"),
                              dup_total=("n_dup_total", "sum"))
    g["decoy_fpr"] = df.groupby("arm").apply(lambda d: d.accepted_decoys.sum() / max(d.decoys_tested.sum(), 1))
    try:
        txt = g.round(3).to_markdown()
    except Exception:
        txt = g.round(3).to_string()
    (work / "summary_arms.md").write_text(txt)
    print(g.round(3).to_string())


def main(argv=None) -> None:
    kaggle = Path("/kaggle/working").exists()
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default="", help='e.g. "R:1-5" or "L-FDR:1-5" or "L-naive:1+3"')
    ap.add_argument("--n", type=int, default=60, help="evaluated proposals per run (pre-registered: 60)")
    ap.add_argument("--work-dir", default="/kaggle/working/af_lab" if kaggle else "./af_lab")
    ap.add_argument("--cache-dir", default="/tmp/af_cache" if kaggle else "./af_cache")
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    ap.add_argument("--start-ollama", action="store_true", default=kaggle)
    ap.add_argument("--install-ollama", action="store_true", default=kaggle)
    ap.add_argument("--mock-llm", action="store_true", help="LLM arms use the random sampler (smoke tests only)")
    ap.add_argument("--max-mins", type=float, default=700.0, help="wall-clock guard; resume by re-running")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a, _ = ap.parse_known_args(argv)                                   # tolerant of Jupyter's -f kernel.json
    work = Path(a.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    if a.report:
        return report(work)
    if a.dry_run:
        rng = np.random.RandomState(0)
        bad = normalize({"op": "count", "keys": ["zz"]})[0]
        ok = all(normalize(sample_spec(rng))[0] for _ in range(200))
        lord = Lord(F["fdr_alpha"])
        log(f"dry-run: grammar ok={ok}, invalid rejected={not bad}, LORD alpha_1={lord.alpha_t(1):.5f}, "
            f"min sign-flip p={signflip_p(np.ones(F['blocks'])):.6f}, materiality={F['materiality']:.6f}, proxy_theta={F['proxy_theta']}")
        return
    jobs_str = a.jobs or os.environ.get("AF_JOBS") or DEFAULT_JOBS
    jobs = parse_jobs(jobs_str) if jobs_str else []
    if not jobs:
        raise SystemExit('nothing to do: set DEFAULT_JOBS at the top of the file or pass --jobs "R:1-5"')
    log(f"jobs: {jobs_str} | n={a.n} | model={a.model}")
    llm = None
    if any(arm not in ("R", "R-naive") for arm, _ in jobs) and not a.mock_llm:
        llm = Ollama(a.ollama_url, a.model)
        llm.ensure_ready(a.start_ollama, a.install_ollama)
        probe = work / "contamination_probe_answer.json"
        if not probe.exists():
            log("contamination probe (log what the model thinks the data is) ...")
            ans = llm.chat([{"role": "user", "content": "Here is an anonymized click log with columns c0..c4 (categorical IDs) and t "
                             "(seconds), target = click converted. Which public dataset or competition is this? Answer briefly."}],
                           temperature=0.0, num_predict=200)
            probe.write_text(json.dumps(dict(model=a.model, answer=ans), indent=2))
            log(f"probe answer: {ans[:160]!r}")
    W = get_window(Path(a.cache_dir))
    lab = Lab(W)
    lab.reset()
    t0 = time.time()
    lab.refit(0)
    log(f"baseline (4 raw IDs) full-track AUC {lab.ref['full_auc']:.5f} (frozen Phase 0: {F['dev_baseline']:.5f})")
    if not TEST and abs(lab.ref["full_auc"] - F["dev_baseline"]) > 0.003:
        log("WARNING: baseline differs from the frozen Phase 0 value by > 0.003 - check the window/obfuscation before trusting results")
    deadline = t0 + 60 * a.max_mins
    for arm, run in jobs:
        if not run_job(lab, arm, run, a.n, work, llm, a.mock_llm, deadline):
            break
    report(work)






# ===================== PHASE 5b: do better gates pay off? (24-block exact LORD++, decoy-empirical-null gate) =====================
# Same proposals / IP subsamples / holdout as Phase 5 (deterministic), so results pair with results/p5 (naive, fdr12).
# Gates: naive | fdr (12 blocks) | fdr24 (24 blocks, exact meet-in-the-middle sign-flip) | decoyq (accept if the candidate's gain is in the
# top 5% of a rolling null of the last 40 decoy gains, with a 20-decoy warm-up at the empty stack; the null mixes stack states: approximation).
# Random proposals from the Phase 3 grammar; the SAME proposal sequence is run under the naive and the LORD++ gate (paired).
# Training size varies through an IP subsample (c0 % K == residue). No proxy screen (every proposal is a full test).
# Per run: DEV gain re-measured with fresh seeds vs the same final stack re-fitted on a holdout day (2017-11-08, already used
# for Phase 3 verdicts: NOT a fresh holdout; 2017-11-07 stays reserved). Accepted decoys are re-permuted on the holdout.
# Gate: Phase 3 gate (12 blocks, LORD++ alpha 0.10, materiality floor in absolute AUC units). With 12 blocks the gate cannot
# accept once alpha_t < 2^-12 (about test 8 without an early accept): reported as frac_tests_dead.
import argparse, json, math, os, time
from pathlib import Path

DEFAULT_CELLS = "2:60:4"
DEFAULT_GATES = "fdr24,decoyq"
HOLDOUT_DAY, HOLDOUT_START_ROW = "2017-11-08", 68_941_878
EVAL_SEEDS = (100, 101, 102)
DEAD_P = 2.0 ** -F["blocks"]
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
        self.stack, self.ref, self.ref_seed, self.null = [], {}, 0, []
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
        t = n_tests + 1
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
        return dict(status="accepted" if accept else ("observed" if gate == "observe" else "rejected"), d_full=round(float(d_full), 6), p=round(float(p), 6),
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


def make_warmup(run, K, n):
    rng = np.random.RandomState(2_000_003 * run + 7_919 * K + 5)
    out, seen, tries = [], set(), 0
    while len(out) < n and tries < 60 * n:
        tries += 1
        ok, _, ns = normalize(sample_spec(rng))
        if ok and canon5(ns) not in seen:
            seen.add(canon5(ns))
            out.append((ns, True, 6_000_000 + 10_000 * run + len(out)))
    return out


def run_one(lab, hold, props, gate, run, K, B):
    t0 = time.time()
    lab.reset()
    lord, log = Lord(F["fdr_alpha"]), []
    if gate == "decoyq":
        for spec, _, dseed in make_warmup(run, K, DQ_WARM):
            lab.evaluate(spec, True, dseed, "observe", lord, -1)
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
        decoys_tested=n_dec, frac_tests_dead=float(np.mean([r["alpha_t"] < (DEAD_P24 if gate == "fdr24" else DEAD_P) for r in log])) if gate in ("fdr", "fdr24") else float("nan"),
        null_n=len(lab.null), warmup=DQ_WARM if gate == "decoyq" else 0,
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
    out = Path(a.out or os.environ.get("AF_TEST_OUT", "") or ("/kaggle/working/p5b" if kaggle else "./p5b_out"))
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
