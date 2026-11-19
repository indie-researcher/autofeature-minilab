# ==== PHASE 3 INSPECT (works with per-notebook downloaded files) ====
import inspect, json, sys, glob
from pathlib import Path

sys.path.insert(0, ".")
import af_phase1_R_naive as p12

print("== 1. p12 signatures ==")
for name in ["load_dev_window", "window_arrays", "Track", "compute",
             "make_decoy", "find_file", "to_seconds", "F", "SEED",
             "SKIP_MARGIN", "WINDOW_DAY", "DAY_START_ROW"]:
    obj = getattr(p12, name, None)
    if obj is None:
        print(f"{name}: NOT FOUND")
    elif inspect.isfunction(obj):
        print(f"def {name}{inspect.signature(obj)}")
    elif inspect.isclass(obj):
        print(f"class {name}:")
        for m, mo in inspect.getmembers(obj, inspect.isfunction):
            if not m.startswith("_"):
                print(f"    def {m}{inspect.signature(mo)}")
    else:
        print(f"{name}: {type(obj).__name__} = {str(obj)[:80]}")

print("\n== 2. Track.fit source (first 40 lines) ==")
print("\n".join(inspect.getsource(p12.Track.fit).splitlines()[:40]))

print("\n== 3. load_dev_window source (first 40 lines) ==")
print("\n".join(inspect.getsource(p12.load_dev_window).splitlines()[:40]))

print("\n== 4. final_*.json schema (first found) ==")
files = sorted(glob.glob("**/final_*.json", recursive=True))
if files:
    d = json.load(open(files[0]))
    for k, v in d.items():
        print(f"  {k}: {type(v).__name__} = {str(v)[:100]}")
else:
    print("No final_*.json found anywhere — upload one as a Kaggle dataset or place it in /kaggle/working.")

print("\n== 5. trials_*.jsonl first record keys ==")
tfiles = sorted(glob.glob("**/trials_*.jsonl", recursive=True))
if tfiles:
    rec = json.loads(open(tfiles[0]).readline())
    for k, v in rec.items():
        print(f"  {k}: {type(v).__name__} = {str(v)[:100]}")
else:
    print("No trials_*.jsonl found anywhere.")

print("\n== 6. spec_freeze.json SEALED block ==")
sf = sorted(glob.glob("**/spec_freeze.json", recursive=True))
if sf:
    s = json.load(open(sf[0]))
    print(json.dumps(s["results"]["SEALED"], indent=2))
else:
    print("spec_freeze.json not found")
