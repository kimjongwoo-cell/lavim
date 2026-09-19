import importlib.util, json, re, sys
from pathlib import Path
CODE = Path("/home/users/whddn12316/wsi_latent_0915_decode_hj/code")
DS_ROOT = Path("/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda")
Q = Path("/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs/expq")
BALANCED = {"gtex", "tcga", "panda"}
spec = importlib.util.spec_from_file_location("m", CODE / "eval/metrics.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
CASE_RE = re.compile(r"^(\d+)_")
DS_CACHE = {}

def load(ds):
    if ds not in DS_CACHE:
        DS_CACHE[ds] = json.loads((DS_ROOT / f"{ds}.json").read_text())
    return DS_CACHE[ds]

def collect(prefix):
    latest = {}
    for root in sorted(Q.glob(prefix + "*")):
        if not root.is_dir():
            continue
        for rj in root.glob("**/result.json"):
            mo = CASE_RE.match(rj.parent.name)
            if not mo: continue
            try: d = json.loads(rj.read_text())
            except Exception: continue
            idx = d.get("dataset_index", int(mo.group(1)))
            if not isinstance(idx, int): idx = int(mo.group(1))
            mt = rj.stat().st_mtime_ns
            if idx not in latest or mt > latest[idx][0]:
                latest[idx] = (mt, d)
    return {i: d for i, (_, d) in latest.items()}

def secs(d):
    calls = d.get("role_calls")
    if not isinstance(calls, list): return None
    tot = [c["elapsed_seconds"] for c in calls
           if isinstance(c, dict) and isinstance(c.get("elapsed_seconds"), (int, float))]
    return sum(tot) if tot else None

ARMS = [a.split("=") for a in sys.argv[1:]]
for arm, ds in ARMS:
    cases = collect(arm)
    if not cases:
        print(json.dumps({"arm": arm, "n": 0, "note": "no results"})); continue
    recs = load(ds); by_class = {}; ss = []
    for idx, r in cases.items():
        if not 0 <= idx < len(recs): continue
        rec = recs[idx]; ch = rec.get("Choice"); gt = rec.get("Answer")
        a = r.get("answer"); pred = a.get("answer") if isinstance(a, dict) else a
        s = secs(r)
        if s is not None: ss.append(s)
        if not (isinstance(ch, list) and isinstance(gt, str) and isinstance(pred, str)):
            continue
        pe = m.expand_letter(pred.strip(), ch); ge = m.expand_letter(gt.strip(), ch)
        by_class.setdefault(ge, []).append(int(bool(m.acc_of_seq(ch, ge, pe))))
    n = sum(len(v) for v in by_class.values())
    micro = sum(sum(v) for v in by_class.values()) / n if n else 0.0
    bacc = sum(sum(v)/len(v) for v in by_class.values()) / len(by_class) if by_class else 0.0
    print(json.dumps({"arm": arm, "ds": ds, "n": n,
                      "metric": round(bacc if ds in BALANCED else micro, 4),
                      "sec": round(sum(ss)/len(ss), 1) if ss else None}))
