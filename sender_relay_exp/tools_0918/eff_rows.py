"""Board EFFROWS fields for one arm/dataset directory."""
import glob, json, os, sys
from statistics import mean
from pathlib import Path
root, ds = sys.argv[1], sys.argv[2]
rows = []
for f in glob.glob(f"{root}/{ds}/gpu*_attempt_*/**/efficiency.json", recursive=True):
    p = Path(f).parent
    try:
        r = json.load(open(p / "result.json")); e = json.load(open(f))
    except Exception:
        continue
    t = {c["role"]: float(c.get("elapsed_seconds") or 0) for c in r.get("role_calls", [])}
    rows.append({"tp": t.get("evidence_planner", 0), "tn": t.get("navigator", 0), "tr": t.get("reasoner", 0),
                 "ta": t.get("answerer", 0), "sec": sum(t.values()),
                 "pre": e["prefill_tokens_total"], "dec": e["decode_steps_total"],
                 "fl": e["flops_total"] / 1e12, "flpre": e["flops_prefill"] / 1e12,
                 "gpu": e["peak_gpu_mem_bytes"] / 2**30})
m = lambda k: round(mean(x[k] for x in rows), 2)
print(json.dumps({"n": len(rows), "sec": m("sec"), "tp": m("tp"), "tn": m("tn"), "tr": m("tr"), "ta": m("ta"),
                  "prefill": m("pre"), "dec": m("dec"), "fl": m("fl"), "flpre": m("flpre"), "gpu": m("gpu")}))
