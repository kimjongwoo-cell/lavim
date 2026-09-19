"""WSI-VQA efficiency table, same fields as the MultiPathQA board (sender_relay_exp/eff_report.case_metrics).
Pipeline arms: latest case dir per dataset_index under runs/<root>/<arm>/wsivqa, optional id file to restrict.
Single: runs/single_wsivqa/qwen3-vl-4b/<run>/efficiency.jsonl (tmp_hj/single_eff_hooks) + predictions timing.
usage: wsivqa_eff.py  (prints JSON {arm: {...means...}})"""
import ast, glob, json, os, statistics as st, sys, types
from pathlib import Path

R = Path("/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp")
src = (R / "eff_report.py").read_text()
tree = ast.parse(src)
tree.body = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.Assign))]
ef = types.ModuleType("eff_defs"); exec(compile(tree, "eff_defs", "exec"), ef.__dict__)

def latest_cases(root):
    best = {}
    for f in glob.glob(f"{root}/gpu*_attempt_*/*/result.json"):
        try:
            i = int(json.load(open(f))["dataset_index"])
        except Exception:
            continue
        t = os.path.getmtime(f)
        if i not in best or t > best[i][0]:
            best[i] = (t, Path(f).parent)
    return {i: d for i, (t, d) in best.items()}

def walls(root):
    w = {}
    for f in glob.glob(f"{root}/gpu*_attempt_*/attempt_metrics.jsonl"):
        for l in open(f):
            try:
                d = json.loads(l)
                w[int(d["dataset_index"])] = float(d["wall_seconds"])
            except Exception:
                pass
    return w

def arm_eff(root, ids=None):
    cases = latest_cases(root)
    if ids is not None:
        cases = {i: d for i, d in cases.items() if i in ids}
    rows = [m for m in (ef.case_metrics(d) for d in cases.values()) if m]
    w = walls(root)
    keys = ["t_total", "t_planner", "t_nav", "t_reasoner", "t_answerer", "reasoner_tokens", "ctx_answer", "kv_answer_mb",
            "decode_tokens", "flops_total", "flops_pre_answer", "m_flops_total", "m_prefill_tokens", "m_decode_steps", "m_peak_gpu_gb"]
    out = {"n": len(rows)}
    for k in keys:
        v = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
        out[k] = round(st.mean(v), 3) if v else None
    wv = [w[i] for i in cases if i in w]
    out["wall"] = round(st.mean(wv), 2) if wv else None
    return out

ids100 = set(json.load(open(R / "runs/nav4_wsivqa/navnp/navnp100_ids.json")))
res = {}
for arm in ("latplan", "plipext", "plipnp"):
    res[f"{arm}|735"] = arm_eff(R / f"runs/nav4_wsivqa/{arm}/wsivqa")
for arm in ("navsel", "plipsel", "navnp", "plipnp", "latplan"):
    res[f"{arm}|100"] = arm_eff(R / f"runs/nav4_wsivqa/{arm}/wsivqa", ids100)
nc = R / "runs/nav4_wsivqa_nc/latplan/wsivqa"
if nc.exists():
    res["latplan_nc|open"] = arm_eff(nc)
for run in sys.argv[1:]:
    e = [json.loads(l) for l in open(R / f"runs/single_wsivqa/qwen3-vl-4b/{run}/efficiency.jsonl")]
    g = lambda k: round(st.mean(x[k] for x in e if x.get(k) is not None), 3)
    res[f"single|{run}"] = {"n": len(e), "wall": g("wall_sec"), "t_total": g("wall_sec"), "ttft": g("ttft_sec"),
                            "m_prefill_tokens": g("prefill_tokens"), "m_decode_steps": g("decode_steps"),
                            "flops_total": g("flops_total"), "kv_mb": g("kv_mb"), "m_peak_gpu_gb": round(g("peak_gpu_mem_bytes") / 2 ** 30, 2)}
print(json.dumps(res, ensure_ascii=False))
