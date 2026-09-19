#!/usr/bin/env python3
"""READ-ONLY dump of per-item predictions for every run dir (MultiPathQA 5 sets).

Per run dir (a directory whose children are ^\\d+_ case dirs with result.json):
  newest result.json per dataset_index (mirrors rescore_dashboard_bacc.collect, but per dir),
  pred = answer['answer'] if dict else answer (raw, kept as-is; None if not str),
  dataset detected by result.slide_id == dataset[idx]['Id'].
Also records run_manifest.json key fields and log header hints.
Writes JSON lines to stdout. No file is modified.
"""
import json, os, re, sys

MP = "/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda"
DSK = ["gtex", "tcga_expert_vqa", "tcga_slidebench", "panda", "tcga"]
DS = {k: json.load(open(f"{MP}/{k}.json")) for k in DSK}
CASE_RE = re.compile(r"^(\d+)_")

roots = sys.argv[1:]


def manifest_info(d):
    p = os.path.join(d, "run_manifest.json")
    out = {}
    if os.path.isfile(p):
        try:
            m = json.load(open(p))
            for k in ("latent_steps", "patch_budget", "variant", "pruning", "reallocation", "max_model_len",
                      "answerer_protocol", "transport_mode", "navigator_control_tokens", "backbone", "model",
                      "model_path", "attn_implementation", "pipeline_mode"):
                if k in m:
                    out[k] = m[k]
            for k, v in m.items():
                if isinstance(v, str) and "Qwen" in v:
                    out.setdefault("model_str", v)
        except Exception as e:
            out["err"] = str(e)
    return out


def scan_rundir(d, case_dirs):
    latest = {}
    for cd in case_dirs:
        rj = os.path.join(d, cd, "result.json")
        try:
            mt = os.stat(rj).st_mtime_ns
            r = json.load(open(rj))
        except Exception:
            continue
        idx = r.get("dataset_index", int(CASE_RE.match(cd).group(1)))
        if not isinstance(idx, int):
            idx = int(CASE_RE.match(cd).group(1))
        prev = latest.get(idx)
        if prev is None or mt > prev[0]:
            latest[idx] = (mt, r)
    items = {}
    dscount = {}
    for idx, (mt, r) in latest.items():
        sid = r.get("slide_id")
        ds = None
        for k in DSK:
            if 0 <= idx < len(DS[k]) and DS[k][idx]["Id"] == sid:
                ds = k
                break
        if ds is None:
            # fallback: gold answer consistency by case dir key
            for k in DSK:
                if 0 <= idx < len(DS[k]) and str(sid).startswith(k + "__") and DS[k][idx]["Answer"] == r.get("gold_answer"):
                    ds = k
                    break
        dscount[ds] = dscount.get(ds, 0) + 1
        ans = r.get("answer")
        pred = ans.get("answer") if isinstance(ans, dict) else ans
        items[idx] = [pred if isinstance(pred, str) else None, ds, mt // 1_000_000_000, r.get("gold_answer")]
    return items, dscount


for root in roots:
    for d, dirs, files in os.walk(root):
        case_dirs = [x for x in dirs if CASE_RE.match(x) and os.path.isfile(os.path.join(d, x, "result.json"))]
        if case_dirs:
            items, dsc = scan_rundir(d, case_dirs)
            # do not descend into case dirs
            dirs[:] = [x for x in dirs if not CASE_RE.match(x)]
            if not items:
                continue
            ds = max(dsc, key=lambda k: dsc[k])
            if ds is None:
                continue
            print(json.dumps({"dir": d, "n": len(items), "ds": ds, "dscount": dsc,
                              "manifest": manifest_info(d), "items": items}), flush=True)
