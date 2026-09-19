"""Headroom analysis library (local, read-only on copied data).

Scorer = exact logic of sender_relay_exp/rescore_dashboard_bacc.py (== 0806 dashboard
multipath_quality_metrics): pred = answer['answer'] if dict else answer; skip non-str pred;
pe = expand_letter(pred.strip(), Choice); ge = expand_letter(Answer.strip(), Choice);
correct = bool(acc_of_seq(Choice, ge, pe)); class key = ge; gtex/tcga/panda -> BACC (macro over
gold classes present), others -> micro. Shard merge = newest mtime per dataset_index.
"""
import importlib.util, json, os, re, collections, math

HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("strict_metrics", os.path.join(HERE, "metrics_remote_copy.py"))
M = importlib.util.module_from_spec(_s); _s.loader.exec_module(M)
DSK = ["gtex", "tcga_expert_vqa", "tcga_slidebench", "panda", "tcga"]
DS = {k: json.load(open(os.path.join(HERE, "ds", f"{k}.json"))) for k in DSK}
BAL = {"gtex", "tcga", "panda"}
SHORT = {"gtex": "GTEx", "tcga_expert_vqa": "EVQA", "tcga_slidebench": "SB", "panda": "PANDA", "tcga": "TCGA"}


def load_dump(path=os.path.join(HERE, "dump.jsonl")):
    rows = []
    for line in open(path):
        rows.append(json.loads(line))
    return rows


def logical_name(d):
    d = re.sub(r"/gpu\d+$", "", d)
    d = re.sub(r"\.s\d+r?(\.killed)?$", "", d)
    base = os.path.basename(d)
    parent = os.path.dirname(d)
    if "/0912/runs" in d or re.search(r"_part\d+_\d+", base):
        base = re.sub(r"_part\d+_\d+", "", base)
        base = re.sub(r"_gpu\d+", "", base)
        base = re.sub(r"_worker\d+", "", base)
    return os.path.join(parent, base)


def merge(rows):
    """logical name -> {ds, dirs, items{idx:[pred, mtime]}} (newest mtime per idx)."""
    L = {}
    for r in rows:
        name = logical_name(r["dir"])
        key = (name, r["ds"])
        e = L.setdefault(key, {"name": name, "ds": r["ds"], "dirs": [], "items": {}, "manifest": r["manifest"]})
        e["dirs"].append(r["dir"])
        for idx, (pred, ds, mt, gold) in r["items"].items():
            if ds != r["ds"]:
                continue
            idx = int(idx)
            prev = e["items"].get(idx)
            if prev is None or mt > prev[1]:
                e["items"][idx] = [pred, mt]
    return L


def judge(ds, idx, pred):
    rec = DS[ds][idx]
    ch = rec["Choice"]; gt = rec["Answer"]
    ge = M.expand_letter(gt.strip(), ch)
    if not isinstance(pred, str):
        return None, ge, 0
    pe = M.expand_letter(pred.strip(), ch)
    return pe, ge, int(bool(M.acc_of_seq(ch, ge, pe)))


def per_item(e):
    ds = e["ds"]; out = {}
    for idx, (pred, mt) in e["items"].items():
        if 0 <= idx < len(DS[ds]):
            out[idx] = judge(ds, idx, pred)
    return out


def score_items(ds, pi, idxs=None):
    by = {}
    for idx, (pe, ge, c) in pi.items():
        if idxs is not None and idx not in idxs:
            continue
        if pe is None:  # canonical scorer skips non-str predictions
            continue
        by.setdefault(ge, []).append(c)
    n = sum(len(v) for v in by.values())
    if n == 0:
        return None, 0
    micro = sum(sum(v) for v in by.values()) / n
    bacc = sum(sum(v) / len(v) for v in by.values()) / len(by)
    return (bacc if ds in BAL else micro) * 100, n
