"""Order 92 Phase A/B evaluation (runs/nav4_wsivqa/lmt/<arm>). usage: python lmt_eval.py [lmt_dir]

Per arm on the 18 broken questions: broken / correct (Order 91 decode_audit rule), crops identical to O0 / B0,
first-content-token (after '{"answer": "') JSD to O0 and B0 and the position t = JSD(X,O0) / (JSD(X,O0)+JSD(X,B0))
(0 = at O0, 1 = at B0), valid-choice margin (best choice-first-token logp - best other-token logp), decision-row
hidden cosine to O0, and Navigator / Reasoner latent K/V relative displacement from O0 (saved layers).
"""
import glob
import json
import os
import re
import sys

import torch

sys.path.insert(0, ".")
from vision_text_mas.prompts import canonical_open_answer_options

W = sys.argv[1] if len(sys.argv) > 1 else "sender_relay_exp/runs/nav4_wsivqa/lmt"
recs = json.load(open("sender_relay_exp/data_wsivqa/wsivqa.json"))
FOR = re.compile(r"[Ѐ-ӿ가-힣぀-ヿ一-鿿]")
REP = re.compile(r"(\b\S{1,20}\b)(?:[\s,\"']+\1){4,}")
IDS = [24, 65, 72, 76, 86, 149, 204, 211, 314, 354, 357, 362, 461, 522, 530, 650, 652, 683]
ORDER = ["O0", "OL", "OV", "OT", "OL_fix", "OLlate", "B0", "BL", "BV", "BT", "BL_fix", "BLlate"]
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/home/users/whddn12316/models/Qwen3-VL-4B-Thinking")


def choices(i):
    return recs[i].get("Choice") or list(canonical_open_answer_options(recs[i]["Question"]))


def load(arm):
    out = {}
    for f in glob.glob(f"{W}/{arm}/run/*/result.json") + glob.glob(f"{W}_redo/{arm}/run/*/result.json"):
        r = json.load(open(f))
        c = json.load(open(os.path.dirname(f) + "/answerer_call.json"))
        i = int(r["dataset_index"])
        a = r["answer"]
        pred = str(a.get("answer", "") if isinstance(a, dict) else a)
        raw = " ".join(c.get("final_outputs") or [])
        crops = tuple((p["magnification"], p["box"]["x"], p["box"]["y"]) for p in r["patches"])
        root = f.split("/run/")[0]
        pt = f"{root}/meas/{i:04d}.pt"
        out[i] = {"pred": pred, "raw": raw, "crops": crops,
                  "m": torch.load(pt) if os.path.exists(pt) else None}
    return out


def broken(i, p, raw):
    return (p not in choices(i)) or bool(FOR.search(raw)) or bool(REP.search(raw))


def jsd(a, b):
    p, q = a.float().exp(), b.float().exp()
    m = 0.5 * (p + q)
    kl = lambda x, lx: float((x * (lx - m.clamp_min(1e-30).log())).sum())
    return 0.5 * kl(p, a.float()) + 0.5 * kl(q, b.float())


def margin(i, lp):
    first = {tok(c, add_special_tokens=False)["input_ids"][0] for c in choices(i) if c}
    lp = lp.float()
    v = max(float(lp[t]) for t in first)
    mask = torch.ones_like(lp, dtype=torch.bool)
    mask[list(first)] = False
    return v - float(lp[mask].max())


def latdisp(x, o, role):
    if x is None or o is None or role not in x["lat"] or role not in o["lat"]:
        return float("nan")
    a, b = x["lat"][role].float(), o["lat"][role].float()
    return float((a - b).norm() / b.norm().clamp_min(1e-9))


arms = {a: load(a) for a in ORDER if os.path.isdir(f"{W}/{a}/run")}
O, B = arms.get("O0", {}), arms.get("B0", {})
rows = []
print(f"{'arm':8s} {'n':>3s} {'broken':>6s} {'correct':>7s} {'=O0crop':>7s} {'=B0crop':>7s} {'JSD_O':>6s} {'JSD_B':>6s} "
      f"{'t(0=O,1=B)':>10s} {'margin':>7s} {'hcosO':>6s} {'N_lat':>6s} {'R_lat':>6s}  broken_ids")
per_case = {}
for a, L in arms.items():
    ids = [i for i in IDS if i in L]
    b = [i for i in ids if broken(i, L[i]["pred"], L[i]["raw"])]
    ok = sum(L[i]["pred"].strip().lower() == recs[i]["Answer"].strip().lower() for i in ids)
    so = sum(1 for i in ids if i in O and O[i]["crops"] == L[i]["crops"])
    sb = sum(1 for i in ids if i in B and B[i]["crops"] == L[i]["crops"])
    jo, jb, ts, mg, hc, nd, rd = [], [], [], [], [], [], []
    for i in ids:
        m = L[i]["m"]
        if m is None:
            continue
        mg.append(margin(i, m["logprob"]))
        if i in O and O[i]["m"] is not None and i in B and B[i]["m"] is not None:
            x, y = jsd(m["logprob"], O[i]["m"]["logprob"]), jsd(m["logprob"], B[i]["m"]["logprob"])
            jo.append(x); jb.append(y); ts.append(x / (x + y) if x + y > 1e-9 else 0.5)
            hc.append(float(torch.nn.functional.cosine_similarity(m["hidden"].float(), O[i]["m"]["hidden"].float(), 0)))
            nd.append(latdisp(m, O[i]["m"], "navigator")); rd.append(latdisp(m, O[i]["m"], "reasoner"))
            per_case.setdefault(i, {})[a] = {"broken": i in b, "t": round(ts[-1], 3), "pred": L[i]["pred"][:40]}
    med = lambda v: float(torch.tensor(v).nanmedian()) if v else float("nan")
    print(f"{a:8s} {len(ids):3d} {len(b):6d} {ok:7d} {so:7d} {sb:7d} {med(jo):6.3f} {med(jb):6.3f} {med(ts):10.3f} "
          f"{med(mg):7.2f} {med(hc):6.3f} {med(nd):6.3f} {med(rd):6.3f}  {b}")
json.dump(per_case, open(f"{W}/per_case.json", "w"), ensure_ascii=False, indent=1)
print("\nper-case broken pattern (x = broken):")
print("case  " + " ".join(f"{a:>7s}" for a in arms))
for i in IDS:
    print(f"#{i:<4d} " + " ".join(f"{('x' if per_case.get(i, {}).get(a, {}).get('broken') else '.') if a in per_case.get(i, {}) else '-':>7s}" for a in arms))
