import json, glob, os, re, sys, importlib.util
from collections import Counter
from pathlib import Path
T = "/home/users/whddn12316/wsi_latent_0915_decode_hj/"
s = importlib.util.spec_from_file_location("rescore", T + "scripts/rescore.py"); rs = importlib.util.module_from_spec(s); sys.modules["rescore"] = rs; s.loader.exec_module(rs)
am = rs.am
R = T + "sender_relay_exp/runs/"
D = "/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda/"
def load(pat):
    out = {}
    for f in glob.glob(pat):
        d = json.load(open(f)); i = d["dataset_index"]; m = os.path.getmtime(f)
        if i in out and out[i]["m"] > m: continue
        a = d["answer"] if isinstance(d["answer"], dict) else {}
        out[i] = {"m": m, "gold": d["gold_answer"], "ans": str(a.get("answer", "")), "rat": str(a.get("rationale", "")), "conf": a.get("confidence"), "dir": os.path.dirname(f)}
    return out
for ds in ("tcga_expert_vqa", "tcga_slidebench"):
    recs = json.load(open(D + ds + ".json"))
    M = load(R + f"canon_reread_nav3/{ds}/gpu*_attempt_*/*/result.json")
    B = load(R + f"nav3_prompt1_base_{ds}_gpu*/*/result.json")
    com = sorted(set(M) & set(B))
    print(f"\n===== {ds} 공통 {len(com)}")
    def kind(i, x):
        ch = {am.normalize(c) for c in recs[i]["Choice"]}
        na = am.normalize(x["ans"])
        if na in ch: return "보기"
        if re.fullmatch(r"(choice.*|text|\{|\}|)", na, re.S): return "형식실패"
        return "보기밖"
    for name, X in (("base", B), ("method", M)):
        kinds = Counter(kind(i, X[i]) for i in com)
        ok = sum(am.normalize(X[i]["ans"]) == am.normalize(X[i]["gold"]) for i in com)
        top = Counter(am.normalize(X[i]["ans"]) for i in com).most_common(5)
        ratlen = sorted(len(X[i]["rat"].split()) for i in com); 
        print(f"  {name:6s} 정답 {ok} · {dict(kinds)} · 답 상위 {top} · rationale 단어 중앙 {ratlen[len(ratlen)//2]}")
    # position of chosen option (index in Choice list)
    def pos(i, X):
        ch = [am.normalize(c) for c in recs[i]["Choice"]]
        a = am.normalize(X[i]["ans"]); return ch.index(a) if a in ch else None
    pb = Counter(pos(i, B) for i in com); pm = Counter(pos(i, M) for i in com); pg = Counter([ [am.normalize(c) for c in recs[i]["Choice"]].index(am.normalize(recs[i]["Answer"])) for i in com])
    print(f"  선택한 보기 번호 분포 base {sorted(pb.items(), key=lambda x: (x[0] is None, x[0]))} · method {sorted(pm.items(), key=lambda x: (x[0] is None, x[0]))} · 정답 번호 {sorted(pg.items())}")
    # changed answers: what did they change to
    brk = [i for i in com if am.normalize(B[i]["ans"]) == am.normalize(B[i]["gold"]) and am.normalize(M[i]["ans"]) != am.normalize(M[i]["gold"])]
    rep = [i for i in com if am.normalize(M[i]["ans"]) == am.normalize(M[i]["gold"]) and am.normalize(B[i]["ans"]) != am.normalize(B[i]["gold"])]
    print(f"  잃음 {len(brk)} 예:", [(i, B[i]['ans'][:25], "→", M[i]['ans'][:25]) for i in brk[:8]])
    print(f"  얻음 {len(rep)} 예:", [(i, B[i]['ans'][:25], "→", M[i]['ans'][:25]) for i in rep[:5]])
    # question type (Task field) breakdown of losses
    tasks = Counter(recs[i].get("Task") for i in brk); tasksR = Counter(recs[i].get("Task") for i in rep)
    print("  잃은 문항 Task:", tasks.most_common(6), "| 얻은 문항 Task:", tasksR.most_common(6))
    # number of choices for broken items
    print("  잃은 문항 보기 수:", Counter(len(recs[i]["Choice"]) for i in brk).most_common(), "| 전체:", Counter(len(recs[i]["Choice"]) for i in com).most_common())
    # did method rationale mention the base answer? confidence
    cb = [B[i]["conf"] for i in brk if isinstance(B[i]["conf"], (int,float))]; cm = [M[i]["conf"] for i in brk if isinstance(M[i]["conf"], (int,float))]
    if cb and cm: print(f"  잃은 문항 confidence 중앙 base {sorted(cb)[len(cb)//2]} method {sorted(cm)[len(cm)//2]}")
