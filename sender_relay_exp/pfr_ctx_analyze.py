#!/usr/bin/env python3
"""pfr_ctx 4-arm (A record / B identity / C matched / D shuffled) 분석 — 데이터셋별.

usage: pfr_ctx_analyze.py [dataset ...]   (기본: 결과가 있는 전 데이터셋)
- 채점: exact (eval/answer_match.py, gold_answer 직접)
- 첫 answer-content 결정 행: 각 arm 자신의 생성 텍스트를 토크나이즈해 처음으로 영숫자를 포함하는
  토큰 j를 찾고, 그것을 예측한 호출(call j; call 0 = prefill 마지막 행)의 통계를 쓴다.
"""
import glob, json, os, statistics as st, sys, warnings
from collections import Counter
warnings.filterwarnings("ignore")
import torch
T = "/home/users/whddn12316/wsi_latent_0915_decode_hj/"
sys.path.insert(0, T)
import importlib.util
spec = importlib.util.spec_from_file_location("answer_match", T + "eval/answer_match.py"); am = importlib.util.module_from_spec(spec); spec.loader.exec_module(am)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/home/users/whddn12316/models/Qwen3-VL-4B-Thinking")
K = T + "sender_relay_exp/runs/pfr_ctx/"
BAL = {"gtex", "tcga", "panda"}

def raw_text(case_dir):
    try:
        a = json.load(open(case_dir + "/answerer_call.json"))
    except Exception:
        return None
    fo = a.get("final_outputs")
    if isinstance(fo, list) and fo:
        fo = fo[-1]
    if isinstance(fo, dict):
        for k in ("text", "raw", "output", "content"):
            if isinstance(fo.get(k), str):
                return fo[k]
    return fo if isinstance(fo, str) else None

def load_arm(ds, arm):
    cases = {}
    for f in glob.glob(f"{K}{ds}/{arm}_attempt_*/*/result.json"):
        d = json.load(open(f)); i = int(d["dataset_index"]); cdir = os.path.dirname(f)
        m = os.path.getmtime(f)
        if i in cases and cases[i]["mtime"] > m: continue
        cases[i] = {"mtime": m, "gold": d["gold_answer"], "ans": str(d["answer"].get("answer", "")), "dir": cdir}
    for i, c in cases.items():
        p = f"{K}{ds}/{arm}.dump/case_{i:04d}.pt"
        c["stats"] = torch.load(p, map_location="cpu")["stats"] if os.path.exists(p) else []
        text = raw_text(c["dir"])
        c["dec"] = None
        if text is not None and c["stats"]:
            ids = tok(text, add_special_tokens=False)["input_ids"]
            j = next((n for n, t in enumerate(ids) if any(ch.isalnum() for ch in tok.decode([t]))), None)
            if j is not None and j < len(c["stats"]):
                c["dec"] = c["stats"][j]
    return cases

def score(ds, cases):
    ok = {i: am.normalize(c["ans"]) == am.normalize(c["gold"]) for i, c in cases.items()}
    if ds in BAL:
        by = {}
        for i, c in cases.items(): by.setdefault(am.normalize(c["gold"]), []).append(ok[i])
        s = 100 * st.mean(sum(v) / len(v) for v in by.values())
    else:
        s = 100 * sum(ok.values()) / max(len(ok), 1)
    return ok, s

def med(vals):
    vals = [v for v in vals if v is not None]
    return None if not vals else st.median(vals)

dss = sys.argv[1:] or [d for d in ("gtex", "tcga_expert_vqa", "tcga_slidebench", "tcga", "panda") if os.path.isdir(K + d)]
for ds in dss:
    arms = {a: load_arm(ds, a) for a in "ABCD"}
    if not arms["A"]: continue
    common = sorted(set.intersection(*(set(v) for v in arms.values() if v)))
    print(f"\n===== {ds}  (공통 케이스 {len(common)})")
    oks = {}
    for a in "ABCD":
        if not arms[a]: continue
        sub = {i: arms[a][i] for i in common}
        ok, s = score(ds, sub); oks[a] = ok
        ch = sum(am.normalize(sub[i]["ans"]) != am.normalize(arms["A"][i]["ans"]) for i in common)
        rep = sum(ok[i] and not oks["A"][i] for i in common) if a != "A" else 0
        brk = sum(oks["A"][i] and not ok[i] for i in common) if a != "A" else 0
        print(f"  {a}: 정답 {sum(ok.values()):2d}/{len(common)} ({'BAcc' if ds in BAL else 'Acc'} {s:5.2f}) · A와 다른 답 {ch} · 새로 맞힘 {rep} · 새로 틀림 {brk}")
    if "C" in oks and "D" in oks:
        repC = {i for i in common if oks["C"][i] and not oks["A"][i]}; brkC = {i for i in common if oks["A"][i] and not oks["C"][i]}
        repD = {i for i in common if oks["D"][i] and not oks["A"][i]}; brkD = {i for i in common if oks["A"][i] and not oks["D"][i]}
        sameCD = sum(am.normalize(arms["C"][i]["ans"]) == am.normalize(arms["D"][i]["ans"]) for i in common)
        print(f"  C vs D: 같은 답 {sameCD}/{len(common)} · repair C{sorted(repC)} D{sorted(repD)} · break C{sorted(brkC)} D{sorted(brkD)}")
    # decision-row statistics
    def dec(a, key): return [arms[a][i]["dec"].get(key) if arms[a][i]["dec"] else None for i in common]
    def pre(a, key): return [arms[a][i]["stats"][0].get(key) if arms[a][i]["stats"] else None for i in common]
    def span(a, key): return [st.mean([s[key] for s in arms[a][i]["stats"] if key in s]) if arms[a][i]["stats"] else None for i in common]
    ndec = sum(1 for i in common if arms["D"].get(i, {}).get("dec"))
    print(f"  첫 content 결정 행 찾음 D {ndec}/{len(common)}")
    for label, fn in (("첫 content 결정 행", dec), ("prefill 결정 행", pre), ("생성 구간 평균", span)):
        print(f"  [{label}] 중앙값  JS(nat,id)@B {med(fn('B','js_nat_id')):.2e} · JS(nat,m)@C {med(fn('C','js_nat_m')):.4f} · "
              f"JS(nat,m)@D {med(fn('D','js_nat_m')):.4f} · JS(nat,s)@D {med(fn('D','js_nat_s')):.4f} · JS(m,s)@D {med(fn('D','js_m_s')):.4f} · "
              f"‖r_s‖/‖r_m‖ {med(fn('D','rnorm_ratio')):.3f} · cos(q,q̂)@C {med(fn('C','cos_q_qC')):.4f} · ρ_V {med(fn('A','rho')):.4f}")
    d = dec("D", "js_m_s"); n_ = dec("D", "js_nat_m")
    frac = [a / b for a, b in zip(d, n_) if a is not None and b]
    print(f"  JS(m,s)/JS(nat,m) 결정 행 케이스별 중앙 {med(frac):.3f} · 범위 {min(frac):.3f}–{max(frac):.3f}" if frac else "")
