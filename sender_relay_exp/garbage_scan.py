import json, glob, re, sys, importlib.util
from collections import Counter
T = "/home/users/whddn12316/wsi_latent_0915_decode_hj/"
s = importlib.util.spec_from_file_location("rescore", T + "scripts/rescore.py"); rs = importlib.util.module_from_spec(s); sys.modules["rescore"] = rs; s.loader.exec_module(rs)
am = rs.am
from pathlib import Path
def scan(name, roots, ds, variant, exclude):
    dirs = rs.run_dirs(rs.resolve_roots(roots), variant, exclude)
    # need full payload -> re-collect latest result paths
    latest = {}
    for d in dirs:
        for p in d.glob("*/result.json"):
            try:
                r = json.loads(p.read_text())
            except Exception:
                continue
            if rs.dataset_from_slide(str(r.get("slide_id", ""))) != ds: continue
            i = int(r["dataset_index"]); m = p.stat().st_mtime
            if i not in latest or m > latest[i][0]: latest[i] = (m, r)
    recs = rs.load_records(ds)
    kinds = Counter(); ex = {}
    rat_loop = 0; rat_long = 0
    for i, (_, r) in latest.items():
        ans = r.get("answer", {}); a = str(ans.get("answer", "")) if isinstance(ans, dict) else str(ans)
        rat = str(ans.get("rationale", "")) if isinstance(ans, dict) else ""
        choices = recs[i]["Choice"] if 0 <= i < len(recs) else []
        na = am.normalize(a)
        if na in {am.normalize(c) for c in choices}: k = "보기와 일치"
        elif na == "": k = "빈 답"
        elif re.search(r"choice|option|\{|\}|answer|<|placeholder|string", na): k = "형식/자리표시자"
        elif any(am.normalize(c) in na for c in choices if am.normalize(c)): k = "보기 포함+군더더기"
        else: k = "보기 밖 텍스트"
        kinds[k] += 1
        if k != "보기와 일치" and k not in ex: ex[k] = a[:80]
        words = rat.split()
        if len(words) > 40:
            grams = Counter(tuple(words[j:j+4]) for j in range(len(words)-3))
            if grams and grams.most_common(1)[0][1] >= 4: rat_loop += 1
        if len(rat) > 1500: rat_long += 1
    n = len(latest)
    print(f"{name:28s} n={n:3d} " + " · ".join(f"{k} {v}" for k, v in kinds.most_common()) + f" | rationale 반복루프 {rat_loop} · 1500자↑ {rat_long}")
    for k, v in ex.items(): print(f"{'':30s}{k}: {v!r}")
R = "0912/runs/"
scan("그분 Base EVQA", [R+"nav2v2_base_expert_*"], "tcga_expert_vqa", "base", [])
scan("그분 Base GTEx", [R+"*nav2*"], "gtex", "base", ["prompt2"])
scan("그분 Base TCGA", [R+"*nav2*"], "tcga", "base", ["prompt2"])
scan("그분 Base PANDA", [R+"*nav2*"], "panda", "base", ["prompt2"])
scan("내 SB sdpa base (진행중)", ["sender_relay_exp/runs/sb_sdpa_base"], "tcga_slidebench", None, [])
scan("타세션 pruning_b+relay3 EVQA", ["sender_relay_exp/runs/pruning_b_reasoner_context_relay3_prompt1/tcga_expert_vqa"], "tcga_expert_vqa", None, [])
