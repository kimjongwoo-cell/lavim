import json, glob, sys, statistics as st
from collections import Counter
R = "/home/users/whddn12316/wsi_latent_0915_decode_hj/"
D = "/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda/"
ds = sys.argv[1] if len(sys.argv) > 1 else "tcga_expert_vqa"
recs = json.load(open(D + ds + ".json"))
rows = {}
for f in glob.glob(R + f"sender_relay_exp/runs/canon_diag/{ds}/diag_gpu*.jsonl"):
    for l in open(f):
        r = json.loads(l); rows[r["case"]] = r          # last write wins (Answerer retries)
n = lambda s: " ".join(str(s).casefold().split())
ARMS = ("native", "canonical", "canonical_t", "shift")
print(f"== {ds}: {len(rows)} cases")
ok = {a: {} for a in ARMS}
for c, r in rows.items():
    g = n(recs[c]["Answer"])
    for a in ARMS:
        if a in r["arms"]:
            ok[a][c] = n(r["arms"][a]["answer"]) == g
for a in ARMS:
    ch = sum(n(rows[c]["arms"][a]["answer"]) != n(rows[c]["arms"]["native"]["answer"]) for c in rows)
    rep = sum(ok[a][c] and not ok["native"][c] for c in rows); brk = sum(ok["native"][c] and not ok[a][c] for c in rows)
    print(f"  {a:12s} 정답(후보 argmax) {sum(ok[a].values()):3d}/{len(rows)} · native와 다른 답 {ch:3d} · 얻음 {rep:2d} · 잃음 {brk:2d}")
def lm(r, a, key, lo, hi):
    v = [x for x in r["arms"][a]["layers"][key][lo:hi + 1] if x is not None]
    return None if not v else sum(v) / len(v)
def med(vals):
    vals = [v for v in vals if v is not None]
    return None if not vals else round(st.median(vals), 4)
flip = [c for c in rows if n(rows[c]["arms"]["canonical"]["answer"]) != n(rows[c]["arms"]["native"]["answer"])]
same = [c for c in rows if c not in flip]
print(f"\n  canonical이 native 답을 바꾼 케이스 {len(flip)} / 안 바꾼 {len(same)}")
for lo, hi, tag in ((0, 17, "L0-17"), (18, 26, "L18-26"), (27, 35, "L27-35")):
    print(f"  [{tag}] 중앙값 (전체 케이스)")
    for key in ("rho_v", "rho_prompt", "rho_lat", "rho_rest", "crop_ent", "crop_max", "cell_corr"):
        print("    " + f"{key:10s} " + " · ".join(f"{a} {med([lm(rows[c], a, key, lo, hi) for c in rows])}" for a in ARMS))
print("\n  DLA (A=native 답, B=canonical 답 또는 native 2위) — 결정 행 첫 토큰이 다른 케이스만")
pairs = [c for c in rows if rows[c]["pair"]["tokens"]]
print(f"  pair 유효 {len(pairs)} (flip 중 {sum(c in flip for c in pairs)})")
for grp, cs in (("flip", [c for c in pairs if c in flip]), ("same", [c for c in pairs if c in same])):
    if not cs: continue
    for a in ARMS:
        prof = []
        for lo, hi in ((0, 17), (18, 26), (27, 35)):
            prof.append(med([sum(x for x in rows[c]["arms"][a]["layers"]["dla_v"][lo:hi + 1] if x is not None) for c in cs]))
        tot = med([sum(x for x in rows[c]["arms"][a]["layers"]["dla_v"] if x is not None) for c in cs])
        att = med([sum(x for x in rows[c]["arms"][a]["layers"]["dla_attn"] if x is not None) for c in cs])
        ld = med([rows[c]["arms"][a].get("logit_diff") for c in cs])
        print(f"    [{grp} n={len(cs)}] {a:12s} visual DLA 합(A−B) L0-17 {prof[0]} · L18-26 {prof[1]} · L27-35 {prof[2]} · 전층 {tot} | attn DLA 전층 {att} | 실제 logit(A)−logit(B) {ld}")
# where does canonical's visual mass go: per-layer rho_v profile native vs canonical (median over cases)
print("\n  층별 rho_v 중앙 (native / canonical):")
print("   " + " ".join(f"L{li}:{med([rows[c]['arms']['native']['layers']['rho_v'][li] for c in rows])}/{med([rows[c]['arms']['canonical']['layers']['rho_v'][li] for c in rows])}" for li in range(0, 36, 3)))
print("  층별 rho_prompt 중앙 (native / canonical):")
print("   " + " ".join(f"L{li}:{med([rows[c]['arms']['native']['layers']['rho_prompt'][li] for c in rows])}/{med([rows[c]['arms']['canonical']['layers']['rho_prompt'][li] for c in rows])}" for li in range(0, 36, 3)))
