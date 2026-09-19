import json, statistics as st
R = "/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/"
rows = [json.loads(l) for l in open(R + "runs/vgain_nav3/vgain_gtex_20260914_165850.jsonl")]
ds = json.load(open(R + "smoke/dcs20_gtex.json"))
n = lambda s: " ".join(str(s).casefold().split())
def gm(x, r):
    lp = x.get("logp"); g = n(ds[r["case"]]["Answer"]); gi = [i for i, c in enumerate(r["cands"]) if n(c) == g]
    if not lp or not gi: return None
    return lp[gi[0]] - max(v for i, v in enumerate(lp) if i != gi[0])
def wm(x, r):  # native-winner margin
    lp = x.get("logp"); a = n(r["arms"]["ident"]["answer"]); ai = [i for i, c in enumerate(r["cands"]) if n(c) == a]
    if not lp or not ai: return None
    return lp[ai[0]] - max(v for i, v in enumerate(lp) if i != ai[0])
sub = [r for r in rows if not r["arms"]["W1"].get("skip")]
print("W 가능 케이스", len(sub), "| skip 사유:", [r["arms"]["W1"].get("skip") for r in rows if r["arms"]["W1"].get("skip")])
for g in ("0.5", "1.5", "2", "4"):
    for fam in ("V", "W", "N"):
        a = fam + g
        ch = sum(n(r["arms"][a]["answer"]) != n(r["arms"]["ident"]["answer"]) for r in sub if not r["arms"][a].get("skip"))
        d = [gm(r["arms"][a], r) - gm(r["arms"]["ident"], r) for r in sub if not r["arms"][a].get("skip") and gm(r["arms"][a], r) is not None]
        w = [wm(r["arms"][a], r) - wm(r["arms"]["ident"], r) for r in sub if not r["arms"][a].get("skip") and wm(r["arms"][a], r) is not None]
        ok = sum(n(r["arms"][a]["answer"]) == n(ds[r["case"]]["Answer"]) for r in sub if not r["arms"][a].get("skip"))
        print(f"  {a:5s} 같은 {len(sub)}케이스: 답 변경 {ch} · 정답 {ok} · gold margin Δ 중앙 {st.median(d):+.3f} · native-winner margin Δ 중앙 {st.median(w):+.3f}")
print("ident 정답(같은 케이스):", sum(n(r['arms']['ident']['answer']) == n(ds[r['case']]['Answer']) for r in sub))
print()
print("케이스별 winner margin(ident) 과 V 곡선 (logp 차: native winner − 2위):")
for r in rows:
    line = []
    for a in ("V0","ident","V1.5","V2","V4"):
        v = wm(r["arms"][a], r); line.append("nan" if v is None else f"{v:+.2f}")
    print(f"  case{r['case']:2d} gold={ds[r['case']]['Answer'][:10]:10s} ident={r['arms']['ident']['answer'][:10]:10s} V4={r['arms']['V4']['answer'][:10]:10s} margin V0/1/1.5/2/4: {' '.join(line)}")
