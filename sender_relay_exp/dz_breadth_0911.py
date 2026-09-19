import json, re, sys, statistics as st
K="/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs/vcut_breadth"
tag=sys.argv[1]; DS=json.load(open(f"/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/smoke/{tag}24.json"))
def norm(s): return re.sub(r'[^a-z0-9]','',str(s).lower())
def rows(arm):
    out={}
    for l in open(f"{K}/{arm}_ver.jsonl"):
        j=json.loads(l); out[j['case_index']]={norm(r['y']):r['log_p_full'] for r in j['rows']}
    return out
A=rows(f'{tag}_id')
for cut in (f'{tag}_zeroA', f'{tag}_zeroA_L12_23', f'{tag}_zeroR'):
    B=rows(cut); dg=[]; dc=[]; mean=[]; disc=[]; rank=[]
    for ci in sorted(A):
        if ci not in B: continue
        g=norm(DS[ci]['Answer']); ks=[k for k in A[ci] if k in B[ci]]
        if g not in ks or len(ks)<2: continue
        dz={k:A[ci][k]-B[ci][k] for k in ks}; comp=max((k for k in ks if k!=g), key=lambda k:A[ci][k])
        dg.append(dz[g]); dc.append(dz[comp]); mean.append(st.mean(dz.values())); others=[dz[k] for k in ks if k!=g]
        disc.append(dz[g]-st.mean(others)); rank.append(1+sum(v>dz[g] for v in others))
    n=len(dg)
    print(f"{cut:20s} n={n} Δz_gold med {st.median(dg):+.2f} | Δz_comp med {st.median(dc):+.2f} | mean(all cand) {st.median(mean):+.2f} | disc*(gold−mean others) med {st.median(disc):+.2f} (>+.5: {sum(x>.5 for x in disc)}, <−.5: {sum(x<-.5 for x in disc)}) | gold rank/{len(ks)} med {st.median(rank):.1f}, rank1 {sum(r==1 for r in rank)}/{n}")
