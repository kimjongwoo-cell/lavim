import json, re, statistics as st, sys, glob, importlib.util
K="/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs/vcut"
DS=json.load(open("/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/smoke/gtex40.json"))
def norm(s): return re.sub(r'[^a-z0-9]','',str(s).lower())
def rows(arm):
    out={}
    for l in open(f"{K}/{arm}_ver.jsonl"):
        j=json.loads(l); out[j['case_index']]={norm(r['y']):(r['y'],r['log_p_full']) for r in j['rows']}
    return out
def decoded(arm):
    out={}
    for f in glob.glob(f"{K}/{arm}/*/result.json"):
        d=json.load(open(f)); a=d['answer']; a=a.get('answer') if isinstance(a,dict) else a; out[d['dataset_index']]=norm(a)
    return out
cut=sys.argv[1] if len(sys.argv)>1 else 'vcut_zeroA'
A=rows('vcut_id'); B=rows(cut); dec=decoded(cut)
print(f"### Δz_k = logP_identity(k) − logP_{cut}(k), 20 gtex classes, per case")
print(f"{'ci':>2s} {'gold':14s} {'Δz*':>6s} {'Δz_comp':>7s} {'Δz_ŷcut':>7s} {'meanΔz':>6s} {'sdΔz':>5s} {'disc*':>6s} {'rank*':>5s} {'argmaxΔz':14s} {'ŷ_id':12s} {'ŷ_cut':12s}")
D=[]; 
for ci in sorted(A):
    if ci not in B: continue
    g=norm(DS[ci]['Answer']); ks=[k for k in A[ci] if k in B[ci]]
    dz={k:A[ci][k][1]-B[ci][k][1] for k in ks}
    top_id=max(ks,key=lambda k:A[ci][k][1]); comp=max((k for k in ks if k!=g),key=lambda k:A[ci][k][1])
    top_cut=max(ks,key=lambda k:B[ci][k][1]); yc=dec.get(ci,top_cut); yc=yc if yc in dz else top_cut
    mean=st.mean(dz.values()); sd=st.pstdev(dz.values()); others=[dz[k] for k in ks if k!=g]
    disc=dz[g]-st.mean(others); rank=1+sum(v>dz[g] for v in others)
    am=max(ks,key=lambda k:dz[k])
    D.append(dict(ci=ci,g=g,dzg=dz[g],dzc=dz[comp],dzy=dz[yc],mean=mean,sd=sd,disc=disc,rank=rank,am=am,top_id=top_id,top_cut=top_cut,dz=dz))
    print(f"{ci:2d} {DS[ci]['Answer'][:14]:14s} {dz[g]:+6.2f} {dz[comp]:+7.2f} {dz[yc]:+7.2f} {mean:+6.2f} {sd:5.2f} {disc:+6.2f} {rank:5d} {A[ci][am][0][:14]:14s} {A[ci][top_id][0][:12]:12s} {B[ci][top_cut][0][:12]:12s}")
print()
print("### 요약")
print(f"Δz_gold median {st.median(d['dzg'] for d in D):+.2f} | Δz_competitor(identity top non-gold) median {st.median(d['dzc'] for d in D):+.2f} | Δz_ŷcut median {st.median(d['dzy'] for d in D):+.2f}")
print(f"mean over 20 classes median {st.median(d['mean'] for d in D):+.2f} (uniform lift) | within-case sd median {st.median(d['sd'] for d in D):.2f}")
print(f"disc* = Δz_gold − mean Δz_others: median {st.median(d['disc'] for d in D):+.2f}; >+0.5 in {sum(d['disc']>0.5 for d in D)}/40, <−0.5 in {sum(d['disc']<-0.5 for d in D)}/40")
print(f"rank of Δz_gold among 20 (1=visual pushed gold most): median {st.median(d['rank'] for d in D)}, rank1 {sum(d['rank']==1 for d in D)}/40, top3 {sum(d['rank']<=3 for d in D)}/40, bottom half {sum(d['rank']>10 for d in D)}/40")
xs=[d['dzg'] for d in D]; ys=[d['dzc'] for d in D]
mx,my=st.mean(xs),st.mean(ys); cov=sum((x-mx)*(y-my) for x,y in zip(xs,ys)); r=cov/((sum((x-mx)**2 for x in xs)*sum((y-my)**2 for y in ys))**0.5)
print(f"corr(Δz_gold, Δz_competitor) = {r:+.2f}")
ws=[d for d in D if d['dzg']-d['dzc']<-0.5]
print(f"\n### wrong-sign(Δm_A<−0.5) {len(ws)}건: argmax_k Δz_k와 상위3")
for d in ws:
    top3=sorted(d['dz'].items(), key=lambda kv:-kv[1])[:3]
    print(f"  ci {d['ci']:2d} gold={DS[d['ci']]['Answer'][:12]:12s} Δz_gold {d['dzg']:+.2f} | argmaxΔz = {A[d['ci']][d['am']][0][:12]:12s} | top3 {[(A[d['ci']][k][0][:10],round(v,2)) for k,v in top3]} | mean {d['mean']:+.2f} sd {d['sd']:.2f}")
# class-mean profile: which classes does the visual path push on average?
agg={}
for d in D:
    for k,v in d['dz'].items(): agg.setdefault(k,[]).append(v)
prof=sorted(((k,st.mean(v)) for k,v in agg.items()), key=lambda kv:-kv[1])
print("\n### 전 케이스 평균 Δz_k (visual path가 평균적으로 미는 class):")
print("  ", [(A[0][k][0][:10],round(v,2)) for k,v in prof])
