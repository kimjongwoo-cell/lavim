import json, glob, os, re, sys, statistics as st, importlib.util
B="/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp"
spec=importlib.util.spec_from_file_location("mm","/home/users/whddn12316/wsi_latent_0915_decode_hj/code/eval/metrics.py"); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
def norm(s): return re.sub(r'[^a-z0-9]','',str(s).lower())
REF={'gtex':('runs/vcut','vcut_id','smoke/gtex40.json'),'evqa':('runs/vcut_breadth','evqa_id','smoke/evqa24.json'),'sb':('runs/vcut_breadth','sb_id','smoke/sb24.json')}
def answers(d,arm,DS):
    out={}
    for f in glob.glob(f"{B}/{d}/{arm}/*/result.json"):
        j=json.load(open(f)); ci=j['dataset_index']; a=j['answer']; a=a.get('answer') if isinstance(a,dict) else a
        ch=DS[ci]['Choice']; ge=m.expand_letter(DS[ci]['Answer'].strip(),ch); pe=m.expand_letter(str(a).strip(),ch)
        out[ci]=(pe,int(bool(m.acc_of_seq(ch,ge,pe))))
    return out
def rows(d,arm):
    out={}
    p=f"{B}/{d}/{arm}_ver.jsonl"
    if not os.path.exists(p): return out
    for l in open(p):
        j=json.loads(l); out[j['case_index']]={norm(r['y']):r['log_p_full'] for r in j['rows']}
    return out
for tag in ('gtex','evqa','sb'):
    d0,a0,ds=REF[tag]; DS=json.load(open(f"{B}/{ds}"))
    A=answers(d0,a0,DS); RA=rows(d0,a0)
    print(f"=== {tag} (10 cases; ref = identity/Full)")
    print(f"{'arm':10s} {'n':>2s} {'correct':>7s} {'ref':>3s} {'flip':>4s} {'margin':>7s} {'ref_m':>6s} {'ΔmA':>6s} {'hurt/help':>9s} {'Δz_gold':>7s} {'Δz_comp':>7s} {'disc*':>6s} {'goldrank':>8s} {'rank1':>5s}")
    for arm in ('context','token','random'):
        T=answers('runs/retr2',f'{tag}_{arm}',DS); RT=rows('runs/retr2',f'{tag}_{arm}')
        cs=sorted(i for i in T if i in A and i<10)
        if not cs: print(f"{arm:10s} (no results yet)"); continue
        ct=sum(T[i][1] for i in cs); cr=sum(A[i][1] for i in cs); fl=sum(T[i][0]!=A[i][0] for i in cs)
        mg=[];mr=[];dg=[];dc=[];disc=[];rank=[]
        for i in cs:
            if i not in RT or i not in RA: continue
            g=norm(DS[i]['Answer']); ks=[k for k in RA[i] if k in RT[i]]
            if g not in ks: continue
            mt=RT[i][g]-max(v for k,v in RT[i].items() if k!=g); mi=RA[i][g]-max(v for k,v in RA[i].items() if k!=g)
            mg.append(mt); mr.append(mi)
            dz={k:RA[i][k]-RT[i][k] for k in ks}; comp=max((k for k in ks if k!=g),key=lambda k:RA[i][k]); others=[dz[k] for k in ks if k!=g]
            dg.append(dz[g]); dc.append(dz[comp]); disc.append(dz[g]-st.mean(others)); rank.append(1+sum(v>dz[g] for v in others))
        dm=[a-b for a,b in zip(mr,mg)]
        f=lambda v: f"{st.median(v):+.2f}" if v else "  n/a"
        print(f"{arm:10s} {len(cs):2d} {ct:7d} {cr:3d} {fl:4d} {f(mg):>7s} {f(mr):>6s} {f(dm):>6s} {sum(x>0.5 for x in dm):4d}/{sum(x<-0.5 for x in dm):<4d} {f(dg):>7s} {f(dc):>7s} {f(disc):>6s} {(st.median(rank) if rank else float('nan')):8.1f} {sum(r==1 for r in rank):5d}")
    for arm in ('context','token','random'):
        p=f"{B}/runs/retr2/{tag}_{arm}.retr.jsonl"
        if os.path.exists(p):
            J=[json.loads(l) for l in open(p)]
            print(f"   {arm}: keep {[j['keep'] for j in J]} root {[j['root'] for j in J]} chosen {[j['chosen'] for j in J][:4]}...")
