import json, os, re, statistics as st, importlib.util
K="/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs/vcut"
DS=json.load(open("/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/smoke/gtex40.json"))
spec=importlib.util.spec_from_file_location("mm","/home/users/whddn12316/wsi_latent_0915_decode_hj/code/eval/metrics.py"); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
import glob
def norm(s): return re.sub(r'[^a-z0-9]','',str(s).lower())
def margins(arm):
    out={}
    for line in open(f"{K}/{arm}_ver.jsonl"):
        j=json.loads(line); ci=j['case_index']; g=norm(DS[ci]['Answer'])
        rows={norm(r['y']):r['log_p_full'] for r in j['rows']}
        if g in rows:
            others={k:v for k,v in rows.items() if k!=g}
            out[ci]=(rows[g]-max(others.values()), rows[g], max(others,key=others.get))
    return out
def answers(arm):
    out={}
    for f in glob.glob(f"{K}/{arm}/*/result.json"):
        d=json.load(open(f)); ci=d['dataset_index']; a=d['answer']; a=a.get('answer') if isinstance(a,dict) else a
        ch=DS[ci]['Choice']; ge=m.expand_letter(DS[ci]['Answer'].strip(),ch); pe=m.expand_letter(str(a).strip(),ch)
        out[ci]=(pe,int(bool(m.acc_of_seq(ch,ge,pe))))
    return out
mid=margins('vcut_id'); aid=answers('vcut_id')
for arm in ['vcut_zeroA','vcut_zeroA_L0_11','vcut_zeroA_L12_23','vcut_zeroA_L24_35']:
    ma=margins(arm); aa=answers(arm)
    cs=sorted(i for i in mid if i in ma)
    d=[mid[i][0]-ma[i][0] for i in cs]          # Δm_A >0 : cut hurts (identity margin higher)
    dg=[mid[i][1]-ma[i][1] for i in cs]         # Δ logP(gold)
    q=lambda v,p: sorted(v)[int(p*(len(v)-1))]
    pos=sum(x>0.5 for x in d); neg=sum(x<-0.5 for x in d); zero=len(d)-pos-neg
    print(f"== {arm}  n={len(cs)}  Δm_A=m_id−m_cut: median {st.median(d):+.2f}  mean {st.mean(d):+.2f}  q10/q25/q75/q90 {q(d,.1):+.2f}/{q(d,.25):+.2f}/{q(d,.75):+.2f}/{q(d,.9):+.2f}")
    print(f"   cut hurts(Δ>0.5): {pos}  neutral(|Δ|≤0.5): {zero}  cut helps(Δ<−0.5): {neg}   | ΔlogP(gold) median {st.median(dg):+.2f}")
    # split by identity correctness and by flip
    for name,sub in (("id correct",[i for i in cs if aid[i][1]==1]),("id wrong",[i for i in cs if aid[i][1]==0]),
                     ("flipped",[i for i in cs if aa.get(i,('',0))[0]!=aid[i][0]]),("not flipped",[i for i in cs if aa.get(i,('',0))[0]==aid[i][0]])):
        if sub:
            dd=[mid[i][0]-ma[i][0] for i in sub]; print(f"   {name:12s} n={len(sub):2d} Δm_A median {st.median(dd):+.2f}  hurts {sum(x>0.5 for x in dd)} / helps {sum(x<-0.5 for x in dd)}")
    if False:
        print("   per-case (ci: m_id → m_cut, Δ, id_ans/cut_ans, gold):")
        for i in cs:
            print(f"     {i:2d}: {mid[i][0]:+6.2f} → {ma[i][0]:+6.2f}  Δ{mid[i][0]-ma[i][0]:+5.2f}  {aid[i][0][:14]:14s}/{aa.get(i,('?',0))[0][:14]:14s} gold={DS[i]['Answer'][:14]}")
