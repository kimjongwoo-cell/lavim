import json, glob, os, statistics as st, re, importlib.util
K="/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs/kill_ocva"
DS=json.load(open("/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/smoke/gtex40.json"))
spec=importlib.util.spec_from_file_location("mm","/home/users/whddn12316/wsi_latent_0915_decode_hj/code/eval/metrics.py"); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
def norm(s): return re.sub(r'[^a-z0-9]','',str(s).lower())
def load(arm):
    out={}
    for f in sorted(glob.glob(f"{K}/{arm}/*/result.json")):
        d=json.load(open(f)); ci=d.get('dataset_index', int(os.path.basename(os.path.dirname(f)).split('_')[0]))
        a=d.get('answer'); a=a.get('answer') if isinstance(a,dict) else a
        rec=DS[ci]; ch=rec['Choice']; ge=m.expand_letter(rec['Answer'].strip(),ch); pe=m.expand_letter(str(a).strip(),ch)
        out[ci]=(pe, ge, int(bool(m.acc_of_seq(ch,ge,pe))))
    return out
def acc(res):
    cls={}
    for pe,ge,ok in res.values(): cls.setdefault(ge,[]).append(ok)
    return sum(ok for _,_,ok in res.values()), len(res), st.mean(sum(v)/len(v) for v in cls.values())
def margin(arm):
    p=f"{K}/{arm}_ver.jsonl"; ms=[]
    if not os.path.exists(p): return None
    for line in open(p):
        j=json.loads(line); ci=j['case_index']; g=norm(DS[ci]['Answer'])
        rows={norm(r['y']):r['log_p_full'] for r in j['rows']}
        if g in rows:
            ms.append(rows[g]-max(v for k,v in rows.items() if k!=g))
    return (st.median(ms), len(ms)) if ms else None
def mv(arm):
    v=[json.load(open(f))['steps'][0]['m_v'] for f in glob.glob(f"{K}/{arm}_access/*_terminal.json") if json.load(open(f)).get('steps')]
    return (st.median(v)*100, len(v)) if v else None
arms=['base','restage','derope','canonical','canonicalt','canonicaltperm']
base=load('base_true'); print(f"{'arm':15s} {'true':>6s} {'BACC':>6s} {'wrong':>6s} {'flipT/W':>8s} {'flip/base':>9s} {'M_V%':>6s} {'margin':>7s} {'n_mg':>4s}")
for a in arms:
    t=load(f"{a}_true"); w=load(f"{a}_wrong"); ct,n,bt=acc(t); cw,nw,bw=acc(w)
    flip=sum(t[i][0]!=w[i][0] for i in t if i in w); fb=sum(t[i][0]!=base[i][0] for i in t if i in base)
    mvv=mv(f"{a}_true"); mg=margin(f"{a}_true")
    print(f"{a:15s} {ct:3d}/{n:<2d} {bt:6.3f} {cw:3d}/{nw:<2d} {flip:4d}/{nw:<2d} {fb:5d}/{n:<2d} {(mvv[0] if mvv else float('nan')):6.2f} {(mg[0] if mg else float('nan')):7.2f} {(mg[1] if mg else 0):4d}")
# offset variance from diag
ov={}
for f in glob.glob(f"{K}/base_true_diag/diag_*.json"):
    j=json.load(open(f))
    for k,v in j['summary'].items(): ov.setdefault(k,[]).append(v['offset_variance'])
print("offset_variance median:", {k:round(st.median(v),3) for k,v in ov.items()}, "n=",len(next(iter(ov.values()))))
