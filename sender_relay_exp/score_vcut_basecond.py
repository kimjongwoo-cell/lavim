import json, glob, os, statistics as st, re, importlib.util
K="/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs/vcut"
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
def bacc(res):
    cls={}
    for pe,ge,ok in res.values(): cls.setdefault(ge,[]).append(ok)
    return st.mean(sum(v)/len(v) for v in cls.values()) if cls else float('nan')
def margins(arm):
    p=f"{K}/{arm}_ver.jsonl"; ms={}
    if not os.path.exists(p): return ms
    for line in open(p):
        j=json.loads(line); ci=j['case_index']; g=norm(DS[ci]['Answer'])
        rows={norm(r['y']):r['log_p_full'] for r in j['rows']}
        if g in rows: ms[ci]=rows[g]-max(v for k,v in rows.items() if k!=g)
    return ms
def mv(arm):
    v=[]
    for f in glob.glob(f"{K}/{arm}_access/*_terminal.json"):
        j=json.load(open(f))
        if j.get('steps'): v.append(j['steps'][0]['m_v'])
    return st.median(v)*100 if v else float('nan')
ref=load('vcut_id_basecond'); mref=margins('vcut_id_basecond')
print(f"{'arm':16s} {'n(common)':>9s} {'correct':>7s} {'id-correct':>10s} {'BACC':>6s} {'flip/id':>8s} {'M_V%':>6s} {'margin':>7s} {'id-margin':>9s}")
for a in ['vcut_id_basecond','vcut_zeroA_base','vcut_maskA_basecond','vcut_zeroRA_basecond']:
    t=load(a)
    if not t: continue
    common=[i for i in t if i in ref] if ref else list(t)
    ct=sum(t[i][2] for i in common); cr=sum(ref[i][2] for i in common) if ref else -1
    fl=sum(t[i][0]!=ref[i][0] for i in common) if ref else -1
    mg=margins(a); mc=[i for i in common if i in mg and i in mref]
    mm=st.median([mg[i] for i in mc]) if mc else float('nan'); mr=st.median([mref[i] for i in mc]) if mc else float('nan')
    print(f"{a:16s} {len(common):9d} {ct:7d} {cr:10d} {bacc({i:t[i] for i in common}):6.3f} {fl:5d}/{len(common):<2d} {mv(a):6.2f} {mm:7.2f} {mr:9.2f}")
