import json, sys, importlib.util
from pathlib import Path
sys.argv=['x']; spec=importlib.util.spec_from_file_location("rs","/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/rescore_dashboard_bacc.py")
src=open("/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/rescore_dashboard_bacc.py").read().split("CELLS = []")[0]
ns={}; exec(compile(src,"rs","exec"),ns)
DS_ROOT=ns['DS_ROOT']; RHJ=ns['RHJ']
for key,fn in (("tcga","tcga.json"),("tcga_slidebench","tcga_slidebench.json")):
    p=DS_ROOT/fn
    if p.exists(): ns['DATASETS'][key]=json.loads(p.read_text())
import glob
for p in glob.glob("/home/users/whddn12316/datasets/MultiPathQA/**/panda*.json", recursive=True)[:3]: print("panda ds candidate", p)
pp=["/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda/panda.json"]
if pp: ns['DATASETS']['panda']=json.loads(open(pp[0]).read())
arm=sys.argv[1] if len(sys.argv)>1 else "routecanon"
for ds,key in (("gtex10","gtex"),("evqa10","tcga_expert_vqa"),("sb10","tcga_slidebench"),("tcga10","tcga"),("panda10","panda")):
    if key not in ns['DATASETS']: print(ds,"no dataset json"); continue
    roots=[RHJ/f"expq/{arm}_{ds}.s{i}" for i in range(3)]
    print(ds, ns['score'](key, ns['collect'](roots)))
