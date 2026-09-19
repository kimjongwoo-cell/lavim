import json, sys
from pathlib import Path
src=open("/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/rescore_dashboard_bacc.py").read().split("CELLS = []")[0]
ns={}; exec(compile(src,"rs","exec"),ns)
DS_ROOT=ns['DS_ROOT']; RHJ=ns['RHJ']
for key in ("tcga","tcga_slidebench","panda"):
    ns['DATASETS'][key]=json.loads((DS_ROOT/f"{key}.json").read_text())
BASE={"gtex":21.41,"tcga_expert_vqa":45.31,"tcga_slidebench":48.22,"tcga":7.33,"panda":18.26}
NAME={"pruning_v3":"Pruning V3","pruning_v2":"Pruning V2","pruning":"Pruning V1"}
rows=[]
print(f"{'arm':12s} {'set':18s} {'n':>4s} {'score':>6s} {'base':>6s} {'Δ':>6s}  (score=BACC gtex/tcga/panda, ACC evqa/sb)")
for v in ("pruning_v3","pruning_v2","pruning"):
    for ds in ("gtex","tcga_expert_vqa","tcga_slidebench","tcga","panda"):
        s=ns['score'](ds, ns['collect']([RHJ/f"prune_v/{v}_{ds}"]))
        if s["n"]==0: continue
        val=s["bacc"] if ds in ("gtex","tcga","panda") else s["micro"]
        print(f"{NAME[v]:12s} {ds:18s} {s['n']:4d} {val*100:6.2f} {BASE[ds]:6.2f} {val*100-BASE[ds]:+6.2f}")
        rows.append({"dataset":ds,"task":f"{NAME[v]} (latent base 위, 0911)","param":"4b","step":10,"n":s["n"],"acc":round(val,4),"sec":None})
json.dump(rows,open("/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs/prune_v/rows.json","w"),ensure_ascii=False,indent=1)
