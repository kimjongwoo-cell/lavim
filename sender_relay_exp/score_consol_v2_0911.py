import json, sys
src=open("/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/rescore_dashboard_bacc.py").read().split("CELLS = []")[0]
ns={}; exec(compile(src,"rs","exec"),ns)
DS_ROOT=ns["DS_ROOT"]; RHJ=ns["RHJ"]
for key in ("tcga","tcga_slidebench","panda"):
    ns["DATASETS"][key]=json.loads((DS_ROOT/f"{key}.json").read_text())
BASE={"gtex":21.41,"tcga_expert_vqa":45.31,"tcga_slidebench":48.22,"tcga":7.33,"panda":18.26}
print("arm        set                   n  score   base      Δ")
for v in sys.argv[1:]:
    for ds in ("gtex","tcga_expert_vqa","tcga_slidebench","tcga","panda"):
        s=ns["score"](ds, ns["collect"]([RHJ/f"consol_v/{v}_{ds}"]))
        if s["n"]==0: continue
        val=s["bacc"] if ds in ("gtex","tcga","panda") else s["micro"]
        print(f"{v:10s} {ds:18s} {s["n"]:4d} {val*100:6.2f} {BASE[ds]:6.2f} {val*100-BASE[ds]:+6.2f}", flush=True)
