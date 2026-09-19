import json, glob, statistics as st, os
R = "/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp"
K = R + "/runs/nav4_wsivqa/latplan/wsivqa"
recs = json.load(open(R + "/data_wsivqa/wsivqa.json"))
rows = []
for f in glob.glob(K + "/gpu*_attempt_20260918_17*/attempt_metrics.jsonl"):
    prev = None
    for l in open(f):
        d = json.loads(l); i = int(d["dataset_index"]); sid = recs[i]["Id"]
        rows.append((d["wall_seconds"], d["role_call_seconds"], sid != prev, sid, i)); prev = sid
slow = [r for r in rows if r[0] > 150]; fast = [r for r in rows if r[0] <= 150]
print("n", len(rows))
for name, g in (("slow>150s", slow), ("fast<=150s", fast)):
    print(name, len(g), "| wall mean", round(st.mean(r[0] for r in g)), "| model role_call mean",
          round(st.mean(r[1] for r in g), 1), "| new-slide frac", round(sum(r[2] for r in g) / len(g), 2))
sz = {}
for r in slow[:8]:
    p = glob.glob(R + f"/data_wsivqa/slides/{r[3]}*DX1*.svs")
    print("slow", r[4], r[3], round(r[0]), "s  svs", round(os.path.getsize(p[0]) / 2**30, 2) if p else "?", "GB")
