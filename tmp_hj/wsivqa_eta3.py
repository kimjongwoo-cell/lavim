import json, glob, statistics as st, time, os
R = "/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp"
K = R + "/runs/nav4_wsivqa/latplan"
plan = json.load(open(K + "/wsivqa_block_plan.json"))
done = set()
for p in glob.glob(K + "/wsivqa/gpu*_attempt_*/*/result.json"):
    try:
        done.add(int(json.load(open(p))["dataset_index"]))
    except Exception:
        pass
print(time.strftime("%H:%M"), "total done", len(done), "/735")
etas = []
for f in sorted(glob.glob(K + "/wsivqa/gpu*_attempt_20260918_2108*/attempt_metrics.jsonl")):
    w = [json.loads(l)["wall_seconds"] for l in open(f)]
    g = os.path.basename(os.path.dirname(f)).split("_")[0]
    el = time.time() - os.path.getmtime(os.path.dirname(f) + "/run_manifest.json") if os.path.exists(os.path.dirname(f) + "/run_manifest.json") else sum(w)
    s = next(k for k, v in plan.items() if any(i in v for i in []) ) if False else None
    print(g, "done", len(w), "mean", round(st.mean(w)) if w else None, "median", round(st.median(w)) if w else None,
          "elapsed/case", round(el / len(w)) if w else None)
for s, v in plan.items():
    left = [i for i in v if i not in done]
    print("shard", s, "left", len(left), "/", len(v))
