import json, glob, statistics as st, time, os
K = "/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs/nav4_wsivqa/latplan/wsivqa"
done = set()
for p in glob.glob(K + "/gpu*_attempt_*/*/result.json"):
    try:
        done.add(int(json.load(open(p))["dataset_index"]))
    except Exception:
        pass
per_gpu = {}
for f in glob.glob(K + "/gpu*_attempt_20260918_17*/attempt_metrics.jsonl"):
    g = os.path.basename(os.path.dirname(f)).split("_")[0]
    for l in open(f):
        try:
            per_gpu.setdefault(g, []).append(json.loads(l)["wall_seconds"])
        except Exception:
            pass
allw = [x for v in per_gpu.values() for x in v]
for g, v in sorted(per_gpu.items()):
    print(g, "n", len(v), "mean", round(st.mean(v)), "median", round(st.median(v)), [round(x) for x in v])
left = 735 - len(done)
print("done", len(done), "/735  left", left)
for name, w in (("지금(공유)", allw), ("GPU6만(단독)", per_gpu.get("gpu6", []))):
    if len(w) >= 2:
        per = st.mean(w)
        print(f"{name}: per case {round(per)}s -> 3샤드 eta {round(left/3*per/3600,2)}h "
              f"({time.strftime('%m-%d %H:%M', time.localtime(time.time()+left/3*per))})")
