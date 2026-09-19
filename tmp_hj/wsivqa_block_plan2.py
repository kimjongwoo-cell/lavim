"""0918: freeze a contiguous, slide-aligned 3-way split of the REMAINING WSI-VQA indices (so each slide is opened once).
Writes runs/nav4_wsivqa/latplan/wsivqa_block_plan.json {"0": [...], "1": [...], "2": [...]}; refuses to overwrite.
Robust to broken result.json (server died mid-write)."""
import glob, json, os, sys
R = "/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp"
ARM = sys.argv[2] if len(sys.argv) > 2 else "latplan"
K = R + "/runs/nav4_wsivqa/" + ARM
PLAN = K + "/wsivqa_block_plan.json"
NS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
if os.path.exists(PLAN):
    sys.exit(f"plan exists: {PLAN}")
recs = json.load(open(R + "/data_wsivqa/wsivqa.json"))
done = set()
for p in glob.glob(K + "/wsivqa/gpu*_attempt_*/*/result.json"):
    try:
        done.add(int(json.load(open(p))["dataset_index"]))
    except Exception:
        pass
rem = [i for i in range(len(recs)) if i not in done]
# group consecutive remaining indices by slide, then cut groups into NS contiguous runs of ~equal size
groups = []
for i in rem:
    if groups and recs[groups[-1][-1]]["Id"] == recs[i]["Id"]:
        groups[-1].append(i)
    else:
        groups.append([i])
target = len(rem) / NS
plan, cur, k = {str(s): [] for s in range(NS)}, 0, 0
for g in groups:
    if k < NS - 1 and len(plan[str(k)]) >= target * (1) and len(plan[str(k)]) > 0:
        k += 1
    plan[str(k)].extend(g)
json.dump(plan, open(PLAN, "w"))
print("done", len(done), "remaining", len(rem), "slide groups", len(groups))
for s, v in plan.items():
    print("shard", s, "n", len(v), "range", v[0], "-", v[-1], "slides", len({recs[i]["Id"] for i in v}))
