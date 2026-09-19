import json, glob, statistics as st, time, os
K = "/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs/nav4_wsivqa/latplan/wsivqa"
done, bad = set(), 0
for p in glob.glob(K + "/gpu*_attempt_*/*/result.json"):
    try:
        done.add(int(json.load(open(p))["dataset_index"]))
    except Exception:
        bad += 1
new, old = [], []
for f in glob.glob(K + "/gpu*_attempt_*/attempt_metrics.jsonl"):
    for l in open(f):
        try:
            d = json.loads(l)
        except Exception:
            continue
        (new if "20260918" in f else old).append(d.get("wall_seconds", 0))
print("unique done", len(done), "/735   bad/empty result.json", bad)
if old:
    print("어제 cases", len(old), "mean", round(st.mean(old)), "median", round(st.median(old)))
print("오늘 cases", len(new), [round(x) for x in new][:20])
per = st.mean(new) if len(new) >= 3 else (st.mean(old) if old else None)
if per:
    left = 735 - len(done)
    print("per case", round(per), "s | left", left, "| eta_h", round(left / 3 * per / 3600, 2),
          "| eta_at", time.strftime("%m-%d %H:%M", time.localtime(time.time() + left / 3 * per)))
