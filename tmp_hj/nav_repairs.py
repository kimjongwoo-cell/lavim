"""nav4 Navigator format repairs / fallback across the 4B base run (and the 2B/8B smokes)."""
import collections
import glob
import json
import os

R = "/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs"
for tag, pat in (("4B EVQA", f"{R}/nav4/base/tcga_expert_vqa/**/result.json"),
                 ("4B SB", f"{R}/nav4/base/tcga_slidebench/**/result.json"),
                 ("2B smoke", f"{R}/nav4_2b/base/smoke_base/**/result.json"),
                 ("8B smoke", f"{R}/nav4_8b/base/smoke_base/**/result.json")):
    reps, calls, secs, fb, keys = collections.Counter(), collections.Counter(), [], 0, collections.Counter()
    n = 0
    for res in glob.glob(pat, recursive=True):
        d = os.path.dirname(res)
        f = os.path.join(d, "round_1/navigator_calls.json")
        if not os.path.exists(f):
            continue
        j = json.load(open(f))
        it = j[0] if isinstance(j, list) else j
        reps[it.get("format_repairs")] += 1
        calls[it.get("physical_calls")] += 1
        secs.append(float(it.get("elapsed_seconds") or 0))
        outs = it.get("final_outputs") or []
        if outs:
            keys["x5_ids" in str(outs[-1])] += 1
        t = os.path.join(d, "round_1/selection_traces.json")
        if os.path.exists(t):
            s = json.dumps(json.load(open(t)))
            if "fallback" in s.lower():
                fb += 1
        n += 1
    if n:
        print(f"{tag:9s} n={n} format_repairs={dict(reps)} physical_calls={dict(calls)} "
              f"nav_sec_mean={sum(secs)/n:.1f} last_output_has_x5_ids={dict(keys)} traces_mention_fallback={fb}")
# one 4B trace to see what 'fallback' looks like
ex = sorted(glob.glob(f"{R}/nav4/base/tcga_expert_vqa/**/round_1/selection_traces.json", recursive=True))[:1]
for t in ex:
    print("example trace:", json.dumps(json.load(open(t)))[:600])
