import glob
import json
import os
import sys

R = "/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs"
ROLES = (("round_1/evidence_planner_call.json", "Planner"), ("round_1/navigator_calls.json", "Navigator"),
         ("round_1/reasoner_call.json", "Reasoner"), ("answerer_call.json", "Answerer"))
for s in ("2b", "8b"):
    for res in sorted(glob.glob(f"{R}/nav4_{s}_pb/base/smoke_base/**/result.json", recursive=True)):
        d = os.path.dirname(res)
        idx = json.load(open(res)).get("dataset_index")
        row = []
        for f, name in ROLES:
            j = json.load(open(os.path.join(d, f)))
            it = j[0] if isinstance(j, list) else j
            row.append("%s %.1fs rep%s calls%s" % (name, it.get("elapsed_seconds", 0), it.get("format_repairs"), it.get("physical_calls")))
        print(s, "idx", idx, " | ".join(row))
        j = json.load(open(os.path.join(d, "round_1/navigator_calls.json")))
        it = j[0] if isinstance(j, list) else j
        for i, o in enumerate(it.get("final_outputs") or []):
            print("   nav output", i, repr(str(o)[:200]))
