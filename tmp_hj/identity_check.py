import glob, json
R = "/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs"
def outs(pattern):
    res = {}
    for rp in glob.glob(f"{R}/{pattern}/**/result.json", recursive=True):
        d = rp[:-len("/result.json")]
        r = json.load(open(rp)); a = json.load(open(d + "/answerer_call.json"))
        res[int(r["dataset_index"])] = (r["answer"]["answer"], (a.get("final_outputs") or [""])[0])
    return res
ref = outs("c1sel_nav3/smoke_ntrs")
print("native NTRS ref (c1sel smoke):", {k: v[0] for k, v in ref.items()})
for name, pat in (("lpvh identity", "lpvh_ntrs_nav3/smoke/identity"), ("lpvh =1", "lpvh_ntrs_nav3/smoke/lpvh"),
                  ("srvw identity", "srvw_ntrs_nav3/smoke/identity"), ("srvw =1", "srvw_ntrs_nav3/smoke/srvw"),
                  ("other lpvh-only identity", "lpvh_nav3/native_obs/smoke_identity"), ("other lpvh-only =1", "lpvh_nav3/native_obs/smoke_lpvh")):
    o = outs(pat)
    same = {k: (o[k][0] == ref[k][0], o[k][1] == ref[k][1]) for k in o if k in ref}
    print(f"{name:26s} answers={ {k: v[0] for k, v in o.items()} } same(answer, full_text) vs NTRS ref={same}")
for f in sorted(glob.glob(f"{R}/c1sel_nav3/smoke_ntrs*.log") + glob.glob(f"{R}/c1sel_nav3/smoke_ntrs/*.log"))[:2]:
    print("ref log", f)
