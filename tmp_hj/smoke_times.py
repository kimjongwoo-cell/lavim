import glob, json
R = "/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs"
for name, pat in (("NTRS ref (c1sel)", "c1sel_nav3/smoke_ntrs"), ("LPVH native", "lpvh_ntrs_nav3/smoke/native"), ("LPVH identity", "lpvh_ntrs_nav3/smoke/identity"),
                  ("LPVH =1", "lpvh_ntrs_nav3/smoke/lpvh"), ("SRVW identity", "srvw_ntrs_nav3/smoke_FAIL_gate005_0915_1130/identity"),
                  ("SRVW =1", "srvw_ntrs_nav3/smoke_FAIL_gate005_0915_1130/srvw")):
    rows = []
    for rp in sorted(glob.glob(f"{R}/{pat}/**/result.json", recursive=True)):
        r = json.load(open(rp)); t = {}
        for rc in r["role_calls"]:
            t[rc["role"]] = t.get(rc["role"], 0) + float(rc.get("elapsed_seconds") or 0)
        a = json.load(open(rp[:-len("result.json")] + "answerer_call.json"))
        rows.append(f"#{r['dataset_index']} R={t.get('reasoner', 0):.1f}s A={t.get('answerer', 0):.1f}s out_chars={len((a.get('final_outputs') or [''])[0])}")
    gpu = sorted({p.split('gpu')[1][0] for p in glob.glob(f"{R}/{pat}/gpu*_attempt_*")})
    print(f"{name:18s} gpu{gpu} " + " | ".join(rows))
