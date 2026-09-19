"""Compare massprobe broken vs clean (runs/nav4_wsivqa/massprobe/{broken,clean}.jsonl)."""
import json
import statistics as st

B = "/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs/nav4_wsivqa/massprobe/"
D = {s: [json.loads(l) for l in open(B + s + ".jsonl")] for s in ("broken", "clean")}


def m(xs):
    return round(st.mean(xs), 3)


for s, rs in D.items():
    jumps = []
    for r in rs:
        p = r["prefill0"]
        jumps.append(max(range(1, len(p)), key=lambda i, p=p: p[i]["c0_norm"] / max(p[i - 1]["c0_norm"], 1e-6)))
    c5 = m([r["prefill0"][5]["c0_norm"] for r in rs])
    c6 = m([r["prefill0"][6]["c0_norm"] for r in rs])
    dims = sorted(set(r["prefill0"][6]["c0_argdim"] for r in rs))
    print(f"== {s} n={len(rs)} sink jump layer set={sorted(set(jumps))} c0 norm L5 {c5} -> L6 {c6} argdim@L6 {dims}")
    keys = ["k_c0", "v_c0", "k_med", "v_med", "k_Plat", "v_Plat", "k_Nlat", "v_Nlat", "k_Rlat", "v_Rlat"]
    for L in (0, 1, 2, 6, 7, 10, 16, 35):
        print(f"  L{L:<2} " + " ".join(f"{k}={m([r['cache'][L][k] for r in rs])}" for k in keys))
    for role in ("evidence_planner", "navigator", "reasoner"):
        lat = [x for r in rs for x in r["latents"].get(role, [])[-10:]]
        ins = [x["in"] for x in lat]
        outs = [m([x["out"][l] for x in lat]) for l in (0, 6, 20, 35)]
        print(f"  {role[:9]:<9} latent in-norm={m(ins)} out-norm L0/L6/L20/L35={outs} (n={len(lat)})")
    print("  prompt-token layer-0 input norm median:",
          {k: m([r["pref_in_med"].get(k, 0) for r in rs]) for k in ("evidence_planner", "navigator", "reasoner")})
