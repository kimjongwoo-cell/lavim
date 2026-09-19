"""Latent tokens: per-layer max |x| dim and value on the sink dim (massprobe *_dims.jsonl)."""
import json, statistics as st, sys, collections
B = "/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs/nav4_wsivqa/massprobe/"
for name in sys.argv[1:]:
    rs = [json.loads(l) for l in open(B + name + ".jsonl")]
    print(f"== {name} n={len(rs)} sink dim(s) at L6: {sorted(set(r['prefill0'][6]['c0_argdim'] for r in rs))}")
    for role in ("evidence_planner", "navigator", "reasoner"):
        lat = [x for r in rs for x in r["latents"].get(role, [])[-10:] if "maxdim" in x]
        if not lat:
            continue
        for L in (0, 3, 5, 6, 7, 16, 35):
            dims = collections.Counter(x["maxdim"][L] for x in lat).most_common(3)
            ma = st.mean(x["maxabs"][L] for x in lat)
            d4 = st.mean(abs(x["d_sink"][L]) for x in lat)
            nrm = st.mean(x["out"][L] for x in lat)
            print(f"  {role[:9]:<9} L{L:<2} norm={nrm:8.1f} max|x|={ma:8.1f} top dims={dims} |x[sink dim]|={d4:8.1f}")
