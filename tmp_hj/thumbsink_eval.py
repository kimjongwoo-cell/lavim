import json, statistics as st
W = "/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs/nav4_wsivqa/thumbsink"
G = ["sink", "P_img", "P_text", "P_lat", "N_img", "N_text", "N_lat", "R_img", "R_text", "R_lat", "A_prompt"]
def load(name):
    return [json.loads(l) for l in open(f"{W}/{name}.jsonl")]
def case_mean(rec, key, layers=None):
    ls = [s for s in rec["layers"] if layers is None or s["layer"] in layers]
    return st.mean(s.get(key, 0.0) for s in ls)
out = {}
for name in ("broken", "clean"):
    recs = load(name); out[name] = recs
    print(f"== {name}: n={len(recs)}  thumb tokens mean={st.mean(r['n_cols']['P_img'] for r in recs):.0f} "
          f"R_img tokens={st.mean(r['n_cols']['R_img'] for r in recs):.0f}")
    for band, ly in (("all", None), ("L0-11", range(0, 12)), ("L12-23", range(12, 24)), ("L24-35", range(24, 36))):
        row = " ".join(f"{g}={st.mean(case_mean(r, g, ly) for r in recs):.4f}" for g in G)
        print(f"  [{band}] {row}")
    tt = [case_mean(r, "thumb_top_mass") for r in recs]
    share = [case_mean(r, "thumb_top_mass") / max(1e-9, case_mean(r, "P_img")) for r in recs]
    vrel = [st.median(s["thumb_top_vnorm_rel"] for s in r["layers"] if "thumb_top_vnorm_rel" in s) for r in recs]
    cols = {}
    for r in recs:
        for s in r["layers"]:
            if s["layer"] >= 12:
                cols[s["thumb_top_col"]] = cols.get(s["thumb_top_col"], 0) + 1
    top_cols = sorted(cols.items(), key=lambda kv: -kv[1])[:5]
    print(f"  thumb top-1 token mass mean={st.mean(tt):.4f}  share of thumb mass={st.mean(share):.2f}  "
          f"its V-norm / thumb median (median over layers)={st.mean(vrel):.2f}  top-1 col offsets (L12+, count)={top_cols}")
b, c = out["broken"], out["clean"]
print("per-case thumb mass (all layers) broken:", sorted(round(case_mean(r, "P_img"), 4) for r in b))
print("per-case thumb mass (all layers) clean :", sorted(round(case_mean(r, "P_img"), 4) for r in c))
