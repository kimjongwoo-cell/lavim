"""Layer-0 Answerer decision-row decomposition, broken vs clean (massprobe {broken,clean}_qk.jsonl -> qk0)."""
import collections
import json
import statistics as st

B = "/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs/nav4_wsivqa/massprobe/"
for name in ("broken_qk", "clean_qk"):
    rs = [json.loads(l) for l in open(B + name + ".jsonl")]
    rs = [r for r in rs if isinstance(r.get("qk0"), dict) and "error" not in r["qk0"]]
    print(f"== {name} n={len(rs)} |q|={st.mean(r['qk0']['q_norm'] for r in rs):.2f} "
          f"valmix_total={st.mean(r['qk0']['valmix_total'] for r in rs):.4f}")
    for g in ("sink", "Plat", "Nlat", "Rlat", "rest"):
        xs = [r["qk0"][g] for r in rs if g in r["qk0"]]
        if not xs:
            continue
        heads = collections.Counter(h for x in xs for h in x["mass_top3_heads"]).most_common(4)
        print(f"  {g:<5} mass={st.mean(x['mass'] for x in xs):.4f} maxhead={st.mean(x['mass_maxhead'] for x in xs):.3f} "
              f"logit={st.mean(x['logit'] for x in xs):7.3f} |k|={st.mean(x['k_norm'] for x in xs):7.2f} "
              f"cos={st.mean(x['cos_qk'] for x in xs):.4f} cosmax={st.mean(x['cos_qk_max'] for x in xs):.4f} "
              f"valmix={st.mean(x['valmix'] for x in xs):.4f} share={st.mean(x['valmix_share'] for x in xs):.3f} "
              f"top heads={heads}")
