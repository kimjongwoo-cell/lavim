#!/usr/bin/env python3
"""VPLI §6 게이트 판독 — ALSI 5셋×20×2arm 산출물에서.

gate 1 (matched-source) : 실제 짝의 destructive interaction 이 shuffled 짝보다 큰가
gate 2 (causal fidelity): local JVP 가 실제 섭동 변위를 예측하는가 (jvp_agree ~ 1)
"""
import json, glob, os, statistics as st
R = "/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs/alsi"

first = json.loads(open(f"{R}/alsi_can_gtex.jsonl").readline())
print("steps[0] 키:", sorted(first["steps"][0]))
print("  예시:", json.dumps(first["steps"][0], ensure_ascii=False)[:420])
print()

for arm in ("nat", "can"):
    print(f"=== arm={arm}")
    print(f"  {'dataset':18s} {'layer':>5s} {'kappa_V 참':>11s} {'kappa_V null':>12s} "
          f"{'Gamma(V,N) 참':>13s} {'null':>8s} {'jvp_agree':>10s}")
    for f in sorted(glob.glob(f"{R}/alsi_{arm}_*.jsonl")):
        ds = os.path.basename(f)[len(f"alsi_{arm}_"):-6]
        rows = [json.loads(l) for l in open(f)]
        by_layer = {}
        for r in rows:
            for s in r.get("steps", []):
                by_layer.setdefault(s.get("layer"), []).append(s)
        for lay in sorted(by_layer):
            S = by_layer[lay]
            def col(k, sub=None):
                out = []
                for s in S:
                    v = s.get(k)
                    if isinstance(v, dict) and sub is not None:
                        v = v.get(sub)
                    if isinstance(v, (int, float)):
                        out.append(float(v))
                return out
            kt = col("kappa_V") or col("kappa", "V")
            kn = col("kappa_V_null") or col("kappa_null", "V") or col("kappa_V_shuf")
            gt = col("Gamma_VN") or col("gamma", "N") or col("cos_VN")
            gn = col("Gamma_VN_null") or col("gamma_null", "N") or col("cos_VN_null")
            ja = col("jvp_agree")
            def m(x):
                return f"{st.median(x):.4f}" if x else "-"
            print(f"  {ds:18s} {lay:5d} {m(kt):>11s} {m(kn):>12s} {m(gt):>13s} "
                  f"{m(gn):>8s} {m(ja):>10s}")
    print()
