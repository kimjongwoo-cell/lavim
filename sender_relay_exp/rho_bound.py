#!/usr/bin/env python3
"""rho_uniq 의 상한 — 기존 ALSI 산출물만으로.

부분공간 투영 ||P_span{R,Q,Y} d_V|| 은 최선의 단일 소스 투영보다 항상 크거나 같으므로
    rho_uniq = 1 - ||P d_V||^2/||d_V||^2  <=  1 - max_s cos^2(d_V, d_s)
정확값은 R,Q,Y 사이 Gram 이 있어야 하고 그건 저장돼 있지 않다.
"""
import json, glob, os, statistics as st
R = "/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs/alsi"

print("rho_uniq 상한 = 1 - max_s cos^2(d_V, d_s).  실제 rho_uniq 는 이 값 이하.")
print("작을수록 '시각 효과가 비시각 subspace 안에 있다'에 가깝다.\n")
for arm in ("nat", "can"):
    print(f"=== arm={arm}")
    print(f"  {'dataset':17s} {'L':>3s} {'상한 p50':>9s} {'p25':>7s}  최대 흡수원")
    agg = {}
    for f in sorted(glob.glob(f"{R}/alsi_{arm}_*.jsonl")):
        ds = os.path.basename(f)[len(f"alsi_{arm}_"):-6]
        by = {}
        for line in open(f):
            for s in json.loads(line).get("steps", []):
                by.setdefault(s["layer"], []).append(s)
        for lay in sorted(by):
            S = by[lay]
            ub = [1.0 - max(s["gamma"][k] ** 2 for k in "RQY") for s in S]
            who = [max("RQY", key=lambda k: abs(s["gamma"][k])) for s in S]
            dom = max(set(who), key=who.count)
            print(f"  {ds:17s} {lay:3d} {st.median(ub):9.3f} "
                  f"{sorted(ub)[len(ub) // 4]:7.3f}  {dom} {who.count(dom)}/{len(who)}")
            a = agg.setdefault(lay, [[], []])
            a[0] += ub
            a[1] += who
    print(f"  {'전 셋':17s}")
    for lay in sorted(agg):
        u, w = agg[lay]
        dom = max(set(w), key=w.count)
        print(f"  {'':17s} {lay:3d} {st.median(u):9.3f} {sorted(u)[len(u) // 4]:7.3f}  "
              f"{dom} {w.count(dom)}/{len(w)}")
    print()
