#!/usr/bin/env python3
"""공통 성분 가설 검정 — ALSI 기존 산출물만 사용, GPU 없음.

가설: d_N 이 d_V 와 같은 방향(cos>0)이므로 visual 고유 정보가 비시각에 묻힌다.
검정: d_V 중 d_N 과 공유되지 않는 성분의 크기.
    공유분   proj = gamma_N * ||d_V||
    고유분   perp = sqrt(1 - gamma_N^2) * ||d_V||
가설이 맞으려면 perp 비율이 작아야 한다(공유가 지배). 크면 가설 기각.
"""
import json, glob, os, math, statistics as st
R = "/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs/alsi"


def med(x):
    return st.median(x) if x else float("nan")


for arm in ("nat", "can"):
    print(f"=== arm={arm}")
    print(f"  {'dataset':17s} {'L':>3s} {'||d_V||':>8s} {'||d_R||':>8s} {'||d_Q||':>8s} "
          f"{'||d_Y||':>8s} | {'g_N':>6s} {'고유분 %':>8s} {'고유분 크기':>11s} {'V 최대?':>8s}")
    agg = {}
    for f in sorted(glob.glob(f"{R}/alsi_{arm}_*.jsonl")):
        ds = os.path.basename(f)[len(f"alsi_{arm}_"):-6]
        by = {}
        for line in open(f):
            for s in json.loads(line).get("steps", []):
                by.setdefault(s["layer"], []).append(s)
        for lay in sorted(by):
            S = by[lay]
            dv = [s["d_norm"]["V"] for s in S]
            dr = [s["d_norm"]["R"] for s in S]
            dq = [s["d_norm"]["Q"] for s in S]
            dy = [s["d_norm"]["Y"] for s in S]
            gn = [s["gamma_N"] for s in S]
            perp_frac = [math.sqrt(max(0.0, 1 - g * g)) for g in gn]
            perp = [p * v for p, v in zip(perp_frac, dv)]
            vmax = sum(1 for s in S
                       if s["d_norm"]["V"] >= max(s["d_norm"][k] for k in "RQY"))
            print(f"  {ds:17s} {lay:3d} {med(dv):8.2f} {med(dr):8.2f} {med(dq):8.2f} "
                  f"{med(dy):8.2f} | {med(gn):6.3f} {100*med(perp_frac):7.1f}% "
                  f"{med(perp):11.2f} {vmax:4d}/{len(S):<4d}")
            a = agg.setdefault(lay, {"pf": [], "dv": [], "vmax": 0, "n": 0, "gn": []})
            a["pf"] += perp_frac; a["dv"] += dv; a["gn"] += gn
            a["vmax"] += vmax; a["n"] += len(S)
    print(f"  {'전 셋':17s}")
    for lay in sorted(agg):
        a = agg[lay]
        print(f"  {'':17s} {lay:3d} {med(a['dv']):8.2f} {'':8s} {'':8s} {'':8s} | "
              f"{med(a['gn']):6.3f} {100*med(a['pf']):7.1f}% {'':11s} "
              f"{a['vmax']:4d}/{a['n']:<4d}")
    print()
print("고유분 % = sqrt(1 - cos^2) — d_V 중 비시각과 공유되지 않는 비율.")
print("공통 성분 가설이 맞으려면 이 값이 작아야 한다.")
