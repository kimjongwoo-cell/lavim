#!/usr/bin/env python3
"""VPLI 전제 검정: non-visual 이 visual 방향을 실제로 상쇄하는가.

kappa_V = -min(0, dbar_V . dbar_N)/||dbar_V||^2 > 0  이어야 'destructive'.
gamma_N = cos(dbar_V, dbar_N) < 0                    이어야 'destructive'.
"""
import json, glob, os, statistics as st
R = "/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs/alsi"


def q(x, p):
    if not x:
        return float("nan")
    x = sorted(x)
    return x[min(len(x) - 1, int(p * len(x)))]


for arm in ("nat", "can"):
    print(f"=== arm={arm}")
    print(f"  {'dataset':17s} {'L':>3s} {'steps':>6s} {'kappa_V>0':>10s} "
          f"{'gamma_N<0':>10s} {'gamma_N p50':>12s} {'p05':>8s} {'jvp p50':>8s}")
    tot = {}
    for f in sorted(glob.glob(f"{R}/alsi_{arm}_*.jsonl")):
        ds = os.path.basename(f)[len(f"alsi_{arm}_"):-6]
        by = {}
        for line in open(f):
            for s in json.loads(line).get("steps", []):
                by.setdefault(s["layer"], []).append(s)
        for lay in sorted(by):
            S = by[lay]
            kv = [float(s["kappa_V"]) for s in S if s.get("kappa_V") is not None]
            gn = [float(s["gamma_N"]) for s in S if s.get("gamma_N") is not None]
            ja = [float(s["jvp_agree"]) for s in S if s.get("jvp_agree") is not None]
            npos = sum(1 for v in kv if v > 1e-9)
            nneg = sum(1 for v in gn if v < 0)
            print(f"  {ds:17s} {lay:3d} {len(S):6d} {npos:4d}/{len(kv):<5d} "
                  f"{nneg:4d}/{len(gn):<5d} {st.median(gn):12.4f} {q(gn, 0.05):8.4f} "
                  f"{st.median(ja):8.4f}")
            t = tot.setdefault(lay, [0, 0, 0, 0, []])
            t[0] += npos; t[1] += len(kv); t[2] += nneg; t[3] += len(gn); t[4] += ja
    print(f"  {'--- 합계':17s}")
    for lay in sorted(tot):
        t = tot[lay]
        print(f"  {'전 셋':17s} {lay:3d} {'':6s} {t[0]:4d}/{t[1]:<5d} {t[2]:4d}/{t[3]:<5d} "
              f"{'':12s} {'':8s} {st.median(t[4]):8.4f}")
    print()
