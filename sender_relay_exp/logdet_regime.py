#!/usr/bin/env python3
"""q 가중이 log-det의 submodular 거동을 살리는가 죽이는가.

log det(I + sum q_i r~ r~^T) 에서 q_i 가 아주 작으면 log(1+x) ~ x 라
log-det(부피/다양성)가 trace(단순 합)로 붕괴한다. gain_first/gain_last 의
감쇠비로 확인한다. 감쇠비 1에 가까우면 한계효용이 평평 = 다양성 항이 죽은 것.
"""
import re, glob, os, statistics as st
K = "/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs"
num = r"[-+0-9.eE]+"
RE = {k: re.compile(rf"{k}=({num})") for k in ("F", "gain_first", "gain_last")}

rows = []
for tag, pat in (("Q-PRIS (q=true)", f"{K}/qpris_full/qt_*.log"),
                 ("PCRIS v2 (q=none)", f"{K}/qpris/qn_*.log")):
    for f in sorted(glob.glob(pat)):
        F, gf, gl = [], [], []
        for line in open(f, errors="ignore"):
            if "[WSIConsol] qpris" not in line:
                continue
            m = {k: r.search(line) for k, r in RE.items()}
            if all(m.values()):
                F.append(float(m["F"].group(1)))
                gf.append(float(m["gain_first"].group(1)))
                gl.append(float(m["gain_last"].group(1)))
        if F:
            ds = os.path.basename(f).replace(".log", "").split("_", 1)[1]
            rows.append((tag, ds, len(F), st.median(F), st.median(gf), st.median(gl),
                         st.median([a / b for a, b in zip(gf, gl) if b > 0])))

print(f"{'arm':18s} {'dataset':18s} {'n':>4s} {'F(총 정보량)':>12s} "
      f"{'gain_first':>11s} {'gain_last':>10s} {'감쇠비':>8s}")
print("-" * 92)
for r in rows:
    print(f"{r[0]:18s} {r[1]:18s} {r[2]:4d} {r[3]:12.2f} {r[4]:11.4f} {r[5]:10.4f} {r[6]:8.2f}")
print("\n감쇠비 = gain_first / gain_last. 1에 가까울수록 한계효용이 평평 = log-det 다양성 항 소멸.")
