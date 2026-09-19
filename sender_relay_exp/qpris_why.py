#!/usr/bin/env python3
"""What Q-PRIS actually kept, per dataset, from the [WSIConsol] qpris log lines."""
import re, glob, os, statistics as st
K = "/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs/qpris_full"
DSS = ("tcga_expert_vqa", "gtex", "tcga", "tcga_slidebench", "panda")

num = r"[-+0-9.eE]+"
RE_MAG = re.compile(r"kept_by_mag=\{5: (\d+), 20: (\d+)\}")
RE_QF = re.compile(r"q_fine=\[([^\]]*)\]")
RE_QR = re.compile(r"q_root=\[([^\]]*)\]")
RE_NU = re.compile(r"nu_fine=\[([^\]]*)\]")
RE_PC = re.compile(r"per_crop=\[([^\]]*)\]")
RE_GF = re.compile(rf"gain_first=({num})")
RE_GL = re.compile(rf"gain_last=({num})")

print(f"{'dataset':18s} {'n':>4s} {'x5 kept':>9s} {'x20 kept':>9s} {'x5 %':>6s} "
      f"{'q_fine uniq':>12s} {'q_root uniq':>12s} {'per_crop min':>12s}")
print("-" * 92)
for ds in DSS:
    f = f"{K}/qt_{ds}.log"
    if not os.path.exists(f):
        continue
    x5, x20, qf, qr, pcmin, pcmax = [], [], set(), set(), [], []
    for line in open(f, errors="ignore"):
        if "[WSIConsol] qpris" not in line:
            continue
        m = RE_MAG.search(line)
        if m:
            x5.append(int(m.group(1))); x20.append(int(m.group(2)))
        m = RE_QF.search(line)
        if m:
            qf.update(v.strip() for v in m.group(1).split(",") if v.strip())
        m = RE_QR.search(line)
        if m:
            qr.update(v.strip() for v in m.group(1).split(",") if v.strip())
        m = RE_PC.search(line)
        if m:
            vals = [int(v) for v in m.group(1).split(",") if v.strip()]
            if vals:
                pcmin.append(min(vals)); pcmax.append(max(vals))
    if not x5:
        continue
    tot = st.mean(x5) + st.mean(x20)
    print(f"{ds:18s} {len(x5):4d} {st.mean(x5):9.1f} {st.mean(x20):9.1f} "
          f"{100 * st.mean(x5) / tot:5.1f}% {len(qf):12d} {len(qr):12d} "
          f"{st.mean(pcmin):12.1f}")

print("\n--- q 항이 실제로 변별하는가 (전 케이스 고유값) ---")
for ds in DSS:
    f = f"{K}/qt_{ds}.log"
    if not os.path.exists(f):
        continue
    qf, qr = set(), set()
    for line in open(f, errors="ignore"):
        if "[WSIConsol] qpris" not in line:
            continue
        m = RE_QF.search(line)
        if m:
            qf.update(v.strip() for v in m.group(1).split(",") if v.strip())
        m = RE_QR.search(line)
        if m:
            qr.update(v.strip() for v in m.group(1).split(",") if v.strip())
    print(f"  {ds:18s} q_fine values={sorted(qf)[:6]}  q_root values={sorted(qr)[:6]}")
