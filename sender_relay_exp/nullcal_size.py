#!/usr/bin/env python3
"""null 기준집합 P_o^- 이 실제로 몇 개짜리인가, 그리고 보정이 빠진 케이스가 있나."""
import re, glob, collections
K = "/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs"
RE_NC = re.compile(r"nullcal_obs=(\d+)")
RE_RK = re.compile(r"ranks=\[([^\]]*)\]")

print(f"{'run':34s} {'cases':>6s} {'child 수':>9s} {'보정된 obs':>11s} {'보정 누락':>9s}")
print("-" * 76)
for pat in (f"{K}/qpris_full/q*_*.log", f"{K}/qpris/qn_*.log"):
    for f in sorted(glob.glob(pat)):
        nc, nchild, miss, n = [], [], 0, 0
        for line in open(f, errors="ignore"):
            if "[WSIConsol] qpris" not in line:
                continue
            m1, m2 = RE_NC.search(line), RE_RK.search(line)
            if not (m1 and m2):
                continue
            ranks = [int(v) for v in m2.group(1).split(",") if v.strip()]
            kids = sum(1 for r in ranks if r > 0)      # 부모 span 이 있는 관찰 = child
            got = int(m1.group(1))
            n += 1
            nc.append(got); nchild.append(kids)
            if got < kids:
                miss += 1
        if n:
            name = f.split("/runs/")[1].replace(".log", "")
            print(f"{name:34s} {n:6d} {sum(nchild)/n:9.2f} {sum(nc)/n:11.2f} {miss:9d}")
print("\n우리 트리: x5 3장 + x20 5장, parents=(-1,-1,-1,0,1,0,1,0)")
print("→ 부모 노릇을 실제로 하는 관찰 = {0, 1} 둘뿐. child 하나당 alts 는 1개.")
print("→ x5 중 obs 2 는 자식이 없어 null 기준에서 아예 빠진다.")
