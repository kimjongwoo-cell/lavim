#!/usr/bin/env python3
"""Does q_i discriminate, or is it at random-subspace level?

q_i = ||U_Q^T r~_i||^2 and W = sqrt(q_i) r~_i, so q_i is a multiplicative weight.
What matters is (a) its spread across tokens and (b) whether q_i/||r~_i||^2 sits at
the rank/d level a random subspace of the same rank would give.
"""
import re, statistics as st
K = "/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs/qpris_full"
DSS = ("tcga_expert_vqa", "gtex", "tcga", "tcga_slidebench", "panda")
D_MODEL = 2560  # Qwen3-VL-4B hidden size ([Backbone] W_a built: (2560, 2560))

def trip(name, line):
    m = re.search(name + r"=\[([^\]]*)\]", line)
    if not m:
        return None
    v = [float(x) for x in m.group(1).split(",") if x.strip()]
    return v if len(v) == 3 else None

print(f"{'dataset':18s} {'q_rank':>7s} {'rank/d':>8s} | "
      f"{'q p50':>9s} {'q p95/p5':>9s} | {'r~ p50':>8s} {'r~ p95/p5':>10s} | {'q/r~^2':>9s}")
print("-" * 100)
for ds in DSS:
    ranks, spread_q, med_q, spread_r, med_r, ratio = [], [], [], [], [], []
    for line in open(f"{K}/qt_{ds}.log", errors="ignore"):
        if "[WSIConsol] qpris" not in line:
            continue
        m = re.search(r"q_rank=(\d+)", line)
        if m:
            ranks.append(int(m.group(1)))
        for qn, rn in (("q_fine", "rtilde_fine"), ("q_root", "rtilde_root")):
            q, r = trip(qn, line), trip(rn, line)
            if not q:
                continue
            med_q.append(q[1])
            if q[0] > 0:
                spread_q.append(q[2] / q[0])
            if r:
                med_r.append(r[1])
                if r[0] > 0:
                    spread_r.append(r[2] / r[0])
                if r[1] > 0:
                    ratio.append(q[1] / (r[1] ** 2))
    if not ranks:
        continue
    rk = st.median(ranks)
    print(f"{ds:18s} {rk:7.0f} {rk / D_MODEL:8.4f} | "
          f"{st.median(med_q):9.5f} {st.median(spread_q):9.2f} | "
          f"{st.median(med_r):8.3f} {st.median(spread_r):10.2f} | "
          f"{st.median(ratio):9.5f}")
print("\nq_rank/d = 같은 rank의 임의 부분공간이 잡을 에너지 비율(기준선).")
print("q/r~^2 이 그 기준선과 같으면, 질문 부분공간이 잔차에 대해 임의 방향과 구별되지 않는다.")
