#!/usr/bin/env python3
"""질문을 어디에 결합해야 하는가 — 오프라인 탐색.

현행 Q-PRIS는 q_i = ||U_Q^T r~_i||^2, 즉 parent-span을 제거한 *잔차*를 질문 span에
투영한다. 잔차는 provenance로 설명되는 성분을 일부러 버리므로, 질문 관련 내용이
버려진 쪽에 있으면 구조적으로 안 잡힌다. 세 표현 × 세 질문방향을 같은 척도로 잰다.

척도 = (투영 에너지 / 전체 에너지) / (rank/d).  1.0이면 무작위 부분공간과 동일.
"""
import sys, glob, torch
sys.path.insert(0, "/home/users/whddn12316/wsi_latent_0915_decode_hj")
from memory.qpris_select import null_calibrated_residuals, question_basis

DUMP = "/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs/qpris/smoke_dump"


def lift(X, U):
    """투영 에너지 비율을 무작위 기준선(rank/d)으로 나눈 값."""
    if U is None or U.shape[1] == 0:
        return float("nan")
    d, k = X.shape[1], U.shape[1]
    num = ((X @ U) ** 2).sum().item()
    den = (X ** 2).sum().item()
    return (num / den) / (k / d)


files = sorted(glob.glob(f"{DUMP}/qpris_*.pt"))
print(f"덤프 {len(files)}건\n")
qs = []
for f in files:
    D = torch.load(f, map_location="cpu", weights_only=False)
    qs.append(D["q"].double())
q_mean = torch.cat(qs, 0).mean(0, keepdim=True)   # 템플릿 방향(케이스 공통)

hdr = f"{'case':6s} {'표현':22s} {'U_Q 원본':>10s} {'템플릿제거':>11s} {'무작위대조':>11s}"
print(hdr); print("-" * len(hdr))
for i, f in enumerate(files):
    D = torch.load(f, map_location="cpu", weights_only=False)
    z = D["z"].double()
    counts, parents = D["token_counts"], D["parents"]
    Q = D["q"].double()
    U_raw = question_basis(Q)
    U_ctr = question_basis(Q - q_mean)                       # 템플릿 성분 제거
    torch.manual_seed(0)
    U_rnd = torch.linalg.qr(torch.randn(z.shape[1], U_raw.shape[1], dtype=torch.float64))[0]

    R, info = null_calibrated_residuals(z, counts, parents, cond="true",
                                        context="parent", nullcal=True)
    zc = z - z.mean(0, keepdim=True)
    P_part = zc - (R * info["nu"][:, None])                  # provenance로 설명되는 성분

    for name, X in (("h (원본 시각상태)", zc),
                    ("P·h (provenance 설명분)", P_part),
                    ("r~ (현행 Q-PRIS 입력)", R)):
        print(f"{i:<6d} {name:22s} {lift(X, U_raw):10.3f} {lift(X, U_ctr):11.3f} "
              f"{lift(X, U_rnd):11.3f}")
    print()
print("1.00 = 같은 rank 무작위 부분공간과 동일. >1 이라야 질문이 뭔가를 잡은 것.")
