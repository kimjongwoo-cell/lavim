# Visual Binding — 사전등록 (protocol steps [1]–[5], 2026-09-03)

## 동결 조건 [1]
- Planner→Navigator→Reasoner→Answerer, patch 8, m=5, refeed 없음, text bridge 없음.
- pruning-v3 / reallocation-v2 전부 OFF (`--variant base`).
- 4B, step5, greedy/seed42, WSI-VQA idx 0–29 (donor = idx 0, TCGA-LD-A74U; idx 1–3은 same-slide donor → matched-vs-wrong 비교에서 별도 표기).

## [2] K/V/KV carrier probe (runs/kv_swap_probe)
같은 케이스의 터미널 디코드를 캐시 사본 위에서 재실행:
(K_x,V_x) = normal, (K_x',V_x) = K-swap, (K_x,V_x') = V-swap, (K_x',V_x') = KV-swap.
측정: ① answer flip 수(파싱된 answer 기준), ② score_* = normal 답의 teacher-forced
sum log-prob (Δ_mode = score_normal − score_mode).
- 원하는 신호: Δ(V-swap) > Δ(K-swap), V-swap flip > 0.
- 전 모드 무변화(flip≈0, Δ≈0)면: 새 handoff를 만들지 않고 [3]에서 어느 layer까지
  visual content가 있는지부터 본다 (사용자 게이트).

## [3] coarse layer probe (같은 런에 포함)
V-swap을 단일 layer로 제한: V_L10 / V_L18 / V_L26 / V_L33 (36층; 28층 예시
{8,14,20,26}의 분율 대응). 판정: 한 layer만 튀면 fragile로 기각, 밴드(연속 2개 이상
비슷한 Δ)가 있어야 [4]의 binding layer 후보.

## [4] Visual Binding (마지막 latent step z_5만)
`VLMAS_VISUAL_BIND=1`: z_5 forward에서 선택 layer의 attention source를 visual KV +
자기 자신으로 제한(4-D additive mask, eager). W_a·append·position·Answerer 무수정,
step 수 불변. off = byte-identical (env 미설정 시 hook 자체가 안 걸림).

## [5] 통제군 4-arm (run_visual_bind_exp.sh)
base / matched / wrong_slide(donor K,V, binding layer만, forward 후 복원) /
shuffled_v(K 유지, visual V 열 순열).

**핵심 게이트 (사전등록):**
- 성공 = matched > base **그리고** matched > wrong_slide **그리고** matched > shuffled_v
  (n=30 count 수준: 정답 수 + paired flip 방향).
- matched ≈ wrong ≈ shuffled > base 형태면 activation-perturbation 효과 → 기각.
- 단일 layer에서만 성공하면 main method로 채택하지 않음.
- n=30이므로 유의성 주장 없음; 통과 시 N 확대 재실행이 다음 단계.

## 이후 (성공 시에만)
[6] z_5 latent-state swap (matched vs mismatched z_5) → Answerer 인과 확인,
[7] raw visual KV 제거 A/B, [8] latency/KV 메모리, [9] compression.
[5] 게이트 실패 시 [6]+ 진행하지 않음.
