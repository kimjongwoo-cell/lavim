# 진행 기록 2026-09-03 (wsi_latent_0915_decode_hj / sender_relay_exp)

모든 수치는 4B step5 canonical(greedy/seed42/patch8/m=5), WSI-VQA idx 0–29 (별도 표기 제외).
태그: [GPU실측] = 원격 실행 결과, [코드읽기] = 코드 확인, [미검증] = 아직 안 돌림.

## 1. 동결 프로토콜 (사용자 지정)
[1] latent-base 고정 → [2] K/V/KV-swap carrier probe → [3] coarse layer probe →
[4] 마지막 latent step Visual Binding → [5] matched/wrong/shuffled 통제군.
게이트: [2]에서 전 모드 무변화면 새 handoff 금지, [5]는 Matched > Base ∧ Wrong ∧ Shuffled.
사전등록: `VISUAL_BIND_PREREG.md`, `CONSOLIDATION_PREREG.md`.

## 2. [2]+[3] KV-swap carrier probe — **NULL 확정** [GPU실측]
런: `runs/kv_swap_probe2/` (30/30, 실패 0; donor=idx0 제외 n=29)

| 스왑 | answer flip | ΔlogP median | 비고 |
|---|---|---|---|
| K-swap (주소만 교란) | 2/29 | −0.017 | flip 2건은 전 모드 공통 포맷변동 → 실질 0 |
| V-swap (내용만 교체) | 2/29 | +0.042 | Δ(V)−Δ(K) median −0.01, V>K 14/29 = 동전 |
| KV-swap | 2/29 | −0.066 | |
| V_L10 / V_L18 / V_L26 / V_L33 | 각 2/29 | −0.14 ~ +0.02 | layer 밴드 없음 |

→ 다른 슬라이드의 visual K/V를 통째로 넣어도 답·logP 모두 불변.
**터미널 readout이 visual KV 열을 인과적으로 읽지 않음** (5중 확인: slide-swap 92% 동일
→ vision-block ΔlogP 0.06% → destination-shuffle 7% → KV-swap flip 0 → relay C≈D).
주의: cache에 내용이 없다는 뜻 아님 — fresh-forward attention은 종양 evidence를 읽음
(Phase0a AUROC 0.84). 죽은 것은 "존재"가 아니라 "사용(readout)".

## 3. ② ROI visual relay 2×2 — **NULL** [GPU실측]
런: `runs/visual_relay_exp1/` (arm당 30/30). visual만 읽도록 masking한 latent state를
append해 Answerer에 전달하는 방식.

| arm | acc(러프) | flips vs base |
|---|---|---|
| A base (=kv_swap_probe2/base_probe) | 8/30 | — |
| B global relay ×1 | 8/30 | 4/30 |
| C roi relay ×8 | 8/30 | 8/30 |
| D roi relay ×8 + cross-slide V | 7/30 | 9/30 |

C vs D flip 1/30 → **내용 교란에 둔감, 구조 교란(B vs C 4/30)에만 반응 = perturbation**.
사전등록 기각 조건 충족. 강제 readout 내용도 답까지 안 감.

## 4-결과. Visual Binding 스윕 — **기각** [GPU실측, 23:10 완주]
| arm | acc | flips vs base |
|---|---|---|
| base | 8/30 | — |
| matched L10 / L18 / L26 | 7 / 9 / 8 | 3 / 1 / 2 |
| wrong_slide L18 | 8/30 | 2/30 |
| shuffled_V L18 | 8/30 | 2/30 |

기각 사유(사전등록 조건 그대로): ① layer 비일관(−1/+1/0 = 노이즈), ② 내용 불감 —
matched_L18 vs wrong 차이 1/30, vs shuffled 1/30, flip 케이스(002)도 세 arm 공통 →
z_5에 강제로 읽힌 visual 내용이 답에 미진입(perturbation). KV-swap → relay → binding
3연속 null로 "visual 내용을 어디서 어떻게 밀어넣어도 터미널 답 불변" 일관.
남은 시험: lavim_method_smoke의 bind_all(§3.4 원형: 전층·R*-only·no-self).

## 4. [4]+[5] Visual Binding — 배선 (참고: 위 결과로 종료)
- 배선: `backbone/qwen3vl.py _visual_bind_step` — z_5(마지막 latent step) forward에서
  선택 layer의 attention을 visual KV+self로 제한(4-D additive mask, eager). W_a·append·
  position·Answerer 무수정, off=byte-identical. 유닛 22/22 (`test_visual_bind.py`).
- [2] null이라 사전등록상 보류였으나 사용자 지시로 실행.
- 런: `runs/visual_bind_exp/` — matched_L10 / matched_L18 / matched_L26 +
  wrong_L18(다른 슬라이드 K,V donor) + shuffled_L18(K 유지, V 열 순열). ~1h.
- 발화 확인: `[VisualBind] layers=[10] mode=matched vis=2048 kv_len=6265`.
- 게이트: Matched > Base ∧ Wrong ∧ Shuffled (아니면 기각). Base = kv_swap_probe2/base_probe.

## 5. Consolidation (Component 1+2) — 배선 완료, binding 뒤 자동 재개
- `memory/consolidate.py` (off=byte-identical, 유닛 21/21, `test_consolidate.py`):
  - **C1 Morphology-Coverage Rescue** (`VLMAS_MORPH_RESCUE=1`): TopB 후 per-image
    facility-location swap R−{r}+{d}, 원본 KV만, 예산 불변(assert), gain>max(0.05,2e-3·n).
  - **C2 Cross-Scale Consolidation** (`VLMAS_CROSS_SCALE=1`): x20 child의 parent-redundancy
    (x5 parent 대비 max cos)를 child 내 순위로만 사용(승수 금지), redundant↔novel swap.
- 첫 발화 [GPU실측]: 케이스당 rescue 스왑 67–91, gain 107–294 → **프루닝 retained set에
  morphology 중복이 실제로 다량 존재**. parents=[-1,-1,-1,0,1,0,1,0] (x5 3장 + x20 5장).
- 실험: `run_consolidation_ablation.sh` — C(rescue)/D(cross)/E(both) × gtex/expert_vqa,
  A/B는 `runs/multipath_qwen4b_step5_20260903/` 재사용(190+128). bind 스윕 종료 후 자동 시작.
- 게이트: GTEx에서 C·E가 pruning 손실을 base 쪽으로 회복 + ExpertVQA 비열화 + 예산 동일.

## 6. MultiPathQA 데이터셋별 질문 유형 [실측 분석]
| 데이터셋 | n | 구조 | 유형 |
|---|---|---|---|
| gtex | 190 | 단일 템플릿, 20지선다 | 100% 장기 식별 (전형/prototype 인식) |
| tcga | 221 | 단일 템플릿, 30지선다 | 100% primary diagnosis (암종 분류) |
| tcga_expert_vqa | 128 | 고유질문 75, 4지선다 | architecture 21 / invasion 17 / presence 13 / 정량 10 / 핵 10 / 괴사·염증 7 / 기타 46 |
| tcga_slidebench | 197 | 고유질문 195, 4지선다 | grade 41 / subtype 19 / architecture 15 / presence 13 / invasion 11 / 기타 83 |

task 분리: 전형 인식 = gtex+tcga ↔ 이질적 detail = expert_vqa+slidebench.
novelty 프루닝이 gtex를 깎는 이유와 정합(전형=다수 morphology를 novelty가 버림).

## 7. Bottleneck 진단 (현재 인과 사슬)
```
patch 픽셀 → encoder(내용 존재 O) → visual KV(cache에 저장 O)
    → [★병목1] Answerer 터미널 readout이 visual 열을 안 읽음 (KV-swap null)
    → [★병목2] 대체 통로인 latent state도 시각을 안 실음
        (reasoner latent 이미지 불변 cos 0.991, relay C≈D)
그나마 살아있는 문:
    - 입력 레벨(무엇을 넣느냐): 프루닝이 GTEx −10 / refeed flip 24% → 인과 O
    - latent 채널 자체: planner→navigator 지배 3/3 → 채널은 살아있음
```
즉 병목 = "visual 정보가 **latent 열에 실리는 지점**". binding 스윕이 정확히 그 지점을
공격하는 마지막 시도이고, 실패 시 남는 지렛대는 입력 레벨(consolidation)임.

## 8. 운영 사고 기록 (재발 방지 조치 포함)
1. 원격 pkill 비표준(usage만 출력) → 좀비 rejoin 루프가 probe와 GPU8 동시 점유 →
   probe 1차 25/30 OOM. **조치**: PID kill로 전환 + `gpu8.lock` flock을 GPU8 스크립트
   전부에 배선(락 못 잡으면 실행 거부, 프로세스 사망 시 자동 해제).
2. pgrep -f가 자기 ssh 명령 문자열도 매칭 → 자기 kill 2회. **조치**: cmdline 검사로
   "bash -c" wrapper 제외 + $$ 가드.
3. probe 출력이 json_prefix 뒤부터 시작하는데 "answer" 키 regex로 파싱해 flip 19/19
   오보(rationale 문구 차이를 flip으로 집계). **조치**: 첫 따옴표 값 파싱으로 정정,
   원문 대조 전 수치 보고 금지.

## 9. 다음
- bind 스윕 결과 집계(게이트 판정) → consolidation 5-arm 표 (내일 오전)
- [2] null의 fallback: layer별 visual content "존재" probe(선형 복호) 설계
- allsteps 매트릭스(GPU 6·7) 완주 후 대시보드 갱신
