# Consolidation (Components 1+2) — 사전등록 (2026-09-03)

## 구성 (memory/consolidate.py, off = byte-identical)
- **C1 Morphology-Coverage Rescue** (`VLMAS_MORPH_RESCUE=1`): TopB 이후 per-image
  facility-location 스왑 R' = R−{r}+{d}, 원본 KV만 사용, 예산 불변(assert),
  gain > max(0.05, 2e-3·n) 일 때만 (noise 재중심화 방지).
- **C2 Cross-Scale Consolidation** (`VLMAS_CROSS_SCALE=1`): x20 child 그룹의
  parent-redundancy(=max cos to x5 parent 그룹)를 **동일 child 이미지 내 순서로만**
  사용(score multiplier 금지 — 배율 systematic offset 상쇄), redundant retained ↔
  novel dropped 스왑, margin 0.05, parent 없는 이미지 무접촉.
- E arm 순서: pruning → C1 → C2 (사용자 다이어그램).

## Arms (4B step5, canonical)
A=base, B=pruning_v3 (runs/multipath_qwen4b_step5_20260903 재사용, 190+128),
C=B+rescue, D=B+cross, E=B+both → runs/consolidation_ablation/.

## Task 분리 (질문유형 분석 0903 기준)
- prototype/typical: **gtex** (100% 장기 식별, 20지선다)
- heterogeneous/detail: **tcga_expert_vqa** (75 고유질문, 구조/침윤/핵/정량)

## 게이트 (dashboard metric = 0806 evaluate() 동일 채점)
- **성공 패턴**: GTEx에서 C(및 E)가 B의 손실을 base 쪽으로 회복(paired flip 수지
  +), ExpertVQA는 비열화(±). KV budget: 전 arm 동일(런타임 assert + audit).
- **기각**: C≈B (rescue 무효) / GTEx 회복이 ExpertVQA 열화로 상쇄 / swap 수가
  0에 가까움(=메커니즘 미발화, audit 라인으로 확인).
- D 단독은 참고용(발화량 자체가 관심사; parent 링크는 x20 5장에만 존재).
- n=318 count 수준 보고, 유의성 주장 없음.

## Caveat (사용자 명시)
두 컴포넌트는 visual memory 구성 개선이지 downstream 활용(vision-inertness)의
직접 해결이 아님. 단, 개입 지점이 encoder-prefill 입력 레벨이라 refeed 실험으로
인과성이 확인된 유일 채널 위에 있음 — pruning이 GTEx를 실제로 깎은 것이 그 증거.
claim은 "pathology-aware latent-memory consolidation"으로 한정.
