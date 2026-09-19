# C2(cross-scale) 단독 재현 배치 — 사전등록 (2026-09-04)

## 배경
0904 판정: 4B step5에서 D(pruning_v3 + VLMAS_CROSS_SCALE=1)가 pruning 손실 회복
(GTEx 46=base, 합계 수지 +7/−1). 단일 배치라 계보의 배치착시 원칙에 따라 재현 필요.
승자 D가 원래 참고용 arm이었으므로 사후 승격 대신 본 재등록으로 검증한다.

## 가설
C2 cross-scale consolidation(단독, C1 없이)은 pruning_v3의 정확도 손실을
step·모델을 바꿔도 일관되게 회복한다.

## 셀 (pruning_v3 대조군 완비 셀만)
4b/step10 · 4b/step20 · 8b/step20 — 각각 GTEx(190)+ExpertVQA(128), canonical config,
`--variant pruning_v3` + `VLMAS_CROSS_SCALE=1`(margin 0.05, 기본값 동일).
대조군 = allsteps_matrix_20260903의 같은 셀 pruning_v3. base 참조 = 0806 대시보드 런.

## 게이트 (count 수준, 대시보드-정합 채점기)
- **유효**: 3셀 중 ≥2셀에서 합계(318) paired 수지 순양(+) AND GTEx 방향 유지(≥0)
  AND ExpertVQA 순손실 ≤1.
- **기각**: 순양 셀 ≤1 → C2는 "4B step5 단일 배치 효과"로 기록, method 편입 보류.
- 스왑 발화(audit cross_swaps>0) 확인 필수 — 발화 없으면 해당 셀 무효 처리.
- 유의성 주장 없음. 수지(+/−)와 회복 문항 겹침만 보고.
