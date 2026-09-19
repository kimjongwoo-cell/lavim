# Hierarchy-Constrained Spatial Pruning — 배선 기록 (0910 19:20~)

**스위치**: `VLMAS_WSI_CONSOL=1 VLMAS_WSI_CONSOL_MODE=hier VLMAS_WSI_CONSOL_KEEP=ρ` (통제군 `MODE=hier_shuf`, 옵션 `VLMAS_WSI_CONSOL_HIER_MIN=m`, 기본 1).
OFF(기본) = 기존과 byte-identical [코드읽기: `_wsi_consolidate_boundary` 안 `elif mode in ("hier","hier_shuf")` 한 분기만 추가, 다른 MODE 경로는 손대지 않음; `test_strat_untouched` PASS].

## 무엇을 하나 (스펙 그대로)
1. **Mandatory S_hier** — 각 crop ≥ m(=1) 토큰(equal-area 중심) ∪ 실제 5×→20× zoom edge마다 부모의 π(c) 토큰(자식 level-0 박스와 footprint 겹침 최대인 부모 grid 셀, 순수 기하). 점수(attention/cosine/morphology) 일절 없음.
2. **나머지 예산** — 기존 strat 규칙 그대로: `allocate_budget`(b_p ∝ |V_p|, ≥1) + `equal_area_grid_indices`. mandatory는 해당 crop 할당량에 포함. mandatory가 할당량을 넘는 crop은 mandatory 유지, 초과분은 여유 있는 crop에서 차감(합=B). B<|S_hier|면 S_hier 전부 유지(예산 초과 허용, 로그로 확인 가능).
3. **타이밍·레이어·crop set·디코딩 = strat와 동일** (Reasoner on_kv 경계 1회, 전 36레이어 동일 index). 바뀌는 건 keep mask 규칙 하나.
4. **hier_shuf 통제** — 부모 index를 실제 anchor들 사이 cyclic derangement(seed 42). 자식의 *상대 위치*를 틀린 부모 프레임에 이식해 π 계산 → edge당 정확히 1개 "기하적으로 무의미한" 부모 토큰 앵커(mandatory 수 동일). ⚠ 초기판은 박스 그대로 겹침 계산 → 겹침 0 → 전부 토큰 0으로 붕괴(mandatory 13→10)했던 결함 수정.

## 파일
- `memory/stratified.py`: `_derange_parents`, `_max_overlap_parent_token`, `hierarchy_keep_mask(..., return_info)` 추가(append only).
- `backbone/qwen3vl.py` `_wsi_consolidate_boundary`: `MODE=hier|hier_shuf` 분기 + 로그 `[WSIConsol] hier mandatory=.. edges=.. parents=[..] per_crop=[..]`.
- `tests_hj_test_hier_prune.py`: 7/7 PASS [GPU실측 venv python] — 예산·≥1/crop·π 기하·shuffle 상이·full-budget identity·박스 없음 degrade·strat 불변·mandatory>quota.

## 스모크 [GPU실측, gtex2, GPU8, CUDA_LAUNCH_BLOCKING=1, KEEP=0.5]
- `smoke_hier`: 2/2 rc=0, traceback 0, result.json 2. 로그: `mandatory=13 edges=5 parents=[-1,-1,-1,0,1,0,1,0] per_crop=[128×8]`, `kept 1024/2048`.
  → 실제 crop 트리: 5× 3장(0,1,2) + 20× 5장(부모 0,1,0,1,0). 5× #2는 자식 없음.
- `smoke_hier_shuf`(초기 결함판): 2/2 rc=0 (mandatory 10 — 붕괴 증거) → 수정 후 `smoke_hier_shuf2` 재실행.

## 비교 설계(스펙) — 큐 투입은 협의 후
같은 B(KEEP=0.5), 같은 crop set·timing·layer·decoding:
| arm | env | 상태 |
|---|---|---|
| Full KV | base | 5셋 있음 |
| Flat Spatial | `MODE=strat KEEP=0.5` | 5셋 있음(gtex 18.69/evqa 45.31/sb 46.19/tcga 7.24/panda 19.03) |
| **Hierarchy** | `MODE=hier KEEP=0.5` | 미실행 |
| Shuffled Hierarchy | `MODE=hier_shuf KEEP=0.5` | 미실행 |
원하는 결과: Hier > Flat 이면서 Hier > Shuffled.
[미검증-가설] 5셋 10잡×3샤드=30잡, 현 큐 뒤 ≈3~4h.

### 스모크 결과 (v2·v3) [GPU실측, gtex2, GPU8, B=96]
| arm | 완주 | 로그 |
|---|---|---|
| wsi | 2/2 rc=0 | `B0=13 edges=5 per_crop≈12×8` |
| wsi_flat | 2/2 rc=0 | `B0=8 edges=0 per_crop=12×8` |
| wsi_shuf | 2/2 rc=0 | `B0=13 edges=5`, parents 뒤바뀜 |
| scale (α=0.5) | 2/2 rc=0 | `B_C=48 B_F=48 coarse=[0,1,2] fine=[3..7] bridges=5 mandatory_C=6 wsi_coords=True per_crop 5×≈15~18/20×≈9~10` |
| scale_shuf | 2/2 rc=0 | 동일 예산, parents 뒤바뀜 |
2케이스 답은 전 arm 동일(Colon / Small Intestine). 유닛 16/16. 풀런은 큐 승인 대기.
