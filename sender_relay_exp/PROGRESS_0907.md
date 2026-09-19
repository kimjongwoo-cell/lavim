# LaViM 진행상황 (2026-09-08 낮 — ★최종 테이블 완성판)

정본 위치: 로컬 `/home/super/hj/0902_2155/sender_relay_exp/PROGRESS_0907.md` = 원격 `wsi_latent_0915_decode_hj/sender_relay_exp/PROGRESS_0907.md`
대시보드(성능표): https://claude.ai/code/artifact/c65c1dbc-0fc9-463d-9239-f3546c9d4783
실험 로그(방법론 카드·이력): https://claude.ai/code/artifact/cc558a62-fee0-4399-844d-63da9efc2c35
채점 규약: EVQA·SB=ACC, GTEx·TCGA·PANDA=BACC (0806 재현 채점기) · **정본 = 4B·step10**
**합격 기준(사용자 등록): 각 셋에서 max(Latent base, VLMAS) + 1pt 이상** — full(C1+C2) arm에 적용

## 1. 확정 방법론 (Method Draft 0907 최종판) — 배선 완료·스모크 통과
- **C1 = Observation-Stratified Visual Compression**: crop별 예산 b_p∝|V_p|(하한 1), equal-area 격자 spatial thinning(dedup 아님), 점수 일절 없음, 노브=B. env: `VLMAS_WSI_CONSOL=1 VLMAS_WSI_CONSOL_KEEP=0.5 VLMAS_WSI_CONSOL_MODE=strat` (통제: `=random_budget`, `=flat`). 코드: memory/stratified.py + qwen3vl `_wsi_consolidate_boundary` 분기.
- **C2 = Cross-Scale Evidence Reassembly**: park 복원 직전, 같은 B토큰을 Navigator parent→child(coarse-to-fine) 순서의 연속 블록으로 재배열 후 recency 배치(K만 Δ-RoPE 보정). env: `VLMAS_KV_RESTAGE=1 VLMAS_KV_PARK=1 VLMAS_KV_RESTAGE_ORDER=hierarchy` (통제: `shuffled_hier|random|magnification|original`). 코드: stratified.reassembly_order + restage_visual_kv 재배열 블록 + park블록 meta fallback(C2-단독용).
- 적용 지점: C1은 crop visual이 캐시에 진입하는 Reasoner 경계 1회(썸네일은 탐색용 입력이라 제외 — 사용자 확정), C2는 터미널 직전 1회.
- **novelty 게이트: full(hierarchy) > s2shuf(shuffled_hier)** 아니면 계층 주장 폐기(스펙 §8).

## 2. step10 정본 앵커 (5셋 완비) [GPU실측]
| | GTEx | EVQA | SB | TCGA | PANDA |
|---|---|---|---|---|---|
| Latent base | 21.41 | 45.31 | 48.22 | 7.33 | 18.26 |
| VLMAS(step무관) | 19.28* | 46.88 | 47.21 | 5.75 | 17.32 |
| **합격선** | 22.41 | 47.88 | 49.22 | 8.33 | 19.26 |
| park(+restage) | 22.51 | 44.53 | 42.64 | 7.90 | — |
| restage 전량 | 24.91 | 45.31 | — | — | — |
(*gtex 4B VLMAS n=153 미완주 표본)

## 3. 매트릭스 진행 (C1/C2/C1+C2 × 5셋, 큐=데이터셋별 3종 세트 순서)
- 완료: strat_gtex .1869(−2.7, random .1972에 패배) · **strat_evqa .4531(base 동률 — 무손실, AR −16.4 대비 설계목표 달성)** · randb_gtex .1972
- 실행 중(0908 01:50): strat_sb10, strat_tcga10
- 대기(순서): reasm_gtex→full_gtex→reasm_evqa→full_evqa→sb세트→tcga세트→panda세트→s2shuf(통제)→tcga/panda restage10→hp_rp→hc→…
- 워커 2대(GPU 6·7), GPU 8=타 세션 realloc 런 점유(불간섭). full_gtex 판정 ~04:50, 매트릭스 완주 ~15:00(8 확보 시 ~09:30).

## 4. 0907 종결된 것 (재시도 금지 — 상세는 실험 로그 §21.5 방법론 카드)
- C2 신형 3형 전멸: bind −2.1 / read(noQ·Q·조합) −1.3~−2.6 / **VG-prefill 단독 −3.1·hp_vg −8.9(바닥 이하)**. 용량-반응 법칙: off-distribution forward 양 ∝ 손해.
- C1 구형: SSC coverage ±0(무정보·raw cosine 이방성 audit 실증) · AR evqa −16.4 붕괴(정량질문=복제본 신호)+shuffle 게이트 실패 · hier_preserve 단독 −0.6.
- 심판 v1/v2/v3, β-scale, 셀 캘리브레이션, pulse: 기존 종결 유지.
- 실패 법칙 5: ①mid-cache 불감 ②frozen 계산 불가침 ③1-토큰 병목 ④saliency 무정보 ⑤존재-보존 압축은 질량(정량) 파괴.
- TCGA park10 완주 .0790(+0.6, raw 15→23) — 부분집합 ×1.9는 착시, "작지만 양성"으로 정정.

## 5. 운영 규칙 (메모리 영구화)
- **풀런 전 2케이스 스모크 필수**(smoke/gtex2.json, CUDA_LAUNCH_BLOCKING=1, 3점검사) — VG 4수정이 교훈.
- **채점→보고→대시보드 = 한 동작**, 완료 감시 워처 상시(reported_runs.txt 대조, sleep180).
- 내가 띄운 프로세스만 개입(GPU 6·7·8, PID 계보 확인) · 배선 게이트키핑 금지(전 변형 opt-in 공존) · 큐 개입은 협의.

## 6. 인프라
- 큐: runs/expq/jobs.txt (NAME|ENVS|DS|MODEL|STEPS, ENVS 빈칸 허용), 워커 run_expq.sh <GPU>
- 스모크: sender_relay_exp/smoke/gtex2.json · 8B OOM 해법=trim-restore · VG류 소비-1회 stash 패턴 주의(_vg_prefill_cols 버그 전례)

## 7. ★최종 판정 (0908 낮)
**승자 구성 = LaViM k75: 관찰-층화 압축(B=0.75) + 원순서 recency 재배치** (계층 재조립 아님)
| | GTEx | EVQA | SB | TCGA | PANDA |
|---|---|---|---|---|---|
| LaViM k75 | **23.65 ✓** | **48.44 ✓** | 44.67 ✗ | 4.72 ✗ | 18.59(+0.3 미달) |
- 2/5 셋 합격(각 base +2.2/+3.1, VLMAS +4.4/+1.6), 시각 KV 25%↓ 동반. avg는 base와 동률.
- 판정 근거 사슬: C1 3-way(층화>flat, random엔 열세 — 비용행), C2 3-way(계층 순서 무기여 — Original 최강), **B-문턱**(EVQA 0.5 붕괴 30.5 → 0.75 합격 48.4 = 압축률×정량질문 용량관계), topo 엣지 게이트 실패(shuffle 무차별).
- 기각 확정: 계층 재조립·엣지 보존·bind/read/VG·(압축0.5+재배치는 정량셋 금지).
- efficiency 자동저장 가동: 케이스별 efficiency.json(FLOPs 분해·carry-saving·TTFT·peak mem) — run_expq에 VLMAS_EFF_DUMP=1 기본.
- 잔여 큐: SB/TCGA/PANDA의 reasm·full 세트(매트릭스 완결용), s2shuf, step10 restage 잔여, 2B park 열, projective(보류).

## 8. Baseline 세션 작업분 (0904–07, 별도 세션 — 이 섹션만 그 세션이 관리)
**결과: Qwen {2B,4B,8B} × 5데이터셋 × {Single, VLMAS, Latent base(s5)} = 45/45셀 완주·대시보드 반영** [GPU실측]
- 데이터셋별 판정: **SlideBench = 유일한 정상 무대**(3사이즈 일관 Single≪VLMAS≈Latent, latent가 VLMAS 동급을 5~8배 빠르게 15초 vs 90~111초) · **PANDA/TCGA = backbone-bound**(18셀 전부 상수 붕괴 — PANDA는 셀마다 "0"/"2"/"3" 도배, 랜덤±3pt; TCGA는 Breast/GBM/Ovarian 쏠림, 2.1~7.9) · GTEx/ExpertVQA 중간 지대(기존 판정 유지)
- **채점기 정본**: `rescore_dashboard_bacc.py` — 0806 대시보드 완전 재현(per-item expand_letter→acc_of_seq, gtex/tcga/panda=BACC macro, latest_cases dedup). latent-base 24셀 라이브 대시보드 표시값과 24/24 일치 검증. 규약: **GTEx·TCGA·PANDA 점수는 반드시 BACC**(plain ACC와 1~4pt 어긋남, C2 step5 "+2.1pt"가 BACC선 +0.04로 소멸한 전례)
- **PANDA 배선 완료(추가만, 0806 무수정)**: `ready_wsivqa/full_no_panda/panda.json`(196) + slides 심링크 + 시드 썸네일 196장(`gen_panda_thumbnails.py`) + `prepare_panda_ready.py`. 고질: 바늘생검 조직후보 소진 실패 ~11/196(결정적, 재실행 무효)
- **hj 트리 수리 이력(전부 이 트리 안, off-path 불변)**: ① `latent_onepass.py` question_focus 240자 컷 ② `execution/cli.py` __main__ 가드(+typer 단일커맨드라 `run` 인자 금지) ③ single 체인 부분복사 구멍 4곳 심링크(models/single_v7 thinking·baselines·wsi_vqa_baselines, single_v7/baselines/inference) ④ single 어댑터 `build_file_index` [:12] 키에 full-stem 키 추가(multipath 긴 Id) ⑤ `wsi_vqa_baselines/prompting.py` 보기 문자 26자 하드코딩 2개소 → `_letter_seq`(A..Z,AA..; TCGA 30지선다) ⑥ `qwen_backend.py` temp≤0→greedy(VLMAS temp0 크래시)
- 운영 교훈: GPU co-tenancy가 OOM 주범(PANDA base 87건, SB 8B latent 145건 — expq 큐와 겹침) → **런은 GPU 단독 점유로**; `--output-root`는 기존 dir 거부(사전 mkdir 금지); pgrep 자기매칭 watcher 무한대기 재발 주의
- 실행 스크립트: `run_mp_trio.sh <ds> <model> <gpu>`(Single→VLMAS→Latent 순차, 기존 dir SKIP), `run_panda_2b8b.sh` · 런 위치: `runs/{panda_full_20260904, mp_tcga_20260905, mp_slidebench_20260905}`
- 부수 완결: allsteps 매트릭스 48/48(8B/5 realloc 재실행 128/128), c2_replication 완주, 시간 컬럼(role_calls 합산) 전 셀 기입

- **[0908 GPU8 점유 공지, baseline 세션]** GPU8에서 멀티백본 SlideBench latent-base(InternVL3-8B-hf→Lingshu, runs/mp_backbones_20260908/, ~15h 예상, 0909 오전까지) 완료(0908 15시, GPU8 반납) — SlideBench latent: InternVL3-8B 43.37(10.0s), Lingshu-8B 46.70(37.3s) vs Qwen8B 46.7(15.2s). ★InternVL은 반드시 -hf 체크포인트(원본=가중치 랜덤초기화 runaway).
