# 0908~0910 세션 정리 — 지시 항목별 배선·실험·수치

기준: 4B · latent step 10 · MultiPathQA (GTEx/TCGA/PANDA = BACC, EVQA/SB = ACC). **gtex base = 21.41**(초기 24.21 오독 정정), EVQA 45.31, SB 48.22, TCGA 7.33, PANDA 18.26.
태그: [GPU실측] 실제 런 / [코드읽기] 코드 확인 / [미검증-가설]. 판정("kill/채택")은 사용자 지시로만 닫음 — 아래는 수치와 상태만.
산출물: 대시보드 `https://claude.ai/code/artifact/c65c1dbc-0fc9-463d-9239-f3546c9d4783` · 실험 로그(§28.x) `https://claude.ai/code/artifact/cc558a62-fee0-4399-844d-63da9efc2c35`
코드 트리: 원격 `wsi_latent_0915_decode_hj`(로컬 `0902_2155` 미러). 모든 배선은 env opt-in, off = byte-identical. 큐 `sender_relay_exp/runs/expq/jobs.txt`, GPU 6·7·8.

---

## 1. Search-tool Navigator (지도교수 스펙: "BEACON / Navigator controller", "Tool-Augmented Multi-Scale Navigation (VoI)")
- 배선: `VLMAS_NAV_SEARCH=1` — Navigator는 쿼리만 내고 frozen `WsiSearchTool`이 크롭을 뽑음(쿼리→x5 3 → 관찰 → 쿼리→ x5 안 x20 2씩 = 9장). `VLMAS_NAV_SEARCH_TOOL=tissue|plip`. 불변조건 5 준수. 파일 `wsi_search_tool.py`, `onepass_navigation_search.py`.
- PLIP(vinid/plip) 플러그: safetensors 변환, cosine 재랭킹(`PlipSearchTool`). `VLMAS_NAV_EVENTS=1`(anchor별 fine 쿼리), `VLMAS_NAV_PLIP_PER_ROOT=N`(x5 풀 확장), 2-hop `VLMAS_NAV_TWOHOP=1`.
- 결과 [GPU실측]:

| 셋 | base | 파이프 tissue(쿼리무시 통제) | 파이프 PLIP | 구조 비용 | PLIP 기여 |
|---|---|---|---|---|---|
| GTEx | 21.41 | 20.49 | **23.75** | −0.9 | +3.3 |
| EVQA | 45.31 | 37.50 | 39.06 | −7.8 | +1.6 |
| SB | 48.22 | 39.09 | 39.09 | −9.1 | 0 |
- anchor별 event 쿼리 21.99, 2-hop 21.44(±0, 픽 59.5%가 셀 1~3). TCGA/PANDA PLIP 진행 중.
- 해석(수치): 손실은 파이프 구조(9장·부모 제한·격리 branch·LLM 격자픽 제거), PLIP 랭킹은 통제 대비 항상 ≥0.
- 사용자 결정: **x20은 x5 안에서만(계층 고정)**. 후속 배선(0910 새벽, 전부 큐/진행):
  - `VLMAS_NAV_DETAIL_APPEND=1` KV 연결(navigator_detail을 공유 캐시에 append, bank=x20만) — 스모크 PASS, latent_audits에 `navigator_detail 929토큰`.
  - `VLMAS_NAV_PLIP_COARSE=0`(x5 tissue / x20 PLIP 진단), `VLMAS_NAV_SEARCH_K5=4 VLMAS_NAV_SEARCH_K20=1`(배분).
  - `VLMAS_NAV_QUERY_FROM_PLAN=1|2`: Planner가 x5/x20 서술을 **실제로 decode**(`decode_plan_targets`, 기본 plan 텍스트는 "latent-planned …" placeholder였음) → x5 쿼리; Navigator가 x5 보고 x20 쿼리.
  - PLIP-wide gtex 22.27(127/190, 부분).

## 2. 원래 Navigator + x20만 PLIP ("x20 뽑는 부분만 고쳐")
- 배선: `VLMAS_NAV_X20_PLIP=1` — grid Navigator·x5 anchor·5장·round-robin·계층 유지, x20 셀만 anchor 내부 tissue-safe 셀을 PLIP 랭킹(쿼리 = Planner 서술(QUERY_FROM_PLAN) 또는 질문). `PlipSearchTool.rank_images`, `onepass_navigation_details.py`.
- 결과 [GPU실측]: gtex 질문쿼리 19.97(127/190) · Planner서술쿼리 20.36 / **EVQA Planner서술쿼리 46.88 (+1.6)** / SB 진행. 랭킹이 anchor마다 달라 구 픽의 {1,5,9,13,17} prior 소멸.

## 3. Counterfactual grounding G_t
- 사용자 기각("포워드 두 번, 버려"). 미배선.

## 4. C1 Observation-Stratified (fixed B) / C2 Cross-Scale Reassembly / Graph-Structured Visual KV Reassembly (projective)
- 배선: `VLMAS_WSI_CONSOL_KEEP`(★기본 0.25 함정), 모드 strat/subspace/hier_preserve/topo, `VLMAS_KV_RESTAGE_ORDER=original|random|hierarchy|projective`(projective MinLA), `VLMAS_PATCH_BUDGET`(crop 25).
- 결과 [GPU실측, gtex]: clean 3-way original 22.51 / projective 22.31 / shuffled 21.79 / hierarchy 21.60 → 순서·그래프 신호 없음. c1b25(crop 25) 전 arm OOM(fixed-B 축 미측정). LaViM k75 방법행 gtex 23.65 / EVQA 48.44 / SB 44.67 / TCGA 4.72 / PANDA 18.59.

## 5. AAVM (Acquisition-Addressed Visual Memory) + 진단 3종
- 배선: `memory/aavm.py`, `aavm_runtime.py`(v2=query-표현 주소), 진단 `identity`/`random_norm`/`nokey`.
- 결과 [GPU실측, gtex, 기준선 NAV_SEARCH+EVENTS 20.49]: identity 20.49(배관 무결) / random_norm 7.77 / nokey 20.49 / v2 3.38 vs shuffled 3.00 → key centroid 이동 자체가 파괴. 비용 +16s/케이스.

## 6. AIVA (append-invariant visual attention)
- 배선 `VLMAS_AIVA=1`(`memory/aiva.py`, 유닛 9/9). GPU 훅 발화 0회(미해결). Access 분해상 B_card −0.28이라 지렛대 아님.

## 7. Access 분해 (a)~(d) + Level 1·2
- `memory/access_diag.py` (`VLMAS_ACCESS_DIAG=<dir>`): M_V = B_comp + B_card, C_V = ‖Σα_j v_j W_O‖.
- Reasoner latent step [n=20/38]: M_V 2~6%, B_comp −3.9~−4.05 지배, B_card −0.28, C_V/C 7~12%.
- **Answerer 단계** 배선 `VLMAS_ACCESS_DIAG_TERMINAL=1` [n=38]: 첫 답 토큰 M_V **0.0046**(Reasoner 평균 0.028의 1/6), 생성 평균 0.017/peak 0.041, C_V/C 0.012 / 0.048 / 0.124.

## 8. KV 레이아웃 / 역할 설명 [코드읽기]
- 역할별 `system → images → text → assistant → latent`; latent step 1개 = KV 1컬럼(원조 LatentMAS·0717·0731 동일). Reasoner 블록 = 텍스트 4 · 크롭 8×256=2048 · 텍스트 1403(56% 빈 JSON 템플릿·27% 좌표) · latent 10. 시각은 Reasoner에서 진입.
- 사용자 희망 레이아웃(visual KV를 Navigator에서 append + 최소 Reasoner 프롬프트)은 `VLMAS_NAV_DETAIL_APPEND`로 일부 구현.

## 9. 3-D MRoPE 재앵커 (method.md §Re-anchoring 규정 미구현 발견)
- 라이브는 `arange().expand(3,-1)` 1-D. 배선 `VLMAS_KV_RESTAGE_MROPE=1|shuffle`(유닛 13/13).
- 결과 [GPU실측, park+restage 위, 190]: 3-D 22.87 vs 1-D 22.51 vs h/w 셔플 21.77 — 방향 일치, 크기 1~2문항.

## 10. Level 1~4 프레임워크 + Pathways causal patching (V→A, V→R, R→A suffix ladder)
- 배선: pass W `VLMAS_RPATH=wrong_visual VLMAS_RPATH_DUMP_KV`(Reasoner 경계 donor swap, latent/visual/close K/V 덤프), pass T `VLMAS_RPATH_PATCH`(vis / R{k}:10 / close / 조합 이식, teacher-forced score, `none`·`vis_drop` 대조, 지오메트리 가드). `memory/rpath_probe.py`, `memory/rpath_patch.py`.
- **★probe 결함 발견·수정**: `generate_terminal_json`이 캐시를 in-place 확장(Answerer 프롬프트+답 ≈540~800컬럼 잔류) → 후속 probe가 자기 답 든 캐시에서 디코드. §10 KV-swap "무반응 2/29"·1차 pass T "flip 0%" 모두 이 결함 산물. 수정: 디코드 직전 길이로 crop(`probe_base_len`). 스모크 `cache 6848→6050`, 항등 2/2. 반증 검토 9/9 반증 실패(잔여 위험 → `none`/`vis_drop`/지오메트리 가드 추가).
- Level 3 [5셋]: latent K/V true-vs-wrong 1−cos gtex ≤0.002, tcga ≤0.026, sb ≤0.028, panda ≤0.037, evqa ≤0.053.
- Level 4 [5셋]: wrong 런 flip gtex 17/37 · evqa 8/25 · tcga 11/44 · sb 7/39 · panda 3/38 (포맷 변형 제외 실질 38).
- **최종표 (실질 flip 38)** [GPU실측]:

| 셋 | flip | vis (V→A) | R1:10 (R→A) | close | vis+R | 항등 |
|---|---|---|---|---|---|---|
| GTEx | 17 | **16** | 0 | 1 | 16 | 17/17 |
| EVQA | 4 | **4** | 0 | 1 | 4 | 4/4 |
| TCGA | 11 | **7** | 5*(prior 회귀) | — | 7 | — |
| SB | 3 | 0 | 0 | 0 | 1 | **3/3** (V·R·close 분산) |
| PANDA | 3 | 2 | 1 | — | 2 | — |
| 합계 | 38 | **29** | 6 | | 30 | 24/24 |
- 무이식 대조군 0/38·flip 0%. §10 프로토콜 crop판 재실행: K 10/37 · V 14/37 · **KV 17/37(=실런 flip률)** · 단층 V 0~4/37.
- 수치 해석: 시각 정보의 답 도달 경로 = Answerer가 Reasoner 프리필의 크롭 KV(2048컬럼)를 직접 읽는 V→A; latent step KV에는 슬라이드 정보가 거의 안 담기고 답에 안 쓰임(SB 3케이스만 분산). Answerer의 visual attention은 0.46%인데 인과는 Answerer.
- 정정된 서사: "Answerer는 mid-cache visual을 안 읽는다"(§10) 철회 → restage 이득은 "recency로 옮기면 더 세게 읽힘". 0901 "Answerer vision 차단 무반응" 메모리와 충돌(재검정 필요). §20 arbitration 채점도 같은 grown-cache 가능성[코드읽기].

## 11. C2 — Receiver-Side Latent Deliberation kill test
- 배선: `VLMAS_REASONER_LATENT_STEPS=r`, `VLMAS_ANSWERER_LATENT_STEPS=a`(터미널 직전 `answerer_latent` 블록; 훅은 `generate_final_text` 경로). r+a=10.
- 결과 [GPU실측, gtex 190]:

| allocation | BACC | Δ | wrong-visual flip | 초/케이스 |
|---|---|---|---|---|
| R=10, A=0 | 21.41 | — | 17/37 = 46% | 26.7 |
| R=5, A=5 | 22.36 | +0.9 | 13/37 = 35% | 20.5 |
| R=0, A=10 | 22.43 | +1.0 | 13/37 = 35% | 21.0 |
- flip 집합: base 17 중 12 공통, A10·R5A5 12/13 겹침. 판정 보류(사용자).

## 12. C2 — Visual Evidence Ratio Decoding
- 배선: `VLMAS_ANSWERER_VER=1` — 후보 y마다 `S_V(y)=log p(y|M_V,M_R,Q) − log p(y|M_R,Q)`(M_V=크롭 2048컬럼 제거 사본), argmax로 answer 필드만 교체. `memory/ver_decode.py`.
- 스모크 [gtex 3케이스, 후보 20, +49s/케이스]: case5 Adipose→Lung(S_V +0.91), case95 Colon→Adrenal(+0.63), case185 Skin→Muscle(+1.48). S_V 전 후보 양수, 후보 간 차 0.2~0.6 nat. gtex·EVQA 풀런 진행. [코드읽기] teacher-forced 절대값은 greedy 디코드와 등가 아님(prefix 토큰화 경계) — S_V는 차분이라 상쇄.

## 13. 기타
- 0806 마지막 프루닝 = `qwen_pruning_v3_rolepreserve_prefill_2block_matrix_20260831`(prefill 인코더 2블록 관찰 group-prune, 생존 23~25%, reasoner_only, 3사이즈×4step 735/735).
- Answerer는 base에서 latent step 0회(C2 arm만 예외).
- 대시보드/로그: 완료 즉시 반영 유지. 메모리 갱신: probe 결함·Pathways 최종·Navigator 스위치·C2 배선.

## 14. 진행 중 큐 (0910 03:40)
VER gtex/EVQA · x20plip EVQA/SB(+Planner쿼리 SB) · 검색파이프 plan-first/K5/x5-tissue/KV연결/wide gtex·EVQA · PLIP panda. 밤새 자동 채점, 아침에 표 일괄 갱신.

---

## 부록 A. 이 창에서 주신 지시·스펙 전체 목록 (시간순, 이름 그대로) — 처리 상태

| # | 주신 것 (이름/요지) | 처리 | 위치 |
|---|---|---|---|
| A1 | "BEACON은 검색 tool로만, 우리는 Navigator controller" 흐름 스펙 (Planner latent → ① 5× search → 관찰 → ② 20× search) | 배선 `VLMAS_NAV_SEARCH=1`, 불변조건 5 | §1 |
| A2 | 대시보드 캡처 이미지 2장 (진행 확인) | 대시보드 즉시 갱신 원칙 적용 | — |
| A3 | `\subsection{Tool-Augmented Multi-Scale Navigation}` (VoI 기반 Planner/Navigator 정의) | 코드와 대조해 "지금 navigator 이렇게 된 거 맞아?" 답변 + lavim_method.md §3.5 반영 | §1, method.md |
| A4 | counterfactual grounding G_t 질문 → "포워드 두 번, 버려" | 기각·미배선 | §3 |
| A5 | C1 Observation-Stratified Visual Compression (fixed B) / C2 Cross-Scale Evidence Reassembly 정의 | strat/subspace/hier_preserve/topo + ORDER original/random/hierarchy/projective 배선, 표 §25·§26, `VLMAS_PATCH_BUDGET` crop25(OOM) | §4 |
| A6 | 실험 로그 링크 "여기에 내가 시킨 거 없네?" → "둘 다 최신 유지" | 대시보드·로그 동시 갱신 체계 | — |
| A7 | `\subsection{Graph-Structured Visual KV Reassembly}` (projective tree) | `ORDER=projective`(MinLA) 배선·clean 3-way | §4 |
| A8 | "4. Acquisition-Addressed Visual Memory" 수식 스펙 §4~7 + `\subsection{AAVM}` ("이거면 충분해? 더 알아야 할 게 있어?") | `memory/aavm.py`·runtime v2, 진단 identity/random_norm/nokey | §5 |
| A9 | "같은 병목(readout이 recency만 읽음) 시각화 가능해?" | 실측 앵커점 기반 시각화 제작(중간·recency 실측, 사이는 해석선 명시) | 로그 §15 계열 |
| A10 | anchor-specific 2-hop 결정 + acquisition-event notation g=(T_g,P_g) | `VLMAS_NAV_EVENTS=1` per-parent fine_query array, provenance JSON | §1 |
| A11 | "배선 계속해", "새로운 파일 만들어서 작성해" | 신규 파일(wsi_search_tool.py, onepass_navigation_search.py, aavm*.py, aiva.py, access_diag.py, rpath_*.py, ver_decode.py) | 전체 |
| A12 | AIVA 스펙 "이것도 배선해서 돌려보세요" | `VLMAS_AIVA=1`, 유닛 9/9, GPU 발화 0(미해결) | §6 |
| A13 | "지금 당장 새 C2보다 diagnosis 먼저: Reasoner step마다 (a) M_V (b) Card (c) B_comp (d) C_V" | `memory/access_diag.py` 배선·20케이스 | §7 |
| A14 | "지금 kv 쌓이는 거 시각화" / "planner 안에서도 vision·latent·prompt 순서" | 역할별 KV 레이아웃 표(§28.5), Planner 블록 순서 | §8 |
| A15 | "5. 두 단계 V→R / R→A" 진단 프레임 | Pathways 설계에 편입(V→R=Level 3, R→A=suffix ladder) | §10 |
| A16 | "latent 10번 돌아서 token 1개?" / "LatentMAS 코드 봐봐" | 원조 `latent_vec.unsqueeze(1)` 동일 확인(latent step 1 = KV 1컬럼) | §8 |
| A17 | "1403토큰 뭐야?" | 56% 빈 JSON 템플릿·27% 좌표 목록 분석 | §8 |
| A18 | "비전이라 달라야 하는 거 / 3d / 0731 interpreter line 211~277·416" | method.md Re-anchoring 규정 vs 라이브 1-D 발견 → 3-D MRoPE 재앵커 배선·판정 | §9 |
| A19 | Level 1~4 프레임워크 (두 번 주심) + "네 경우 표" | Level 1·2 Reasoner/Answerer, 3, 4 전부 측정; 우리 상태 = "Reasoner insensitive, Answerer dependent" | §7, §10 |
| A20 | "LaViM에서 가장 먼저 Pathways causal patching" (V→A, V→R, R suffix ladder) | pass W/T 배선, probe 결함 발견·수정, 5셋 최종표 | §10 |
| A21 | "먼저 kv를 이용해서 visual info 어떤 causal pathway 쓰는지 분석" / "다른 데이터셋에서도" | 5셋 완료 | §10 |
| A22 | "4개 agent 각각 역할" / "kv 끊기는 곳 있어? + 희망 다이어그램" / "원래 코드 8장 어떻게" | 역할·끊김 4곳 설명, `VLMAS_NAV_DETAIL_APPEND`로 연결 | §1, §8 |
| A23 | "결론" / "casual pathway 뭐가 문제" / "내가 시킨 거 table로" | 사다리 표·결론(V→A) 보고 | §10 |
| A24 | "구조 비용이 지배 — 무슨 소리?" / "PLIP이 낮다는 거야?" / "뽑는 구조가 뭔데" / "어쩌라는 거" / "8장 문제 있잖아" | 파이프 분해 설명, "구조는 base·선택은 PLIP" 제안 | §1, §2 |
| A25 | "x20은 x5 안에서 뽑게" (계층 고정) → "해봐" | PLIP_COARSE=0 · K5/K20 배선·큐 | §1 |
| A26 | "Planner가 x5/x20 특징 뱉고 → x5 → Navigator → x20, 이렇게 된 거야?" → "해봐" | `VLMAS_NAV_QUERY_FROM_PLAN` + Planner 실제 decode(`decode_plan_targets`) | §1 |
| A27 | "지금 kv 계속 이어져야 하는데 연결 가능?" | `VLMAS_NAV_DETAIL_APPEND=1` 배선·스모크 PASS | §1 |
| A28 | "navigator 원래 버전에서 x20 뽑는 부분만 고쳐볼래?" | `VLMAS_NAV_X20_PLIP=1` 배선·gtex/EVQA/SB 런 | §2 |
| A29 | Slack용 답변 3종 (visual KV 출처 / latent에 안 담김 / Answerer가 vis KV만) | 문안 작성 | §10 |
| A30 | "latent step에서 reallocation으로 시각적 도움 되는 걸 뽑아내면?" (교수) | 실측 근거로 의견(Reasoner 자리 무효 근거, C2 위 realloc 제안) | 본문 §11 참고 |
| A31 | "결론 니 맘대로 닫지 마" | 이후 수치만 보고, 판정 보류 | 전체 |
| A32 | C2 — Receiver-Side Latent Deliberation kill test "제일 우선" | 배선·3 allocation 완료 | §11 |
| A33 | C2 — Visual Evidence Ratio Decoding "배선 가능?" | 배선·스모크·풀런 진행 | §12 |
| A34 | "kv vision/text 분리 + 프루닝: 스텝 간 차이 큰 곳 = 변화한 visual?" | 실측(스텝간 0.0077 vs 시각 0.0003)으로 답, 크롭별 S_V 제안 | 본문 §10 |
| A35 | "0806 원격 마지막 pruning 뭐야?" / "answerer도 latent step 돌아?" | rolepreserve_prefill_2block_matrix_0831 설명 / base는 0회 | §13 |
| A36 | "너가 작업한 거 처음부터 끝까지 md로, 이름 포함" | 이 문서 | — |

이전 창(요약 승계) 지시 중 이 창에서 이어진 것: "각각 3개 한번에 GPU" 운영, "내가 시킨 실험부터", "바로바로 대시보드에 써", "method 형식으로 md 업데이트"(`lavim_method.md` §3.5~3.7 + 검증현황표 — `0902_2155/scratchpad/lavim_method.md`에 복사), "PLIP은?", "C1 C2 뭐야", "Cross-Scale Evidence Reassembly 효과 없었어?".

---

## 15. 이 창(0909 밤~0910) 지시·진행분 — 위 §1~§14는 다른 창, 아래는 이 창에서 처리한 것

기준 동일: 4B·step10, gtex=BACC/evqa·sb=ACC/tcga·panda=BACC, base gtex 21.41·EVQA 45.31·SB 48.22·TCGA 7.33·PANDA 18.26. 태그 [GPU실측]/[코드읽기]/[미검증-가설].

| # | 주신 지시 (요지, 시간순) | 처리·결과 | 위치 |
|---|---|---|---|
| B1 | PROGRESS_0907.md 읽고 진행상황 파악 | 원격 큐·워커·GPU 실측 대조, 정본 동기화 확인 | — |
| B2 | AAVM v2 스펙("acquisition query의 pre-RoPE **query** representation 주소") 배선, 영향 없게 | `VLMAS_AAVM=2`(memory/aavm.py `reduce_query_heads`, aavm_runtime `query_address_v2`), 유닛 22/22, 스모크 9/9 crops·events=4. 풀런 gtex **3.38**(base 21.41)·shuffled 3.00, EVQA 48.44 → 진단: query/key 부분공간 cos≈0.34·주소 이방성 0.98·centroid 이동 자체 파괴(다른 창 진단 3종과 합쳐 **폐기**) [GPU실측] | expq/aavm2_*, aavm2shuf_* |
| B3 | "코드 문제 없는지 확인" | py_compile·유닛·off-arm byte-identical 검증; 스펙 편차 2건(회전 방식·u_g 추출 맥락) 보고 | — |
| B4 | 에이전트별 프롬프트 전문 / "언제부터 x20 parent별?" | 실런 산출물에서 5역할 프롬프트 원문 덤프; parent별 fine query = `VLMAS_NAV_EVENTS=1` 분기(0908 23:52 추가), 기본은 단일 fine query 유지 | — |
| B5 | 로그·대시보드 읽고 최대 문제 분석 → ISSUES_0909.md | 문제 3건(방법 스토리 공동화·§10 probe 결함 소급·EVQA 귀속) + P0~P2 | sender_relay_exp/ISSUES_0909.md |
| B6 | session summary+작업상황으로 "5셋 전부 올리는 법" 분석·점검 → ANALYSIS_5SETS_0910.md | 셋별 갭·성분 상충 매트릭스·Pathways 처방; 점검에서 **VER 폐기 신호**(gtex .1002)·TCGA hier 군집 9.8 탈붕괴 실증 | sender_relay_exp/ANALYSIS_5SETS_0910.md |
| B7 | "하나의 방법론으로 통합" + "+5 목표" | Receiver-Side Latent Deliberation 단일 방법 제안 → 승인: `{A10,R5A5}×{EVQA,SB,TCGA,PANDA}×3 = 24잡` 등록(진행 중) | expq/c2ra_* |
| B8 | C1(Observation-Stratified)+C2(Provenance-Factorized Re-Read) 배선 여부 확인만 | C1=strat 배선·실측 완료 / Re-Read = AAVM v1(key-공간 주소) 배선 완료·풀런 미실행(큐 aavmv1 9잡 대기) 보고 | — |
| B9 | Receiver-Conditioned Visual Paging 배선 | `ORDER=demand`(memory/paging.py: boundary probe→E_p log-mean-exp→오름차순 recency, 페이지별 3-D 앵커) 유닛 14/14·스모크 PASS(E_p 2.5~5.1 변별). ★일괄 앵커는 순서 무효 함정 발견 | memory/paging.py |
| B10 | Counterfactual Dual-Address 배선 | `ORDER=counterfactual`(memory/cf_address.py: p_N/p_-p/p_pR JS-divergence, 2P+1 forward) 유닛 11/11·스모크 8/8 receiver(U_R≈100×U_N) | memory/cf_address.py |
| B11 | Single-Pass Dual-Address Routing 5셋 + "pruning 얹은 버전"으로 10행 | **base+ROUTE**: gtex 23.34(+1.9)·EVQA 46.88(+1.6)·SB 47.72(−0.5)·TCGA 7.61(+0.3)·PANDA 17.53(−0.7) / **strat k75+ROUTE**: 20.19·46.88·46.70·7.61·19.05 — 대시보드 반영 [GPU실측] | expq/route_*, stratroute_* |
| B12 | "Address resolution 결과는?" | 로그 집계: 5셋 **99% receiver**, N은 L1·L7에만; strat 시 PANDA만 94.8%로 갈림 | — |
| B13 | "왜 하이퍼파라미터 없어? 왜 안 돼" → "해봐" | 원인=RoPE 거리감쇠가 raw G_R−G_N을 지배(스펙이 threshold 금지). 위치보정 τ(`VLMAS_KV_ROUTE_TAU`) 배선: 스모크 R 99%→37% | expq/routetau0_* (5셋 큐) |
| B14 | Position-Bias-Invariant Dual-Address (Λ=log π_R−log π_N, softmax shift-invariance) 구현 | `VLMAS_KV_ROUTE_MODE=share` 배선: 스모크 **R≈43%**, Λ −1.9~+1.9. 5셋 15잡 큐(진행 중) | expq/routeshare_* |
| B15 | Observation-Canonical Visual Addressing(o_p 제거, (h,w) 유지, query를 c=1+max u에서 읽기) 배선 | `ROUTE_MODE=canonical`(동치 key 재배치 r_new=(b−c)+u, 프롬프트 프리필 전 1회) 스모크 2/2(c=16, n=2048). 캐시 길이 가드 추가(이전 케이스 bookkeeping 재사용 CUDA assert 수정). 5셋 15잡 큐 | expq/routecanon_* |
| B16 | "큰 실험 전에 kill test 30~50케이스: Base/Restage/Full-deRoPE/Canonical × offset variance·정답 margin·wrong-slide 민감도·visual mass" | gtex40 서브셋 생성, `ROUTE_MODE=derope`(통제)·`ROUTE_MODE=diag`(offset-variance 해석 계산: δ∈{0,256,1024,4096})·VER score-only(`VLMAS_VER_SCORE_ONLY=1`, 답 교체 없이 후보 logP) 배선. **12런(6 arm×{true,wrong}) 완주 → 최종표 KILL_TEST_0910.md: base 13/restage 14/deRoPE 8/canonical 11/canonical-crop 12/π(p) 통제 11 (정답/40), M_V 0.45/3.5/28.7/11.2/13.5/13.7%, margin −3.94/−4.16/−8.91/−1.91/−1.62/−1.58 — B 쪽(접근↑·정답 불변)** | runs/kill_ocva/ |
| B17 | Hierarchy-Constrained Spatial Pruning(5×anchor→20×children 트리 유지: 각 crop ≥1 + 실제 zoom edge마다 부모 π(c) 토큰 mandatory, 나머지는 기존 strat 규칙 b_p∝|V_p| equal-area; 점수 없음; timing·layer 동일) 배선 | `VLMAS_WSI_CONSOL_MODE=hier`(통제 `hier_shuf`=부모 derangement+상대위치 이식) 배선, 유닛 7/7, 스모크 gtex2 2/2(mandatory=13 edges=5, kept 1024/2048). 초기 hier_shuf 겹침0 붕괴 결함 수정 후 재스모크. 5셋 비교(Full/Flat/Hier/Shuf, KEEP=0.5)는 큐 협의 대기 | runs/hier_prune/, HIER_PRUNE_0910.md |
| B18 | WSI-Aware Visual KV Compression 스펙 v2(center seed + bridge(자식 중심 사영) + greedy covering-radius 채움, B≥B₀ 가드, 절대 예산 B∈{64,96,128}, flat/shuffled 통제) 배선 | `MODE=wsi|wsi_flat|wsi_shuf` + `VLMAS_WSI_CONSOL_BUDGET` 배선, 유닛 11/11, 스모크 gtex2 B=96 3 arm(결과 HIER_PRUNE_0910.md) | runs/hier_prune/smoke_wsi*, HIER_PRUNE_0910.md |
| B19 | Scale-Structured Visual KV Compression 스펙 v3(α로 B_C/B_F 분할; coarse=bridge+childless one-center 후 공통 WSI 프레임 전역 FPS; fine=child별 b_c∝N_c one-center+FPS) 배선 | `MODE=scale|scale_shuf` + `VLMAS_WSI_CONSOL_ALPHA`(기본 0.5) 배선, 유닛·스모크 결과 HIER_PRUNE_0910.md | runs/hier_prune/smoke_scale*, HIER_PRUNE_0910.md |
| B20 | (사용자 지시) utilization failure 3단계 절단: ① R-off/A-off ② r_V zeroing ③ depth survival + W_a 수축 | `memory/visual_cut.py`(VLMAS_VCUT=zero\|mask\|identity, STAGE=R\|A\|RA, LAYERS=a-b) 배선. GTEx40 canonical: identity 12/40 · zeroR 13(flip 3) · maskR 12 · zeroA 10(flip 17, margin −1.78→−3.69) · maskA 8 · zeroRA 9 → Reasoner 무관, Answerer direct read causal. W_a≈I(‖W_a−I‖/‖I‖ 2e-6) → alignment 축 종결. 사고 2건: contextmanager 데코레이터 위치·route 북키핑 소실(복원) | runs/vcut/, VCUT_0910.md |
| B21 | case-wise Δm_A·class-space Δz_k·SVD/PCA·mean-direction 제거 (사용자 지시) | Δz_gold +1.78 vs comp +1.68(변별 +0.1), PC1 71%(중심화 69%), cos(Δz,μ_V) .85, argmax Adrenal 19/40, corr(μ_V, −zeroA prior) +.83; μ 제거 후 gold>all 6/40 → **class-biased(non-discriminative) visual readout**; 고정 방향은 L12-23(PC1 89%, μ 에너지 58%) | VCUT_0910.md |
| B22 | breadth: EVQA·SB 24케이스 identity/zeroR/zeroA/zeroA L12-23 | zeroR≈identity 3셋 공통(flip 1/24·1/24); zeroA≪identity gtex·EVQA(−2/−3, flip 42/58%), SB는 방향성 없음(8/8); Δz_gold≈Δz_comp 3셋 공통, gold rank1 6/24=우연 | runs/vcut_breadth/ |
| B23 | Retrieval kill test(latent q·pre-RoPE k → 5×root+20×children 그룹, Full/Random/Token/Context; 10케이스×3셋) | 1차: crop 0이 31/31 1위(순서 편향) → LOO 위치편향 제거+예산 일치 재실행(retr2): Context>Token>Random 불성립(gtex 3/4/5, evqa 4/5/4, sb 6/6/6; disc* context가 random 못 이김) → 음성 | memory/retrieval_group.py, runs/retr, retr2 |
| B24 | (오독) preset pruning_v3/v2/`pruning`(v1) 5셋 on latent base | v3: 18.54/43.75/48.22/6.99/18.59 · v2: 16.69/45.31/46.70/6.17/17.83 · v1: 12.76/44.53/49.20(187)/10.00(101)/19.94(87) — v1 잔여는 사용자 지시로 중단 | runs/prune_v/ |
| B25 | (사용자 지시 정정) 스펙 v3 scale(B=96 α.5) → v2 wsi(B=96) → v1 hier(KEEP .5), latent base 위 5셋 | 15잡 미니 큐 GPU 6/7/8 진행 중(0911 11:15~) | runs/consol_v/ |

운영 사고 기록: jobs.txt 개행 접합 2회(수리·백업), 세션 간 동시편집(aavm_runtime 00:15), 런처 `$O` 미확장(kill test 1차 폐기·재가동), pgrep 자기매칭으로 ssh 세션 자살 2회(패턴 앵커링으로 해결).
