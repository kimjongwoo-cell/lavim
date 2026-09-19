# 인수인계 0915 — C1 #14 QASC 5셋 · salience 위치편향 원인 · #14.5~14.9 오프라인 · C2 Signal Gate · SBSH 배선

> 작성 09-15 07:50 (KST), 08:10 갱신(§9). 이전 문서: `HANDOVER_0914.md` (규칙·기준선·nav3 세팅은 그대로 유효, 여기서는 반복하지 않은 것 위주).
> 태그: [실측] 실행·산출물 확인 / [코드읽기] 코드로 확인 / [미검증] 확인 안 됨 / [해석] 추론.

---

## 0. 규칙 (0914 문서 §0 + 이번 세션 추가)

- 한국어로 답한다. 사용자가 **질문하면 답만** 하고, 허락 없이 작업을 진행하지 않는다.
- 주장마다 태그를 붙인다. **p값 언급 금지.**
- 원격 수정은 A 트리 `/home/users/whddn12316/wsi_latent_0915_decode_hj` 안에서만 한다.
- `0912/`는 읽기만 한다. nav2 파일은 만지지 않는다.
- GPU는 6·7·8만 쓰고, GPU당 한 런이다. **내가 띄운 프로세스만** kill한다(PID 계보 + output-root 이중 확인).
- 원격 `pkill`은 HPC 도구이므로 `kill PID`만 쓴다.
- 새 배선은 2케이스 스모크 후 풀런한다. 결과는 노션 Experiment Log에 8섹션 형식으로 기록한다.
- **런이나 데이터셋이 끝나면 요청 없이** exact 채점 → board_0913 병합 게시.
- (0915 추가) **백엔드 `backbone/qwen3vl.py`를 다른 세션이 계속 수정 중이다.**
  - md5 이력: 05:46 `f67158b2`, 07:26 `52352a58`
  - 이번 세션 배선은 백엔드를 건드리지 않고 `memory/` 모듈만으로 했다. 백엔드를 고쳐야 하면 md5 가드 필수.
- (0915 추가) 원격 CPU 분석은 `OMP_NUM_THREADS=8`로 제한한다. 제한하지 않으면 22코어를 먹어 load 58, 케이스당 1.5분 걸렸다(제한 시 7초).
- (0915 추가) 원격에는 **scipy·umap이 없다.** UMAP 그림은 로컬(`/home/super/hj/venvs/wsi-latentmas-py312`)에서 그린다.
- (0915 추가) **hot 8칸 정의를 섞지 말 것**(§3.2).

## 1. 접속·경로

| 항목 | 값 |
|---|---|
| ssh | `ssh isyse_94_jw` |
| A 트리 | `/home/users/whddn12316/wsi_latent_0915_decode_hj` |
| python | `/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python` |
| 로컬 작업 복사본 | `/tmp/claude-1000/-home-super-hj/a67317d1-9994-4cb0-b72c-699fc25d19c9/scratchpad/rss/` (이번 세션 스크립트 원본) |
| 로컬 그림 | `/home/super/hj/0902_2155/scratchpad/vistok_viz/fig/` |
| 보드 | https://claude.ai/code/artifact/079769f9-cbc1-4858-88e6-1424d5dac92a (로컬 `0902_2155/scratchpad/board_0913.html`) |

## 2. 지금 돌고 있는 것 (07:50)

| GPU | 런 | 상태 |
|---|---|---|
| 6 | QASC 몫 0 · PANDA | 07:19 시작, 66문항 |
| 7 | SBSH 스모크(재) → #14.6 ntrs 스모크 → #14.8 ssda 스모크 | 07:4x 시작. 백그라운드 대기 task `bykj0j0l8`가 `ALL SMOKES DONE`을 기다리는 중 |
| 8 | QASC 몫 2 · PANDA | 07:36 시작, 65문항 |

- QASC 몫 1(GPU7)은 07:25에 전부 끝났다. 전체 QASC 완료 예상은 약 08:30 [추정].
- 스모크 로그 위치:
  - `runs/sbsh_nav3/chain.log`
  - `runs/c1sel_nav3/chain.log`, `runs/c1sel_nav3/smoke_{ntrs,ssda}_gate.txt`
- 할 일: 스모크 통과 여부를 보고한다. QASC PANDA가 끝나면 채점 → 보드 → 노션.

## 3. C1 라인

### 3.1 C1 #14 QASC 5셋 런 (`VLMAS_C1_QASC=1`) [실측, exact, 같은 문항의 nav3 base 대비]

**구현** (`memory/qasc_select.py`) [코드읽기]
- Reasoner 프리필에서 vision block 23의 received attention a_i를 가중치로 쓴다.
- 배율 간 중복 계수 n_cross를 곱하고, 정규화한 merger feature z로 log det(I + Σ a z zᵀ)를 greedy로 최대화해 25%(768/3072)만 남긴다.

**런처·출력:** `sender_relay_exp/qasc_nav3.sh <gpu> <shard> <smoke|wait>` → `runs/qasc_nav3/`

| 셋 | 진행 | QASC | nav3 base | 얻음 / 잃음 | 같은 답 |
|---|---|---|---|---|---|
| ExpertVQA | 128/128 | 42.19% (54) | 39.84% (51) | +5 / −2 | 117 |
| SlideBench | 197/197 | 44.16% (87) | 43.65% (86) | +7 / −6 | 183 (패치 동일 180문항: 44.44 vs 43.33) |
| GTEx | 190/190 | BAcc 19.90 (45) | 22.85 (51) | +1 / **−7** | 173 |
| TCGA | 221/221 | BAcc 5.26 (8) | 5.42 (9) | +1 / −2 | 193 |
| PANDA | 111/196 | 16.67 (33) | 16.67 (33) | 0 / 0 | 111 (둘 다 전부 '0') |

- **GTEx 손실:** skin 2, colon 2, adipose 2, liver 1이고 잃은 답 대부분이 colon으로 갔다. 예전 Q-PRIS 소수 클래스 손실과 같은 패턴이다.
- **원인은 미확정이다.** 규칙 탓인지 25% 예산 탓인지 가르려면 **25% Random 대조 런**이 필요하다. 사용자에게 제안했고 아직 답을 받지 못했다.
- **QASC 성질** [실측]:
  - 선택이 salience 순위에 지배된다(스모크 cover_last 0.87).
  - hot 8칸 몫 0.106(균등 0.031), 군집 커버 0.928(Random 0.990).
  - GTEx 5/20 케이스에서 nav3에도 배율 간 겹침이 있어 20× crop이 최소 4개까지 줄었다. "nav3엔 겹침 0"이라는 옛 기록은 GTEx에서 틀렸다.

### 3.2 salience a_i 위치 편향과 원인 [실측]

**audit** (`runs/vit_sal_audit/full`, crop 1182장, 노션 "Vision-encoder question-agnostic salience audit")
- 가장자리 8칸이 crop salience의 11%를 가져간다.
- 템플릿 상관 +0.33, 조직 상관 ≈ 0.

**원인 검사 4종** (`sender_relay_exp/vit_sink_probe.py`, `runs/vit_sink_probe/{smoke,short}`)
1. 빈 이미지에서도 같은 칸이 뜬다(block 23 템플릿 상관 0.45~0.54).
2. 이미지를 1칸 밀면 hot 칸은 제자리에 남는다(2.9× vs 1.07×, 71/72).
3. 해상도를 바꾸면 절대 index가 아니라 상대 좌표를 따른다(384/768: 0.38/0.29 vs 0.13/0.06).
4. block 23에서 hidden norm과 상관 −0.17 → register 토큰이 아니다.
- **결론:** 위치에 묶인 구조 편향이다. pos_embed 보간인지 2D RoPE인지는 미구별.

**hot 8칸 정의가 세 가지다.** 서로 7/8칸 겹친다.
- audit: 실제 crop, block 평균
- null 보정 표: 실제 crop, block 23
- 그림·할당 표: 빈 이미지 템플릿 b, block 23 = (2,0)(2,15)(4,15)(5,15)(13,0)(13,15)(15,0)(15,7)

### 3.3 #14.5 ~ #14.9 오프라인 점검 (노션 Experiment Log "Null-calibrated salience 오프라인 검증" §4·§7에 전부 기록)
공통 입력: GTEx nav3 20케이스 merger feature `runs/vit_ntrs_gtex/arrays/<key>_arrays.npz` (X fp16, a_raw, w_page, tissue, 기존 선택들), 예산은 케이스 토큰의 25%.

| 버전 | 핵심 | 결과 [실측] |
|---|---|---|
| #14.5 흰 이미지 prior로 나누기 | salience / white prior | 위치 분산 몫 0.256 → **0.566(역전·증가)**, 테두리 0.094 → **기각 방향** |
| #14.6 null-template 성분만 빼기 (γ̂ 사영) + 영역별 facility location | residual p + (1+cos)/2 coverage, 전역 greedy | γ̂ 중앙 0.34, 위치 몫 0.124. 할당 sd **4.8**(51–78, Random 5.9보다 균등). d_min **0.1154**, 군집 커버 0.982 |
| #14.7 영역별 log-det | q = √w·z | 선택 97.6%가 residual Top-k와 같음(w≈0.003 → 선형 영역). c를 키워도 겹침 ≥76%, d_min 0.129 |
| 사용자 식: crop spectrum water-filling | 무가중 공분산, 전역 top-B 고유값 | k_g 56–74, sd 3.3, ρ(k, trace) +0.50 |
| #14.8 SSDA | 가중 spectrum → k_g + crop 안 FL | sd 3.0, **#14.6과 선택 97.8% 동일**, d_min 0.1155, 군집 0.981 |
| #14.9 Spatial-Graph RD | 격자 graph geodesic, eccentricity 초기값 | (25% 예산) sd 3.6, #14.6 겹침 0.49, d_min 0.1228. 위치 성분 제거(β 0.185)해도 배경/조직 edge cost 0.424/0.383 불변 |

**주의 (#14.9)**
- HSV 조직 마스크는 **지방 세포 내부를 배경으로 센다.** 그래서 "graph가 배경을 복잡하다고 본다"는 해석은 철회했다.
- 첫 #14.9 로컬 점검은 예산 768 고정 오류가 있었다(9-crop 케이스 과선택). 정정해 기록했다.

**그림**
- 공간 선택 지도(하늘색 = hot 8칸):
  - `runs/vit_ntrs_gtex/*_spatial.png`
  - `runs/vit_ssda_gtex/*_spatial.png`
  - `runs/vit_sgrd_gtex/{gtex_00,04,08,12}_spatial.png`
  - 옛 8-crop: 로컬 `fig/gtex10_ntrs/`
- spectrum: `runs/vit_waterfill_gtex/`(20장 + aggregate), 축 수정판 `runs/vit_waterfill_gtex_fig/`
- UMAP 2D: 로컬 `fig/gtex_nav3_div2d/`, `fig/gtex_nav3_div2d_ssda/`, `fig/gtex10_div2d/`

**사용자 판단 대기**
- "#14.8로 해도 문제없지?" → 결함 없음, 다만 #14.6과 사실상 같다고 답했다.
- 어느 쪽을 GPU로 돌릴지는 미정.

### 3.4 C1 런타임 배선 (GPU 미실행, 스모크 대기)
- `memory/ssda_select.py` (md5 `ca5cdba7`), null 템플릿 `memory/ssda_null_block23.npz`
  - `VLMAS_C1_QASC=1 VLMAS_C1_SELECT=ssda` → #14.8, 로그 `[SSDA] kept`
  - `VLMAS_C1_QASC=1 VLMAS_C1_SELECT=ntrs` → #14.6, 로그 `[NTRS] kept`
- `memory/qasc_select.py` (md5 `17274ab8`): 맨 앞에 env 분기만 추가. env가 없으면 기존 QASC와 동일하다.
  - 백업: `.bak_0915_ssda`, `.bak_0915_ntrs`
- 테스트:
  - `tests_hj_test_ssda.py` 27/27 (water-filling 전수 최적, 가중 고유값, FL greedy, 실제 GTEx에서 #14.6 오프라인과 100% 일치)
  - `tests_hj_test_qasc.py` 23/23
- 스모크: `sender_relay_exp/c1sel_smoke.sh "<gpus>" ntrs ssda` → `runs/c1sel_nav3/`

## 4. C2 라인

### 4.1 canonical / re-read (0914 문서 이후 결론) [실측]
- **canonical만 켠 런 (nav3):**
  - ExpertVQA 32.81 (42) · SB 39.59 (78) · GTEx 24.54 (52)
  - base는 39.84 / 43.65 / 22.85
  - Canonical·ReRead 런과 같은 답이 127/128 · 195/197 · 188/190 → **re-read 몫 ≈ 0**
  - TCGA·PANDA는 사용자 지시로 kill했다.
- **canon_diag:** key를 다시 회전하는 것만으로 답의 1/4~1/3이 바뀐다. **canonical은 C2로 실패.**
- **re-read 4-arm:** 40/40 같은 답.

### 4.2 Role-Sensitive Support Signal Gate (RSS, 노션 페이지 갱신 완료) [실측]
- `memory/rss_diag.py` (md5 `a5bf76bb`), 런처 `rss_gate_gtex.sh` → `runs/rss_gate_gtex/`
- **Stage A:** PASS (identity 오차 ≤ 2.8e−6)
- **Stage B (25케이스):** ρ(Ŝ, mass) 0.94, 최대 support x5 24/25
- **Stage C (LOO 10케이스):**
  - Answerer JS 기준 ρ: Ŝ 0.28, mass 0.20. 짝지은 Δ −0.001, top-1 1/10.
  - Reasoner state 기준 ρ ≈ 0. ρ(S_R, S_A) 0.08.
- **판정:** Gate 1 불통과, Gate 2(Navigator) 미실행 → RSVMH 성능 런 조건 미충족.
- RSVMH(`memory/rsvmh.py`, md5 `44f3c225`)는 배선만 되어 있고 실행하지 않았다.

### 4.3 Consumer-Conditioned WSI Support Handoff (11.8) — 리뷰만
사용자에게 지적한 점:
1. Stage C LOO는 Reasoner replay가 섞여 있어 Answerer oracle이 아니다.
2. "restage + 원래 순서" 대조군이 필요하다.
3. 효과가 작다(top 토큰 변화 11/120).
4. Gate 1의 gradient 점수는 Ŝ와 같은 항이다.
5. C2 headroom이 작다.

제안: 6-arm Gate 0. 미실행.

### 4.4 State-Backed WSI Support Handoff (SBSH, 11.9) — 배선 완료, GPU 스모크 재실행 중
- **`memory/sbsh.py`** (md5 `75d0aa6b`)
  - crop마다 노출 좌표 u_g를 두고 모든 층·모든 Reasoner latent step의 attention logit에 더한다.
  - S_g = ‖∂z̄_R/∂u_g‖²를 Hutchinson(K=16, 기본) 또는 FD(eps 0.1)로 추정한다.
  - 캐시는 deepcopy 후 crop(pre_len)으로만 쓴다.
- **연결:**
  - `rss_diag.ReasonerProbe.after_step`(마지막 step) → `ctx.sbsh`
  - `rsvmh` `VLMAS_KV_RESTAGE_ORDER=state_backed` / `state_backed_rev`
- **env:**
  - `VLMAS_SBSH=1`, `VLMAS_SBSH_EST=hutchinson|fd|both`, `_K`, `_EPS`
  - **probe가 생기려면 `VLMAS_RSS=<jsonl> VLMAS_RSS_NO_LOO=1`이 반드시 필요하다.** 백엔드 게이트가 옛 ORDER 튜플만 인식한다.
- **07:26 첫 GPU 스모크: OOM**(GPU 48GB).
  - 원인: SDPA에 mask를 주면 KV를 GQA 배수만큼 복제하고 math kernel로 떨어진다.
  - 수정: replay 동안만 `ALL_ATTENTION_FUNCTIONS[impl]`을 gated attention(브로드캐스트 GQA)으로 교체하고 끝나면 복원한다.
  - 테스트 `tests_hj_test_sbsh.py` 32/32 (zero gate = native, gate −∞ = 열 삭제, autograd = FD, Hutchinson ≈ exact, gated = sdpa, 등록부 복원). RSS 44/44 · RSVMH 26/26 회귀 통과.
  - 실패한 출력: `runs/sbsh_nav3_OOM_0915_0726`
  - 함정: RMSNorm이 내부에서 float32로 캐스팅한다 → FD eps 1e−5는 틀리고 0.03~0.1이어야 autograd와 맞는다.
- **런처:** `sender_relay_exp/sbsh_nav3.sh "7 6 8"` → `runs/sbsh_nav3/`
  - smoke_signal (EST=both) → smoke_handoff (ORDER=state_backed, MROPE=1)
- **스모크 통과 후** 게이트(S vs mass·LOO)나 성능 런은 사용자 확인을 받고 진행한다.

## 5. 노션 (이번 세션에 만들거나 고친 페이지)

| 페이지 | 내용 |
|---|---|
| Vision-encoder question-agnostic salience audit (3dbd771b-c9e2-8116-…) | 원인 검사 4종 추가, 가설 정정 |
| Null-calibrated salience 오프라인 검증 (3dbd771b-c9e2-81ea-b824-ca4604818830) | #14.5 본문 + §7에 #14.6·#14.7·crop 할당 원인·2D·water-filling·#14.8·#14.9(+정정) 누적 |
| [C2 구조 검증] Role-Sensitive WSI Support Signal Gate (3dbd771b-c9e2-81dd-…) | Stage A~C 결과 8섹션, 사전등록 원문은 아카이브로 |
| (0914 세션 작성, 참고) Context-Updated Answerer Query Re-Read / nav3 Canonical·ReRead 5셋 / Canonical addressing 모델 단 진단 | — |

- 미기록: QASC 5셋 최종(PANDA 끝나면 작성), SBSH·NTRS·SSDA GPU 스모크 결과.

## 6. 보드 (board_0913)
- QASC 행과 필터("C1 #14 QASC")를 메인 표에 추가했다.
- 07:45 갱신: ExpertVQA·SB·GTEx·TCGA 완료, PANDA 111/196.
- 게시 전 `node --check`로 JS 문법을 확인한다. 같은 파일 경로로 재게시하면 같은 URL이 유지된다.

## 7. 남은 일 (우선순위)
1. **스모크 결과 확인·보고** (`bykj0j0l8`): SBSH(재), NTRS, SSDA 각각 PASS/FAIL. SBSH는 floor_cos, Hutchinson–FD 순위 상관, ρ(S, mass), 소요 시간을 본다.
2. **QASC PANDA 완료** → exact 채점 → 보드 → 노션 Experiment Log(QASC 5셋 8섹션). GTEx 클래스별 손실 포함.
3. **사용자 결정 대기:**
   - 25% Random 대조 런
   - #14.6 vs #14.8 GPU 5셋 런
   - SBSH 게이트 / RSVMH 성능 런
   - Consumer-Conditioned Gate 0
4. **위치 부기 버그** (WSI_CONSOL 뒤 restage/park/route의 pos 오인덱싱) chip `task_8eb0ed4f` 사용자 대기.
5. `reread_only_nav3.sh`는 준비만 되어 있고 실행하지 않았다.

## 8. 코드·md5 요약 (A 트리)

| 파일 | md5(앞 8) | 비고 |
|---|---|---|
| memory/qasc_select.py | 17274ab8 | ssda/ntrs 분기 |
| memory/ssda_select.py | ca5cdba7 | #14.8 + #14.6 런타임 |
| memory/ssda_null_block23.npz | 40b8fe5f | 빈 이미지 5장 block 23 지도 |
| memory/sbsh.py | 75d0aa6b | gated attention 교체판 |
| memory/rss_diag.py | a5bf76bb | SBSH 호출·기록 |
| memory/rsvmh.py | 44f3c225 | state_backed 순서 |
| backbone/qwen3vl.py | 52352a58 (07:26) | 다른 세션이 수정 중. 이번 세션 무수정 |

- 분석 스크립트(`sender_relay_exp/`): `vit_sink_probe.py`, `vit_nullcal.py`, `vit_ntrs_gtex.py`, `vit_waterfill_gtex.py`, `vit_ssda_gtex.py`, `vit_sgrd_gtex.py`, 런처 `qasc_nav3.sh`, `rss_gate_gtex.sh`, `sbsh_nav3.sh`, `c1sel_smoke.sh`
- 로컬 그림 스크립트: `0902_2155/scratchpad/vistok_viz/{vistok_ntrs.py, vistok_div2d.py}`

## 9. 08:10 갱신
- **GPU 스모크 결과** [실측, GPU6 07:57~08:03]:
  - #14.6 ntrs: PASS. 결과 2/2, `[NTRS] kept 768/3072`, crop별 57–83, 선택 0.5초
  - #14.8 ssda: PASS. crop별 58–71, 다만 선택에 케이스당 **120~133초** 걸림
  - 원인: N×d SVD가 파이프라인 프로세스 안에서 BLAS 스레드 경합으로 느려짐
  - 수정: 256×256 Gram 행렬 eigvalsh로 교체. `ssda_select.py` md5 `1e854634`, 백업 `.bak_0915_gram`, 테스트 27/27. GPU 재확인은 안 함.
- **SBSH 재스모크(07:55, GPU6)도 OOM이었다.**
  - attention 교체만으로는 부족했다. 원인은 DynamicCache가 매 층·step마다 KV 전체를 concat 복사하고, matmul이 query 그룹 수만큼 KV를 펼쳐 저장하기 때문이다.
  - 수정: `SplitCache`로 교체. Reasoner 시작 prefix KV는 한 번만 연속 메모리로 두고 공유하며, latent KV만 리스트로 쌓는다. query 그룹은 query 축으로 접어 KV 확장이 없다.
  - `sbsh.py` md5 `773caf1a`, 백업 `.bak_0915_split`, 테스트 34/34. prefix 800 vs 40에서 역전파 저장량 증가가 KV 복사 비용의 1/12.
  - **GPU 재스모크는 아직 안 했다.** `sbsh_nav3.sh`로 다시 돌려야 한다.
- **nav3 x5 clamp(PANDA Navigator 실패 11문항):** §memory `wsi-latentmas-nav3-x5-clamp`. `tmp_hj/nav3_x5_clamp/`(원본 무수정 sitecustomize 훅), 테스트 20/20. GPU 런은 안 함.
  - 켜기: `PYTHONPATH=<repo>/tmp_hj/nav3_x5_clamp:<repo> VLMAS_NAV3_X5_CLAMP=1`
- **사용자 결정: "14.6 쓸거야 우선"** → `sender_relay_exp/ntrs_nav3.sh <shard> 3 "6 8 7"` 3개를 08:08에 띄웠다. 출력 `runs/ntrs_nav3/`.
  - 각 몫이 빈 GPU를 먼저 잡는다.
  - **08:04부터 다른 세션 posgap이 GPU6·7을 사용 중**이라, 현재는 GPU8(QASC PANDA 끝나는 대로)만 받을 수 있다.
- **QASC:** GPU8 PANDA가 마지막이다. 끝나면 exact 채점 → 보드 → 노션.
