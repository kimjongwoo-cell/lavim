# 인수인계 0914→0915 — 채점기 통일 · SB sdpa · nav3 Canonical/ReRead · canonical 진단 · a_i 감사

> **09-15 02:00 갱신**: §9(09-14 오후~09-15 작업), §7(남은 일), §0 규칙 추가분.

> 다른 창에서 이어받기용. **0913 대시보드(board_0913.html)와 wsi_latentmas_0913 트리 작업은 제외**했다.
> 태그: [실측] 실제 실행·산출물 확인 / [코드읽기] 코드로 확인 / [미검증] 확인 안 됨.

---

## 0. 먼저 읽을 규칙 (사용자 지시)

- 답은 **한국어**. 사용자가 **질문하면 답만** 하고, 허락 없이 작업을 진행하지 않는다.
- 주장마다 [실측]/[코드읽기]/[미검증] 태그. **p값 언급 금지**(효과는 문항 수로).
- 원격 수정은 **A 트리 안에서만**: `/home/users/whddn12316/wsi_latent_0915_decode_hj`
- `0912/` 폴더(이미지 보낸 사람 작업)는 **읽기만**. `sender_relay_exp/nav2v2_mine_run.sh` 등 **nav2 파일은 만지지 말 것**.
- GPU는 **6·7·8만**, GPU당 한 런. **내가 띄운 프로세스만** kill(PID 계보 + output-root 이중 확인).
  - 09-14 02:06부터 다른 세션이 `cross_scale_router_pruning_b_relay2_prompt1` 를 GPU 0/6/7/8에서 돌리고 있다. 건드리지 말 것.
- 실험은 데이터셋별 20케이스(5셋×20), 새 배선은 2케이스 스모크 후 풀런.
- 실험 결과는 노션 Experiment Log에 8섹션 형식으로 기록.
- (09-14 추가) GPU **6·7·8 전부 써도 됨**("678 다 써도 되니까"). 사용자가 "방법론 세팅"이라 하면 4-arm 실험이 아니라 **arm 하나**.
- (09-14 추가) **런·데이터셋이 끝나면 요청 없이 바로** exact 채점 → board_0913 병합 게시(§9.2).
- 위 "다른 세션 cross_scale_router_pruning_b_relay2_prompt1" 메모는 09-14 기준. 09-15 01:55에는 GPU 6·7·8 전부 내 체인(§9.10).

## 1. 접속·경로

| 항목 | 값 |
|---|---|
| ssh | `ssh isyse_94_jw` |
| A 트리 | `/home/users/whddn12316/wsi_latent_0915_decode_hj` |
| python | `/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python` |
| 데이터 | `/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda/{gtex,tcga,panda,tcga_expert_vqa,tcga_slidebench}.json` (+`slides/`) |
| 모델 | `/home/users/whddn12316/models/Qwen3-VL-4B-Thinking` |
| 로컬 미러 | `/home/super/hj/0902_2155` (이 파일 위치) |

ssh 주의: 원격 명령 안에서 `&` 로 띄우고 같은 줄에서 또 ssh 를 부르면 로컬 명령이 안 끝난다.
띄우는 명령과 확인 명령은 ssh 호출을 분리한다. glob 은 `"0912/runs/*nav2*"` 처럼 따옴표로 감싼다.

## 2. 실험 기준 세팅

Nav2 · patch 12 · max-model-len 12288 · latent step 10 · **SDPA** · Qwen3-VL-4B-Thinking. [코드읽기]

```
VLMAS_ATTN_IMPLEMENTATION=sdpa  VLMAS_NAV2=1   (VLMAS_PROMPT_SET 미설정 = prompt1 과 같음)
python -m wsi_latentmas.pipeline.latent_mas --variant base --backbone qwen3-vl \
  --latent-steps 10 --patch-budget 12 --max-model-len 12288 --navigator-control-tokens 512 \
  --temperature 0.0 --top-p 1.0 --seed 42 --deterministic --answerer-greedy --no-answerer-thinking \
  --answerer-max-new-tokens 512 --answerer-rationale --answerer-protocol structured_json \
  --canonical-open-options --navigator-kv --io-pipeline --no-save-navigation-pngs --case-retries 0
```

- attention 백엔드는 로그·run_manifest 어디에도 기록되지 않는다 → 런처가 env 를 따로 남겨야 한다.
- `backbone/qwen3vl.py` 기본값은 **eager**. env 를 안 주면 eager 로 돈다.
- `latent_agent_prompts.prompt2_enabled()` 는 `VLMAS_PROMPT_SET=="prompt2"` 일 때만 True.
- `--output-root` 는 **존재하지 않는 새 폴더**여야 한다(아니면 rc=2 "output root must be new").

## 3. 기준선 = 이미지 보낸 사람의 Base (0912/live_dashboard.py)

사용자가 이 Base를 기준선으로 채택했다. 4셋은 sdpa·Nav2로 확인됐고, SlideBench만 grid navigator라 재실행 중이다(§5).

정확일치(exact) 채점 기준, `live_dashboard.collect_uncached()` 의 Base 행 [실측 09-14 02:0x]:

| dataset | 지표 | n | 정답 | 점수 | 초/문항 | 읽는 런 폴더 (`0912/runs/`) |
|---|---|---|---|---|---|---|
| tcga_expert_vqa | Acc | 128 | 52 | 40.62 | 17.8 | `nav2v2_base_expert_*` |
| gtex | BAcc | 190 | 44 | 19.83 | 13.39 | `*nav2*` 중 prompt2 제외, variant=base |
| tcga | BAcc | 221 | 15 | 5.75 | 21.3 | 위와 같음 |
| panda | BAcc | 196 | 53 | 16.67 | 14.3 | 위와 같음 |
| tcga_slidebench | Acc | 197 | 89 | 45.18 | 12.73 | `tcga_slidebench_base_12patch_*` (**grid — 교체 예정**) |

- live_dashboard 규칙: `normalized(v)=" ".join(v.casefold().strip().split())`, index별 result.json mtime 최신 1개,
  gtex/tcga/panda BAcc(gold 라벨별 recall 평균), 나머지 Acc. [코드읽기]
- live_dashboard 를 import 하려면 `sys.modules["live"]=module` 등록 후 exec 해야 한다(dataclass 오류).

## 4. 채점기 — 새로 만든 파일 (09-14)

### 4.1 배경
채점기가 셋이 섞여 있었다.
- **old** `eval/metrics.py` `expand_letter`→`acc_of_seq` (`scripts/dashboard_rows.py` 가 사용): 버그 2종.
  `quick_ratio` 동점이면 정답 → `CHOICE`·`text`·`{` 같은 형식 실패도 정답 처리.
  `expand_letter` 가 "B-cell" 의 B 를 보기 B 로 읽음. **쓰지 말 것.** 공용 파일이라 수정도 금지.
- **exact** 이미지 대시보드 방식. 사용자가 "이미지 보낸 사람 채점기랑 똑같이" 로 통일을 지시했다.
- gigapixel-goblin(GIANT) 방식은 채택 안 함.

exact 가 놓치는 진짜 정답은 띄어쓰기·구두점만 다른 경우다. 그래서 정리 level 을 추가했다.

### 4.2 파일 (전부 A 트리, 신규 — 기존 파일은 수정 안 함)

| 파일 | 역할 |
|---|---|
| `eval/answer_match.py` | `normalize(text, level)`, `collisions(choices, level)`, `judge(pred, gold, choices, level)->(맞음, exact로 되돌아감)` |
| `scripts/rescore.py` | 런 폴더를 exact/norm/nospace 세 level 로 채점. 집계는 live_dashboard 와 동일 |
| `tests_hj_test_answer_match.py` | 유닛 + live_dashboard Base 5행 재현. **57/57 통과** [실측] |

level 정의:
- `exact` = live_dashboard `normalized()` 와 같음.
- `norm` = NFKC → casefold·공백 정리 → 같은 구두점 연속(`,,`) 하나로 → 앞뒤 `. , ; :` 제거.
- `nospace` = norm + 공백 전부 제거.
- 지우지 않는 것: 하이픈, `+/-`, 괄호, 가운데 소수점, 관사. 유사도 비교는 쓰지 않는다.
- 한 문항 보기 둘이 같은 문자열로 합쳐지고 정답이 거기 걸리면 그 문항은 exact 로 판정한다.
  exact 로 맞으면 모든 level 에서 맞다(단조).
- BAcc 클래스 키는 모든 level 에서 exact 정리한 gold.
- 보기는 result.json 에 없어서 데이터셋 JSON 에서 `dataset_index` 로 찾고, Answer 가 gold_answer 와 같을 때만 쓴다.
  **20케이스 부분집합 런(dcs20_*)은 인덱스가 부분집합 기준**이므로 `--dataset-json sender_relay_exp/smoke/dcs20_<ds>.json` 필수.

사용법:
```bash
cd /home/users/whddn12316/wsi_latent_0915_decode_hj
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
$PY tests_hj_test_answer_match.py
$PY scripts/rescore.py --collisions
$PY scripts/rescore.py --root "0912/runs/nav2v2_base_expert_*" --dataset tcga_expert_vqa --variant base --changed
$PY scripts/rescore.py --root "0912/runs/*nav2*" --exclude prompt2 --dataset gtex --dataset tcga --dataset panda --variant base
$PY scripts/rescore.py --root sender_relay_exp/runs/nav2v2_mine/sweep/csis_1024 --dataset gtex --dataset-json sender_relay_exp/smoke/dcs20_gtex.json
```

### 4.3 확인한 결과 [실측]
- **보기 충돌: 5셋 전부, 세 level 모두 0건.** SlideBench gold 라벨 1쌍(`T2N0M0.` / `T2N0M0`)만 norm 에서 합쳐지는데,
  SB 는 Acc 이고 클래스 키가 exact 라 점수에 영향 없음.
- **이미지 보낸 사람 Base 5셋: exact = norm = nospace, 한 문항도 안 바뀜.**
- 옛 eager base 런(`sender_relay_exp/runs/expq/`)에서는 바뀐다. 사용자에게 설명한 4건이 이것이다:

| 런 | exact | norm | nospace | 바뀐 문항 |
|---|---|---|---|---|
| `expq/strat_evqa10` EVQA | 32.81 (42) | 32.81 | 33.59 (43) | idx26 `MultipleSmall to medium…` (nospace) |
| `expq/sb_base10` SB | 43.65 (86) | 44.67 (88) | 45.18 (89) | idx12 `pT2 pNXpMX`(nospace), idx23 `…gastritis`(norm), idx129 `adenocarcinoma,, NOS`(norm) |

### 4.4 결정 — **exact 로 평가** (사용자 09-14: "내가 준 이미지랑 똑같은 걸로 평가해")
- 보고 수치는 전부 `exact`(= live_dashboard). norm/nospace 는 참고용으로 파일에만 남긴다.
- exact 재채점 결과 [실측] — `sender_relay_exp/exact_rescore.py`:

Q-PRIS 전체 5셋 (eager·budget8·mlen8192·step10, B=512; 같은 index 집합)
base 출처: TCGA·SB·PANDA = `expq/{tcga,sb,panda}_base10`, GTEx·EVQA = 0806 트리 `multipathqa_*_qwen_base_allsteps_*/…/4b/step10`(대시보드 기준선 원천). EVQA는 old 45.31을 내는 폴더가 여럿이라 exact로 갈린다 — `expq/strat_evqa10` 기준이면 32.81, Δ 0.00(2/2).
20케이스 3 arm 옛 표(`qpris_table.py`)는 dcs20 index를 전체 JSON에 매겨 18~19/20 gold 불일치 → 무효.

| ds | n | old base→Q | **exact base** | **exact Q-PRIS** | Δ | 얻음/잃음 |
|---|---|---|---|---|---|---|
| EVQA (Acc) | 128 | 45.31→42.97 | 34.38 (44) | 32.81 (42) | −1.57 | 2/4 |
| GTEx (BAcc) | 190 | 21.41→16.62 | 21.41 (46) | 16.62 (36) | −4.79 | 2/12 |
| TCGA (BAcc) | 220 | 7.33→7.34 | 6.53 (14) | 6.54 (13) | +0.01 | 3/4 |
| SB (Acc) | 197 | 48.22→47.72 | 43.65 (86) | 43.15 (85) | −0.50 | 2/3 |
| PANDA (BAcc) | 185 | 18.26→19.03 | 16.01 (49) | 16.34 (50) | +0.33 | 2/1 |

Q-PRIS 20케이스 3 arm (qn=none / qt=true / qs=shuffled): EVQA 9/10/9, GTEx 3/1/1, TCGA 1/1/1, SB 6/6/6, PANDA 5/5/5 (정답 수/20).

CSIS 스윕 GTEx dcs20 (BAcc, 정답/20, base와 같은 답):
- budget8 트리 B=512: base 6.41 (2) · pcris_v2 6.06 (2/18, 12/18) · csis_256 5.13 (2, 15/20) · csis_1024 12.82 (3, 15/20) · csis_4096 5.13 (2, 14/20)
- nav2v2 세팅 B=768: base 14.10 (3) · csis_256 14.10 (3, 18/20) · csis_1024 11.54 (2, 17/20) · csis_4096 11.54 (2, 16/20)
- 20케이스라 1문항 차이가 BAcc 수 pt 로 보인다. 결론 금지.

- 09-14 03:1x 0913 보드(최신판 위에 norm/nospace 표기 제거·exact 확정 문구)·노션 Q-PRIS 페이지 §1·§4·§5·§7·Key result 를 exact 로 교체 완료.

## 5. SlideBench sdpa Base 재실행 (완료)

> **09-14 04:26 완료** — SlideBench sdpa·Nav2 Base 197/197 (rc=0, tb 0, OOM 0, selection trace 197/197 `nav2-global-joint`). exact **46.70% (92/197)**, 모델호출 14.2s/문항. grid Base(45.18, 89) 대비 같은 답 171/197, Nav2만 맞춤 10 · grid만 맞춤 7. 0913 보드 BASELINE12 에 채택값으로 반영(다른 세션의 QACSIS 섹션 위에 병합).

- 런처: `sender_relay_exp/sb_sdpa_base.sh <gpu> [first last]` — 위 §2 플래그, sdpa, NAV2=1, 197문항.
  실행마다 `runs/sb_sdpa_base/partFFF_LLL/attempt_<시각>/` 에 쓰고 meta·log 는 그 옆 `attempt_<시각>.meta.json/.log`.
  이미 result.json 이 있는 index 는 건너뛴다(이어 돌리기).
- 대기열: `sender_relay_exp/sb_wait_launch.sh` — GPU 6/7/8 중 <2GB 가 60초 유지되면 그 카드로 `exec sb_sdpa_base.sh`.
  **09-14 02:07 재투입, PID 2770536 대기 중** [실측]. 상태: `runs/sb_sdpa_base/chain.log`.
- 이력: 01:31 첫 실행이 출력 폴더에 run_meta.json 을 먼저 만들어 rc=2 로 즉시 종료(결과 0).
  런처를 attempt 폴더 방식으로 고쳤고, 실패 흔적은 `part000_196/attempt_20260914_013144_FAILED_rc2.*`,
  원본 스크립트는 `runs/sb_sdpa_base/sb_sdpa_base.sh.bak_0914_0131fail`.
- 수정 후 실제 GPU 실행은 아직 안 됐다 [미검증]. 시작되면 `attempt_*.log` 에 traceback 없는지, result.json 이 쌓이는지 먼저 본다.
- 끝나면: `scripts/rescore.py --root sender_relay_exp/runs/sb_sdpa_base --dataset tcga_slidebench` 로 채점 → SB 기준선 교체.

확인 명령:
```bash
ssh isyse_94_jw 'tail -5 /home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs/sb_sdpa_base/chain.log; pgrep -af "sb_wait_launch|sb_sdpa_base"; find /home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs/sb_sdpa_base -name result.json | wc -l'
```

## 6. C1 선택 실험 현황 (A 트리)

### Q-PRIS — 종결
- `VLMAS_WSI_CONSOL_MODE=qpris`, `VLMAS_QPRIS_Q={none,true,shuffled}`, B=512/2048, **eager**, step10, budget8, mlen8192.
- 런: `sender_relay_exp/runs/qpris`(20케이스 3 arm), `runs/qpris_full/qt_<ds>`(전체 5셋 true arm). 드라이버 `qpris_full_chain.sh`.
- 결과: 질문 투영이 같은 rank 무작위 부분공간과 구별 안 됨, qt = shuffled 99/100 동일. 전체 5셋 3하락 2동률,
  GTEx −4.79 = 소수 클래스 10문항. (old 채점기 수치)
- 노션 Q-PRIS Experiment Log `3dad771b-c9e2-8150-bb1c-c1c4ce158cf9` (§9 결과분석까지).

### PCRIS v2 — 사용자가 "안 해도 돼". 중단.

### CSIS (Cross-Scale Innovation–Support Selection) — 배선·스모크 완료, 정확도 채점 미완
- 코드: `memory/csis_select.py`(신규), `backbone/qwen3vl.py` 에 `elif mode == "csis":` 분기(opt-in, 끄면 byte-identical).
- env: `VLMAS_WSI_CONSOL=1 VLMAS_WSI_CONSOL_MODE=csis VLMAS_WSI_CONSOL_BUDGET=<B>`,
  `VLMAS_CSIS_COND/CONTEXT/NULLCAL/SIGMA/SIGMA_SCALE/DUMP/SELF_PARENT`. 로그 마커 `[WSIConsol] csis`.
- 식: L = (r̃r̃ᵀ) ⊙ exp(−‖x_i−x_j‖²/2σ²), F = logdet(I+L_S) 커널 greedy. 기본 σ = 가장 작은 crop 변(1024px).
- 테스트 `tests_hj_test_csis.py` 31/31.
- 스윕(GTEx dcs20 20케이스) [실측 로그]:
  - `runs/csis/sweep` (budget8 트리, B=512, GPU8, 20:00 완료): base / pcris_v2(rc=143, 18/20) / csis_256·1024·4096 전부 20/20.
  - `runs/nav2v2_mine/sweep` (§2 nav2 세팅, B=768, GPU6, 20:14 완료): base / csis_256·1024·4096 전부 20/20, selfpar=51.
- ★ Nav2 x20 후보는 부모가 선택 안 되면 anchor fallback 으로 self-parent 가 된다(nav2 런 x20 의 27%). CSIS 는 `drop_self_parents` 로 root 처리.
- **정확도는 아직 유효 채점 없음** — 이전 재채점이 전체 데이터셋 인덱스로 잘못 매겨 무효. 채점 level 확정 후 `--dataset-json dcs20_gtex.json` 로 채점.
- 노션 기록 아직 없음.

### VPLI — 전제 기각(Killed)
- ALSI 기존 산출물로 공통 성분 가설 검정 → 기각. 노션 VPLI Method Evolution `3dad771b-c9e2-812d-9f3d-c1530c9f3716` (Status Killed).

### ALSI — v1 만 실행
- `memory/alsi_diag.py` 에 `gram`, `gram_null` 측정 추가(측정 전용).
- 기존 산출물은 절대 ε [1.0, 0.5] 인 **v1**. v2(ε 상대화 + gram)는 미실행. 런: `runs/alsi*`.
- 노션 ALSI Experiment Log `3d9d771b-c9e2-8131-a3fe-f37a3f00f4e2` 에 [결과해석] 추가됨.

### 효율
- `VLMAS_EFF_DUMP=1` 사이드카로 FLOPs/prefill/decode/TTFT 기록. PCRIS v2 B=512 wall −24% [실측].

## 7. 남은 일 (09-15 02:00 기준, 우선순위 순)

1. **posgap 가설 검증**(§9.7b) 04:45 이후 시작 → 스모크 게이트 확인 → 완료 후 `posgap_analyze.py` → 보고 → canon_diag 노션 페이지 §7 추가 → 보드 반영.
   (a_i 감사 1단계는 완료·노션·보드 반영, 2단계 top/random/bottom 은 미실행)
2. ~~canon_diag SlideBench~~ → 완료·분석·노션 기록(§9.7). canon_only 결과 나오면 그 노션 페이지 §7 에 3자 비교 추가.
3. **canon_only_nav3**(§9.6) 5셋 완료 → 채점 → nav3 base / canonical만 / canonical+reread 3자 비교 → **보드 자동 갱신**.
4. Context re-read 4-arm 결과(§9.4)를 노션 `3dbd771b-c9e2-815a-97ba-d6ee42ced077` 에 기록(아직 Planned 상태).
5. 보드 요약 "Navigator 효과" 칸을 nav3 base SB 197 기준으로 재계산할지 사용자 확인 대기.
6. (이전부터) CSIS 정확도 노션 기록 · QA-CSIS parent-free 구현 · ALSI v2 · 로짓 렌즈 노션 페이지 토큰 종류별 정정.
   - SB sdpa Base 는 완료(§5), 채점 level 은 exact 확정(§4.4).

## 8. 함정 모음

- `scripts/dashboard_rows.py` 는 old 채점기이고, `arm.glob("**/result.json")` + 폴더명 `^\d+_` 필요(심링크 뷰 미추적).
- dcs20 부분집합 런을 전체 데이터셋 JSON 으로 인덱싱하면 틀린다.
- eager 와 sdpa 끼리 섞어 비교하지 말 것(sdpa 만으로 base 가 움직인 전례).
- 0912 쪽 래퍼 스크립트만 보고 attn 을 판단하지 말 것. inner runner(`run_dataset_nav2_prompt1.sh` 등)의 `VLMAS_ATTN_IMPLEMENTATION=${…:-sdpa}` 를 봐야 한다.
- 새 런처는 `--output-root` 를 새 폴더로 넘기고 meta 는 그 밖에 쓴다.
- `pgrep -f`/cmdline grep 으로 kill 대상 찾을 때 자기 ssh 쉘까지 잡힌다 → `grep "[c]anon..."` 패턴.
- ssh 단일따옴표 안 python heredoc 에 어포스트로피가 있으면 깨진다 → 스크립트는 scp 로 보낸다.
- 보드 게시가 "최신판 충돌"로 거부되면 Artifact read_file 로 받아 병합 후 재게시(다른 세션도 같은 보드를 고친다).

---

## 9. 09-14 오후 ~ 09-15 작업 (이 세션)

### 9.1 nav3 구조 — 알아둘 사실 [코드읽기+실측]
- nav3 = `VLMAS_NAV3_INDEPENDENT=1` + `VLMAS_NAV2=1 VLMAS_CROSS_SCALE_ROUTER=0 VLMAS_PROMPT_SET=prompt1` + sdpa,
  patch 12 · mlen 12288 · step 10. 원본 러너 `0912/run_nav3_prompt1_base_both_gpu78.sh` → `0912/run_dataset_nav2_prompt1.sh`(읽기만).
- x5 root 와 x20 root 를 **독립**으로 고른다. `source_x5_anchor = patch_id`(자기 자신) → **cross-scale 부모가 없다**.
  x20 의 92~98% 가 선택된 x5 밖. 그래서 nav3 에서는 `cross_scale_consolidate` 교환 0회, QA-CSIS innovation 항·PSAR support 가 사라진다.
- Pruning-B(현재 C1): 비전 블록 4까지 전체 토큰 → 점수 = novelty + 0.25·texture(**질문 미사용**) →
  `hierarchy_prefill_keep_mask`(크롭당 최소 8 + 가지 점수 + 전역 25% = 768/3072) → `morphology_coverage_rescue`(facility location)
  → `cross_scale_consolidate`(부모 중복도, margin 0.05; nav3 에선 무동작).

### 9.2 대시보드 board_0913 갱신 [실측]
- 로컬 `scratchpad/board_0913.html`, artifact https://claude.ai/code/artifact/079769f9-cbc1-4858-88e6-1424d5dac92a
- 추가한 것: **"맞힘 (Acc)" 열**(BAcc 셋 GTEx/TCGA/PANDA 는 괄호에 Acc %), `CANON` 테이블(nav3+Canonical+ReRead 5셋),
  `NAV3K` 맞힘 수, PSAR k 필드, R0902/R0902_S10/NAV2 맞힘 수, `PANDA_NOTE`.
- PANDA 는 상수 붕괴라 BAcc 와 맞힘 수가 반대로 간다: VL-MAS 178/185 가 '3' → BAcc 17.32·맞힘 24,
  nav3 base 185/185 가 '0' → BAcc 16.67·맞힘 51. (사용자가 "정확도 낮은게 더 많이 맞혀?"로 지적한 부분)
- nav3 base 행 통일: SB **43.65% (86/197, 23.6s)**, TCGA **5.42% (9/221)**. PSAR vs nav3 base 같은 답 909/921.

### 9.3 VGain — 사용자 지시로 중단 [실측]
- o = o_N + γ·o_V. nav3 GTEx 16/20 케이스: γ=4 정답 2→3, 답 변경 8/16 이 공통 attractor 로 → 판정 보류.
- 런 `runs/vgain_nav3/vgain_gtex_20260914_165850.jsonl`, 분석 `vgain_analyze.py`·`vgain_analyze_controls.py`.
  노션 `3dad771b-c9e2-81e7-aaee-d38619006f85` (Status Done, 8섹션).

### 9.4 Context-Updated Answerer Query Visual Re-Read — 4-arm (Nav2, canonical 없음, layer 27) [실측]
- 식: q̂ = W_Q·LN(x + W_O·m_C), visual 열만 다시 softmax, ρ_V(visual mass) 유지.
  arm A record / B identity / C "1"(matched) / D shuffled(고정 derangement, 상대 진행도 정렬).
- 코드 `memory/pfr_attention.py`(백업 `runs/pfr_ctx/pfr_attention.py.bak_0914`), env `VLMAS_ANSWERER_PFR=1 _LAYER/_DUMP/_DONOR_DIR/_PERM`,
  로그 `[PFRctx]`. 테스트 `tests_hj_test_pfr_ctx.py` 35/35. 체인 `pfr_ctx_gpu6_chain.sh`, 분석 `pfr_ctx_analyze.py`.
- GTEx 20·EVQA 20: **답이 native 와 20/20 동일**. 결정 행 JS(nat,matched) 0.0089/0.0076 ≈ JS(nat,shuffled) 0.0094/0.0076,
  JS(matched,shuf)/JS(nat,matched) 중앙 0.086/0.30 → **문맥 특이성 약함**. 결정 행 visual mass 0.77%/0.47%. SB 는 도중 중단.
- 결론: re-read 단독은 답에 영향 없음. canonical 4-arm 판(`pfr_ctx_can_*.sh`)은 게이트가 SKIP 을 실패로 쳐서 폐기.

### 9.5 방법론 런: nav3 + Canonical + ReRead (전체 5셋, 완료) [실측]
- `sender_relay_exp/canon_reread_nav3.sh <gpu> <shard> <smoke|wait>` (index % 3, GPU 6/7/8), 출력 `runs/canon_reread_nav3/`.
- env: nav3 + `VLMAS_KV_ROUTE=1 VLMAS_KV_ROUTE_MODE=canonical VLMAS_KV_ROUTE_TERMINAL_ONLY=1 VLMAS_ANSWERER_PFR=1 VLMAS_ANSWERER_PFR_LAYER=27`.
- ★ **canonical stale 부기 버그**: `_route_vis_cols/_route_vis_pos/_route_pages` 가 Reasoner 경계에서 기록되고 안 지워져
  다음 케이스 Navigator 디코드까지 canonical 이 걸렸다. → `latent_terminal.py` 에 `VLMAS_KV_ROUTE_TERMINAL_ONLY=1` 게이트
  (`backbone._route_terminal_call`, 엔진에서 try/finally 로 세팅) + `MODE=bookkeep`(부기만, 디코드 native) 추가.
  오염 출력은 `runs/canon_reread_nav3_STALE_navcanon_0914_2148/` 로 격리.
- 채점 `canon_reread_score.py`(nav3 base 대비, 같은 패치), 하락 분석 `canon_reread_drop.py`.

| 셋 | nav3 base | Canonical+ReRead |
|---|---|---|
| ExpertVQA | 39.84 (51) | **32.81 (42)** |
| GTEx (BAcc) | 22.85 | 25.25 |
| SlideBench | 43.65 (86) | **38.58 (76)** |
| TCGA (BAcc) | 5.42 (9) | 6.55 (13) |
| PANDA (BAcc) | 16.67 (51) | 17.20 (37; 예측 '3' 89 · '0' 83) |

### 9.6 대조 런: nav3 + Canonical만 (진행 중)
- **ExpertVQA 완료(02:24)**: 32.81% (42/128) = Canonical+ReRead 와 같은 답 127/128, re-read 로 얻음 0·잃음 0 → EVQA 하락 전부 canonical 몫.
- **SB 완료**: 39.59% (78/197) vs nav3 base 43.65(86), 같은 답 149 · 얻음 13 · 잃음 21, Canonical+ReRead 와 같은 답 195/197 (re-read 얻음 0·잃음 2).
- **GTEx 완료**: 24.54 vs 22.85, 같은 답 135 · 얻음 9 · 잃음 8, +ReRead 와 같은 답 188/190 (re-read 얻음 1·잃음 0).
- ★ **09-15 04:06:53 TCGA(206/221)·PANDA(19/196) 도중 세 체인 동시 외부 kill** — traceback 없음, 체인 bash 까지 소멸. 직후 04:03~04:09 다른 세션이
  `rss_gate_gtex.sh`(GPU6)·`qasc_nav3.sh`(GPU7) 투입(같은 claim 디렉터리 `runs/vgain/claim_gpu*` 사용). 내가 kill 한 것 아님. 재투입은 사용자 확인 대기.
  TCGA 206문항만: 7.02 vs 5.80(정답 13 vs 9). 보드에 "중단" 상태로 표시.
  채점 `sender_relay_exp/canon_only_score.py [ds]`(nav3 base / canonical만 / +reread 3자, 출력 /tmp/canon_only_score_out.json). 보드 `CONLY` 행으로 반영.
- `sender_relay_exp/canon_only_nav3.sh` = 9.5 에서 PFR env 만 제거. 출력 `runs/canon_only_nav3/`, 순서 EVQA→SB→GTEx→TCGA→PANDA.
  게이트: canonical 마커 == 2, pfr == 0. 스모크 PASS(09-15 01:45).
- `canon_diag_then_only.sh <gpu> <shard>` 가 9.7 진단 뒤 같은 GPU·샤드에서 이어서 실행. 09-15 01:55 GPU7·8 EVQA 진행, GPU6 은 진단 SB 끝나면 시작.
- 목적: 9.5 의 하락 중 canonical 몫과 re-read 몫 분리.

### 9.7 canonical 이 왜 떨어뜨리나 — 모델 단 진단 canon_diag
- 코드 `memory/canon_diag.py`(신규), 엔진 VGain 블록 뒤 `VLMAS_CANON_DIAG` 블록(백업 `runs/canon_diag/*.bak_0915`). 테스트 `tests_hj_test_canon_diag.py` 18/18.
- 같은 캐시에서 arm native / canonical / canonical_t / shift(pos + (b−1−max)) 를 teacher-forced 로 비교:
  층별 rho_v·prompt·lat·rest mass, crop_ent, crop_max, cell_corr(Spearman vs h+w, 동점 평균순위), 결정 행 DLA(visual 직접 / attn 전체).
- 런처 `canon_diag_nav3.sh`(EVQA·SB, env `VLMAS_KV_ROUTE=1 MODE=bookkeep VLMAS_CANON_DIAG=<ds>/diag_gpu$G.jsonl`), 분석 `canon_diag_analyze.py <ds>`.
- **EVQA 128 결과 [실측]**: 정답 native 52 · canonical 45 · canonical_t 50 · shift 50, 답 변경 43/44/48.
  L0–17 rho_v 0.6%→20%(L6 46%), prompt 53%→43%(L6 73%→37%), cell_corr +0.03→+0.25, crop_ent 0.90→0.96.
  flip 37케이스 DLA: visual 직접 기여 +0.027(작음), attn 전체 +2.74→−0.20, logit diff 1.5→−0.75.
- 해석: canonical 이 **앞쪽 층 attention 을 질문·보기 텍스트에서 visual 위치로 끌어온다**(위치 편향). visual 증거가 늘어서가 아니라
  텍스트 처리가 교란돼 옆 보기로 옮겨간다. 주소를 조금만 바꿔도(shift) 답의 1/3 이 바뀐다 = 과민.
- **SB 197 결과 [실측, 09-15 01:57 완료]**: 정답 native 95 · canonical 75 · canonical_t 78 · shift 62, 답 변경 50/54/58 (얻음/잃음 8/28 · 12/29 · 6/39).
  L0–17 rho_v 0.56%→20%, prompt 53%→43%, cell_corr +0.02→+0.25 (EVQA 와 거의 같은 값). flip 28 DLA: visual 직접 +0.03, attn 전체 +2.78→−0.81.
- ★ **해석 수정**: shift 는 rho_v 를 5.5% 만 올리는데도 답을 가장 많이 바꾸고(EVQA 48, SB 58) SB 에서 −33 → attention 끌림은 canonical 고유 현상일 뿐
  하락의 필요조건이 아니다. 원인은 **cached visual key 를 사후에 재회전하는 것 자체**에 더 가깝다 [미검증-가설: value·깊은 층 hidden 은 원래 위치로 계산됨].
- 노션 Experiment Log `3dbd771b-c9e2-81a5-b4bd-c2977aa6fa4c` (EVQA+SB, 8섹션, 판정 보류).

### 9.7b 가설 검증: 사후 key 재회전 vs 일관 계산 (posgap, 대기 중)
- 가설(§9.7): 답이 흔들리는 원인은 **key 만 사후에 돌려 value·깊은 층·뒤 토큰과 어긋나는 것**인가, 아니면 모델이 visual 위치 자체에 민감한가.
- 같은 위치 변화 "visual 블록 뒤에 빈 위치 G칸"(= visual 과 그 앞 전부가 Answerer 에서 G 멀어짐)을 두 방식으로:
  - **posthoc**: native 런, Answerer 경계에서 캐시 복사본의 열 0..마지막 visual 을 −G 재회전 (`canon_diag` arm `gap<G>`; arms native,gap512,gap2048,shift)
  - **consistent**: `VLMAS_POS_GAP=2048` — Reasoner prefill 에서 마지막 visual 토큰 뒤 위치를 +G, latent·Answerer 까지 그 기하로 계산 (arm native)
  - canonical/shift 는 visual 을 뒤 토큰보다 미래 위치로 보내 일관 계산판이 비정상이 되므로 gap 기하를 택함. G=2048 ≈ shift 이동량 중앙 2186.
  - RoPE 등가(열 −G 재회전 == query +G) 테스트로 확인.
- 코드: `backbone/qwen3vl.py` Reasoner prefill 에 `VLMAS_POS_GAP` 블록(로그 `[PosGap] reasoner`), `memory/canon_diag.py` `parse_gap_arm`·gap arm·rec["pos_gap"].
  백업 `runs/posgap/*.bak_0915_posgap`. 테스트 `tests_hj_test_canon_diag.py` 25/25. 끄면(env 없음) 코드 경로 불변.
- 런처 `sender_relay_exp/posgap_nav3.sh <gpu> <shard> <smoke|wait>` — 5셋×20(`runs/posgap/ids_<ds>.txt` = vit_sal 과 같은 dcs20 매핑), 샤드 % 3.
  스모크 게이트: posthoc 2 / 같은 설정 재실행 2 (native logp 차 < 1e-3·같은 답 = **결정성 바닥**) / consistent 2 ([PosGap] 2줄).
  09-15 02:34 GPU8 smoke · GPU6·7 wait 로 투입 — canon_only 체인이 claim 을 풀면(≈04:45) 시작. 출력 `runs/posgap/{smoke,posthoc,consistent2048}/<ds>/`.
- 분석 `sender_relay_exp/posgap_analyze.py` (P0/Pg512/Pg2048/Pshift/C2048: 정답·바뀜·얻음·잃음·JS·ρ, 생성 답 비교). 기존 canon_diag 로 52/50/48/15/17 재현 확인.
- 판정 기준: Pg2048 만 많이 바뀌고 C2048 은 적게 → 불일치가 원인(가설 지지) / 둘 다 비슷 → 위치 민감.
- **스모크 PASS 04:12**(재실행 native logp 차 0.00 → 결정성 바닥 0). GPU8 샤드 2 완료 05:02 (5셋×6 = 30케이스). GPU6·7 샤드는 다른 세션 qasc_nav3 가 GPU 점유 중이라 대기.
- **중간 결과 [실측, n=30, 1/3]**: native 대비 답 바뀜 Pg512 8 · Pg2048 7 · Pshift 10 · **C2048 10**. JS(native,posthoc2048) 0.076 · JS(native,consistent) 0.069 · **JS(posthoc,consistent) 0.012**.
  posthoc2048 과 consistent2048 같은 답 26/30, posthoc 가 바꾼 7케이스를 consistent 도 전부 바꿈(+3). → 사후 회전이 일관 계산과 거의 같은 결과 = **불일치 가설 비지지, 위치 자체 민감**(잠정).
  gap 기하(멀어짐)만 검정됨 — canonical/shift(가까워짐)의 일관판은 원리상 불가.

### 9.8 Vision-encoder salience a_i 위치 편향 감사 (진행 중, 09-15 01:53 시작)
- 출처: 노션 C1 #14 "Question-Agnostic Salience–Coverage Log-Det Selection" Validation 절 —
  "5×·20× 평균 salience heatmap, salience 와 token index · crop-center distance correlation". 대응 Experiment Log `3dbd771b-c9e2-8116-a742-c0333a2cbdc0`.
- a_i = 고정 vision 층에서 같은 크롭의 모든 query 가 token i 에 주는 attention 의 헤드·query 평균(received attention), merge 후 2×2 패치 집계.
  09-12 `vistok_viz/vit_indeg.py` 와 같은 양. 크롭 간 배분엔 원리적으로 침묵(cu_seqlens 블록 대각 → 크롭마다 합 1).
- **LLM 안 돌림**: nav3 result.json 의 크롭 box 를 `render_box(max_side=512)` 로 다시 그리고 비전 타워만 **CPU float32** 로 통과
  (GPU 6·7·8 사용 중). 재현 확인: 다시 그린 128px tissue_fraction 이 Navigator 기록값과 차이 0, 다른 런 소스와 크롭 좌표 일치.
- 파일(A 트리 `sender_relay_exp/`): `vit_indeg.py`(복사), `vit_sal_stats.py`(순수 통계), `vit_sal_extract.py`, `vit_sal_analyze.py`,
  테스트 `tests_hj_test_vit_sal.py` 30/30. 케이스 = `smoke/dcs20_<ds>.json` 5셋×20 → 전체 JSON 인덱스로 매핑(Id+Question+Choice).
- 지표(크롭별, 24블록 + 블록 평균): r_idx, r_dist(음수 = 중심 선호), r_tissue, r_dist|tissue, r_idx|tissue(조직 순위 제거),
  r_template(다른 케이스 같은 셋·배율 평균맵과의 상관 = 고정 공간 템플릿), 중심/테두리 비, 최대값, argmax 위치 최빈 비율.
- 출력 `runs/vit_sal_audit/{smoke,full}/` (cases/*.npz, manifest.json, summary.json, fig_heatmap_mean.png, fig_heatmap_blocks.png, fig_corr_blocks.png).
  스모크 2케이스 PASS(케이스당 ~18s). 전체 100케이스 ≈ 30분. 로그 `runs/vit_sal_audit/full.log`.
- 다음 단계(노션): 같은 예산에서 top-salience vs random vs bottom-salience, 5×/20× 따로 → top>random>bottom 이 안정적일 때만 F(S) 에 a_i 투입.
- **결과 [실측, 09-15 02:26 완료, 100/100 케이스 · 1182 crop]**: 평균맵이 셋·배율 무관하게 같은 고정 패턴.
  가장자리 8칸 (2,15)(2,0)(5,15)(13,15)(13,0)(4,15)(15,7)(15,8) 이 crop mass 11%(균등 3.1%), top-25% 에 7/8 포함. block 3 은 (2,15) 가 crop 99% 에서 1위.
  crop별 Spearman 중앙(블록 평균): idx +0.10 · 중심거리 +0.16(가장자리 선호, 조직 통제 후 동일) · 조직 ≈0(PANDA 만 +0.26~0.44) · 다른 케이스 평균맵 +0.33.
  top-25% 선택 조직 비율 − crop 평균 −0.005. 추가 분석 `vit_sal_topk.py`. 노션 페이지 갱신(Ongoing, 그림 3장). 보드 요약 카드·출처 문단에 반영.

### 9.9 질문 답하며 확인한 사실
- 로짓 렌즈: 스텝의 65% 가 단어 뒷조각. 첫 content 토큰은 L33 에서 1위. "L24–30 후보 형성" 주장은 토큰 종류별로 다시 봐야 함(`lens_tokcat.py`).
- 디코딩 쓰레기: 그분 Base TCGA 17/221 불일치(형식 실패 12), EVQA·PANDA 0 (`garbage_scan.py`).
- q2v salience 는 크롭 내용이 아니라 슬롯 위치를 따른다(r ≈ −0.8) — 09-12 GTEx 10케이스.
- Relay2 dual: probe layer 20/24/28/32 스텝별 visual mass → 중앙값+1×MAD 초과 스텝 K/V 를 36층 전부 다음 역할 캐시 끝에 붙임.
  cumulative 모드라 중복 주입 → attention 약 2배.
- Q-PRIS exact 재채점, 20케이스 옛 표 무효(§4.4, 노션 교체 완료).

### 9.10 GPU 현황 (09-15 01:55) [실측]

| GPU | 작업 | 다음 |
|---|---|---|
| 6 | canon_diag SlideBench 샤드 0 (01:57 완료) → canon_only_nav3 샤드 0 | EVQA→SB→GTEx→TCGA→PANDA |
| 7 | canon_only_nav3 샤드 1 ExpertVQA (43, 01:52 시작) | SB→GTEx→TCGA→PANDA |
| 8 | canon_only_nav3 샤드 2 ExpertVQA (42, 01:45 시작) | SB→GTEx→TCGA→PANDA |
| CPU | vit_sal_extract 5셋×20 (01:53 시작) | vit_sal_analyze |

GPU 4·5 는 다른 사용자/세션, 0–3 은 비어 있으나 사용 금지.

확인 명령:
```bash
ssh isyse_94_jw 'cd /home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs && tail -2 canon_diag/chain_gpu6.log canon_only_nav3/chain_gpu{6,7,8}.log; tail -2 vit_sal_audit/full.log'
```
