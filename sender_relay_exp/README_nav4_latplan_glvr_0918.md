# WSI LatentMAS — nav4 · latplan · NOVA ρ · GLVR 실행 가이드

병리 WSI(Whole Slide Image) 질의응답을 위한 4-역할 latent 멀티에이전트 파이프라인의 실행·채점 방법을 정리한 문서입니다. 기준선(nav4 base)부터 성능 표와 같은 순서로 한 단계씩 기능을 더해 가며, 각 단계를 어떻게 재현하는지 설명합니다.

- 대상 데이터셋: **TCGA ExpertVQA (128문항)**, **TCGA SlideBench (197문항)** — 모두 객관식
- 모델: Qwen3-VL-4B-Thinking
- 작성 기준일: 2026-09-18

---

## 목차

1. [파이프라인 개요와 용어](#1-파이프라인-개요와-용어)
2. [빠른 시작](#2-빠른-시작)
3. [환경](#3-환경)
4. [단계별 구성](#4-단계별-구성)
5. [결과](#5-결과)
6. [채점과 효율 측정](#6-채점과-효율-측정)
7. [문제 해결](#7-문제-해결)

---

## 1. 파이프라인 개요와 용어

### 1.1 4-역할 파이프라인

하나의 질문에 대해 네 역할이 **공유 KV 캐시**를 이어 쓰면서 순서대로 동작합니다. 역할 사이에는 텍스트 대신 latent 상태(모델 hidden state를 입력 임베딩으로 되돌린 것)를 넘깁니다.

| 순서 | 역할 | 하는 일 |
|---|---|---|
| 1 | **Planner** | 슬라이드 썸네일을 보고 어디를 확대해 볼지 계획 (latent 10스텝) |
| 2 | **Navigator** | 번호가 매겨진 격자 두 장(5×, 20×)에서 볼 칸을 JSON으로 고름 (nav4) |
| 3 | **Reasoner** | 고른 패치 12장(B=12)을 보고 형태학적 소견을 latent 10스텝으로 정리 |
| 4 | **Answerer** | 앞의 KV 전체를 보고 `{"answer": ...}` JSON으로 답을 생성 |

### 1.2 용어

| 용어 | 뜻 |
|---|---|
| **nav4** | 두 개의 번호 격자에서 칸 ID를 JSON으로 디코드하는 Navigator 버전 |
| **디코딩 수정 (decode fix)** | Answerer 답이 비었을 때 숫자를 억지로 재디코드하던 동작을 제거하고, 답 토큰 예산을 512로 늘린 수정 (§4.1) |
| **latplan** | Planner가 확대 목표를 텍스트로 디코드하지 않고 latent KV로만 넘기는 설정 |
| **C1** | 시각 토큰 선택(pruning) 계열. Reasoner가 볼 시각 토큰 수를 줄임 |
| **NOVA ρ** | C1 방법. 시각 토큰 3072개 중 25%(768개)를, 배경(흰 영역) 비중을 연속값으로 보정한 log-det 탐욕 선택으로 고름 |
| **C2** | 역할 간 latent 전달 계열. Answerer가 앞 역할의 정보를 더 잘 쓰게 함 |
| **GLVR** | Grounded Latent Visual Relay. C2 방법. 아래 두 부분으로 구성 |
| ─ **formation** | Reasoner latent를 질문과 관련된 시각 상태 쪽으로 몇 스텝 최적화(grounding)한 뒤 다시 캐시에 씀 |
| ─ **relay** | Reasoner latent가 시각 토큰에서 읽은 정보 u_V,k를 Answerer에 넘김: `o' = o + λ(ρ_B · gᵀU_V − m_B)` |
| **H_C (donor)** | Answerer attention 중 sink·시각·현재 프롬프트를 제외한 나머지 KV. relay가 이 몫을 시각 정보로 바꿔 넣음 |
| **λ** | relay 비율. 0이면 원래 모델, 1이면 H_C 몫을 전부 교체. 기본 0.75 |
| **arm** | 런처에서 한 설정을 가리키는 이름 (`base`, `latplan`, ...) |
| **exact 채점** | 모델 답 문자열을 정규화한 뒤 정답 보기와 정확히 일치할 때만 정답 |

### 1.3 단계 구성 (성능 표 순서)

```
nav4 base (디코딩 수정)                                   arm: base
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
latplan · Planner 텍스트 경로 제거                          arm: latplan
 └ + C1 NOVA ρ                                           arm: nova_rho_latplan
    └ + C2 GLVR (formation + relay)                      arm: glvr_rho
    └ + C2 GLVR relay만 (formation 없음)                  arm: glvr_rho_nf
       └ 효율 버전 (glvr_rho_nf와 같은 방법, 계산만 절약)    arm: glvr_rho_fast
```

각 단계는 바로 위 단계에 **한 가지만** 더합니다.

---

## 2. 빠른 시작

모든 명령은 94번 서버의 `sender_relay_exp/` 폴더에서 실행합니다.

```bash
ssh isyse_94_jw
```
```bash
cd /home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp
```

### 2.1 전체 단계를 순서대로 한 번에 — `run_stack_0918.sh`

각 단계마다 **2문항 스모크 테스트(통과 필수) → 3샤드 풀런 → 완료 대기 → 바로 위 단계 대비 채점** 순으로 진행하고 다음 단계로 넘어갑니다. 스모크가 실패하면 그 자리에서 멈춥니다.

먼저 실행될 명령만 확인합니다(아무것도 실행하지 않음).

```bash
DRY_RUN=1 bash run_stack_0918.sh
```

실제 실행:

```bash
nohup bash run_stack_0918.sh > stack_nohup.log 2>&1 &
```

- 결과는 새 폴더 `runs/stack_<월일_시분>/<arm>/`에 쌓이고, 기존 결과는 건드리지 않습니다.
- 진행 로그: `runs/stack_*/stack.log` / 단계별 채점: `runs/stack_*/scores.jsonl`
- 소요 시간(추정): 디스크가 정상일 때 단계당 1~1.5시간, 다섯 단계 합계 6~8시간

| 환경변수 | 기본값 | 설명 |
|---|---|---|
| `OUTROOT` | `runs/stack_<월일_시분>` | 결과 폴더 |
| `ARMS` | `base latplan nova_rho_latplan glvr_rho glvr_rho_nf` | 돌릴 단계. 일부만 돌리면 비교 대상 단계가 같은 `OUTROOT`에 있어야 채점됨. 효율 버전은 `glvr_rho_fast` 추가 |
| `GPUS` | `6 7 8` | 사용할 GPU 번호 |
| `NSHARD` | `3` | 샤드 수 |
| `MAXUSED` | `20000` | GPU 사용 메모리(MiB)가 이 값 미만일 때만 해당 GPU를 사용 |
| `SPECS` | `tcga_expert_vqa:128 tcga_slidebench:197` | 데이터셋과 문항 수 |
| `SLIDE_ROOT` | 데이터셋의 `slides/` | 슬라이드 경로 (§7.1) |

### 2.2 한 단계만 돌리기

```bash
MAXUSED=20000 bash ./rtask_nav4_nonav2_arm.sh <arm> smoke 3 "6"
```
```bash
for s in 0 1 2; do MAXUSED=20000 nohup bash ./rtask_nav4_nonav2_arm.sh <arm> $s 3 "6 7 8" tcga_expert_vqa:128 tcga_slidebench:197 > /dev/null 2>&1 < /dev/null & done
```

- 인자: `<arm> <샤드 번호 | smoke> <샤드 수> "<GPU 목록>" [데이터셋:문항수 ...]`
- 샤드는 스모크 결과 파일 `<결과폴더>/<arm>/SMOKE_GATE_<arm>`이 `PASS`로 시작해야 출발하므로 스모크를 먼저 돌립니다.
- 이미 끝난 문항(`result.json` 있음)은 건너뛰므로, 중단돼도 같은 명령으로 이어서 돌 수 있습니다.
- `nova_rho_latplan`만 별도 런처 `rtask_nav4_nova_rho_latplan_arm.sh`를 씁니다(인자 형식 동일).

---

## 3. 환경

| 항목 | 경로 |
|---|---|
| 서버 | `isyse_94_jw` |
| 코드 트리 (`$REPO`) | `/home/users/whddn12316/wsi_latent_0915_decode_hj` |
| Python | `/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python` |
| 모델 | `/home/users/whddn12316/models/Qwen3-VL-4B-Thinking` |
| 데이터 (문항 JSON) | `/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda/{tcga_expert_vqa,tcga_slidebench}.json` |
| 슬라이드 | 같은 폴더의 `slides/` (원본 .svs로 가는 심볼릭 링크) |
| 결과 | `$REPO/sender_relay_exp/runs/` |

GPU 한 장당 한 런에 약 14GB가 필요합니다. 한 장에 두 런을 같이 올리려면 `MAXUSED`를 20000~30000으로 둡니다.

### 3.1 코드 버전 (md5 앞 8자리)

같은 결과를 재현하려면 아래 파일의 md5가 같아야 합니다. 런처는 GPU를 잡을 때 주요 파일 md5를 로그(`chain_<arm>_shard<k>.log`)에 남깁니다.

| 파일 | md5 | 역할 |
|---|---|---|
| `vision_text_mas/onepass_navigation_nav4.py` | `4f6e14a4` | nav4 Navigator |
| `vision_text_mas/onepass_navigator.py` | `198bf899` | Navigator 분기 (nav4 우선) |
| `vision_text_mas/latent_answerer.py` | `5f0b363f` | Answerer (디코딩 수정 포함) |
| `vision_text_mas/latent_qwen_engine.py` | `085f2bc2` | 4-역할 엔진 |
| `vision_text_mas/latent_terminal.py` | `addc7084` | Answerer 생성 루틴 |
| `backbone/qwen3vl.py` | `9d2f3990` | Qwen3-VL 백엔드 |
| `memory/nova_rho_select.py` | `21423886` | C1 NOVA ρ |
| `memory/nova_select.py` | `29b4b911` | C1 NOVA (v1, ρ의 바탕) |
| `memory/secld_select.py` | `9e902868` | 선택 증거 계산 (NOVA ρ가 교체) |
| `memory/qasc_select.py` | `e97a4535` | C1 선택 진입점 |
| `memory/rnlcr.py` | `d76efbdd` | GLVR formation(grounding)과 replay |
| `tmp_hj/lpvh_ntrs_hooks/sitecustomize.py` | `c18b9492` | base 기본 훅 |
| `tmp_hj/latplan_hooks/sitecustomize.py` | `dc91f05e` | latplan 훅 |
| `tmp_hj/nova_rho_latplan_hooks/sitecustomize.py` | `fcdbb974` | NOVA ρ + latplan 훅 |
| `tmp_hj/glvr_fast2/` | `glvr_diag.py ca21eadb` · `sitecustomize.py 6982860d` · `glvr_core.py 3e6c7d7f` | GLVR (glvr_rho, glvr_rho_nf) |
| `tmp_hj/glvr_fast3/` | `glvr_diag.py 2b16dd7a` · `sitecustomize.py a7ba4a12` · `glvr_coverage.py 73c6b52c` | GLVR 효율 버전 |

`tmp_hj/*` 훅은 `PYTHONPATH` 맨 앞에 두는 `sitecustomize.py` 방식이라 원본 코드를 수정하지 않고 동작을 바꿉니다. 환경변수가 없으면 아무것도 바꾸지 않습니다.

---

## 4. 단계별 구성

### 4.0 모든 단계 공통 설정

런처가 모든 arm에 아래 설정을 넣습니다.

```bash
CUDA_VISIBLE_DEVICES=$G PYTHONPATH=$HOOKS:$REPO PYTHONUNBUFFERED=1 <arm별 env> \
VLMAS_PROMPT_SET=prompt1 VLMAS_ATTN_IMPLEMENTATION=sdpa \
VLMAS_NAV4=1 VLMAS_NAV2=1 VLMAS_CROSS_SCALE_ROUTER=0 \
VLMAS_NAV3_X5_CLAMP=1 VLMAS_EFF_DUMP=1 VLMAS_ANSWERER_GRADE_RULE=1 \
VLMAS_REASONER_TASK="<런처 안의 Reasoner 지시문>" \
$PY -m wsi_latentmas.pipeline.latent_mas \
  --variant base --backbone qwen3-vl --dataset "$DATA/<ds>.json" --slide-root "${SLIDE_ROOT:-$DATA/slides}" \
  --model "$MODEL" --output-root "<새 폴더>" --device cuda:0 \
  --latent-steps 10 --patch-budget 12 --max-model-len 12288 --navigator-control-tokens 512 \
  --temperature 0.0 --top-p 1.0 --seed 42 --deterministic --answerer-greedy --no-answerer-thinking \
  --answerer-max-new-tokens 512 --answerer-rationale --answerer-protocol structured_json \
  --canonical-open-options --navigator-kv --io-pipeline --no-save-navigation-pngs --case-retries 0
```

- attention 구현은 반드시 `sdpa`입니다. 환경변수를 빼면 백엔드 기본값인 eager로 돌아 결과가 달라집니다.
- 디코딩은 greedy(temperature 0, seed 42)라 같은 코드·입력이면 같은 답이 나옵니다.
- `--output-root`는 존재하지 않는 새 폴더여야 합니다. 런처가 `gpu<G>_attempt_<시각>` 이름으로 만듭니다.
- `VLMAS_NAV2=1`은 Planner가 확대 목표를 텍스트로 디코드하게 하는 스위치입니다. latplan 단계부터는 훅이 이 경로를 끕니다.
- `VLMAS_NAV3_X5_CLAMP`는 PANDA 데이터셋용 설정이라 ExpertVQA·SlideBench에는 영향이 없습니다.

### 4.1 nav4 base (디코딩 수정) — `base`

- 훅: `tmp_hj/lpvh_ntrs_hooks` (기본)
- 추가 env: 없음
- 디코딩 수정 내용 (이 코드 트리의 기본값):
  - **빈 답 재디코드 제거.** 예전에는 Answerer 답이 비면 한 자리 숫자로 강제 재디코드해서 1·2번 보기를 지어냈고, 10번 이상 보기는 고를 수 없었습니다. 이제 빈 답은 `<no-answer>`로 저장되고 오답 처리됩니다. 예전 동작 재현용 스위치는 `VLMAS_TERMINAL_DIGIT_REPAIR=1`입니다.
  - **답 토큰 예산 512.** 예전 32토큰으로는 Thinking 모델이 추론 뒤 JSON 답까지 도달하지 못했습니다.
  - Answerer 출력은 `{"answer":`를 앞에 고정(teacher-force)해 JSON 형식을 강제합니다.

### 4.2 latplan — `latplan`

- 훅: `tmp_hj/latplan_hooks` (런처가 자동 선택)
- 추가 env: `VLMAS_PLAN_LATENT_ONLY=1`
- 바뀌는 것: Planner가 5×/20× 확대 목표를 텍스트로 디코드하지 않고, Navigator 프롬프트의 목표 줄 두 개도 지웁니다. Planner의 판단은 latent KV로만 Navigator에 전달됩니다. **이후 모든 단계에 들어갑니다.**

### 4.3 + C1 NOVA ρ — `nova_rho_latplan`

- 런처: `rtask_nav4_nova_rho_latplan_arm.sh` / 결과 기본 폴더: `runs/nav4_rho_latplan/`
- 훅: `tmp_hj/nova_rho_latplan_hooks` (latplan 훅 포함)
- 추가 env: `VLMAS_C1_QASC=1 VLMAS_C1_SELECT=nova VLMAS_NOVA_KEEP=0.25 VLMAS_C1_META_BRIDGE=1 VLMAS_NOVA_RHO=1 VLMAS_PLAN_LATENT_ONLY=1`
- 바뀌는 것: Reasoner의 시각 토큰 3072개 중 768개(25%)만 남깁니다.
  1. 각 셀의 흰 픽셀 비율을 연속값 ρ로 계산합니다.
  2. crop 안에서 정규화한 가우시안으로 이웃과 평균해 배경 정도 b를 구합니다.
  3. 조직 정도 e = 1 − b로 가중한 log-det 탐욕 선택으로 서로 다른 조직 상태를 고르게 남깁니다.

### 4.4 + C2 GLVR (formation + relay) — `glvr_rho`

- 훅: `HOOKS_OVERRIDE=$REPO/tmp_hj/glvr_fast2` (NOVA ρ + latplan 훅을 이어서 불러옴)
- 추가 env: 4.3 전부 + `VLMAS_RNLCR=ground VLMAS_GLVR=<로그 jsonl> VLMAS_GLVR_APPLY=0.75 VLMAS_GLVR_CHAIN=nova_rho_latplan_hooks`
- 바뀌는 것:
  1. **formation:** Reasoner latent 10개를, 질문과 관련도가 높은 시각 상태 평균(P)에 가까워지고 낮은 쪽(N)에서 멀어지도록 Adam 5스텝 최적화한 뒤 캐시에 다시 씁니다.
  2. **relay:** 1의 재기록 과정에서 각 latent가 시각 토큰에서 읽은 벡터 u_V,k를 기록합니다. Answerer의 모든 층에서 H_C 몫(m_B)을 빼고, 같은 비중(ρ_B)만큼을 latent 가중치 g로 섞은 시각 읽기 gᵀU_V로 채웁니다(λ=0.75).
- λ는 5문항 진단에서 0 / 0.25 / 0.5 / 0.75 / 1을 비교해 골랐습니다.

### 4.5 + C2 GLVR relay만 (formation 없음) — `glvr_rho_nf`

- 훅: 4.4와 같음
- 추가 env: 4.4 + `VLMAS_RNLCR_S1_STEPS=0`
- 바뀌는 것: formation 최적화를 0스텝으로 끄고 relay만 남깁니다. formation과 relay 각각의 기여를 나누기 위한 단계입니다.

### 4.6 효율 버전 — `glvr_rho_fast`

- 훅: `HOOKS_OVERRIDE=$REPO/tmp_hj/glvr_fast3`
- 추가 env: 4.3 전부 + `VLMAS_RNLCR=diag VLMAS_GLVR=<로그 jsonl> VLMAS_GLVR_GENCAP=1 VLMAS_GLVR_APPLY=0.75 VLMAS_GLVR_ANSWER_ONLY=1 VLMAS_GLVR_CHAIN=nova_rho_latplan_hooks`
- 4.5와 방법은 같고 계산만 줄였습니다.
  - u_V,k를 Reasoner latent 생성 중에 바로 기록해서 캐시 재기록(replay)을 없앴습니다.
  - relay를 답 구간(가장 긴 보기의 토큰 수 + 2)에만 적용합니다. 채점은 answer 필드만 봅니다.
- 상태: 5문항에서 4.5와 답이 모두 같았습니다. 전체 문항 실행은 아직입니다.

---

## 5. 결과

exact 채점, ExpertVQA 128문항 · SlideBench 197문항.

| 방법 | ExpertVQA | SlideBench | 문항당 모델 시간 (EVQA · SB) |
|---|---|---|---|
| nav4 base (디코딩 수정) | 40.62% (52) | 46.19% (91) | 13.2 · 13.4초 |
| **━━━━━━━━━━━━━━━━** | | | |
| Single (썸네일 한 장) ※참고 | 36.72% (47) | 39.59% (78) | — |
| latplan | 40.62% (52) | 46.70% (92) | 15.4 · 13.4초 |
| latplan └ + C1 NOVA ρ | 42.19% (54) | 45.69% (90) | 15.7 · 16.1초 |
| latplan └└ + C1 NOVA ρ + C2 GLVR (λ 0.75) | **42.97% (55)** | 47.21% (93) | 26.1 · 23.9초 |
| latplan └└ + C1 NOVA ρ + C2 GLVR relay만 | 41.41% (53) | **48.22% (95)** | 23.1 · 24.0초 |
| └ 효율 버전 | 5문항 답 동일 | — | 미측정 |

바로 위 단계와의 문항 단위 비교 (위 단계가 틀리고 이 단계가 맞은 수 / 그 반대):

| 비교 | ExpertVQA | SlideBench |
|---|---|---|
| latplan ← nav4 base | +5 / −5 | +9 / −8 |
| NOVA ρ ← latplan | +4 / −2 | +2 / −4 |
| GLVR ← NOVA ρ | +4 / −3 | +7 / −4 |
| GLVR relay만 ← NOVA ρ | +4 / −5 | +6 / −1 |
| GLVR relay만 ← GLVR | 0 / −2 | +3 / −1 |

- **Single**은 이 가이드의 런처가 아니라 별도 파이프라인(썸네일 한 장만 보는 단일 모델)으로 얻은 참고값입니다.
- **문항당 모델 시간**은 네 역할 호출 시간의 합을 문항 평균한 값입니다. 슬라이드 파일 읽기 시간은 빠져 있어서, 실제 경과 시간은 디스크 상황에 따라 더 깁니다.
- 해석 요약:
  - ExpertVQA의 GLVR 이득은 formation에서, SlideBench 이득은 relay에서 나옵니다(formation을 끄면 ExpertVQA −2, SlideBench +2).
  - 진단 결과 Reasoner latent 10개가 시각 토큰에서 읽는 내용은 서로 거의 같습니다(유효 차원 10 중 2.4–4.3). relay가 전달하는 것은 latent별로 다른 근거라기보다 공통 시각 성분에 가깝습니다.

---

## 6. 채점과 효율 측정

```bash
cd /home/users/whddn12316/wsi_latent_0915_decode_hj
```

exact 짝비교 (`PAIRS=<arm>:<비교 대상>,...`, `DSS=<데이터셋>,...`):

```bash
PAIRS=glvr_rho_nf:nova_rho_latplan,nova_rho_latplan:latplan DSS=tcga_expert_vqa,tcga_slidebench /home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python sender_relay_exp/tools_0918/score_pairs.py
```

- 출력(JSON 한 줄씩): `arm`·`ref`의 정확도와 맞은 수, `gain`(arm만 맞음), `loss`(ref만 맞음), `same`(답 문자열이 같은 문항 수), `n_common`(공통 문항 수).
- 판정은 `scripts/rescore.py`의 exact 규칙을 씁니다.
- 기본 arm 경로는 `runs/nav4/<arm>`(NOVA ρ는 `runs/nav4_rho_latplan/nova_rho_latplan`)입니다. `ARM_ROOT=<폴더>`를 주면 모든 arm을 `<폴더>/<arm>`에서 찾습니다(§2.1 결과 채점용). 새 arm은 파일 안 `ARMS`에 경로를 추가합니다.

효율 (역할별 시간, prefill 토큰 수, decode 스텝, FLOPs, 최대 GPU 메모리의 문항 평균):

```bash
cd sender_relay_exp && python3 tools_0918/eff_rows.py runs/nav4/glvr_rho_nf tcga_slidebench
```

---

## 7. 문제 해결

### 7.1 문항당 시간이 수백 초로 늘어날 때

`/home`이 있는 공용 디스크에 다른 작업의 쓰기가 몰리면, 슬라이드에서 타일을 읽을 때마다 대기가 생겨 문항당 수백 초가 걸릴 수 있습니다(모델 계산 시간은 그대로). 다음 명령으로 확인합니다. `r_await`(읽기 대기 ms)가 수 ms면 정상, 수십~수백 ms면 디스크 경합입니다.

```bash
iostat -x 3 2 | grep sdb
```

해결 방법:
- 경합이 끝날 때까지 기다립니다.
- 또는 필요한 슬라이드만 다른 디스크로 복사하고 `SLIDE_ROOT`를 그쪽으로 지정합니다. 큰 블록 직접 읽기(`dd bs=64M iflag=direct`)는 경합 중에도 빠릅니다. 예를 들어 ExpertVQA 슬라이드 76개(79GB)를 약 4분에 복사해 문항당 330초 → 51초가 됐습니다.
  - `SLIDE_ROOT`에는 `slides/`와 같은 이름의 링크들이 있는 폴더를 줍니다(원본 `slides/`의 링크 이름을 유지하고 대상만 복사본으로 바꿈).
  - `/tmp` 같은 공용 시스템 디스크는 다른 사용자와 같이 씁니다. **대용량 복사 전에는 관리자나 팀과 먼저 상의하고, 실행이 끝나면 바로 삭제하세요.**

### 7.2 GPU 메모리 부족(OOM)

- 여러 런을 동시에 띄울 때는 **같은 `CLAIMDIR`을 공유**해야 합니다. claim 폴더를 따로 쓰면 각 런이 같은 GPU를 비어 있다고 보고 한꺼번에 올라가 OOM이 납니다.
- 한 장에 두 런이면 약 28GB, 세 런이면 약 42GB입니다(49GB GPU 기준).

### 7.3 기타

- 중단된 런은 같은 명령으로 다시 띄우면 남은 문항만 이어서 돕니다.
- 실행 중인 런을 멈출 때는 자기 프로세스의 PID만 `kill <PID>`로 종료합니다. 이 서버의 `pkill`은 일반 리눅스 명령이 아닙니다.
- `backbone/qwen3vl.py` 등 공용 코드는 다른 작업에서 바뀔 수 있습니다. 재현 전에 §3.1 md5를 확인하세요.
