# nav4_latplan — 기준 실행 스크립트

**nav4 + 디코딩 수정 + latplan** 설정 하나만 돌리는 독립 스크립트입니다. 실행에 필요한 설정(환경변수, 실행 인자, Reasoner 지시문)이 모두 `run_latplan.sh` 한 파일에 들어 있습니다. 새 방법은 이 파일을 고치지 않고 환경변수로 위에 얹어서 쌓습니다(§3).

- 위치: `/home/users/whddn12316/wsi_latent_0915_decode_hj/nav4_latplan/` (서버 `isyse_94_jw`)
- 전체 단계(NOVA ρ, GLVR 등)와 결과는 `sender_relay_exp/README_nav4_latplan_glvr_0918.md` 참고

## 1. 이 설정이 하는 일

| 구성 | 내용 |
|---|---|
| 파이프라인 | Planner → Navigator → Reasoner → Answerer, 역할 간 latent KV 전달, latent 10스텝, 패치 12장 |
| nav4 | Navigator가 번호 격자 두 장(5×, 20×)에서 칸 ID를 JSON으로 고름 |
| 디코딩 수정 | 빈 답을 숫자로 억지 재디코드하지 않음(`<no-answer>`로 저장, 오답 처리), 답 토큰 예산 512, `{"answer":` 고정 |
| latplan | Planner가 확대 목표를 텍스트로 디코드하지 않고, Navigator 프롬프트의 목표 줄도 제거. Planner 판단은 latent KV로만 전달 (`VLMAS_PLAN_LATENT_ONLY=1`, 훅 `tmp_hj/latplan_hooks`) |
| 디코딩 | greedy (temperature 0, seed 42), attention `sdpa` |

기존 결과(`sender_relay_exp/runs/nav4/latplan`): ExpertVQA 40.62% (52/128), SlideBench 46.70% (92/197), exact 채점.

## 2. 실행

```bash
cd /home/users/whddn12316/wsi_latent_0915_decode_hj/nav4_latplan
```

1) 스모크 (2문항, GPU 한 장, 약 2분). 결과가 `PASS`로 시작해야 다음 단계가 돕니다.

```bash
bash run_latplan.sh smoke
```

2) 전체 실행 (샤드 3개가 GPU를 하나씩 잡고 병렬로 돌고, 모두 끝날 때까지 기다림).

```bash
nohup bash run_latplan.sh run > run_nohup.log 2>&1 &
```

3) 진행 확인

```bash
bash run_latplan.sh status
```

- 실행 전에 명령만 보고 싶으면 앞에 `DRY_RUN=1`을 붙입니다.
- 결과: `runs/<NAME>/<데이터셋>/gpu*_attempt_*/<문항>/result.json`, 로그 `runs/<NAME>/run.log`
- 중단돼도 같은 명령을 다시 실행하면 끝난 문항은 건너뛰고 남은 것만 돕니다.
- 같은 `OUTROOT` 아래 런들은 GPU claim 폴더(`runs/claims`)를 공유해서, 한 GPU에 한 런씩만 올라갑니다.

| 환경변수 | 기본값 | 설명 |
|---|---|---|
| `NAME` | `latplan` | 실행 이름. 결과가 `runs/<NAME>`에 쌓임 |
| `OUTROOT` | `./runs` | 결과 루트 |
| `GPUS` | `6 7 8` | 사용할 GPU 후보 |
| `NSHARD` | `3` | 병렬 샤드 수 |
| `MAXUSED` | `20000` | 사용 메모리(MiB)가 이 값보다 작은 GPU만 사용. 한 런 ≈ 14GB |
| `SPECS` | `tcga_expert_vqa:128 tcga_slidebench:197` | 데이터셋:문항 수 |
| `SLIDE_ROOT` | 데이터셋 `slides/` | 슬라이드 폴더 |
| `HOOKS` | `tmp_hj/latplan_hooks` | `PYTHONPATH` 맨 앞에 둘 훅 폴더 |
| `EXTRA_ENV` | 없음 | 위에 얹을 방법의 환경변수 |

## 3. 위에 쌓기

기준 설정(`BASE_ENV`, `ARGS`)은 그대로 두고 `NAME`·`HOOKS`·`EXTRA_ENV`만 바꿉니다. 이름을 다르게 주면 결과 폴더가 분리됩니다.

**+ C1 NOVA ρ**

```bash
NAME=nova_rho HOOKS=/home/users/whddn12316/wsi_latent_0915_decode_hj/tmp_hj/nova_rho_latplan_hooks EXTRA_ENV="VLMAS_C1_QASC=1 VLMAS_C1_SELECT=nova VLMAS_NOVA_KEEP=0.25 VLMAS_C1_META_BRIDGE=1 VLMAS_NOVA_RHO=1" bash run_latplan.sh smoke
```

**+ C1 NOVA ρ + C2 GLVR relay (formation 없음)**

```bash
NAME=glvr_nf HOOKS=/home/users/whddn12316/wsi_latent_0915_decode_hj/tmp_hj/glvr_fast2 EXTRA_ENV="VLMAS_C1_QASC=1 VLMAS_C1_SELECT=nova VLMAS_NOVA_KEEP=0.25 VLMAS_C1_META_BRIDGE=1 VLMAS_NOVA_RHO=1 VLMAS_RNLCR=ground VLMAS_RNLCR_S1_STEPS=0 VLMAS_GLVR=$PWD/runs/glvr_nf/glvr.jsonl VLMAS_GLVR_APPLY=0.75 VLMAS_GLVR_CHAIN=nova_rho_latplan_hooks" bash run_latplan.sh smoke
```

스모크가 `PASS`면 같은 환경변수로 `run`을 실행합니다.

새 방법을 만들 때:
1. 원본 코드를 고치지 않으려면 `tmp_hj/<새 훅>/sitecustomize.py`를 만들고, 그 안에서 `latplan_hooks/sitecustomize.py`를 먼저 불러온 뒤 필요한 함수만 바꿉니다(`nova_rho_latplan_hooks/sitecustomize.py`가 예시입니다). 기능은 환경변수로 켜고 끄게 합니다.
2. `HOOKS=<새 훅 폴더> EXTRA_ENV="<새 스위치>" NAME=<이름>`으로 스모크 → 실행합니다.
3. 스모크 판정은 결과 2개, Traceback 0, 빈 답 0, latplan 훅 적용 확인을 봅니다. 새 방법의 적용 여부는 해당 로그를 따로 확인하세요.

## 4. 채점

```bash
cd /home/users/whddn12316/wsi_latent_0915_decode_hj
```

같은 `runs/` 아래 두 실행을 exact로 비교합니다(`PAIRS=<실행>:<기준>`).

```bash
ARM_ROOT=$PWD/nav4_latplan/runs PAIRS=nova_rho:latplan DSS=tcga_expert_vqa,tcga_slidebench /home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python sender_relay_exp/tools_0918/score_pairs.py
```

`ARM_ROOT`를 주면 `PAIRS`에 쓴 이름을 모두 `ARM_ROOT/<이름>` 폴더에서 찾습니다.

## 5. 주의

- `backbone/qwen3vl.py` 등 공용 코드는 다른 작업에서 바뀔 수 있습니다. `run.log`의 CLAIM 줄에 매번 md5가 남으니, 비교할 실행끼리 md5가 같은지 확인하세요. 기준 md5: nav4 `4f6e14a4`, answerer `5f0b363f`, backbone `9d2f3990`, engine `085f2bc2`, latplan 훅 `dc91f05e`.
- 문항당 시간이 수백 초로 늘면 공용 디스크 경합일 수 있습니다: `iostat -x 3 2 | grep sdb`에서 `r_await`가 수십 ms 이상이면 경합입니다.
- 실행을 멈출 때는 자기 프로세스 PID만 `kill <PID>`로 종료합니다.
