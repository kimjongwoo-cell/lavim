# 0806 onepass think-off 디코드 붕괴 — 조사 기록 (2026-08-07)

> 규칙: **검증된 것만 단정.** 각 항목에 근거 태그 — `[GPU실측]` / `[코드]` / `[미검증-가설]`.

## 질문
0717은 think-off로 latent KV를 이어받아 디코드해도 답이 멀쩡한데,
0806 onepass는 think-off(`cumulative_nothink`)면 답이 template 토큰으로 도배됨.
**왜?**

## 결론 (검증 완료)
0806 onepass 한정 진범 = **m latent step을 `<think>` 열어놓고 생성해야 함(D1).**
답 register(think 닫힘)에서 latent을 생성하면 carried KV가 오염되고, 그 뒤 무엇을
해도(닫음 경계·realign·답 format) answerer가 도배됨. `cumulative`만 이 조건 충족.
※ 0717이 왜 think-closed로도 되는지는 **여전히 별개 미확정**(0806 onepass 구조 한정 결론).

---

## ✅ 검증된 사실

### GPU 실측 (동일 세팅: WSI-VQA idx 0–4, latent-steps 5, 512 tok, greedy, no-rationale, seed 42, Qwen3-VL-4B-Thinking)
| 조건 | 결과 | 근거 |
|---|---|---|
| `cumulative` (think ON) | **5/5 클린** (`invasive ductal carcinoma`, `50mm`, `low`, `no`, …) | [GPU실측] smoke_ab |
| `cumulative_nothink` (think OFF) | **5/5 도배** (`rule rule rule…`, `root root…`, `text text…`, `choice choice…`) | [GPU실측] smoke_ab |
| `cumulative_nothink` + **realign=wa** (0717과 동일 realign) | **5/5 도배** (여전히 붕괴) | [GPU실측] smoke_wa |
| **`cumulative` + 닫음 `</think>` 제거** (CLOSE_THINK=off) | **5/5 클린** | [GPU실측] smoke_d1d2 |
| `cumulative_nothink` + **JSON 답변**(`{"answer":`) | **5/5 도배** (`{"answer":rule rule…`) | [GPU실측] smoke_njson |
| **`cumulative` + JSON 답변** | **5/5 클린** (`{"answer":"invasive ductal carcinoma"}`) | [GPU실측] smoke_cjson |

- confound 제거해도 붕괴 재현: 옛 0/5 주장에 있던 `--answerer-max-new-tokens 32`(→512로), `set -e` 미실행 이슈 전부 제거해도 동일. [GPU실측]
- **answerer 프롬프트는 clean/flood 모드에서 바이트 동일**(670자) → 범인은 **carried KV 내용**이지 프롬프트/답format 아님. [GPU실측] answerer_call.json 비교
- **D1(latent register)이 유일 원인, D2(닫음 경계) 무관**: cumulative에서 `</think>` 경계를 빼도 클린 → latent를 `<think>` 열고 생성했느냐만이 결정. [GPU실측]

### 코드 읽기
- 0806 terminal `generate_on_kv`은 0717과 사실상 동일 — 둘 다 프롬프트 끝 `<think>\n`이면 `</think>\n\n` 추가해 닫음. [코드] `0806/backbone/qwen3vl.py:1204` ↔ `0717/backbone/qwen3vl.py:1082`
- 0717 reference(point.sh)는 think-off로 돌음 — `--think`/`--reasoner_think` 계열 플래그 없음, `get_thinking` 기본 reasoner=False. [코드] `0717/scripts/point.sh`, `0717/utils/thinking.py:19-25`
- **0717도 full KV carry** — `--carry_prune none`, Navigator→Reasoner→Verifier로 latent KV 이어받음. (0806만 쌓는 게 아님) [코드] `0717/scripts/point.sh:87,135`
- realign 기본값: 0717 = `wa`, 0806 onepass = `identity_norm` 하드코딩. [코드] `0717/backbone/qwen3vl.py:38`, `0806/vision_text_mas/latent_qwen_engine.py:190`
- 0806 nothink 디코드는 `continuation_user_turn=True` → 시스템을 `"Next-stage role instructions:\n{system}\n\n{user}"`로 접어 user 턴에 넣음. 0717은 실제 `[system, user]` 턴 재방출. [코드] `0806/backbone/qwen3vl.py:280,1186`

---

## ❌ 실험으로 소거된 원인 (범인 아님)
- **답 format (tag/JSON)** — nothink+JSON(`{"answer":`)도 5/5 도배. carried KV가 이미 오염이라 껍데기 무의미. [GPU실측]
- **닫음 경계 `</think>` (D2)** — cumulative에서 빼도 클린. [GPU실측]
- **realign 방식** — wa(0717것)로 바꿔도 think-off는 5/5 도배. W_a 실제 빌드됨(2560×2560, identity 아님)에도. [GPU실측]
- **answerer 프롬프트** — clean/flood 바이트 동일. [GPU실측]
- **terminal decode 코드** — 0717과 바이트 동일 이식 확인. [코드]
- **anti-repetition / max-token confound** — 512로 제거해도 그대로. [GPU실측]
- **"onepass가 latent를 쌓아서 / 0717은 fresh reprefill이라서"** — 반증됨: 0717도 kv carry로 latent 쌓음. [코드]

---

## ⚠️ 여전히 미검증 (별개 질문)
- **0717은 왜 think-closed latent로도 되는가** — 0717 `_embed_prompt`는 think-off일 때 빈 `</think>` 뒤에 latent 생성(0806 nothink와 같은 구조)인데 작동함. 0806 onepass엔 D1이 확정됐지만 0717과의 그 차이는 아직 격리 안 됨. [미검증]

---

## ✅ 실무 결론 (검증된 범위)
- 0806 onepass에서 **클린 디코드 = latent를 `<think>` 열고 생성하는 `cumulative` 뿐.** [GPU실측]
- think-off 계열(upstream/nothink/nothink+wa/nothink+json) 전부 도배. [GPU실측]
- **0717식 JSON 출력을 원하면**: `--transport-mode cumulative --answerer-protocol json` → `{"answer":"..."}` 5/5 클린. [GPU실측]
- CLI 기본값 `cumulative_nothink`(붕괴)는 바꿔야 함. `vision_text_mas/latent_onepass_cli.py:177` [코드]

## 재현 (scratchpad/)
- `smoke_nothink_ab.sh` — cumulative vs cumulative_nothink
- `smoke_nothink_wa.sh` — nothink + realign=wa (ONEPASS_REALIGN env, 조사 후 원복)
- `smoke_d1d2.sh` — cumulative + CLOSE_THINK=off (D1 vs D2 격리; env 조사 후 원복)
- `smoke_nothink_json.sh` / `smoke_cum_json.sh` — 답 format 격리
