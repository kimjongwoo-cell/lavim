# thinking/ — qwen single baseline with the thinking trace captured

`baselines/` 파일은 **한 줄도 수정하지 않습니다.** 여기 있는 `run_thinking.py`가 기존
어댑터(`baselines/inference/qwen3-vl-4b-inference.py`)를 모듈로 임포트해서 런타임에
패치하고, `run_thinking.sh`는 기존 `baselines/qwen3-vl-4b.sh`를 그대로 재사용하되
`QWEN_ADAPTER`만 이쪽으로 돌립니다. 되돌리려면 이 폴더를 지우면 끝입니다.

## 실행

```bash
# 2문항 스모크
THINKING_MAX_ITEMS=2 CONDA_ENV=latentmas CUDA_VISIBLE_DEVICES=0 \
  bash /home/super/hj/wsi_latentmas_0731/models/single_v7/thinking/run_thinking.sh

# 전체 WSI-VQA
CONDA_ENV=latentmas CUDA_VISIBLE_DEVICES=0 \
  bash /home/super/hj/wsi_latentmas_0731/models/single_v7/thinking/run_thinking.sh

# BCNB
DATASET=bcnb CONDA_ENV=latentmas CUDA_VISIBLE_DEVICES=0 \
  bash /home/super/hj/wsi_latentmas_0731/models/single_v7/thinking/run_thinking.sh
```

결과: `thinking/results/<dataset>/qwen3-vl-4b/<RUN_ID>/predictions/`

| 파일 | 내용 |
| --- | --- |
| `qwen3vl_predictions.jsonl` | 기존 baseline과 동일한 스키마 (thinking 없음) |
| `thinking.jsonl` | 사고 과정 원문 + 토큰 수 + truncated 플래그 |
| `predictions_with_thinking.jsonl` | 위 둘을 question_id 기준으로 병합 (자동 생성) |

## 무엇을 패치하는가 (3곳)

1. **`QwenVLRunner._close_thinking` → identity.** 원본은 `apply_chat_template` 직후
   `"\n</think>\n\n"`를 붙여 첫 토큰 샘플링 전에 thinking 블록을 닫아버립니다.

2. **`QwenVLRunner.generate` → 래퍼.** 이건 장식이 아니라 **필수**입니다. 채팅 템플릿이
   여는 `<think>`를 *프롬프트 쪽에* 넣고(`add_generation_prompt`가
   `<|im_start|>assistant\n<think>\n`로 끝남), 어댑터는 생성된 토큰만 디코드하므로
   완성문은 여는 태그 없이

   ```
   <사고과정...></think>\n\nanswer: ...
   ```

   형태가 됩니다. `strip_thinking()`은 `<think>.*?</think>` 쌍을 찾기 때문에 **이걸 못
   지웁니다.** 래퍼가 `</think>` 기준으로 잘라 뒷부분만 어댑터에 돌려주지 않으면 사고
   과정이 그대로 answer/explanation에 섞여 평가가 조용히 오염됩니다.
   `num_output_tokens`는 thinking을 포함한 진짜 총량을 유지합니다(효율 지표 정직성).

3. **`load_pseudo_json` → `THINKING_MAX_ITEMS`로 앞부분만 자르기.** 스모크용.
   원본 어댑터엔 `--max-items`가 없어서 추가했습니다.

## 주의

- **기존 single baseline 숫자와 직접 비교 금지.** 기록된 single 결과는 전부
  thinking-off 조건입니다. 이건 별도 arm으로 취급하세요.
- **토큰 예산.** 기본 reasoning 예산은 4096입니다. 이 예산을 소진했는데 명시적인
  `answer:` 또는 `\\boxed{}`가 없으면 같은 이미지와 프롬프트를 thinking-off 512토큰으로
  한 번만 최종화합니다. 따라서 8192로 전체 생성을 늘릴 필요가 없습니다. 이미 실패가
  확인된 행만 복구할 때는 정상 행을 answers 파일에 남겨 resume한 뒤
  `SINGLE_DIRECT_FINAL_ONLY=1 MAX_NEW_TOKENS=512`를 사용합니다.
- **경로 고정.** `baselines/qwen3-vl-4b.sh`의 `LATENTMAS_ROOT` 기본값은
  `wsi_latentmas_0709`, `scripts/run_all_baselines.sh`의 기본값은 `_0701`입니다.
  `run_thinking.sh`는 자기 위치 기준으로 `models/single_v7`에 고정하므로 stale 디렉터리로 새지
  않습니다.
- **API 경로 미지원.** `LLM_URL`(OpenAI 호환 서버)로 돌리면 thinking이 서버 측
  `chat_template_kwargs`로 꺼지는데 그건 패치 대상이 아니라, 명시적으로 거부합니다.
- **resume 시 `thinking.jsonl`은 append**됩니다. 같은 파일에 이전 런이 섞이면
  `join_thinking.py`가 `unmatched traces` 개수를 알려줍니다.

## 검증 상태

- 모델 없는 배선 테스트 14/14 PASS (split / 패치 / sink / join / API 거부)
- GPU e2e 스모크 2문항 PASS — `results/wsivqa/qwen3-vl-4b/smoke_154152/` 에 남아 있음.
  `raw_prediction` 깨끗, MCQ 파싱 정상, trace 2/2 매칭, truncated 0
