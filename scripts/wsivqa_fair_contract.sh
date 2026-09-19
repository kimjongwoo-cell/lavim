#!/usr/bin/env bash

fair_contract_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FAIR_CONTRACT_FILE="${FAIR_CONTRACT_FILE:-$fair_contract_root/config/wsivqa_fair_v1.json}"

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required to load $FAIR_CONTRACT_FILE" >&2
  return 2 2>/dev/null || exit 2
fi

FAIR_CONTRACT_ID="$(jq -er '.contract_id' "$FAIR_CONTRACT_FILE")"
FAIR_DATASET="${WSIVQA_DATASET:-$(jq -er '.dataset.path' "$FAIR_CONTRACT_FILE")}"
FAIR_EXPECTED_CASES="$(jq -er '.dataset.expected_cases' "$FAIR_CONTRACT_FILE")"
FAIR_SLIDE_ROOT="${WSIVQA_SLIDE_ROOT:-$(jq -er '.dataset.slide_root' "$FAIR_CONTRACT_FILE")}"
FAIR_CHECKPOINT="${WSIVQA_MODEL:-$(jq -er '.model.checkpoint' "$FAIR_CONTRACT_FILE")}"
FAIR_SERVED_MODEL="$(jq -er '.model.served_name' "$FAIR_CONTRACT_FILE")"
FAIR_DTYPE="$(jq -er '.model.dtype' "$FAIR_CONTRACT_FILE")"
FAIR_MAX_MODEL_LEN="$(jq -er '.model.max_model_len' "$FAIR_CONTRACT_FILE")"
FAIR_TEMPERATURE="$(jq -er '.sampling.temperature' "$FAIR_CONTRACT_FILE")"
FAIR_TOP_P="$(jq -er '.sampling.top_p' "$FAIR_CONTRACT_FILE")"
FAIR_SEED="$(jq -er '.sampling.seed' "$FAIR_CONTRACT_FILE")"
FAIR_NATIVE_THINKING="$(jq -er '.generation.native_thinking // true' "$FAIR_CONTRACT_FILE")"
FAIR_MAX_NEW_TOKENS="$(jq -er '.generation.max_new_tokens_per_request // 4096' "$FAIR_CONTRACT_FILE")"
FAIR_VLLM_BASE_URL="$(jq -er '.execution.vllm_base_url' "$FAIR_CONTRACT_FILE")"
FAIR_WORKER_COUNT="$(jq -er '.execution.worker_count' "$FAIR_CONTRACT_FILE")"
FAIR_MAX_ROUNDS="$(jq -er '.execution.max_verifier_rounds' "$FAIR_CONTRACT_FILE")"
configured_evaluator="$(jq -er '.evaluation.script' "$FAIR_CONTRACT_FILE")"
if [[ -f "$configured_evaluator" ]]; then
  FAIR_EVALUATOR="$configured_evaluator"
else
  FAIR_EVALUATOR="${WSIVQA_EVALUATOR:-$fair_contract_root/eval/metrics.py}"
fi
FAIR_METRIC_SCOPE="$(jq -er '.evaluation.metric_scope' "$FAIR_CONTRACT_FILE")"

readonly FAIR_CONTRACT_FILE FAIR_CONTRACT_ID FAIR_DATASET FAIR_EXPECTED_CASES
readonly FAIR_SLIDE_ROOT FAIR_CHECKPOINT FAIR_SERVED_MODEL FAIR_DTYPE
readonly FAIR_MAX_MODEL_LEN FAIR_TEMPERATURE FAIR_TOP_P FAIR_SEED
readonly FAIR_NATIVE_THINKING FAIR_MAX_NEW_TOKENS
readonly FAIR_VLLM_BASE_URL FAIR_WORKER_COUNT FAIR_MAX_ROUNDS
readonly FAIR_EVALUATOR FAIR_METRIC_SCOPE
