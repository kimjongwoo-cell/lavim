#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# LatentMAS official Single reasoning adapted to WSI, with full trace saved.
#
# Reuses baselines/qwen3-vl-4b.sh unchanged and only redirects QWEN_ADAPTER to
# thinking/run_thinking.py, which patches the stock adapter at runtime. It uses
# one sampled generation and the last boxed answer. Nothing under baselines/ is
# modified.
#
# It also pins LATENTMAS_ROOT to THIS directory's repo: the stock script
# defaults to wsi_latentmas_0709 and run_all_baselines.sh defaults to _0701, so
# without pinning you would run stale code and write results there.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash thinking/run_thinking.sh                 # wsi-vqa
#   DATASET=bcnb CUDA_VISIBLE_DEVICES=0 bash thinking/run_thinking.sh    # bcnb
#   THINKING_MAX_ITEMS=2 bash thinking/run_thinking.sh                   # smoke
#
# Notes:
#   - Sampling values come from the canonical launcher through SINGLE_* overrides.
# ─────────────────────────────────────────────────────────────────────────────

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LATENTMAS_ROOT="${LATENTMAS_ROOT:-$(dirname "$HERE")}"
FAIR_REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
source "$FAIR_REPO_ROOT/scripts/wsivqa_fair_contract.sh"

python_bin="${PYTHON_BIN:-python}"
if ! python_path="$(command -v "$python_bin")"; then
  echo "PYTHON_BIN is not executable: $python_bin" >&2
  exit 2
fi
export PATH="$(dirname "$python_path"):$PATH"

export REPO_ROOT="${REPO_ROOT:-/home/super/hj}"
export HJ_ROOT="${HJ_ROOT:-$REPO_ROOT/HJ}"
export LATENTMAS_ROOT
export QWEN_ADAPTER="${QWEN_ADAPTER:-$HERE/run_thinking.py}"
export GENERAL_EVALUATOR="$FAIR_EVALUATOR"

export CHECKPOINT="$FAIR_CHECKPOINT"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-$FAIR_MAX_NEW_TOKENS}"
export SINGLE_FINALIZATION_MAX_NEW_TOKENS="${SINGLE_FINALIZATION_MAX_NEW_TOKENS:-$FAIR_MAX_NEW_TOKENS}"
export TEMPERATURE="${SINGLE_TEMPERATURE:-$FAIR_TEMPERATURE}"
export TOP_P="${SINGLE_TOP_P:-$FAIR_TOP_P}"
export DO_SAMPLE="${DO_SAMPLE:-true}"
export SEED="${SINGLE_SEED:-$FAIR_SEED}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-$FAIR_MAX_MODEL_LEN}"
SINGLE_BACKEND="${SINGLE_BACKEND:-vllm}"
case "$SINGLE_BACKEND" in
  vllm)
    backend="vllm"
    export LLM_URL="$FAIR_VLLM_BASE_URL"
    export LLM_MODEL="$FAIR_SERVED_MODEL"
    ;;
  native-vllm)
    backend="native-vllm-direct-engine"
    export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
    export LLM_URL="inprocess://vllm"
    export LLM_MODEL="$CHECKPOINT"
    export SINGLE_DIRECT_VLLM=1
    ;;
  transformers|hf)
    backend="transformers"
    export LLM_URL=""
    export LLM_MODEL=""
    ;;
  hf-service)
    backend="transformers-batched"
    export LLM_URL="${HF_SERVICE_URL:-http://127.0.0.1:8123/v1}"
    export LLM_MODEL="$FAIR_SERVED_MODEL"
    ;;
  *)
    echo "unknown SINGLE_BACKEND=$SINGLE_BACKEND (expected: vllm | native-vllm | transformers | hf-service)" >&2
    exit 2
    ;;
esac
export SINGLE_NATIVE_THINKING="${SINGLE_NATIVE_THINKING:-false}"
export SINGLE_CANONICAL_OPEN_OPTIONS="${SINGLE_CANONICAL_OPEN_OPTIONS:-1}"
export SINGLE_ANSWERER_PROTOCOL="${SINGLE_ANSWERER_PROTOCOL:-structured_json}"
export SINGLE_OUTPUT_PROTOCOL="${SINGLE_OUTPUT_PROTOCOL:-structured_json}"
export SINGLE_DIRECT_FINAL_ONLY="${SINGLE_DIRECT_FINAL_ONLY:-1}"
export SINGLE_OFFICIAL_ONE_CALL="${SINGLE_OFFICIAL_ONE_CALL:-false}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export RUN_ID="${RUN_ID:-$(date '+%Y%m%d_%H%M%S')}"

DATASET="${DATASET:-wsivqa}"
BCNB_DIR="${BCNB_DIR:-/media/super/4TB/hj/SlideBench_BCNB}"
WSIVQA_JSON="$FAIR_DATASET"
WSIVQA_SLIDES="$FAIR_SLIDE_ROOT"

case "$DATASET" in
  wsivqa)
    export INPUT_JSON="${INPUT_JSON:-$WSIVQA_JSON}"
    export TEST_JSON="${TEST_JSON:-$WSIVQA_JSON}"
    export SLIDE_DIR="${SLIDE_DIR:-$WSIVQA_SLIDES}"
    export IMAGE_ROOT="${IMAGE_ROOT:-}"
    export PROMPT_STYLE="${PROMPT_STYLE:-wsi}"
    export MCQ_STYLE="${MCQ_STYLE:-text}"
    export DATASET_TAG="${DATASET_TAG:-WsiVQA_test}"
    ;;
  bcnb)
    export INPUT_JSON="${INPUT_JSON:-$BCNB_DIR/SlideBench_BCNB_test.json}"
    export IMAGE_ROOT="${IMAGE_ROOT:-$BCNB_DIR/WSIs/WSIs}"
    export THUMBNAIL_DIR="${THUMBNAIL_DIR:-$BCNB_DIR/thumbnails}"
    export SLIDE_DIR="${SLIDE_DIR:-}"
    export PROMPT_STYLE="${PROMPT_STYLE:-bcnb}"
    export DATASET_TAG="${DATASET_TAG:-SlideBench_BCNB}"
    ;;
  *)
    echo "unknown DATASET=$DATASET (expected: wsivqa | bcnb)" >&2
    exit 2
    ;;
esac

# Pin WORK_DIR so the join step below can find the outputs the stock script wrote.
export RESULTS_ROOT="${RESULTS_ROOT:-$HERE/results/$DATASET}"
export WORK_DIR="${WORK_DIR:-$RESULTS_ROOT/qwen3-vl-4b/$RUN_ID}"

mkdir -p "$WORK_DIR"
jq -n \
  --arg contract_id "$FAIR_CONTRACT_ID" \
  --arg contract_file "$FAIR_CONTRACT_FILE" \
  --arg dataset "$INPUT_JSON" \
  --arg slide_root "$SLIDE_DIR" \
  --arg model "$CHECKPOINT" \
  --arg evaluator "$GENERAL_EVALUATOR" \
  --arg backend "$backend" \
  --arg started_at "$(date --iso-8601=seconds)" \
  --arg git_commit "$(git -C "$FAIR_REPO_ROOT" rev-parse HEAD)" \
  --argjson native_thinking "$SINGLE_NATIVE_THINKING" \
  --argjson max_new_tokens "$MAX_NEW_TOKENS" \
  --argjson finalization_max_new_tokens "$SINGLE_FINALIZATION_MAX_NEW_TOKENS" \
  --argjson temperature "$TEMPERATURE" \
  --argjson top_p "$TOP_P" \
  --argjson seed "$SEED" \
  --argjson max_model_len "$MAX_MODEL_LEN" \
  --argjson official_one_call "$SINGLE_OFFICIAL_ONE_CALL" \
  --arg output_protocol "$SINGLE_ANSWERER_PROTOCOL" \
  '{contract_id:$contract_id,contract_file:$contract_file,dataset:$dataset,slide_root:$slide_root,model:$model,backend:$backend,evaluator:$evaluator,started_at:$started_at,git_commit:$git_commit,native_thinking:$native_thinking,max_new_tokens:$max_new_tokens,finalization_max_new_tokens:$finalization_max_new_tokens,temperature:$temperature,top_p:$top_p,seed:$seed,max_model_len:$max_model_len,official_one_call:$official_one_call,output_protocol:$output_protocol}' \
  >"$WORK_DIR/run_manifest.json"

echo "LatentMAS-compatible qwen WSI Single"
echo "  repo       : $LATENTMAS_ROOT"
echo "  adapter    : $QWEN_ADAPTER"
echo "  dataset    : $DATASET  ($INPUT_JSON)"
echo "  checkpoint : $CHECKPOINT"
echo "  max tokens : $MAX_NEW_TOKENS"
echo "  max context: $MAX_MODEL_LEN"
echo "  backend    : $backend"
echo "  sampling   : $DO_SAMPLE  temp=$TEMPERATURE  top_p=$TOP_P  seed=$SEED"
echo "  one call   : $SINGLE_OFFICIAL_ONE_CALL  protocol=$SINGLE_ANSWERER_PROTOCOL"
echo "  work dir   : $WORK_DIR"
echo "  gpu        : $CUDA_VISIBLE_DEVICES"
echo "  (detailed output goes to \$WORK_DIR/logs/qwen3_vl_4b.log)"

CONDA_ENV="${CONDA_ENV:-}"
if [[ -n "$CONDA_ENV" ]] && command -v conda >/dev/null 2>&1; then
  conda run -n "$CONDA_ENV" --no-capture-output bash "$LATENTMAS_ROOT/baselines/qwen3-vl-4b.sh"
else
  bash "$LATENTMAS_ROOT/baselines/qwen3-vl-4b.sh"
fi

PRED="$WORK_DIR/predictions/qwen3vl_predictions.jsonl"
TRACE="$WORK_DIR/predictions/thinking.jsonl"
if [[ -f "$PRED" && -f "$TRACE" ]]; then
  echo ""
  echo "merging thinking trace into predictions"
  python "$HERE/join_thinking.py" --predictions "$PRED" --thinking "$TRACE"
else
  echo "skipping join (missing $PRED or $TRACE)"
fi
