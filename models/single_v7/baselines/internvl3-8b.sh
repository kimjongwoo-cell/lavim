#!/usr/bin/env bash
set -euo pipefail

#######################################
# InternVL3-8B (HF-native) single-model WSI-VQA inference + eval.
#
# Mirror of baselines/qwen3-vl-4b.sh — same thumbnail/eval scaffolding and the
# unified wsi_vqa_baselines prompting, so single-model numbers are directly
# comparable to the qwen baseline. Only the checkpoint + adapter differ.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash baselines/internvl3-8b.sh
#   MAX_ITEMS is NOT supported here (baseline adapter runs the full json); use a
#   trimmed INPUT_JSON for a smoke.
#
# BCNB:
#   PROMPT_STYLE=bcnb INPUT_JSON=/media/super/4TB/hj/SlideBench_BCNB/SlideBench_BCNB_test.json \
#   IMAGE_ROOT=/media/super/4TB/hj/SlideBench_BCNB/WSIs/WSIs bash baselines/internvl3-8b.sh
#######################################

REPO_ROOT="${REPO_ROOT:-/home/super/hj}"
HJ_ROOT="${HJ_ROOT:-$REPO_ROOT/HJ}"
LATENTMAS_ROOT="${LATENTMAS_ROOT:-$REPO_ROOT/wsi_latentmas_0711}"
GENERAL_EVALUATOR="${GENERAL_EVALUATOR:-$LATENTMAS_ROOT/eval/metrics.py}"
INTERNVL_ADAPTER="${INTERNVL_ADAPTER:-$LATENTMAS_ROOT/baselines/inference/internvl3-8b-inference.py}"

TEST_JSON="${TEST_JSON:-/home/super/hj/HJ/WSI-VQA/dataset/WSI_captions/WsiVQA_test.json}"
INPUT_JSON="${INPUT_JSON:-$TEST_JSON}"
CHECKPOINT="${CHECKPOINT:-/media/super/4TB/hj/InternVL3-8B-hf}"
LLM_URL="${LLM_URL:-}"
LLM_MODEL="${LLM_MODEL:-}"
API_KEY="${API_KEY:-${OPENAI_API_KEY:-}}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-300}"

IMAGE_PATH="${IMAGE_PATH:-}"
IMAGE_ROOT="${IMAGE_ROOT:-}"
SLIDE_DIR="${SLIDE_DIR:-/media/super/4TB/hj/DATA_SVS}"
THUMBNAIL_SIZE="${THUMBNAIL_SIZE:-1024}"
SHARED_THUMBNAIL_DIR="${SHARED_THUMBNAIL_DIR:-$HJ_ROOT/baseline_results/wsi-vqa/shared_thumbnails_dx1}"

RUN_ID="${RUN_ID:-$(date '+%Y%m%d_%H%M%S')}"
DATASET_TAG="${DATASET_TAG:-$(basename "$INPUT_JSON")}"
DATASET_TAG="${DATASET_TAG%.*}"
RESULTS_ROOT="${RESULTS_ROOT:-$LATENTMAS_ROOT/results/wsi-vqa}"
DEFAULT_WORK_DIR="$RESULTS_ROOT/internvl3-8b/${RUN_ID}"
WORK_DIR="${WORK_DIR:-$DEFAULT_WORK_DIR}"

THUMBNAIL_DIR="${THUMBNAIL_DIR:-$SHARED_THUMBNAIL_DIR}"
ANSWERS_FILE="${ANSWERS_FILE:-$WORK_DIR/predictions/internvl_predictions.jsonl}"
VIS_DIR="${VIS_DIR:-$WORK_DIR/vis}"
EVAL_DIR="${EVAL_DIR:-$WORK_DIR/eval}"
EVAL_JSON="${EVAL_JSON:-$EVAL_DIR/internvl_summary.json}"
EVAL_CSV="${EVAL_CSV:-$EVAL_DIR/internvl_details.csv}"
GENERAL_OUTPUT_JSONL="${GENERAL_OUTPUT_JSONL:-$EVAL_DIR/internvl_general_eval.jsonl}"
INFERENCE_TIME_JSON="${INFERENCE_TIME_JSON:-$EVAL_DIR/inference_time.json}"
GENERAL_EVAL_DIR="${GENERAL_EVAL_DIR:-$EVAL_DIR/general}"
GENERAL_EVAL_NAME="${GENERAL_EVAL_NAME:-internvl3_8b_${DATASET_TAG}}"
GENERAL_EVAL_TXT="${GENERAL_EVAL_TXT:-$EVAL_DIR/general_eval.txt}"
LOG_DIR="${LOG_DIR:-$WORK_DIR/logs}"
RUN_LOG="${RUN_LOG:-$LOG_DIR/internvl3_8b.log}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
NUM_BEAMS="${NUM_BEAMS:-1}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
MODEL_CLASS="${MODEL_CLASS:-}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-false}"

DO_SAMPLE="${DO_SAMPLE:-false}"
FORCE_CHOICE_ANSWER="${FORCE_CHOICE_ANSWER:-true}"
MCQ_STYLE="${MCQ_STYLE:-text}"
PROMPT_STYLE="${PROMPT_STYLE:-wsi}"
OVERWRITE_THUMBNAILS="${OVERWRITE_THUMBNAILS:-false}"
OVERWRITE_PREDICTIONS="${OVERWRITE_PREDICTIONS:-false}"
SKIP_INFERENCE="${SKIP_INFERENCE:-false}"
RUN_GENERAL_EVAL="${RUN_GENERAL_EVAL:-true}"
GENERAL_EVAL_METRIC_SCOPE="${GENERAL_EVAL_METRIC_SCOPE:-both}"
PRINT_OUT="${PRINT_OUT:-false}"
SEED="${SEED:-0}"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

timestamp_stream() {
  while IFS= read -r line; do
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$line"
  done
}

setup_logging() {
  mkdir -p "$LOG_DIR" "$EVAL_DIR" "$(dirname "$ANSWERS_FILE")"
  exec > >(timestamp_stream >> "$RUN_LOG") 2>&1
}

log() { echo "$*"; }

require_file() {
  local path="$1"; local label="$2"
  [[ -f "$path" ]] || { log "Missing $label: $path"; return 1; }
}
require_dir() {
  local path="$1"; local label="$2"
  [[ -d "$path" ]] || { log "Missing $label: $path"; return 1; }
}
require_python_module() {
  local m="$1"; local hint="$2"
  python -c "import ${m}" >/dev/null 2>&1 || { log "Missing module: ${m} ($hint)"; return 1; }
}
flag_enabled() {
  case "$1" in 1|true|True|yes|YES|y|Y) return 0;; *) return 1;; esac
}

check_inputs() {
  require_file "$INPUT_JSON" "input JSON"
  require_file "$INTERNVL_ADAPTER" "InternVL inference adapter"
  require_file "$GENERAL_EVALUATOR" "general WSI-VQA evaluator"
  if [[ -z "$LLM_URL" ]]; then
    require_dir "$CHECKPOINT" "InternVL checkpoint directory"
    require_file "$CHECKPOINT/config.json" "InternVL config"
  fi
  [[ -n "$IMAGE_ROOT" ]] && require_dir "$IMAGE_ROOT" "image root"
  [[ -n "$IMAGE_PATH" ]] && require_file "$IMAGE_PATH" "single image"
  if [[ -z "$IMAGE_ROOT" && -z "$IMAGE_PATH" ]]; then
    require_dir "$SLIDE_DIR" "slide directory"
  fi
  require_python_module "PIL" "pip install pillow"
  if [[ -z "$LLM_URL" ]]; then
    require_python_module "torch" "pip install torch"
    require_python_module "transformers" "pip install transformers accelerate"
  fi
}

run_internvl() {
  local args=(
    "$INTERNVL_ADAPTER"
    --checkpoint "$CHECKPOINT"
    --llm-url "$LLM_URL"
    --llm-model "$LLM_MODEL"
    --api-key "$API_KEY"
    --request-timeout "$REQUEST_TIMEOUT"
    --input-json "$INPUT_JSON"
    --thumbnail-dir "$THUMBNAIL_DIR"
    --answers-file "$ANSWERS_FILE"
    --vis-dir "$VIS_DIR"
    --output-csv "$EVAL_CSV"
    --output-json "$EVAL_JSON"
    --general-jsonl "$GENERAL_OUTPUT_JSONL"
    --inference-time-json "$INFERENCE_TIME_JSON"
    --thumbnail-size "$THUMBNAIL_SIZE"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --num-beams "$NUM_BEAMS"
    --temperature "$TEMPERATURE"
    --repetition-penalty "$REPETITION_PENALTY"
    --seed "$SEED"
    --device-map "$DEVICE_MAP"
    --torch-dtype "$TORCH_DTYPE"
  )
  [[ -n "$SLIDE_DIR" ]]   && args+=(--slide-dir "$SLIDE_DIR")
  [[ -n "$IMAGE_ROOT" ]]  && args+=(--image-root "$IMAGE_ROOT")
  [[ -n "$IMAGE_PATH" ]]  && args+=(--image-path "$IMAGE_PATH")
  [[ -n "$TOP_P" ]]       && args+=(--top-p "$TOP_P")
  [[ -n "$MODEL_CLASS" ]] && args+=(--model-class "$MODEL_CLASS")
  flag_enabled "$TRUST_REMOTE_CODE"   && args+=(--trust-remote-code)
  flag_enabled "$DO_SAMPLE"           && args+=(--do-sample)
  flag_enabled "$FORCE_CHOICE_ANSWER" || args+=(--no-force-choice-answer)
  args+=(--mcq-style "$MCQ_STYLE")
  args+=(--prompt-style "$PROMPT_STYLE")
  flag_enabled "$OVERWRITE_THUMBNAILS"  && args+=(--overwrite-thumbnails)
  flag_enabled "$OVERWRITE_PREDICTIONS" && args+=(--overwrite-predictions)
  flag_enabled "$SKIP_INFERENCE"        && args+=(--skip-inference)
  flag_enabled "$PRINT_OUT"             && args+=(--print-out)
  python "${args[@]}"
}

run_general_eval() {
  flag_enabled "$RUN_GENERAL_EVAL" || { log "Skipping general eval"; return 0; }
  log "Running eval/metrics.py"
  mkdir -p "$GENERAL_EVAL_DIR"
  python "$GENERAL_EVALUATOR" \
    --input "$GENERAL_OUTPUT_JSONL" \
    --questions "$GENERAL_OUTPUT_JSONL" \
    --output-dir "$GENERAL_EVAL_DIR" \
    --name "$GENERAL_EVAL_NAME" \
    --inference-time-json "$INFERENCE_TIME_JSON" \
    --metric-scope "$GENERAL_EVAL_METRIC_SCOPE" | tee "$GENERAL_EVAL_TXT" || log "General eval failed (non-fatal)"
}

main() {
  setup_logging
  log "InternVL3-8B single-model WSI-VQA baseline"
  log "Checkpoint: $CHECKPOINT"
  log "Adapter: $INTERNVL_ADAPTER"
  log "Input JSON: $INPUT_JSON"
  log "Thumbnail dir: $THUMBNAIL_DIR"
  log "Answers file: $ANSWERS_FILE"
  log "Work dir: $WORK_DIR"
  log "Log: $RUN_LOG"
  check_inputs
  run_internvl
  run_general_eval
  log "Done."
  log "Answers: $ANSWERS_FILE"
  log "Summary JSON: $EVAL_JSON"
  log "General eval dir: $GENERAL_EVAL_DIR"
}

main "$@"
