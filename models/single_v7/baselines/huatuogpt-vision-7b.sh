#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/super/hj}"
HJ_ROOT="${HJ_ROOT:-$REPO_ROOT/HJ}"
LATENTMAS_ROOT="${LATENTMAS_ROOT:-$REPO_ROOT/wsi_latentmas_0701}"
GENERAL_EVALUATOR="${GENERAL_EVALUATOR:-$LATENTMAS_ROOT/eval/metrics.py}"
HUATUO_ADAPTER="${HUATUO_ADAPTER:-$LATENTMAS_ROOT/baselines/inference/huatuogpt-vision-7b-inference.py}"
HUATUO_CODE_DIR="${HUATUO_CODE_DIR:-/media/super/4TB/hj/huatuogpt-vision}"
PYTHON_BIN="${PYTHON_BIN:-/home/super/miniforge3/envs/latentmas/bin/python}"

TEST_JSON="${TEST_JSON:-/home/super/hj/HJ/WSI-VQA/dataset/WSI_captions/WsiVQA_test.json}"
INPUT_JSON="${INPUT_JSON:-$TEST_JSON}"
CHECKPOINT="${CHECKPOINT:-/media/super/4TB/hj/HuatuoGPT-Vision-7B}"

IMAGE_PATH="${IMAGE_PATH:-}"
IMAGE_ROOT="${IMAGE_ROOT:-}"
SLIDE_DIR="${SLIDE_DIR:-/media/super/4TB/hj/DATA_SVS}"
THUMBNAIL_SIZE="${THUMBNAIL_SIZE:-1024}"
SHARED_THUMBNAIL_DIR="${SHARED_THUMBNAIL_DIR:-$HJ_ROOT/baseline_results/wsi-vqa/shared_thumbnails_dx1}"

RUN_ID="${RUN_ID:-$(date '+%Y%m%d_%H%M%S')}"
DATASET_TAG="${DATASET_TAG:-$(basename "$INPUT_JSON")}"
DATASET_TAG="${DATASET_TAG%.*}"
RESULTS_ROOT="${RESULTS_ROOT:-$LATENTMAS_ROOT/results/wsi-vqa}"
DEFAULT_WORK_DIR="$RESULTS_ROOT/huatuogpt-vision-7b/${RUN_ID}"
WORK_DIR="${WORK_DIR:-$DEFAULT_WORK_DIR}"

THUMBNAIL_DIR="${THUMBNAIL_DIR:-$SHARED_THUMBNAIL_DIR}"
ANSWERS_FILE="${ANSWERS_FILE:-$WORK_DIR/predictions/huatuogpt_vision_7b_predictions.jsonl}"
VIS_DIR="${VIS_DIR:-$WORK_DIR/vis}"
EVAL_DIR="${EVAL_DIR:-$WORK_DIR/eval}"
EVAL_JSON="${EVAL_JSON:-$EVAL_DIR/huatuogpt_vision_7b_summary.json}"
EVAL_CSV="${EVAL_CSV:-$EVAL_DIR/huatuogpt_vision_7b_details.csv}"
GENERAL_OUTPUT_JSONL="${GENERAL_OUTPUT_JSONL:-$EVAL_DIR/huatuogpt_vision_7b_general_eval.jsonl}"
INFERENCE_TIME_JSON="${INFERENCE_TIME_JSON:-$EVAL_DIR/inference_time.json}"
GENERAL_EVAL_DIR="${GENERAL_EVAL_DIR:-$EVAL_DIR/general}"
GENERAL_EVAL_NAME="${GENERAL_EVAL_NAME:-huatuogpt_vision_7b_${DATASET_TAG}}"
GENERAL_EVAL_TXT="${GENERAL_EVAL_TXT:-$EVAL_DIR/general_eval.txt}"
LOG_DIR="${LOG_DIR:-$WORK_DIR/logs}"
RUN_LOG="${RUN_LOG:-$LOG_DIR/huatuogpt_vision_7b.log}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
NUM_BEAMS="${NUM_BEAMS:-1}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
DEVICE="${DEVICE:-cuda}"

DO_SAMPLE="${DO_SAMPLE:-false}"
FORCE_CHOICE_ANSWER="${FORCE_CHOICE_ANSWER:-true}"
MCQ_STYLE="${MCQ_STYLE:-letter}"
PROMPT_STYLE="${PROMPT_STYLE:-wsi}"
# PROMPT_STYLE="${PROMPT_STYLE:-navigate}"
OVERWRITE_THUMBNAILS="${OVERWRITE_THUMBNAILS:-false}"
OVERWRITE_PREDICTIONS="${OVERWRITE_PREDICTIONS:-false}"
SKIP_INFERENCE="${SKIP_INFERENCE:-false}"
RUN_GENERAL_EVAL="${RUN_GENERAL_EVAL:-true}"
GENERAL_EVAL_METRIC_SCOPE="${GENERAL_EVAL_METRIC_SCOPE:-both}"
PRINT_OUT="${PRINT_OUT:-false}"
SEED="${SEED:-0}"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export CUDA_VISIBLE_DEVICES

timestamp_stream() {
  while IFS= read -r line; do
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$line"
  done
}

setup_logging() {
  mkdir -p "$LOG_DIR" "$EVAL_DIR" "$(dirname "$ANSWERS_FILE")"
  # Detailed output -> log file only; terminal stays clean for the tqdm bar.
  exec > >(timestamp_stream >> "$RUN_LOG") 2>&1
}

log() {
  echo "$*"
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    log "Missing $label: $path"
    return 1
  fi
}

require_dir() {
  local path="$1"
  local label="$2"
  if [[ ! -d "$path" ]]; then
    log "Missing $label: $path"
    return 1
  fi
}

require_python_module() {
  local module_name="$1"
  local install_hint="$2"
  if ! "$PYTHON_BIN" -c "import ${module_name}" >/dev/null 2>&1; then
    log "Missing Python module: ${module_name}"
    log "Install it in the current environment: ${install_hint}"
    return 1
  fi
}

flag_enabled() {
  case "$1" in
    1|true|True|yes|YES|y|Y)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

check_inputs() {
  require_file "$INPUT_JSON" "input JSON"
  require_file "$HUATUO_ADAPTER" "HuatuoGPT-Vision inference adapter"
  require_file "$PYTHON_BIN" "Python interpreter"
  require_file "$HUATUO_CODE_DIR/cli.py" "HuatuoGPT-Vision cli.py"
  require_file "$GENERAL_EVALUATOR" "general WSI-VQA evaluator"

  require_dir "$CHECKPOINT" "HuatuoGPT-Vision checkpoint directory"
  require_file "$CHECKPOINT/config.json" "HuatuoGPT-Vision config"

  if [[ -n "$IMAGE_ROOT" ]]; then
    require_dir "$IMAGE_ROOT" "image root"
  fi
  if [[ -n "$IMAGE_PATH" ]]; then
    require_file "$IMAGE_PATH" "single image"
  fi
  if [[ -z "$IMAGE_ROOT" && -z "$IMAGE_PATH" ]]; then
    require_dir "$SLIDE_DIR" "slide directory"
  fi

  require_python_module "PIL" "pip install pillow"
  require_python_module "torch" "pip install torch"
  require_python_module "transformers" "pip install transformers accelerate"
  require_python_module "shortuuid" "pip install shortuuid"
}

run_huatuogpt() {
  local args=(
    "$HUATUO_ADAPTER"
    --checkpoint "$CHECKPOINT"
    --huatuo-code-dir "$HUATUO_CODE_DIR"
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
    --device "$DEVICE"
  )

  if [[ -n "$SLIDE_DIR" ]]; then
    args+=(--slide-dir "$SLIDE_DIR")
  fi
  if [[ -n "$IMAGE_ROOT" ]]; then
    args+=(--image-root "$IMAGE_ROOT")
  fi
  if [[ -n "$IMAGE_PATH" ]]; then
    args+=(--image-path "$IMAGE_PATH")
  fi
  if [[ -n "$TOP_P" ]]; then
    args+=(--top-p "$TOP_P")
  fi
  if flag_enabled "$DO_SAMPLE"; then
    args+=(--do-sample)
  fi
  if ! flag_enabled "$FORCE_CHOICE_ANSWER"; then
    args+=(--no-force-choice-answer)
  fi
  args+=(--mcq-style "$MCQ_STYLE")
  args+=(--prompt-style "$PROMPT_STYLE")
  if flag_enabled "$OVERWRITE_THUMBNAILS"; then
    args+=(--overwrite-thumbnails)
  fi
  if flag_enabled "$OVERWRITE_PREDICTIONS"; then
    args+=(--overwrite-predictions)
  fi
  if flag_enabled "$SKIP_INFERENCE"; then
    args+=(--skip-inference)
  fi
  if flag_enabled "$PRINT_OUT"; then
    args+=(--print-out)
  fi

  "$PYTHON_BIN" "${args[@]}"
}

run_general_eval() {
  if ! flag_enabled "$RUN_GENERAL_EVAL"; then
    log "Skipping general evaluator because RUN_GENERAL_EVAL=$RUN_GENERAL_EVAL"
    return 0
  fi

  log "Running eval/metrics.py"
  mkdir -p "$GENERAL_EVAL_DIR"
  "$PYTHON_BIN" "$GENERAL_EVALUATOR" \
    --input "$GENERAL_OUTPUT_JSONL" \
    --questions "$GENERAL_OUTPUT_JSONL" \
    --output-dir "$GENERAL_EVAL_DIR" \
    --name "$GENERAL_EVAL_NAME" \
    --inference-time-json "$INFERENCE_TIME_JSON" \
    --metric-scope "$GENERAL_EVAL_METRIC_SCOPE" | tee "$GENERAL_EVAL_TXT" || log "General eval failed (non-fatal)"
}

main() {
  setup_logging
  log "HuatuoGPT-Vision-7B WSI-VQA baseline"
  log "Checkpoint: $CHECKPOINT"
  log "Huatuo code dir: $HUATUO_CODE_DIR"
  log "Python: $PYTHON_BIN"
  log "Input JSON: $INPUT_JSON"
  log "Slide dir: ${SLIDE_DIR:-<none>}"
  log "Image root: ${IMAGE_ROOT:-<none>}"
  log "Image path: ${IMAGE_PATH:-<none>}"
  log "Thumbnail dir: $THUMBNAIL_DIR"
  log "Answers file: $ANSWERS_FILE"
  log "Seed: $SEED"
  log "Work dir: $WORK_DIR"
  log "Log: $RUN_LOG"

  check_inputs
  run_huatuogpt
  run_general_eval

  log "Done."
  log "Answers: $ANSWERS_FILE"
  log "Summary JSON: $EVAL_JSON"
  log "Details CSV: $EVAL_CSV"
  log "General eval JSONL: $GENERAL_OUTPUT_JSONL"
  log "General eval dir: $GENERAL_EVAL_DIR"
}

main "$@"
