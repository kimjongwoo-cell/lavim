#!/usr/bin/env bash
set -euo pipefail

if (( $# < 1 )); then
  echo "usage: $0 NEW_EXPERIMENT_ROOT [DATASET_INDEX ...]" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export FAIR_CONTRACT_FILE="${FAIR_CONTRACT_FILE:-$ROOT/config/wsivqa_fair_v2.json}"
source "$ROOT/scripts/wsivqa_fair_contract.sh"

experiment_root="$1"
shift
if [[ -e "$experiment_root" ]]; then
  echo "refusing to overwrite existing output: $experiment_root" >&2
  exit 2
fi

indices=("$@")
if (( ${#indices[@]} == 0 )); then
  dataset_cases="$(jq 'length' "$FAIR_DATASET")"
  if [[ ! "$dataset_cases" =~ ^[1-9][0-9]*$ ]]; then
    echo "dataset must be a non-empty JSON array: $FAIR_DATASET" >&2
    exit 2
  fi
  mapfile -t indices < <(seq 0 "$((dataset_cases - 1))")
fi
SINGLE_HF_BACKEND="${SINGLE_HF_BACKEND:-hf-service}"
case "$SINGLE_HF_BACKEND" in
  hf-service)
    worker_count="${HF_CASE_WORKERS:-6}"
    backend="transformers-batched"
    HF_SERVICE_URL="${HF_SERVICE_URL:-http://127.0.0.1:8123/v1}"
    HF_BATCH_SIZE="${HF_BATCH_SIZE:-4}"
    max_batch_size="$HF_BATCH_SIZE"
    worker_env=("SINGLE_BACKEND=$SINGLE_HF_BACKEND" "HF_SERVICE_URL=$HF_SERVICE_URL")
    ;;
  transformers)
    worker_count="${HF_CASE_WORKERS:-2}"
    backend="transformers-parallel-b1"
    HF_SERVICE_URL=""
    max_batch_size=1
    worker_env=("SINGLE_BACKEND=$SINGLE_HF_BACKEND")
    ;;
  *)
    echo "SINGLE_HF_BACKEND must be hf-service or transformers: $SINGLE_HF_BACKEND" >&2
    exit 2
    ;;
esac
if [[ ! "$worker_count" =~ ^[1-9][0-9]*$ ]]; then
  echo "HF_CASE_WORKERS must be a positive integer: $worker_count" >&2
  exit 2
fi
if (( worker_count > ${#indices[@]} )); then
  worker_count="${#indices[@]}"
fi
mkdir -p "$experiment_root/shards"
thumbnail_seed_dir="${SINGLE_THUMBNAIL_SEED_DIR:-}"
if [[ -n "$thumbnail_seed_dir" && ! -d "$thumbnail_seed_dir" ]]; then
  echo "SINGLE_THUMBNAIL_SEED_DIR is not a directory: $thumbnail_seed_dir" >&2
  exit 2
fi
requested_indices="$(printf '%s\n' "${indices[@]}" | jq -Rsc 'split("\n") | map(select(length > 0) | tonumber)')"
single_max_new_tokens="${SINGLE_MAX_NEW_TOKENS:-$FAIR_MAX_NEW_TOKENS}"
single_max_model_len="${SINGLE_MAX_MODEL_LEN:-$FAIR_MAX_MODEL_LEN}"
single_temperature="${SINGLE_TEMPERATURE:-$FAIR_TEMPERATURE}"
single_top_p="${SINGLE_TOP_P:-$FAIR_TOP_P}"
single_seed="${SINGLE_SEED:-$FAIR_SEED}"
single_native_thinking="${SINGLE_NATIVE_THINKING:-$FAIR_NATIVE_THINKING}"
single_output_protocol="${SINGLE_OUTPUT_PROTOCOL:-labeled}"
single_direct_final_only="${SINGLE_DIRECT_FINAL_ONLY:-0}"

jq -n \
  --arg contract_id "$FAIR_CONTRACT_ID" \
  --arg contract_file "$FAIR_CONTRACT_FILE" \
  --arg backend "$backend" \
  --arg service_url "$HF_SERVICE_URL" \
  --arg dataset "$FAIR_DATASET" \
  --arg slide_root "$FAIR_SLIDE_ROOT" \
  --arg model "$FAIR_CHECKPOINT" \
  --arg evaluator "$FAIR_EVALUATOR" \
  --arg started_at "$(date --iso-8601=seconds)" \
  --arg git_commit "$(git -C "$ROOT" rev-parse HEAD)" \
  --argjson worker_count "$worker_count" \
  --argjson max_batch_size "$max_batch_size" \
  --argjson max_model_len "$single_max_model_len" \
  --argjson temperature "$single_temperature" \
  --argjson top_p "$single_top_p" \
  --argjson seed "$single_seed" \
  --argjson native_thinking "$single_native_thinking" \
  --argjson max_new_tokens "$single_max_new_tokens" \
  --arg output_protocol "$single_output_protocol" \
  --arg direct_final_only "$single_direct_final_only" \
  --argjson requested_indices "$requested_indices" \
  '{contract_id:$contract_id,contract_file:$contract_file,backend:$backend,service_url:$service_url,dataset:$dataset,slide_root:$slide_root,model:$model,evaluator:$evaluator,started_at:$started_at,git_commit:$git_commit,worker_count:$worker_count,max_batch_size:$max_batch_size,max_model_len:$max_model_len,temperature:$temperature,top_p:$top_p,seed:$seed,native_thinking:$native_thinking,max_new_tokens:$max_new_tokens,output_protocol:$output_protocol,direct_final_only:($direct_final_only == "1" or $direct_final_only == "true"),requested_indices:$requested_indices}' \
  >"$experiment_root/run_manifest.json"

pids=()
for ((worker_id=0; worker_id<worker_count; worker_id++)); do
  shard="$experiment_root/shards/worker_${worker_id}.json"
  jq \
    --argjson requested "$requested_indices" \
    --argjson workers "$worker_count" \
    --argjson worker "$worker_id" \
    '. as $dataset | [range(0; $requested | length) | select(. % $workers == $worker) | $requested[.] as $index | $dataset[$index]]' \
    "$FAIR_DATASET" >"$shard"
  if [[ -n "$thumbnail_seed_dir" ]]; then
    mkdir -p "$experiment_root/worker_$worker_id/thumbnails"
    cp -a --reflink=auto "$thumbnail_seed_dir"/. "$experiment_root/worker_$worker_id/thumbnails/"
  fi
  env \
    "${worker_env[@]}" \
    DATASET=wsivqa \
    INPUT_JSON="$shard" \
    TEST_JSON="$shard" \
    WORK_DIR="$experiment_root/worker_$worker_id" \
    THUMBNAIL_DIR="$experiment_root/worker_$worker_id/thumbnails" \
    RUN_ID="worker_$worker_id" \
    RUN_GENERAL_EVAL=false \
    MAX_MODEL_LEN="$single_max_model_len" \
    SINGLE_TEMPERATURE="$single_temperature" \
    SINGLE_TOP_P="$single_top_p" \
    SINGLE_SEED="$single_seed" \
    MAX_NEW_TOKENS="$single_max_new_tokens" \
    SINGLE_FINALIZATION_MAX_NEW_TOKENS="$single_max_new_tokens" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    bash "$ROOT/models/single_v7/thinking/run_thinking.sh" \
    >"$experiment_root/worker_${worker_id}.log" 2>&1 &
  pids+=("$!")
done
printf '%s\n' "${pids[@]}" >"$experiment_root/pids.txt"

failed_workers=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed_workers=$((failed_workers + 1))
  fi
done
if (( failed_workers > 0 )); then
  echo "$failed_workers Single worker(s) failed; refusing evaluation" >&2
  exit 1
fi

prediction_buffer="$experiment_root/predictions.unsorted.jsonl"
general_buffer="$experiment_root/general_eval.unsorted.jsonl"
thinking_buffer="$experiment_root/thinking.unsorted.jsonl"
: >"$prediction_buffer"
: >"$general_buffer"
: >"$thinking_buffer"
for ((worker_id=0; worker_id<worker_count; worker_id++)); do
  worker_root="$experiment_root/worker_$worker_id"
  for required in "$worker_root/predictions/qwen3vl_predictions.jsonl" "$worker_root/eval/qwen3vl_general_eval.jsonl" "$worker_root/predictions/thinking.jsonl"; do
    if [[ ! -f "$required" ]]; then
      echo "missing required worker artifact: $required" >&2
      exit 1
    fi
  done
  jq -c \
    --argjson requested "$requested_indices" \
    --argjson worker "$worker_id" \
    --argjson workers "$worker_count" \
    '((.question_id | tonumber) * $workers + $worker) as $position | $requested[$position] as $index | .question_id = ($index | tostring) | .sample_index = $index' \
    "$worker_root/predictions/qwen3vl_predictions.jsonl" >>"$prediction_buffer"
  jq -c \
    --argjson requested "$requested_indices" \
    --argjson worker "$worker_id" \
    --argjson workers "$worker_count" \
    '((.question_id | tonumber) * $workers + $worker) as $position | $requested[$position] as $index | .question_id = ($index | tostring)' \
    "$worker_root/eval/qwen3vl_general_eval.jsonl" >>"$general_buffer"
  jq -c \
    --argjson requested "$requested_indices" \
    --argjson worker "$worker_id" \
    --argjson workers "$worker_count" \
    '(.seq * $workers + $worker) as $position | $requested[$position] as $index | .seq = $index' \
    "$worker_root/predictions/thinking.jsonl" >>"$thinking_buffer"
done

mkdir -p "$experiment_root/predictions" "$experiment_root/eval/general"
jq -sc 'sort_by(.question_id | tonumber)[]' "$prediction_buffer" >"$experiment_root/predictions/qwen3vl_predictions.jsonl"
jq -sc 'sort_by(.question_id | tonumber)[]' "$general_buffer" >"$experiment_root/eval/qwen3vl_general_eval.jsonl"
jq -sc 'sort_by(.seq)[]' "$thinking_buffer" >"$experiment_root/predictions/thinking.jsonl"
rm -f "$prediction_buffer" "$general_buffer" "$thinking_buffer"

complete="$(jq -s --argjson requested "$requested_indices" '([.[].question_id | tonumber] | sort) == ($requested | sort)' "$experiment_root/eval/qwen3vl_general_eval.jsonl")"
if [[ "$complete" != true ]]; then
  echo "merged Single results do not exactly match requested indices" >&2
  exit 1
fi
if (( ${#indices[@]} != FAIR_EXPECTED_CASES )); then
  echo "completed Single direct-HF batch smoke: ${#indices[@]} unique rows"
  exit 0
fi

python "$FAIR_EVALUATOR" \
  --input "$experiment_root/eval/qwen3vl_general_eval.jsonl" \
  --questions "$experiment_root/eval/qwen3vl_general_eval.jsonl" \
  --output-dir "$experiment_root/eval/general" \
  --name qwen3_vl_4b_WsiVQA_test \
  --metric-scope "$FAIR_METRIC_SCOPE" \
  >"$experiment_root/eval/general_eval.txt"

echo "completed fair Single v7 direct-HF batched: $FAIR_EXPECTED_CASES unique rows"
