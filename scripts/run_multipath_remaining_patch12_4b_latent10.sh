#!/usr/bin/env bash
# Queue the remaining MultiPathQA benchmarks after the current ExpertVQA
# Reallocation A run.  Each dataset uses the same 4B / patch-12 / latent-10
# protocol.  GPU 5 executes Base, Pruning B, Reallocation A, and their Both
# combination in sequence so no shared GPU is touched.
set -euo pipefail

project_root=/home/users/whddn12316/wsi_latent_0915_decode_hj
venv_python=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
data_root=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
slide_root="$data_root/slides"
model=/home/users/whddn12316/models/Qwen3-VL-4B-Thinking
queue_root="$project_root/runs/multipath_remaining_patch12_4b_latent10_20260908"
expert_root="$project_root/runs/expertvqa_reallocation_a_direct_relay_x5_4_x20_8_latent10_gpu8_mlen12288_full"
datasets=(gtex panda tcga tcga_slidebench)
minimum_free_mib=36000

common_args=(
  --backbone qwen3-vl
  --model "$model"
  --slide-root "$slide_root"
  --device cuda:0
  --latent-steps 10
  --patch-budget 12
  --max-model-len 12288
  --navigator-control-tokens 512
  --temperature 0.0
  --top-p 1.0
  --seed 42
  --deterministic
  --answerer-greedy
  --no-answerer-thinking
  --answerer-max-new-tokens 512
  --answerer-rationale
  --answerer-protocol structured_json
  --canonical-open-options
  --navigator-kv
  --io-pipeline
  --no-save-navigation-pngs
  --case-retries 0
  --resume
)

wait_for_free_gpu() {
  local gpu=$1
  local free_mib
  while true; do
    free_mib=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -dc '0-9')
    if (( free_mib >= minimum_free_mib )); then
      return
    fi
    echo "[$(date '+%F %T')] GPU $gpu busy (${free_mib} MiB free); waiting" >&2
    sleep 60
  done
}

gpu_workload_pids() {
  nvidia-smi -i 5 --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
    | tr -d ' ' \
    | rg -v '^6273$' || true
}

wait_for_expertvqa_reallocation() {
  local expected completed failed
  expected=$(jq 'length' "$data_root/tcga_expert_vqa.json")
  while true; do
    completed=$(find "$expert_root" -name result.json -type f 2>/dev/null | wc -l)
    failed=$(find "$expert_root" -name failed.json -type f 2>/dev/null | wc -l)
    if (( completed + failed >= expected )); then
      return
    fi
    echo "[$(date '+%F %T')] ExpertVQA Reallocation A ${completed}/${expected}; waiting" >&2
    sleep 60
  done
}

run_variant() {
  local gpu=$1
  local dataset=$2
  local variant=$3
  local output_root=$4
  local base_manifest=${5:-}
  local log="$queue_root/logs/${dataset}_${variant}.log"
  local manifest_args=()

  if [[ -n "$base_manifest" ]]; then
    manifest_args=(--expected-base-manifest "$base_manifest")
  fi

  mkdir -p "$(dirname "$log")"
  echo "[$(date '+%F %T')] START dataset=$dataset variant=$variant gpu=$gpu" | tee -a "$log"
  CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 "$venv_python" \
    -m wsi_latentmas.pipeline.latent_mas \
    --variant "$variant" \
    --dataset "$data_root/$dataset.json" \
    --output-root "$output_root" \
    "${common_args[@]}" \
    "${manifest_args[@]}" \
    >>"$log" 2>&1 &
  local worker_pid=$!
  local own_gpu_pid=""
  local foreign_pid=""
  local -a gpu_pids=()

  while kill -0 "$worker_pid" 2>/dev/null; do
    mapfile -t gpu_pids < <(gpu_workload_pids)
    if [[ -z "$own_gpu_pid" ]]; then
      if (( ${#gpu_pids[@]} == 1 )); then
        own_gpu_pid=${gpu_pids[0]}
      elif (( ${#gpu_pids[@]} > 1 )); then
        foreign_pid=${gpu_pids[1]}
      fi
    else
      for gpu_pid in "${gpu_pids[@]}"; do
        if [[ "$gpu_pid" != "$own_gpu_pid" ]]; then
          foreign_pid=$gpu_pid
          break
        fi
      done
    fi
    if [[ -n "$foreign_pid" ]]; then
      echo "[$(date '+%F %T')] GPU 5 external PID $foreign_pid detected; stopping $dataset/$variant" | tee -a "$log"
      kill -TERM "$worker_pid"
      wait "$worker_pid" || true
      return
    fi
    sleep 5
  done

  if wait "$worker_pid"; then
    echo "[$(date '+%F %T')] DONE dataset=$dataset variant=$variant gpu=$gpu" | tee -a "$log"
  else
    echo "[$(date '+%F %T')] INCOMPLETE dataset=$dataset variant=$variant gpu=$gpu; continuing queue" | tee -a "$log"
  fi
}

cd "$project_root"
mkdir -p "$queue_root/logs"

wait_for_expertvqa_reallocation

for dataset in "${datasets[@]}"; do
  base_root="$queue_root/$dataset/latent_base"
  pruning_root="$queue_root/$dataset/pruning_b"
  reallocation_root="$queue_root/$dataset/reallocation_a"
  both_root="$queue_root/$dataset/both"
  base_manifest="$base_root/run_manifest.json"

  wait_for_free_gpu 5
  run_variant 5 "$dataset" base "$base_root"

  wait_for_free_gpu 5
  run_variant 5 "$dataset" pruning_b "$pruning_root" "$base_manifest"

  wait_for_free_gpu 5
  run_variant 5 "$dataset" reallocation_a "$reallocation_root" "$base_manifest"

  wait_for_free_gpu 5
  run_variant 5 "$dataset" pruning_b_reallocation_a "$both_root" "$base_manifest"
done

echo "[$(date '+%F %T')] remaining MultiPathQA queue completed" | tee -a "$queue_root/queue.log"
