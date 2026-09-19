#!/usr/bin/env bash
set -euo pipefail

repo=/home/users/whddn12316/wsi_latent_0915_decode_hj
run_dir="$repo/0912/runs"
runner="$repo/0912/run_gtex_nav2_sparse_resume.sh"
gpus=(0 0 1 1 2 2 3 3 6 6 7 8 8)
declare -A done=()
selections=("" "" "" "" "" "" "" "" "" "" "" "" "")

while IFS= read -r result; do
  index="$(jq -r '.dataset_index' "$result")"
  done["$index"]=1
done < <(find "$run_dir" -path '*/nav2_prompt2_gtex_both*/*/result.json' -type f)

worker=0
for ((index=0; index<190; index++)); do
  [[ -n "${done[$index]:-}" ]] && continue
  if [[ -n "${selections[$worker]}" ]]; then
    selections[$worker]+=",$index"
  else
    selections[$worker]="$index"
  fi
  worker=$(((worker + 1) % ${#gpus[@]}))
done

for worker in "${!gpus[@]}"; do
  [[ -z "${selections[$worker]}" ]] && continue
  gpu="${gpus[$worker]}"
  name="nav2_prompt2_gtex_both13_worker${worker}_gpu${gpu}_0913"
  VLMAS_PROMPT_SET=prompt2 bash "$runner" \
    "$gpu" pruning_b_latent_kv_relay2_dual "${selections[$worker]}" \
    "$run_dir/$name" >"$run_dir/$name.log" 2>&1 &
done

wait
