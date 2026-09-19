#!/usr/bin/env bash
set -euo pipefail

repo=/home/users/whddn12316/wsi_latent_0915_decode_hj
runner="$repo/0912/run_dataset_nav2_prompt2.sh"
run_root="$repo/0912/runs"

run_phase() {
  local dataset="$1"
  local variant="$2"
  local size="$3"
  local label
  case "$variant" in
    base) label=base ;;
    pruning_b_latent_kv_relay2) label=both ;;
    *) echo "Unsupported variant: $variant" >&2; return 2 ;;
  esac

  mapfile -t done_ids < <(
    find "$run_root" -maxdepth 3 -type f -path "*/nav2_ra_prompt2_${dataset}_${label}_*/*/result.json" -print0 2>/dev/null \
      | xargs -0 -r -n1 dirname \
      | sed -E 's#.*/([0-9]+)_.*#\1#' \
      | sed 's/^0*//' \
      | sed 's/^$/0/' \
      | sort -n -u
  )

  declare -A done=()
  local id
  for id in "${done_ids[@]}"; do done["$id"]=1; done

  local ids6=() ids8=() toggle=0
  for ((id=0; id<size; id++)); do
    [[ -n "${done[$id]:-}" ]] && continue
    if ((toggle == 0)); then ids6+=("$id"); toggle=1; else ids8+=("$id"); toggle=0; fi
  done

  echo "[$dataset/$label] complete=${#done_ids[@]} remaining=$((${#ids6[@]} + ${#ids8[@]})) gpu6=${#ids6[@]} gpu8=${#ids8[@]}"
  ((${#ids6[@]} == 0 && ${#ids8[@]} == 0)) && return 0

  local sel6 sel8 out6 out8 pid6='' pid8=''
  sel6=$(IFS=,; echo "${ids6[*]}")
  sel8=$(IFS=,; echo "${ids8[*]}")
  out6="$run_root/nav2_ra_prompt2_${dataset}_${label}_resume_gpu6_0913"
  out8="$run_root/nav2_ra_prompt2_${dataset}_${label}_resume_gpu8_0913"

  if ((${#ids6[@]})); then
    bash "$runner" 6 "$dataset" "$variant" "$sel6" "$out6" >"$out6.log" 2>&1 & pid6=$!
  fi
  if ((${#ids8[@]})); then
    bash "$runner" 8 "$dataset" "$variant" "$sel8" "$out8" >"$out8.log" 2>&1 & pid8=$!
  fi
  [[ -z "$pid6" ]] || wait "$pid6"
  [[ -z "$pid8" ]] || wait "$pid8"
}

cd "$repo"
run_phase tcga base 221
run_phase tcga pruning_b_latent_kv_relay2 221
run_phase tcga_slidebench base 197
run_phase tcga_slidebench pruning_b_latent_kv_relay2 197
run_phase panda base 196
run_phase panda pruning_b_latent_kv_relay2 196
