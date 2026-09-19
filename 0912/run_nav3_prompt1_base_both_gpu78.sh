#!/usr/bin/env bash
set -euo pipefail

repo=/home/users/whddn12316/wsi_latent_0915_decode_hj
runner="$repo/0912/run_dataset_nav2_prompt1.sh"
run_root="$repo/sender_relay_exp/runs"
stamp=20260914

datasets=(tcga_expert_vqa gtex panda tcga_slidebench tcga)
sizes=(128 190 196 197 221)

completed_ids() {
  local prefix="$1"
  find "$run_root" -path "*/${prefix}*/*" \
    \( -name result.json -o -name failure.json \) -type f -print0 2>/dev/null |
    xargs -0 -r -n1 dirname |
    sed -n 's#.*/\([0-9][0-9][0-9]\)_.*#\1#p' |
    sed 's/^0*//; s/^$/0/' | sort -nu
}

run_condition() {
  local dataset="$1" size="$2" label="$3" variant="$4"
  local prefix="nav3_prompt1_${label}_${dataset}_"
  local done_file remaining_file
  done_file=$(mktemp)
  remaining_file=$(mktemp)
  completed_ids "$prefix" >"$done_file"
  awk 'FILENAME == ARGV[1] { done[$1] = 1; next } !($1 in done)' \
    "$done_file" <(seq 0 $((size - 1))) >"$remaining_file"

  local remaining
  remaining=$(wc -l <"$remaining_file")
  if (( remaining == 0 )); then
    echo "DONE $dataset $label ($size/$size already complete)"
    rm -f "$done_file" "$remaining_file"
    return
  fi

  local ids7 ids8
  ids7=$(awk 'NR % 2 == 1' "$remaining_file" | paste -sd, -)
  ids8=$(awk 'NR % 2 == 0' "$remaining_file" | paste -sd, -)
  echo "START $dataset $label remaining=$remaining gpu7=$(awk 'NR % 2 == 1' "$remaining_file" | wc -l) gpu8=$(awk 'NR % 2 == 0' "$remaining_file" | wc -l)"

  local pids=()
  if [[ -n "$ids7" ]]; then
    VLMAS_NAV3_INDEPENDENT=1 VLMAS_NAV2=0 \
      bash "$runner" 7 "$dataset" "$variant" "$ids7" \
      "$run_root/${prefix}gpu7_${stamp}" \
      >"$run_root/${prefix}gpu7_${stamp}.log" 2>&1 &
    pids+=("$!")
  fi
  if [[ -n "$ids8" ]]; then
    VLMAS_NAV3_INDEPENDENT=1 VLMAS_NAV2=0 \
      bash "$runner" 8 "$dataset" "$variant" "$ids8" \
      "$run_root/${prefix}gpu8_${stamp}" \
      >"$run_root/${prefix}gpu8_${stamp}.log" 2>&1 &
    pids+=("$!")
  fi

  local failed=0 pid
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  rm -f "$done_file" "$remaining_file"
  if (( failed )); then
    echo "FAILED $dataset $label"
    return 1
  fi
  echo "DONE $dataset $label"
}

for i in "${!datasets[@]}"; do
  dataset="${datasets[$i]}"
  size="${sizes[$i]}"
  run_condition "$dataset" "$size" base base
  run_condition "$dataset" "$size" both pruning_b_latent_kv_relay2_dual
done
