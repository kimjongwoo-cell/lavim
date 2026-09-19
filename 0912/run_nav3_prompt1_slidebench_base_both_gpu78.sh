#!/usr/bin/env bash
set -euo pipefail

repo=/home/users/whddn12316/wsi_latent_0915_decode_hj
runner="$repo/0912/run_dataset_nav2_prompt1.sh"
run_root="$repo/sender_relay_exp/runs"
dataset=tcga_slidebench
size=197
stamp=20260914

run_condition() {
  local label="$1" variant="$2"
  local prefix="nav3_prompt1_${label}_${dataset}_"
  local remaining_file
  remaining_file=$(mktemp)

  python - "$run_root" "$prefix" "$size" >"$remaining_file" <<'PY'
import glob, json, os, sys
root, prefix, size = sys.argv[1], sys.argv[2], int(sys.argv[3])
done = set()
for path in glob.glob(os.path.join(root, prefix + "*", "*", "*.json")):
    if os.path.basename(path) not in {"result.json", "failure.json"}:
        continue
    try:
        with open(path) as f:
            done.add(int(json.load(f)["dataset_index"]))
    except Exception:
        parent = os.path.basename(os.path.dirname(path))
        try:
            done.add(int(parent.split("_", 1)[0]))
        except Exception:
            pass
for index in range(size):
    if index not in done:
        print(index)
PY

  local remaining ids7 ids8
  remaining=$(wc -l <"$remaining_file")
  if (( remaining == 0 )); then
    echo "DONE $dataset $label ($size/$size already complete)"
    rm -f "$remaining_file"
    return
  fi
  ids7=$(awk 'NR % 2 == 1' "$remaining_file" | paste -sd, -)
  ids8=$(awk 'NR % 2 == 0' "$remaining_file" | paste -sd, -)
  echo "START $dataset $label remaining=$remaining"

  local pids=()
  if [[ -n "$ids7" ]]; then
    VLMAS_NAV3_INDEPENDENT=1 VLMAS_NAV2=0 bash "$runner" 7 "$dataset" "$variant" "$ids7" \
      "$run_root/${prefix}gpu7_${stamp}" >"$run_root/${prefix}gpu7_${stamp}.log" 2>&1 &
    pids+=("$!")
  fi
  if [[ -n "$ids8" ]]; then
    VLMAS_NAV3_INDEPENDENT=1 VLMAS_NAV2=0 bash "$runner" 8 "$dataset" "$variant" "$ids8" \
      "$run_root/${prefix}gpu8_${stamp}" >"$run_root/${prefix}gpu8_${stamp}.log" 2>&1 &
    pids+=("$!")
  fi
  local pid failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  rm -f "$remaining_file"
  (( failed == 0 )) || return 1
  echo "DONE $dataset $label"
}

run_condition base base
run_condition both pruning_b_latent_kv_relay2_dual
