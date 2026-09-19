#!/bin/bash
# Recover from the double-worker OOM on GPU 8:
# the remote pkill is non-standard (prints an EXAMPLE usage instead of
# killing), so the smoke script's rejoin loop survived preemption and ran a
# matrix 8B job CONCURRENTLY with the KV probe -> probe cases 5..29 OOMed.
# This script kills EVERY worker loop/python on GPU 8 by PID (never pkill),
# requeues their jobs, reruns the probe alone, then starts exactly ONE
# rejoin worker.
set -u
TREE=/home/users/whddn12316/wsi_latent_0915_decode_hj
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
M=$TREE/sender_relay_exp/runs/allsteps_matrix_20260903
OUT=$TREE/sender_relay_exp/runs/kv_swap_probe2

for loop_pid in $(pgrep -f "run_worker8_rejoin.sh"; pgrep -f "run_visual_relay_smoke.sh"); do
  echo "killing worker loop $loop_pid"
  kill -9 $loop_pid 2>/dev/null
done
for p in $(pgrep -f "wsi_latentmas.pipeline.latent_mas"); do
  if tr "\0" "\n" < /proc/$p/environ 2>/dev/null | grep -q "^CUDA_VISIBLE_DEVICES=8$"; then
    dest=$(tr "\0" "\n" < /proc/$p/cmdline | grep -A1 -- "--output-root" | tail -1)
    job=$(echo "$dest" | sed -E "s|.*/allsteps_matrix_20260903/([^/]+)/([^/]+)/([a-z0-9_]+)_step([0-9]+)$|\\2 \\4 \\3 \\1|")
    echo "killing gpu8 python $p ($dest)"
    kill $p 2>/dev/null; sleep 3; kill -9 $p 2>/dev/null
    if [ -n "$dest" ] && [[ "$dest" == *allsteps_matrix* ]]; then
      rm -rf "$dest"
      flock $M/jobs.lock bash -c "echo \"$job\" >> $M/jobs.txt"
      echo "requeued: $job"
    fi
  fi
done
sleep 10

mkdir -p $OUT
IDX=""; for i in $(seq 0 29); do IDX="$IDX --dataset-index $i"; done
VLMAS_KV_SWAP_PROBE=1 VLMAS_KV_SWAP_PROBE_OUT=$OUT/kv_swap_probe.jsonl \
VLMAS_KV_SWAP_PROBE_BANDS=10,18,26,33 VLMAS_KV_SWAP_PROBE_SCORE=1 \
PYTHONPATH=$TREE PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=8 \
  $PY -m wsi_latentmas.pipeline.latent_mas \
  --variant base --backbone qwen3-vl \
  --dataset /home/users/whddn12316/datasets/WSI-VQA/WsiVQA_test.json \
  --slide-root /home/users/whddn12316/datasets/WSI-VQA/DATA_SVS \
  --model /home/users/whddn12316/models/Qwen3-VL-4B-Thinking --device cuda:0 \
  --latent-steps 5 --patch-budget 8 --max-model-len 8192 --navigator-control-tokens 512 \
  --temperature 0.0 --top-p 1.0 --seed 42 --answerer-max-new-tokens 512 \
  --terminal-no-repeat-ngram-size 0 --terminal-repetition-penalty 1.0 \
  --terminal-antiloop-method none --answerer-protocol structured_json \
  --transport-mode cumulative --realign-method wa --case-retries 0 \
  --deterministic --answerer-greedy --no-answerer-thinking --canonical-open-options \
  --navigator-kv --io-pipeline --answerer-rationale --no-save-navigation-pngs \
  $IDX --output-root $OUT/base_probe > $OUT/base_probe.log 2>&1
echo "KV PROBE2 rc=$?"
nohup bash $TREE/sender_relay_exp/run_worker8_rejoin.sh >> $TREE/sender_relay_exp/allsteps_driver.log 2>&1 &
echo "ORCHESTRATION2 DONE (worker8 rejoined)"
