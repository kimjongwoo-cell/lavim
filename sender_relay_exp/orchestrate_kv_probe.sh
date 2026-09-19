#!/bin/bash
# Wait for the relay smoke to finish, preempt its auto-rejoin worker on GPU 8,
# run the KV-swap carrier probe (Step 1), then start the standalone worker-8
# rejoin so the allsteps matrix returns to 3 GPUs.
set -u
TREE=/home/users/whddn12316/wsi_latent_0915_decode_hj
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
DRV=$TREE/sender_relay_exp/visual_relay_driver.log
MOUT=$TREE/sender_relay_exp/runs/allsteps_matrix_20260903
OUT=$TREE/sender_relay_exp/runs/kv_swap_probe
until grep -q "RELAY SMOKE DONE" $DRV 2>/dev/null; do sleep 60; done
echo "relay smoke done — preempting rejoin worker"
# find the gpu8 python (matrix job the rejoin just started), requeue it
for p in $(pgrep -f "wsi_latentmas.pipeline.latent_mas"); do
  if tr "\0" "\n" < /proc/$p/environ 2>/dev/null | grep -q "^CUDA_VISIBLE_DEVICES=8$"; then
    dest=$(tr "\0" "\n" < /proc/$p/cmdline | grep -A1 -- "--output-root" | tail -1)
    job=$(echo "$dest" | sed -E "s|.*/allsteps_matrix_20260903/([^/]+)/([^/]+)/([a-z0-9_]+)_step([0-9]+)$|\\2 \\4 \\3 \\1|")
    echo "preempt pid=$p dest=$dest job=$job"
    pkill -f run_visual_relay_smoke.sh
    kill $p 2>/dev/null; sleep 3; kill -9 $p 2>/dev/null
    if [ -n "$dest" ] && [[ "$dest" == *allsteps_matrix* ]]; then
      rm -rf "$dest"
      flock $MOUT/jobs.lock bash -c "echo \"$job\" >> $MOUT/jobs.txt"
    fi
  fi
done
sleep 5
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
echo "KV PROBE rc=$?"
nohup bash $TREE/sender_relay_exp/run_worker8_rejoin.sh >> $TREE/sender_relay_exp/allsteps_driver.log 2>&1 &
echo "ORCHESTRATION DONE (worker8 rejoined)"
