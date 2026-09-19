#!/bin/bash
# §3.3 selection arm (Method-faithful, TopB 대체) on MultiPathQA — the missing
# head-to-head vs pruning_v3 / D_cross at the SAME budget (keep 0.25).
# s_c caveat: base variant has no q2v capture (prefill full-attention would not
# fit), so this run is the s_c=uniform (pure-coverage) version of §3.3;
# task-weighted s_c needs a memory-safe selective capture (follow-up).
# Usage: run_consol33.sh <GPU>
set -u
GPU=${1:?usage: run_consol33.sh GPU}
TREE=/home/users/whddn12316/wsi_latent_0915_decode_hj
if [ "$GPU" = "8" ]; then
  exec 9>>$TREE/sender_relay_exp/gpu8.lock
  flock -n 9 || { echo "GPU8 LOCK HELD — refusing to start"; exit 1; }
fi
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
OUT=$TREE/sender_relay_exp/runs/consol33
MP=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda

run_job () {
  local ds=$1
  local dest=$OUT/$ds/4b/consol33_step5
  if [ -d "$dest" ]; then echo "SKIP $ds (exists)"; return; fi
  local t0=$SECONDS
  VLMAS_WSI_CONSOL=1 VLMAS_WSI_CONSOL_KEEP=0.25 \
  PYTHONPATH=$TREE PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$GPU \
    $PY -m wsi_latentmas.pipeline.latent_mas \
    --variant base --backbone qwen3-vl \
    --dataset $MP/$ds.json --slide-root $MP/slides \
    --model /home/users/whddn12316/models/Qwen3-VL-4B-Thinking --device cuda:0 \
    --latent-steps 5 --patch-budget 8 --max-model-len 8192 \
    --navigator-control-tokens 512 --temperature 0.0 --top-p 1.0 --seed 42 \
    --answerer-max-new-tokens 512 --terminal-no-repeat-ngram-size 0 \
    --terminal-repetition-penalty 1.0 --terminal-antiloop-method none \
    --answerer-protocol structured_json --transport-mode cumulative \
    --realign-method wa --case-retries 0 --deterministic --answerer-greedy \
    --no-answerer-thinking --canonical-open-options --navigator-kv \
    --io-pipeline --answerer-rationale --no-save-navigation-pngs \
    --output-root $dest > $OUT/$ds.4b.step5.log 2>&1
  echo "CONSOL33-JOB $ds/4b/step5 gpu=$GPU rc=$? elapsed=$((SECONDS-t0))s"
}

mkdir -p $OUT
run_job gtex
run_job tcga_expert_vqa
echo "CONSOL33 DONE gpu=$GPU"
