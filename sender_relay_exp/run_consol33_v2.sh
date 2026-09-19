#!/bin/bash
# §3.3 SPEC-COMPLETE arm: task-weighted s_c (hook q2v) + unshifted kappa.
# Fix targets from the consol33-uniform failure (BACC .1585 < v3 .1914 on gtex):
#   VLMAS_WSI_CONSOL_SC=1     s_c = question->vision attention (spec §3.3.3)
#   VLMAS_WSI_CONSOL_KAPPA=cos  orthogonal tokens cover nothing (junk cannot
#                               absorb budget; spec's (1+cos)/2 gave floor 0.5)
# Usage: run_consol33_v2.sh <GPU>
set -u
GPU=${1:?usage: run_consol33_v2.sh GPU}
TREE=/home/users/whddn12316/wsi_latent_0915_decode_hj
if [ "$GPU" = "8" ]; then
  exec 9>>$TREE/sender_relay_exp/gpu8.lock
  flock -n 9 || { echo "GPU8 LOCK HELD — refusing to start"; exit 1; }
fi
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
OUT=$TREE/sender_relay_exp/runs/consol33_v2
MP=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
mkdir -p $OUT
for ds in gtex tcga_expert_vqa; do
  dest=$OUT/$ds/4b/consol33v2_step5
  if [ -d "$dest" ]; then echo "SKIP $ds (exists)"; continue; fi
  t0=$SECONDS
  VLMAS_WSI_CONSOL=1 VLMAS_WSI_CONSOL_KEEP=0.25 \
  VLMAS_WSI_CONSOL_SC=1 VLMAS_WSI_CONSOL_KAPPA=cos \
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
  echo "CONSOL33V2-JOB $ds rc=$? elapsed=$((SECONDS-t0))s"
done
echo "CONSOL33 V2 DONE gpu=$GPU"
