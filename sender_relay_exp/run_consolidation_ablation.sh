#!/bin/bash
# Pathology-structured consolidation ablation (Components 1+2), GPU 8.
# Arms (A=base, B=pruning_v3 REUSED from runs/multipath_qwen4b_step5_20260903):
#   C = pruning + Morphology-Coverage Rescue      (VLMAS_MORPH_RESCUE=1)
#   D = pruning + Cross-Scale Consolidation       (VLMAS_CROSS_SCALE=1)
#   E = pruning + both                            (rescue then cross)
# Datasets: gtex (190, prototype/typical) + tcga_expert_vqa (128, detail).
# KV budget is asserted unchanged inside the backbone ([Consolidate] audit).
set -u
TREE_LOCK=/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/gpu8.lock
exec 9>>$TREE_LOCK
flock -n 9 || { echo "GPU8 LOCK HELD by another process — refusing to start"; exit 1; }
TREE=/home/users/whddn12316/wsi_latent_0915_decode_hj
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
OUT=$TREE/sender_relay_exp/runs/consolidation_ablation
MP=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
COMMON="--variant pruning_v3 --backbone qwen3-vl \
 --model /home/users/whddn12316/models/Qwen3-VL-4B-Thinking --device cuda:0 \
 --latent-steps 5 --patch-budget 8 --max-model-len 8192 --navigator-control-tokens 512 \
 --temperature 0.0 --top-p 1.0 --seed 42 --answerer-max-new-tokens 512 \
 --terminal-no-repeat-ngram-size 0 --terminal-repetition-penalty 1.0 \
 --terminal-antiloop-method none --answerer-protocol structured_json \
 --transport-mode cumulative --realign-method wa --case-retries 0 \
 --deterministic --answerer-greedy --no-answerer-thinking --canonical-open-options \
 --navigator-kv --io-pipeline --answerer-rationale --no-save-navigation-pngs"

run_arm () {
  local ds=$1 name=$2 rescue=$3 cross=$4
  local dest=$OUT/$ds/$name
  if [ -d "$dest" ]; then echo "SKIP $ds/$name (exists)"; return; fi
  local t0=$SECONDS
  VLMAS_MORPH_RESCUE=$rescue VLMAS_CROSS_SCALE=$cross \
  PYTHONPATH=$TREE PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=8 \
    $PY -m wsi_latentmas.pipeline.latent_mas $COMMON \
    --dataset $MP/$ds.json --slide-root $MP/slides \
    --output-root $dest > $OUT/$ds.$name.log 2>&1
  echo "CONSOL-ARM $ds/$name rc=$? elapsed=$((SECONDS-t0))s"
}

mkdir -p $OUT
for ds in gtex tcga_expert_vqa; do
  run_arm $ds C_rescue 1 0
  run_arm $ds D_cross  0 1
  run_arm $ds E_both   1 1
done
echo "CONSOLIDATION ABLATION DONE"
