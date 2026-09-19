#!/bin/bash
# GPU8 follow-ups in order: ② restage replication at 8B/step20 (gtex+evqa),
# ③ evqa wrong-slide control (4B/step5). Solo-occupancy assumed (8B + restage
# copies need headroom; the gpu8 flock guards our own scripts).
set -u
TREE=/home/users/whddn12316/wsi_latent_0915_decode_hj
exec 9>>$TREE/sender_relay_exp/gpu8.lock
flock -n 9 || { echo "GPU8 LOCK HELD — refusing to start"; exit 1; }
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
OUT=$TREE/sender_relay_exp/runs/restage_followups
MP=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
COMMON="--variant base --backbone qwen3-vl --slide-root $MP/slides --device cuda:0 \
 --patch-budget 8 --max-model-len 8192 --navigator-control-tokens 512 \
 --temperature 0.0 --top-p 1.0 --seed 42 --answerer-max-new-tokens 512 \
 --terminal-no-repeat-ngram-size 0 --terminal-repetition-penalty 1.0 \
 --terminal-antiloop-method none --answerer-protocol structured_json \
 --transport-mode cumulative --realign-method wa --case-retries 0 \
 --deterministic --answerer-greedy --no-answerer-thinking --canonical-open-options \
 --navigator-kv --io-pipeline --answerer-rationale --no-save-navigation-pngs"
mkdir -p $OUT

run_one () {
  local name=$1 ds=$2 model=$3 steps=$4; shift 4
  local dest=$OUT/$name
  if [ -d "$dest" ]; then echo "SKIP $name (exists)"; return; fi
  local t0=$SECONDS
  env "$@" VLMAS_KV_RESTAGE=1 \
  PYTHONPATH=$TREE PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=8 \
    $PY -m wsi_latentmas.pipeline.latent_mas $COMMON \
    --dataset $MP/$ds.json --latent-steps $steps \
    --model /home/users/whddn12316/models/Qwen3-VL-${model}-Thinking \
    --output-root $dest > $OUT/$name.log 2>&1
  echo "FOLLOWUP $name rc=$? elapsed=$((SECONDS-t0))s"
}

# ② replication at the cell where C2 also replicated strongly
run_one restage_8b20_gtex gtex 8B 20 VLMAS_DUMMY=1
run_one restage_8b20_evqa tcga_expert_vqa 8B 20 VLMAS_DUMMY=1
# ③ evqa wrong-slide control (4B step5; donor = evqa idx 0, matched — exclude)
run_one wrong_evqa_4b5 tcga_expert_vqa 4B 5 VLMAS_KV_RESTAGE_MODE=wrong_slide
echo "RESTAGE FOLLOWUPS DONE"
