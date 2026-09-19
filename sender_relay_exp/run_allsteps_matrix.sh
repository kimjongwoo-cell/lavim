#!/bin/bash
# Pruning-V3 + Reallocation-V2 allsteps matrix (dashboard-matching setting):
# {2B,4B,8B} x steps {5,10,20,30} x {tcga_expert_vqa, gtex}, canonical config.
# flock job queue over GPUs 6/7/8. Results:
#   sender_relay_exp/runs/allsteps_matrix_20260903/<dataset>/<model>/<arm>_step<N>/
set -u
TREE=/home/users/whddn12316/wsi_latent_0915_decode_hj
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
MP=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
OUT=$TREE/sender_relay_exp/runs/allsteps_matrix_20260903
QUEUE=$OUT/jobs.txt
LOCK=$OUT/jobs.lock
mkdir -p $OUT

if [ ! -s $QUEUE ]; then
  : > $QUEUE
  # 8B first (longest) so the tail of the schedule stays balanced
  for model in 8b 4b 2b; do
    for step in 30 20 10 5; do
      for arm in pruning_v3 reallocation_v2; do
        for ds in gtex tcga_expert_vqa; do
          # already done this session: 4b step5 pruning_v3 / reallocation arms
          [ "$model/$step/$arm" = "4b/5/pruning_v3" ] && continue
          echo "$model $step $arm $ds" >> $QUEUE
        done
      done
    done
  done
fi

model_path () {
  case $1 in
    2b) echo /home/users/whddn12316/models/Qwen3-VL-2B-Thinking;;
    4b) echo /home/users/whddn12316/models/Qwen3-VL-4B-Thinking;;
    8b) echo /home/users/whddn12316/models/Qwen3-VL-8B-Thinking;;
  esac
}

worker () {
  local gpu=$1
  while true; do
    local job
    job=$(flock $LOCK bash -c "head -1 $QUEUE; sed -i '1d' $QUEUE")
    [ -z "$job" ] && break
    set -- $job; local model=$1 step=$2 arm=$3 ds=$4
    local dest=$OUT/$ds/$model/${arm}_step${step}
    if [ -d "$dest" ]; then echo "SKIP $job (exists)"; continue; fi
    local t0=$SECONDS
    PYTHONPATH=$TREE PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$gpu \
      $PY -m wsi_latentmas.pipeline.latent_mas \
      --variant $arm --backbone qwen3-vl \
      --dataset $MP/$ds.json --slide-root $MP/slides \
      --model $(model_path $model) --device cuda:0 \
      --latent-steps $step --patch-budget 8 --max-model-len 8192 \
      --navigator-control-tokens 512 --temperature 0.0 --top-p 1.0 --seed 42 \
      --answerer-max-new-tokens 512 --terminal-no-repeat-ngram-size 0 \
      --terminal-repetition-penalty 1.0 --terminal-antiloop-method none \
      --answerer-protocol structured_json --transport-mode cumulative \
      --realign-method wa --case-retries 0 --deterministic --answerer-greedy \
      --no-answerer-thinking --canonical-open-options --navigator-kv \
      --io-pipeline --answerer-rationale --no-save-navigation-pngs \
      --output-root $dest > $OUT/$ds.$model.${arm}_step${step}.log 2>&1
    echo "JOB $ds/$model/${arm}_step${step} gpu=$gpu rc=$? elapsed=$((SECONDS-t0))s"
  done
  echo "WORKER gpu=$gpu drained"
}

worker 6 & worker 7 & worker 8 &
wait
echo "MATRIX ALL DONE"
