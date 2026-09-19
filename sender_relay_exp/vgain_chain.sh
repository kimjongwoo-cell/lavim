#!/bin/bash
# [C2 causal test] Answerer Visual-Path Gain Intervention — native addressing fixed.
# Notion Experiment Log 3dad771b-c9e2-81e7-aaee-d38619006f85. Probe: memory/vgain_diag.py.
#
# Pipeline = the 0914 baseline setting (HANDOVER_0914 §2): Nav2 · patch 12 · max-model-len 12288 ·
# latent step 10 · SDPA · Qwen3-VL-4B-Thinking. The gain arms run inside the Answerer on copies
# of the pre-Answerer cache (float32 recompute hook), so the pipeline's own answer is the native arm.
#
# usage: vgain_chain.sh <gpu> <name:dataset.json> ...    (dataset.json relative to sender_relay_exp/smoke)
set -u
GPU=${1:?usage: vgain_chain.sh GPU name:dataset.json ...}; shift
TREE=/home/users/whddn12316/wsi_latent_0915_decode_hj
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
MP=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
K=$TREE/sender_relay_exp/runs/vgain
SM=$TREE/sender_relay_exp/smoke
mkdir -p $K
export VLMAS_ATTN_IMPLEMENTATION=sdpa
export VLMAS_NAV2=1
echo "START $(date +%F_%T) gpu=$GPU jobs=$*" >> $K/chain_gpu$GPU.log
for job in "$@"; do
  NAME=${job%%:*}; DSJ=${job#*:}
  STAMP=$(date +%Y%m%d_%H%M%S)
  OUT=$K/${NAME}_$STAMP
  JL=$K/vgain_${NAME}_$STAMP.jsonl
  "$PY" - "$OUT.meta.json" "$GPU" "$SM/$DSJ" "$JL" <<'PYMETA'
import datetime, json, os, sys
json.dump({"started_at": datetime.datetime.now().isoformat(timespec="seconds"),
           "script": "sender_relay_exp/vgain_chain.sh", "gpu": sys.argv[2], "dataset": sys.argv[3],
           "jsonl": sys.argv[4],
           "env": {k: os.environ.get(k) for k in ("VLMAS_ATTN_IMPLEMENTATION", "VLMAS_NAV2", "VLMAS_PROMPT_SET",
                                                   "VLMAS_VGAIN_GRID", "VLMAS_VGAIN_ARMS", "VLMAS_VGAIN_NO_GEN")}},
          open(sys.argv[1], "w"), indent=1)
PYMETA
  cd $TREE && env VLMAS_VGAIN=$JL PYTHONPATH=$TREE PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout 28800 \
    $PY -m wsi_latentmas.pipeline.latent_mas --variant base --backbone qwen3-vl --dataset $SM/$DSJ \
    --slide-root $MP/slides --model /home/users/whddn12316/models/Qwen3-VL-4B-Thinking --device cuda:0 \
    --latent-steps 10 --patch-budget 12 --max-model-len 12288 --navigator-control-tokens 512 \
    --temperature 0.0 --top-p 1.0 --seed 42 --deterministic --answerer-greedy --no-answerer-thinking \
    --answerer-max-new-tokens 512 --answerer-rationale --answerer-protocol structured_json \
    --canonical-open-options --navigator-kv --io-pipeline --no-save-navigation-pngs --case-retries 0 \
    --output-root $OUT > $OUT.log 2>&1 < /dev/null
  rc=$?
  echo "DONE $NAME rc=$rc out=$(basename $OUT) rows=$(wc -l < $JL 2>/dev/null || echo 0) markers=$(grep -c '^\[VGain\] case [0-9]* slide' $OUT.log) skips=$(grep -c '^\[VGain\] case .*SKIP' $OUT.log) tb=$(grep -c Traceback $OUT.log) oom=$(grep -ci 'out of memory' $OUT.log) results=$(find $OUT -name result.json 2>/dev/null | wc -l) $(date +%F_%T)" >> $K/chain_gpu$GPU.log
done
echo "ALL DONE $(date +%F_%T) gpu=$GPU" >> $K/chain_gpu$GPU.log
