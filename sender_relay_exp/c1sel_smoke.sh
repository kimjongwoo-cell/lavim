#!/bin/bash
# GPU smoke for the runtime C1 selectors dispatched from memory/qasc_select.py:
#   ntrs  = #14.6 Null-Template Residual Salience + Physical-Support Constrained Morphology Coverage
#   ssda  = #14.8 Spectral Support-Demand Allocation + Residual-Salience Representative Selection
# nav3 env identical to qasc_nav3.sh (VLMAS_C1_QASC=1) + VLMAS_C1_SELECT=<mode>; gtex2 smoke cases 0-1.
# Claims the first free GPU of the list (memory < 2 GB). Gate: results 2, [NTRS]/[SSDA] kept lines >= 2, tb 0.
# usage: c1sel_smoke.sh "<gpu list>" <mode...>
set -u
GPUS=${1:?gpus}; shift
REPO=/home/users/whddn12316/wsi_latent_0915_decode_hj
DATA=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
MODEL=/home/users/whddn12316/models/Qwen3-VL-4B-Thinking
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
K=$REPO/sender_relay_exp/runs/c1sel_nav3
CLAIM=$REPO/sender_relay_exp/runs/vgain
LOG=$K/chain.log
mkdir -p $K $CLAIM
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG; }
mem() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' -v g=$1 '$1==g{print $2}'; }
log "WAIT one of gpus [$GPUS] modes [$*]"
G=""
while [ -z "$G" ]; do
  for g in $GPUS; do
    u=$(mem $g)
    if [ -n "$u" ] && [ "$u" -lt 2000 ] && mkdir $CLAIM/claim_gpu$g 2>/dev/null; then G=$g; break; fi
  done
  [ -z "$G" ] && sleep 60
done
trap 'rmdir $CLAIM/claim_gpu$G 2>/dev/null' EXIT
log "CLAIM gpu$G qasc_md5=$(md5sum $REPO/memory/qasc_select.py | cut -c1-8) ssda_md5=$(md5sum $REPO/memory/ssda_select.py | cut -c1-8) backbone_md5=$(md5sum $REPO/backbone/qwen3vl.py | cut -c1-8)"
for mode in "$@"; do
  TAG=$(echo $mode | tr a-z A-Z)
  out=$K/smoke_$mode/gpu${G}_attempt_$(date +%Y%m%d_%H%M%S)
  mkdir -p $K/smoke_$mode
  log "START smoke $mode -> ${out#$K/}"
  cd $REPO && env \
    CUDA_VISIBLE_DEVICES=$G PYTHONPATH=$REPO PYTHONUNBUFFERED=1 \
    VLMAS_PROMPT_SET=prompt1 VLMAS_ATTN_IMPLEMENTATION=sdpa VLMAS_NAV2=1 VLMAS_NAV3_INDEPENDENT=1 \
    VLMAS_CROSS_SCALE_ROUTER=0 VLMAS_C1_QASC=1 VLMAS_C1_SELECT=$mode \
    $PY -m wsi_latentmas.pipeline.latent_mas \
    --variant base --backbone qwen3-vl --dataset $REPO/sender_relay_exp/smoke/gtex2.json --slide-root "$DATA/slides" \
    --model "$MODEL" --output-root "$out" --device cuda:0 \
    --latent-steps 10 --patch-budget 12 --max-model-len 12288 --navigator-control-tokens 512 \
    --temperature 0.0 --top-p 1.0 --seed 42 --deterministic --answerer-greedy --no-answerer-thinking \
    --answerer-max-new-tokens 512 --answerer-rationale --answerer-protocol structured_json \
    --canonical-open-options --navigator-kv --io-pipeline --no-save-navigation-pngs --case-retries 0 \
    --dataset-index 0 --dataset-index 1 > "$out.log" 2>&1 < /dev/null
  rc=$?
  res=$(find $out -name result.json | wc -l); kept=$(grep -c "^\[$TAG\] kept" $out.log); tb=$(grep -c Traceback $out.log)
  st=FAIL; [ "$res" -eq 2 ] && [ "$kept" -ge 2 ] && [ "$tb" -eq 0 ] && st=PASS
  log "DONE smoke $mode rc=$rc results=$res kept_lines=$kept qasc_lines=$(grep -c '^\[QASC\] kept' $out.log) tb=$tb oom=$(grep -ci 'out of memory' $out.log) -> $st"
  grep "^\[$TAG\] kept" $out.log | cut -c1-300 >> $LOG
  echo $st > $K/smoke_${mode}_gate.txt
done
log "ALL SMOKES DONE"
