#!/bin/bash
# C1 #14.6 only: nav3 + Null-Template Residual Salience + Physical-Support Constrained Morphology Coverage
#   (memory/ssda_select.py select_ntrs via VLMAS_C1_QASC=1 VLMAS_C1_SELECT=ntrs; B = 25% of Reasoner visual tokens,
#   same prefill site as QASC, no relay, no C2). Base = nav3 base runs (runs/nav3_prompt1_base_*).
#   nav3 = VLMAS_NAV3_INDEPENDENT=1 + VLMAS_NAV2=1 + CROSS_SCALE_ROUTER=0 + prompt1, sdpa; 4B Thinking · patch 12 ·
#   max-model-len 12288 · latent step 10 · structured_json. GPU smoke passed 09-15 07:58 (runs/c1sel_nav3/smoke_ntrs).
# Full sets sharded by index % NSHARD; each shard claims the first free GPU of the list (memory < 2 GB).
# usage: ntrs_nav3.sh <shard> <nshard> "<gpu list>"
# Re-running resumes (indices with result.json under runs/ntrs_nav3/<ds>/ are skipped).
set -u
SHARD=${1:?shard}; NSHARD=${2:?nshard}; GPUS=${3:?gpus}
REPO=/home/users/whddn12316/wsi_latent_0915_decode_hj
DATA=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
MODEL=/home/users/whddn12316/models/Qwen3-VL-4B-Thinking
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
K=$REPO/sender_relay_exp/runs/ntrs_nav3
CLAIM=$REPO/sender_relay_exp/runs/vgain
LOG=$K/chain_shard$SHARD.log
mkdir -p $K $CLAIM
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG; }
mem() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' -v g=$1 '$1==g{print $2}'; }
log "WAIT one of gpus [$GPUS] shard $SHARD/$NSHARD"
G=""
while [ -z "$G" ]; do
  for g in $GPUS; do
    u=$(mem $g)
    if [ -n "$u" ] && [ "$u" -lt 2000 ] && mkdir $CLAIM/claim_gpu$g 2>/dev/null; then G=$g; break; fi
  done
  [ -z "$G" ] && sleep 60
done
trap 'rmdir $CLAIM/claim_gpu$G 2>/dev/null' EXIT
log "CLAIM gpu$G shard=$SHARD nav3_md5=$(md5sum $REPO/vision_text_mas/onepass_navigator.py | cut -c1-8) qasc_md5=$(md5sum $REPO/memory/qasc_select.py | cut -c1-8) ssda_md5=$(md5sum $REPO/memory/ssda_select.py | cut -c1-8) backbone_md5=$(md5sum $REPO/backbone/qwen3vl.py | cut -c1-8) engine_md5=$(md5sum $REPO/vision_text_mas/latent_qwen_engine.py | cut -c1-8)"

run() {
  local tag=$1 dsjson=$2 dest=$3; shift 3
  local out=$dest/gpu${G}_attempt_$(date +%Y%m%d_%H%M%S)
  mkdir -p $dest
  log "START $tag pending=$(( $# / 2 )) -> ${out#$K/}"
  cd $REPO && env \
    CUDA_VISIBLE_DEVICES=$G PYTHONPATH=$REPO PYTHONUNBUFFERED=1 \
    VLMAS_PROMPT_SET=prompt1 VLMAS_ATTN_IMPLEMENTATION=sdpa VLMAS_NAV2=1 VLMAS_NAV3_INDEPENDENT=1 \
    VLMAS_CROSS_SCALE_ROUTER=0 VLMAS_C1_QASC=1 VLMAS_C1_SELECT=ntrs \
    $PY -m wsi_latentmas.pipeline.latent_mas \
    --variant base --backbone qwen3-vl --dataset "$dsjson" --slide-root "$DATA/slides" \
    --model "$MODEL" --output-root "$out" --device cuda:0 \
    --latent-steps 10 --patch-budget 12 --max-model-len 12288 --navigator-control-tokens 512 \
    --temperature 0.0 --top-p 1.0 --seed 42 --deterministic --answerer-greedy --no-answerer-thinking \
    --answerer-max-new-tokens 512 --answerer-rationale --answerer-protocol structured_json \
    --canonical-open-options --navigator-kv --io-pipeline --no-save-navigation-pngs --case-retries 0 \
    "$@" > "$out.log" 2>&1 < /dev/null
  local rc=$?
  log "DONE $tag rc=$rc results=$(find $out -name result.json | wc -l) ntrs=$(grep -c '^\[NTRS\] kept' $out.log) qasc=$(grep -c '^\[QASC\] kept' $out.log) tb=$(grep -c Traceback $out.log) oom=$(grep -ci 'out of memory' $out.log)"
}

for spec in tcga_expert_vqa:128 tcga_slidebench:197 gtex:190 tcga:221 panda:196; do
  ds=${spec%%:*}; n=${spec##*:}
  done_ids=$(find $K/$ds -name result.json 2>/dev/null -exec $PY -c \
    'import json,sys; [print(json.load(open(f))["dataset_index"]) for f in sys.argv[1:]]' {} + | sort -n -u)
  ids=()
  for ((i=0; i<n; i++)); do
    [ $((i % NSHARD)) -eq "$SHARD" ] || continue
    grep -qx "$i" <<< "$done_ids" || ids+=(--dataset-index "$i")
  done
  if [ ${#ids[@]} -eq 0 ]; then log "SKIP $ds shard $SHARD complete"; continue; fi
  run $ds $DATA/$ds.json $K/$ds "${ids[@]}"
done
log "ALL DONE shard $SHARD"
touch $K/SHARD${SHARD}_DONE
