#!/bin/bash
# Model-level diagnosis of canonical addressing on nav3 (memory/canon_diag.py).
# Pipeline decode stays native nav3 (VLMAS_KV_ROUTE=1 VLMAS_KV_ROUTE_MODE=bookkeep only records the visual
# bookkeeping); at the Answerer boundary the probe copies the cache into native / canonical / canonical_t /
# shift arms and records choice log-probs plus per-layer decision-row attention masses, crop entropy,
# cell-position correlation and visual DLA. ExpertVQA then SlideBench (full sets), index % 3 over GPU 6/7/8.
# usage: canon_diag_nav3.sh <gpu> <shard 0|1|2> <smoke|wait>
set -u
G=${1:?gpu}; SHARD=${2:?shard}; ROLE=${3:?smoke|wait}
REPO=/home/users/whddn12316/wsi_latent_0915_decode_hj
DATA=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
MODEL=/home/users/whddn12316/models/Qwen3-VL-4B-Thinking
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
K=$REPO/sender_relay_exp/runs/canon_diag
CLAIM=$REPO/sender_relay_exp/runs/vgain
LOG=$K/chain_gpu$G.log
mkdir -p $K $CLAIM
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG; }
mem() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' -v g=$G '$1==g{print $2}'; }

if [ "$ROLE" = wait ]; then
  log "WAIT smoke gate"
  until [ -f $K/smoke_gate.txt ]; do sleep 30; done
  grep -q ^PASS $K/smoke_gate.txt || { log "smoke gate not PASS — exit"; exit 1; }
fi
log "WAIT gpu$G"
until u=$(mem); [ -n "$u" ] && [ "$u" -lt 2000 ] && mkdir $CLAIM/claim_gpu$G 2>/dev/null; do sleep 60; done
trap 'rmdir $CLAIM/claim_gpu$G 2>/dev/null' EXIT
log "CLAIM gpu$G shard=$SHARD nav3_md5=$(md5sum $REPO/vision_text_mas/onepass_navigator.py | cut -c1-8) pfr_md5=$(md5sum $REPO/memory/pfr_attention.py | cut -c1-8) route_md5=$(md5sum $REPO/memory/route_address.py | cut -c1-8) engine_md5=$(md5sum $REPO/vision_text_mas/latent_qwen_engine.py | cut -c1-8)"

# run <tag> <dataset-json> <outdir> <ids...>
run() {
  local tag=$1 dsjson=$2 dest=$3; shift 3
  local stamp; stamp=$(date +%Y%m%d_%H%M%S)
  local out=$dest/gpu${G}_attempt_$stamp
  mkdir -p $dest
  log "START $tag pending=$(( $# / 2 )) -> ${out#$K/}"
  cd $REPO && env \
    CUDA_VISIBLE_DEVICES=$G PYTHONPATH=$REPO PYTHONUNBUFFERED=1 \
    VLMAS_PROMPT_SET=prompt1 VLMAS_ATTN_IMPLEMENTATION=sdpa VLMAS_NAV2=1 VLMAS_NAV3_INDEPENDENT=1 \
    VLMAS_CROSS_SCALE_ROUTER=0 \
    VLMAS_KV_ROUTE=1 VLMAS_KV_ROUTE_MODE=bookkeep \
    VLMAS_CANON_DIAG=$dest/diag_gpu$G.jsonl \
    $PY -m wsi_latentmas.pipeline.latent_mas \
    --variant base --backbone qwen3-vl --dataset "$dsjson" --slide-root "$DATA/slides" \
    --model "$MODEL" --output-root "$out" --device cuda:0 \
    --latent-steps 10 --patch-budget 12 --max-model-len 12288 --navigator-control-tokens 512 \
    --temperature 0.0 --top-p 1.0 --seed 42 --deterministic --answerer-greedy --no-answerer-thinking \
    --answerer-max-new-tokens 512 --answerer-rationale --answerer-protocol structured_json \
    --canonical-open-options --navigator-kv --io-pipeline --no-save-navigation-pngs --case-retries 0 \
    "$@" > "$out.log" 2>&1 < /dev/null
  local rc=$?
  local canon; canon=$(grep -c '^\[CanonDiag\] case' $out.log)
  local pfr; pfr=$(grep -c '^\[KVCanon\] canonical_t\? addressing\|^\[KVCanon\] canonical addressing\|^\[KVCanon\] canonical_t addressing' $out.log)
  local res; res=$(find $out -name result.json | wc -l)
  log "DONE $tag rc=$rc results=$res diag=$canon kvcanon_lines=$pfr tb=$(grep -c Traceback $out.log) oom=$(grep -ci 'out of memory' $out.log)"
  echo "$res $canon $pfr $(grep -c Traceback $out.log)"
}

if [ "$ROLE" = smoke ]; then
  r=$(run smoke $REPO/sender_relay_exp/smoke/gtex2.json $K/smoke --dataset-index 0 --dataset-index 1)
  read res canon pfr tb <<< "$r"
  if [ "$res" -eq 2 ] && [ "$canon" -eq 2 ] && [ "$pfr" -ge 4 ] && [ "$tb" -eq 0 ]; then
    echo PASS > $K/smoke_gate.txt; log "SMOKE PASS results=$res canonical=$canon pfr=$pfr tb=$tb"
  else
    echo "FAIL results=$res canonical=$canon pfr=$pfr tb=$tb" > $K/smoke_gate.txt; log "SMOKE FAIL results=$res canonical=$canon pfr=$pfr tb=$tb"; exit 1
  fi
fi

for spec in tcga_expert_vqa:128 tcga_slidebench:197; do
  ds=${spec%:*}; n=${spec##*:}
  done_ids=$(find $K/$ds -name result.json 2>/dev/null -exec $PY -c \
    'import json,sys; [print(json.load(open(f))["dataset_index"]) for f in sys.argv[1:]]' {} + | sort -n -u)
  ids=()
  for ((i=0; i<n; i++)); do
    [ $((i % 3)) -eq "$SHARD" ] || continue
    grep -qx "$i" <<< "$done_ids" || ids+=(--dataset-index "$i")
  done
  if [ ${#ids[@]} -eq 0 ]; then log "SKIP $ds shard $SHARD complete"; continue; fi
  run $ds $DATA/$ds.json $K/$ds "${ids[@]}" > /dev/null
done
log "ALL DONE gpu$G shard $SHARD"
