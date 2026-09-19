#!/bin/bash
# Method setting: nav3 · Canonical addressing + Context-Updated Answerer Query Visual Re-Read (single arm).
#   navigator  nav3  = VLMAS_NAV3_INDEPENDENT=1 (with the runner's VLMAS_NAV2=1, CROSS_SCALE_ROUTER=0, prompt1),
#                      same env as 0912/run_nav3_prompt1_base_both_gpu78.sh -> 0912/run_dataset_nav2_prompt1.sh
#   C2 read    canonical visual addressing (VLMAS_KV_ROUTE=1 VLMAS_KV_ROUTE_MODE=canonical), applied to the Answerer
#              decode ONLY (VLMAS_KV_ROUTE_TERMINAL_ONLY=1; 09-14 21:48 first smoke showed stale Reasoner bookkeeping
#              re-rotating nav3 navigator decodes of the next case — those outputs moved to *_STALE_navcanon_*)
#              + matched context-updated query re-read at layer 27 with native visual mass kept
#                (VLMAS_ANSWERER_PFR=1 VLMAS_ANSWERER_PFR_LAYER=27)
#   base       4B Thinking · patch 12 · max-model-len 12288 · latent step 10 · sdpa · structured_json
# Full sets (EVQA 128 · GTEx 190 · SB 197 · TCGA 221 · PANDA 196), sharded by index % 3 over GPU 6/7/8.
#
# usage: canon_reread_nav3.sh <gpu> <shard 0|1|2> <smoke|wait>
#   smoke: 2-case gtex2 smoke + gate first (writes runs/canon_reread_nav3/smoke_gate.txt)
#   wait : wait for smoke_gate.txt == PASS
# Re-running resumes (indices with result.json under runs/canon_reread_nav3/<ds>/ are skipped).
set -u
G=${1:?gpu}; SHARD=${2:?shard}; ROLE=${3:?smoke|wait}
REPO=/home/users/whddn12316/wsi_latent_0915_decode_hj
DATA=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
MODEL=/home/users/whddn12316/models/Qwen3-VL-4B-Thinking
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
K=$REPO/sender_relay_exp/runs/canon_reread_nav3
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
    VLMAS_KV_ROUTE=1 VLMAS_KV_ROUTE_MODE=canonical VLMAS_KV_ROUTE_TERMINAL_ONLY=1 \
    VLMAS_ANSWERER_PFR=1 VLMAS_ANSWERER_PFR_LAYER=27 \
    $PY -m wsi_latentmas.pipeline.latent_mas \
    --variant base --backbone qwen3-vl --dataset "$dsjson" --slide-root "$DATA/slides" \
    --model "$MODEL" --output-root "$out" --device cuda:0 \
    --latent-steps 10 --patch-budget 12 --max-model-len 12288 --navigator-control-tokens 512 \
    --temperature 0.0 --top-p 1.0 --seed 42 --deterministic --answerer-greedy --no-answerer-thinking \
    --answerer-max-new-tokens 512 --answerer-rationale --answerer-protocol structured_json \
    --canonical-open-options --navigator-kv --io-pipeline --no-save-navigation-pngs --case-retries 0 \
    "$@" > "$out.log" 2>&1 < /dev/null
  local rc=$?
  local canon; canon=$(grep -c '^\[KVCanon\] canonical addressing' $out.log)
  local pfr; pfr=$(grep -c '^\[PFRctx\] {"case": [0-9]*, "mode": "1"' $out.log)
  local res; res=$(find $out -name result.json | wc -l)
  log "DONE $tag rc=$rc results=$res canonical=$canon pfr=$pfr tb=$(grep -c Traceback $out.log) oom=$(grep -ci 'out of memory' $out.log)"
  echo "$res $canon $pfr $(grep -c Traceback $out.log)"
}

if [ "$ROLE" = smoke ]; then
  r=$(run smoke $REPO/sender_relay_exp/smoke/gtex2.json $K/smoke --dataset-index 0 --dataset-index 1)
  read res canon pfr tb <<< "$r"
  if [ "$res" -eq 2 ] && [ "$canon" -eq 2 ] && [ "$pfr" -ge 2 ] && [ "$tb" -eq 0 ]; then
    echo PASS > $K/smoke_gate.txt; log "SMOKE PASS results=$res canonical=$canon pfr=$pfr tb=$tb"
  else
    echo "FAIL results=$res canonical=$canon pfr=$pfr tb=$tb" > $K/smoke_gate.txt; log "SMOKE FAIL results=$res canonical=$canon pfr=$pfr tb=$tb"; exit 1
  fi
fi

for spec in tcga_expert_vqa:128 gtex:190 tcga_slidebench:197 tcga:221 panda:196; do
  ds=${spec%%:*}; n=${spec##*:}
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
