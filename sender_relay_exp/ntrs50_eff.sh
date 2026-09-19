#!/bin/bash
# nav3 + C1 #14.6 NTRS with a 50% visual budget (VLMAS_QASC_KEEP=0.5; memory/ssda_select.py budget = ceil(0.5 N)), no C2,
#   VLMAS_EFF_DUMP=1. Default dataset: GTEx (190). Compare with nav3 base and NTRS 25% (runs/ntrs_nav3) on the same questions.
# Smoke gate (once, shared lock): gtex2 cases 0-1 -> results 2, [NTRS] kept 1536/3072 x2, tb 0, eff 2.
# usage: ntrs50_eff.sh <shard> <nshard> "<gpu list>" [dataset:n ...]
set -u
SHARD=${1:?shard}; NSHARD=${2:?nshard}; GPUS=${3:?gpus}; shift 3
SPECS=${*:-gtex:190}
REPO=/home/users/whddn12316/wsi_latent_0915_decode_hj
DATA=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
MODEL=/home/users/whddn12316/models/Qwen3-VL-4B-Thinking
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
K=$REPO/sender_relay_exp/runs/ntrs50_nav3
GATE=$K/smoke/gate.txt
CLAIM=$REPO/sender_relay_exp/runs/vgain
LOG=$K/chain_shard$SHARD.log
mkdir -p $K $CLAIM
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG; }
mem() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' -v g=$1 '$1==g{print $2}'; }

SMOKER=0
log "WAIT smoke gate or smoke lock shard $SHARD/$NSHARD"
until [ -f $GATE ]; do
  if mkdir $K/smoke.lock 2>/dev/null; then SMOKER=1; break; fi
  sleep 30
done
if [ $SMOKER = 0 ] && ! grep -q ^PASS $GATE; then log "smoke gate not PASS ($(cat $GATE)) - exit"; exit 1; fi
log "WAIT one of gpus [$GPUS] shard $SHARD/$NSHARD smoker=$SMOKER"
G=""
while [ -z "$G" ]; do
  for g in $GPUS; do
    u=$(mem $g)
    if [ -n "$u" ] && [ "$u" -lt 2000 ] && mkdir $CLAIM/claim_gpu$g 2>/dev/null; then G=$g; break; fi
  done
  [ -z "$G" ] && sleep 60
done
trap 'rmdir $CLAIM/claim_gpu$G 2>/dev/null' EXIT
log "CLAIM gpu$G shard=$SHARD keep=0.5 qasc_md5=$(md5sum $REPO/memory/qasc_select.py | cut -c1-8) ssda_md5=$(md5sum $REPO/memory/ssda_select.py | cut -c1-8) engine_md5=$(md5sum $REPO/vision_text_mas/latent_qwen_engine.py | cut -c1-8) backbone_md5=$(md5sum $REPO/backbone/qwen3vl.py | cut -c1-8) nav3_md5=$(md5sum $REPO/vision_text_mas/onepass_navigator.py | cut -c1-8)"

run() {  # run <tag> <dataset-json> <dest> <ids...>
  local tag=$1 dsjson=$2 dest=$3; shift 3
  local out=$dest/gpu${G}_attempt_$(date +%Y%m%d_%H%M%S)
  mkdir -p $dest
  log "START $tag keep=0.5 pending=$(( $# / 2 )) -> ${out#$K/}"
  cd $REPO && env \
    CUDA_VISIBLE_DEVICES=$G PYTHONPATH=$REPO PYTHONUNBUFFERED=1 \
    VLMAS_PROMPT_SET=prompt1 VLMAS_ATTN_IMPLEMENTATION=sdpa VLMAS_NAV2=1 VLMAS_NAV3_INDEPENDENT=1 \
    VLMAS_CROSS_SCALE_ROUTER=0 VLMAS_C1_QASC=1 VLMAS_C1_SELECT=ntrs VLMAS_QASC_KEEP=0.5 VLMAS_EFF_DUMP=1 \
    $PY -m wsi_latentmas.pipeline.latent_mas \
    --variant base --backbone qwen3-vl --dataset "$dsjson" --slide-root "$DATA/slides" \
    --model "$MODEL" --output-root "$out" --device cuda:0 \
    --latent-steps 10 --patch-budget 12 --max-model-len 12288 --navigator-control-tokens 512 \
    --temperature 0.0 --top-p 1.0 --seed 42 --deterministic --answerer-greedy --no-answerer-thinking \
    --answerer-max-new-tokens 512 --answerer-rationale --answerer-protocol structured_json \
    --canonical-open-options --navigator-kv --io-pipeline --no-save-navigation-pngs --case-retries 0 \
    "$@" > "$out.log" 2>&1 < /dev/null
  local rc=$?
  LAST_OUT=$out
  log "DONE $tag rc=$rc results=$(find $out -name result.json | wc -l) ntrs=$(grep -c '^\[NTRS\] kept' $out.log) kept50=$(grep -c '^\[NTRS\] kept 1536/3072' $out.log) tb=$(grep -c Traceback $out.log) oom=$(grep -ci 'out of memory' $out.log) eff=$(find $out -name efficiency.json | wc -l)"
}

if [ $SMOKER = 1 ]; then
  run smoke $REPO/sender_relay_exp/smoke/gtex2.json $K/smoke --dataset-index 0 --dataset-index 1
  o=$LAST_OUT
  r=$(find $o -name result.json | wc -l); k=$(grep -c '^\[NTRS\] kept 1536/3072' $o.log); t=$(grep -c Traceback $o.log); e=$(find $o -name efficiency.json | wc -l)
  line="results=$r kept1536=$k tb=$t eff=$e $(grep -m1 '^\[NTRS\] kept' $o.log | cut -c1-120)"
  if [ "$r" -eq 2 ] && [ "$k" -eq 2 ] && [ "$t" -eq 0 ] && [ "$e" -eq 2 ]; then echo "PASS $line" > $GATE; else echo "FAIL $line" > $GATE; fi
  log "SMOKE $(cat $GATE)"
  grep -q ^PASS $GATE || exit 1
fi

for spec in $SPECS; do
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
