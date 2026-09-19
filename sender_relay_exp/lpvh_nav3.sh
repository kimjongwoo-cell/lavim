#!/bin/bash
# C2 11.9 Latent-Provenance Visual Handoff (memory/lpvh.py, VLMAS_LPVH=1) on nav3 base (no C1 by default):
#   Reasoner: A[t,g] = mean_{l,h} sum_{j in G_g} alpha^R_{t,j} over the m latent steps (probe, no extra forward)
#   Answerer: pi = b_{A->latent} A ; visual mass re-allocated between supports, rho_V and within-support
#             distribution conserved (hooks on all decoder layers, rows=gen).
# nav3 = VLMAS_NAV3_INDEPENDENT=1 + VLMAS_NAV2=1 + CROSS_SCALE_ROUTER=0 + prompt1, sdpa; 4B Thinking · patch 12 ·
# max-model-len 12288 · latent step 10 · structured_json. Base = runs/nav3_prompt1_base_* (same questions).
# usage:
#   lpvh_nav3.sh smoke "<gpu list>"                 identity gate (maxdiff) then VLMAS_LPVH=1 on gtex2 cases 0-1
#   lpvh_nav3.sh <shard> <nshard> "<gpu list>"      5 full sets sharded by index % NSHARD, resumable
# extra env passed through: VLMAS_LPVH_SUPPORT (obs|branch|shuffled) VLMAS_LPVH_PROV (native|shuffled|uniform|
#   reasoner_mass) VLMAS_LPVH_ROWS (gen|all) VLMAS_LPVH_LAYERS (all|a-b) LPVH_C1=ntrs (adds VLMAS_C1_QASC=1
#   VLMAS_C1_SELECT=ntrs) LPVH_TAG=<name> (output subdir, default from PROV/SUPPORT)
#   LPVH_AFTER=<glob> (do not claim a GPU before a file matching the glob exists, e.g. runs/ntrs_nav3/SHARD*_DONE)
set -u
REPO=/home/users/whddn12316/wsi_latent_0915_decode_hj
DATA=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
MODEL=/home/users/whddn12316/models/Qwen3-VL-4B-Thinking
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
PROV=${VLMAS_LPVH_PROV:-native}; SUP=${VLMAS_LPVH_SUPPORT:-obs}
TAG=${LPVH_TAG:-${PROV}_${SUP}${LPVH_C1:+_c1$LPVH_C1}}
K=$REPO/sender_relay_exp/runs/lpvh_nav3/$TAG
CLAIM=$REPO/sender_relay_exp/runs/vgain
mkdir -p $K $CLAIM
C1ENV=""; [ "${LPVH_C1:-}" = ntrs ] && C1ENV="VLMAS_C1_QASC=1 VLMAS_C1_SELECT=ntrs"
if [ "$1" = smoke ]; then MODE=smoke; GPUS=${2:?gpus}; LOG=$K/smoke.log
else MODE=full; SHARD=${1:?shard}; NSHARD=${2:?nshard}; GPUS=${3:?gpus}; LOG=$K/chain_shard$SHARD.log; fi
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG; }
mem() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' -v g=$1 '$1==g{print $2}'; }
if [ -n "${LPVH_AFTER:-}" ]; then
  log "WAIT for file glob $LPVH_AFTER before claiming"
  until compgen -G "$LPVH_AFTER" > /dev/null; do sleep 120; done
fi
log "WAIT one of gpus [$GPUS] mode=$MODE tag=$TAG"
G=""
while [ -z "$G" ]; do
  for g in $GPUS; do
    u=$(mem $g)
    if [ -n "$u" ] && [ "$u" -lt 2000 ] && mkdir $CLAIM/claim_gpu$g 2>/dev/null; then G=$g; break; fi
  done
  [ -z "$G" ] && sleep 60
done
trap 'rmdir $CLAIM/claim_gpu$G 2>/dev/null' EXIT
log "CLAIM gpu$G lpvh_md5=$(md5sum $REPO/memory/lpvh.py | cut -c1-8) engine_md5=$(md5sum $REPO/vision_text_mas/latent_qwen_engine.py | cut -c1-8) backbone_md5=$(md5sum $REPO/backbone/qwen3vl.py | cut -c1-8) nav3_md5=$(md5sum $REPO/vision_text_mas/onepass_navigator.py | cut -c1-8)"

# run <lpvh mode 1|identity> <tag> <dataset-json> <outdir> <ids...>
run() {
  local lmode=$1 tag=$2 dsjson=$3 dest=$4; shift 4
  local out=$dest/gpu${G}_attempt_$(date +%Y%m%d_%H%M%S)
  mkdir -p $dest
  log "START $tag lpvh=$lmode pending=$(( $# / 2 )) -> ${out#$K/}"
  cd $REPO && env \
    CUDA_VISIBLE_DEVICES=$G PYTHONPATH=$REPO PYTHONUNBUFFERED=1 \
    VLMAS_PROMPT_SET=prompt1 VLMAS_ATTN_IMPLEMENTATION=sdpa VLMAS_NAV2=1 VLMAS_NAV3_INDEPENDENT=1 \
    VLMAS_CROSS_SCALE_ROUTER=0 $C1ENV \
    VLMAS_LPVH=$lmode VLMAS_LPVH_SUPPORT=$SUP VLMAS_LPVH_PROV=$PROV \
    VLMAS_LPVH_ROWS=${VLMAS_LPVH_ROWS:-gen} VLMAS_LPVH_LAYERS=${VLMAS_LPVH_LAYERS:-all} VLMAS_LPVH_LOG=${VLMAS_LPVH_LOG:-0} \
    $PY -m wsi_latentmas.pipeline.latent_mas \
    --variant base --backbone qwen3-vl --dataset "$dsjson" --slide-root "$DATA/slides" \
    --model "$MODEL" --output-root "$out" --device cuda:0 \
    --latent-steps 10 --patch-budget 12 --max-model-len 12288 --navigator-control-tokens 512 \
    --temperature 0.0 --top-p 1.0 --seed 42 --deterministic --answerer-greedy --no-answerer-thinking \
    --answerer-max-new-tokens 512 --answerer-rationale --answerer-protocol structured_json \
    --canonical-open-options --navigator-kv --io-pipeline --no-save-navigation-pngs --case-retries 0 \
    "$@" > "$out.log" 2>&1 < /dev/null
  local rc=$?
  RES=$(find $out -name result.json | wc -l); LP=$(grep -c '^\[LPVH\] case' $out.log); SK=$(grep -c '^\[LPVH\] SKIP' $out.log)
  log "DONE $tag lpvh=$lmode rc=$rc results=$RES lpvh_lines=$LP skip=$SK tb=$(grep -c Traceback $out.log) oom=$(grep -ci 'out of memory' $out.log)"
  grep '^\[LPVH\]' $out.log | cut -c1-400 >> $LOG
  LAST_LOG=$out.log
}

if [ "$MODE" = smoke ]; then
  SM=$REPO/sender_relay_exp/smoke/gtex2.json
  run identity smoke_identity $SM $K/smoke_identity --dataset-index 0 --dataset-index 1
  MD=$(grep -o 'maxdiff_identity=[0-9.e+-]*' $LAST_LOG | sed 's/.*=//' | sort -g | tail -1)
  MR=$(grep -o 'maxrel_identity=[0-9.e+-]*' $LAST_LOG | sed 's/.*=//' | sort -g | tail -1)
  run 1 smoke_lpvh $SM $K/smoke_lpvh --dataset-index 0 --dataset-index 1
  st=FAIL
  R2=$(find $K/smoke_lpvh -name result.json | wc -l); L2=$(grep -c '^\[LPVH\] case' $LAST_LOG); T2=$(grep -c Traceback $LAST_LOG)
  if [ "$R2" -ge 2 ] && [ "$L2" -ge 2 ] && [ "$T2" -eq 0 ] && [ -n "$MR" ] && awk -v m="$MR" 'BEGIN{exit !(m < 0.02)}'; then st=PASS; fi
  log "SMOKE $st identity_maxdiff=$MD identity_maxrel=$MR results=$R2 lpvh_lines=$L2 tb=$T2"
  echo "$st identity_maxdiff=$MD identity_maxrel=$MR results=$R2 lpvh_lines=$L2 tb=$T2" > $K/smoke_gate.txt
  log "ALL SMOKES DONE"
  exit 0
fi

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
  run 1 $ds $DATA/$ds.json $K/$ds "${ids[@]}"
done
log "ALL DONE shard $SHARD"
touch $K/SHARD${SHARD}_DONE
