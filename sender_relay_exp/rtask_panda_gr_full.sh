#!/bin/bash
# 0916 PANDA grade-rule test (VLMAS_ANSWERER_GRADE_RULE=1): decode-fix base, PANDA 196 cases, 3 shards, resume-aware (nav3, patch 12, step 10, NO C1/NTRS), greedy decode (same as board base)
#   only change vs the board base: bullet-format Reasoner task (VLMAS_REASONER_TASK)
#   + bullet-format Reasoner task (VLMAS_REASONER_TASK), copy tree wsi_latent_0915_decode_hj.
#   Digit repair is removed in this tree (empty answer -> <no-answer>). PANDA uses the inline x5 clamp.
# usage: sample_base_rtask_full.sh <shard> <nshard> "<gpus>"   (resumes: done dataset_index skipped)
set -u
SHARD=${1:?shard}; NSHARD=${2:?nshard}; GPUS=${3:?gpus}; HGPU=; HPID=
REPO=/home/users/whddn12316/wsi_latent_0915_decode_hj
DATA=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
MODEL=/home/users/whddn12316/models/Qwen3-VL-4B-Thinking
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
K=$REPO/sender_relay_exp/runs/rtask_panda_graderule
CLAIM=${CLAIMDIR:-/home/users/whddn12316/wsi_latent_0902_2155_hj/sender_relay_exp/runs/vgain}
LOG=$K/chain_gr_shard$SHARD.log
RTASK=$'For each patch, examine its image quality, architecture, cellularity, cytology, stroma, necrosis, tissue boundary, artifacts, and relevance to the question. Focus only on visually observable findings and distinguish true tissue features from artifacts. State your best morphological read for every feature even when uncertain; do not report a feature as unresolvable.\n\nFormat your response as follows:\n- One bullet point per patch, starting with its patch ID, using terse 2-4 word phrases per feature.\n- Consistent findings: findings shared across patches and the patch IDs that support them.\n- Contradictions: disagreements between patches.\n- Unresolved evidence: what remains uncertain.\n- Diagnostically usable patches: patch IDs.\n\nDo not diagnose, choose an answer, or navigate.\nNow, output your response below:'
mkdir -p $K $CLAIM
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG; }
mem() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' -v g=$1 '$1==g{print $2}'; }
G=""
if [ -n "$HGPU" ] && [ -n "$HPID" ]; then
  log "HANDOVER wait for pid $HPID on gpu$HGPU (previous launcher, same shard) then take its claim"
  while kill -0 "$HPID" 2>/dev/null; do sleep 5; done
  rmdir $CLAIM/claim_gpu$HGPU 2>/dev/null
  if mkdir $CLAIM/claim_gpu$HGPU 2>/dev/null; then G=$HGPU; fi
fi
log "WAIT one of gpus [$GPUS] shard $SHARD/$NSHARD (handover=${G:-none})"
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
    VLMAS_CROSS_SCALE_ROUTER=0 VLMAS_EFF_DUMP=1 VLMAS_REASONER_TASK="$RTASK" VLMAS_ANSWERER_GRADE_RULE=1 \
    $PY -m wsi_latentmas.pipeline.latent_mas \
    --variant base --backbone qwen3-vl --dataset "$dsjson" --slide-root "$DATA/slides" \
    --model "$MODEL" --output-root "$out" --device cuda:0 \
    --latent-steps 10 --patch-budget 12 --max-model-len 12288 --navigator-control-tokens 512 \
    --temperature 0.0 --top-p 1.0 --seed 42 --deterministic --answerer-greedy --no-answerer-thinking \
    --answerer-max-new-tokens 512 --answerer-rationale --answerer-protocol structured_json \
    --canonical-open-options --navigator-kv --io-pipeline --no-save-navigation-pngs --case-retries 0 \
    "$@" > "$out.log" 2>&1 < /dev/null
  local rc=$?
  log "DONE $tag rc=$rc results=$(find $out -name result.json | wc -l) graderule=$(grep -c 'Choose exactly one grade value' $out.log) empty=$(grep -c 'EMPTY terminal answer' $out.log) tb=$(grep -c Traceback $out.log) oom=$(grep -ci 'out of memory' $out.log) eff=$(find $out -name efficiency.json | wc -l)"
}

done_ids=$(find $K/panda -name result.json 2>/dev/null -exec $PY -c \
  'import json,sys; [print(json.load(open(f))["dataset_index"]) for f in sys.argv[1:]]' {} + | sort -n -u)
ids=()
for ((i=0; i<196; i++)); do
  [ $((i % NSHARD)) -eq "$SHARD" ] || continue
  grep -qx "$i" <<< "$done_ids" || ids+=(--dataset-index "$i")
done
log "RESUME shard $SHARD/$NSHARD pending=${#ids[@]} already_done=$(wc -w <<< "$done_ids")"
if [ ${#ids[@]} -eq 0 ]; then log "SKIP shard $SHARD complete"; exit 0; fi
run panda $DATA/panda.json $K/panda "${ids[@]}"
log "ALL DONE shard $SHARD"
log "ALL DONE shard $SHARD"
touch $K/SHARD${SHARD}_DONE
