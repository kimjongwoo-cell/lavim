#!/bin/bash
# State-Backed WSI Support Handoff (memory/sbsh.py) — GPU smoke on nav3 GTEx (native bank, --variant base, no C1).
#   A signal : gtex40 0-1, VLMAS_SBSH_EST=both (Hutchinson K=16 + central difference eps 0.1), no restage,
#              VLMAS_RSS=<jsonl> VLMAS_RSS_NO_LOO=1 (probe bookkeeping + floor replay)
#   B handoff: gtex40 0-1, VLMAS_KV_RESTAGE=1 VLMAS_KV_RESTAGE_ORDER=state_backed VLMAS_KV_RESTAGE_MROPE=1
# Claims the first of the given GPUs whose memory drops below 2 GB (existing chains keep their claim).
# usage: sbsh_nav3.sh "<gpu list>"
set -u
GPUS=${1:-"7 6 8"}
REPO=/home/users/whddn12316/wsi_latent_0915_decode_hj
DATA=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
MODEL=/home/users/whddn12316/models/Qwen3-VL-4B-Thinking
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
DSJSON=$REPO/sender_relay_exp/smoke/gtex40.json
K=$REPO/sender_relay_exp/runs/sbsh_nav3
CLAIM=$REPO/sender_relay_exp/runs/vgain
LOG=$K/chain.log
mkdir -p $K $CLAIM
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG; }
mem() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' -v g=$1 '$1==g{print $2}'; }

log "WAIT one of gpus [$GPUS]"
G=""
while [ -z "$G" ]; do
  for g in $GPUS; do
    u=$(mem $g)
    if [ -n "$u" ] && [ "$u" -lt 2000 ] && mkdir $CLAIM/claim_gpu$g 2>/dev/null; then G=$g; break; fi
  done
  [ -z "$G" ] && sleep 60
done
trap 'rmdir $CLAIM/claim_gpu$G 2>/dev/null' EXIT
log "CLAIM gpu$G backbone_md5=$(md5sum $REPO/backbone/qwen3vl.py | cut -c1-8) sbsh_md5=$(md5sum $REPO/memory/sbsh.py | cut -c1-8) rss_md5=$(md5sum $REPO/memory/rss_diag.py | cut -c1-8) rsvmh_md5=$(md5sum $REPO/memory/rsvmh.py | cut -c1-8)"

# run <stage> <first> <last> <extra env...>
run() {
  local stage=$1 lo=$2 hi=$3; shift 3
  local stamp; stamp=$(date +%Y%m%d_%H%M%S)
  local dest=$K/$stage
  local out=$dest/gpu${G}_attempt_$stamp
  mkdir -p $dest
  local args=()
  for ((i=lo; i<=hi; i++)); do args+=(--dataset-index "$i"); done
  log "START $stage cases=$lo-$hi env=[$*] -> ${out#$K/}"
  cd $REPO && env \
    CUDA_VISIBLE_DEVICES=$G PYTHONPATH=$REPO PYTHONUNBUFFERED=1 \
    VLMAS_PROMPT_SET=prompt1 VLMAS_ATTN_IMPLEMENTATION=sdpa VLMAS_NAV2=1 VLMAS_NAV3_INDEPENDENT=1 \
    VLMAS_CROSS_SCALE_ROUTER=0 \
    VLMAS_RSS=$dest/rss_gtex.jsonl VLMAS_RSS_NO_LOO=1 "$@" \
    $PY -m wsi_latentmas.pipeline.latent_mas \
    --variant base --backbone qwen3-vl --dataset "$DSJSON" --slide-root "$DATA/slides" \
    --model "$MODEL" --output-root "$out" --device cuda:0 \
    --latent-steps 10 --patch-budget 12 --max-model-len 12288 --navigator-control-tokens 512 \
    --temperature 0.0 --top-p 1.0 --seed 42 --deterministic --answerer-greedy --no-answerer-thinking \
    --answerer-max-new-tokens 512 --answerer-rationale --answerer-protocol structured_json \
    --canonical-open-options --navigator-kv --io-pipeline --no-save-navigation-pngs --case-retries 0 \
    "${args[@]}" > "$out.log" 2>&1 < /dev/null
  local rc=$?
  log "DONE $stage rc=$rc results=$(find $out -name result.json | wc -l) sbsh=$(grep -c '^\[SBSH\] supports' $out.log) sbsh_err=$(grep '^\[SBSH\]' $out.log | grep -vc 'error=None') rsvmh=$(grep -c 'order=state_backed' $out.log) rss=$(grep -c '^\[RSS\] case' $out.log) tb=$(grep -c Traceback $out.log) oom=$(grep -ci 'out of memory' $out.log)"
  grep '^\[SBSH\]\|order=state_backed\|\[RSVMH\]' "$out.log" | cut -c1-400 >> $LOG
}

run smoke_signal 0 1 VLMAS_SBSH=1 VLMAS_SBSH_EST=both VLMAS_SBSH_K=16 VLMAS_SBSH_EPS=0.1
run smoke_handoff 0 1 VLMAS_KV_RESTAGE=1 VLMAS_KV_RESTAGE_ORDER=state_backed VLMAS_KV_RESTAGE_MROPE=1 VLMAS_SBSH_K=16
log "SMOKE DONE"
touch $K/SMOKE_DONE
