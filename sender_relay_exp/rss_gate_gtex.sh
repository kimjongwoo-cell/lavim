#!/bin/bash
# [C2 구조 검증] Role-Sensitive WSI Support Signal Gate — Reasoner role only (user decision 09-15:
# Navigator-side supports deferred), native visual bank (nav3, --variant base, no C1), GTEx.
#   Stage A: gtex40 cases 0-2, proxy + identity checks + no-deletion replay floor (VLMAS_RSS_NO_LOO=1) -> gate
#   Stage B: gtex40 cases 0-24, cheap Reasoner profile only (VLMAS_RSS_NO_LOO=1) -> profile report
#   Stage C: gtex40 cases 0-9 (fixed before Stage B), full Reasoner support LOO + Answerer decision row
# usage: rss_gate_gtex.sh <gpu>
set -u
G=${1:?gpu}
REPO=/home/users/whddn12316/wsi_latent_0915_decode_hj
DATA=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
MODEL=/home/users/whddn12316/models/Qwen3-VL-4B-Thinking
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
DSJSON=$REPO/sender_relay_exp/smoke/gtex40.json
K=$REPO/sender_relay_exp/runs/rss_gate_gtex
CLAIM=$REPO/sender_relay_exp/runs/vgain
LOG=$K/chain.log
mkdir -p $K $CLAIM
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG; }
mem() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' -v g=$G '$1==g{print $2}'; }

log "WAIT gpu$G"
until u=$(mem); [ -n "$u" ] && [ "$u" -lt 2000 ] && mkdir $CLAIM/claim_gpu$G 2>/dev/null; do sleep 60; done
trap 'rmdir $CLAIM/claim_gpu$G 2>/dev/null' EXIT
log "CLAIM gpu$G backbone_md5=$(md5sum $REPO/backbone/qwen3vl.py | cut -c1-8) engine_md5=$(md5sum $REPO/vision_text_mas/latent_qwen_engine.py | cut -c1-8) rss_md5=$(md5sum $REPO/memory/rss_diag.py | cut -c1-8) nav3_md5=$(md5sum $REPO/vision_text_mas/onepass_navigator.py | cut -c1-8)"

# run <stage> <no_loo 0|1> <first> <last>
run() {
  local stage=$1 noloo=$2 lo=$3 hi=$4
  local stamp; stamp=$(date +%Y%m%d_%H%M%S)
  local dest=$K/$stage
  local out=$dest/gpu${G}_attempt_$stamp
  mkdir -p $dest
  local args=()
  for ((i=lo; i<=hi; i++)); do args+=(--dataset-index "$i"); done
  local extra=""
  [ "$noloo" = 1 ] && extra="VLMAS_RSS_NO_LOO=1"
  log "START $stage cases=$lo-$hi no_loo=$noloo -> ${out#$K/}"
  cd $REPO && env \
    CUDA_VISIBLE_DEVICES=$G PYTHONPATH=$REPO PYTHONUNBUFFERED=1 \
    VLMAS_PROMPT_SET=prompt1 VLMAS_ATTN_IMPLEMENTATION=sdpa VLMAS_NAV2=1 VLMAS_NAV3_INDEPENDENT=1 \
    VLMAS_CROSS_SCALE_ROUTER=0 \
    VLMAS_RSS=$dest/rss_gtex.jsonl $extra \
    $PY -m wsi_latentmas.pipeline.latent_mas \
    --variant base --backbone qwen3-vl --dataset "$DSJSON" --slide-root "$DATA/slides" \
    --model "$MODEL" --output-root "$out" --device cuda:0 \
    --latent-steps 10 --patch-budget 12 --max-model-len 12288 --navigator-control-tokens 512 \
    --temperature 0.0 --top-p 1.0 --seed 42 --deterministic --answerer-greedy --no-answerer-thinking \
    --answerer-max-new-tokens 512 --answerer-rationale --answerer-protocol structured_json \
    --canonical-open-options --navigator-kv --io-pipeline --no-save-navigation-pngs --case-retries 0 \
    "${args[@]}" > "$out.log" 2>&1 < /dev/null
  local rc=$?
  log "DONE $stage rc=$rc results=$(find $out -name result.json | wc -l) rss_lines=$(grep -c '^\[RSS\] case' $out.log) tb=$(grep -c Traceback $out.log) oom=$(grep -ci 'out of memory' $out.log)"
}

if [ ! -f $K/stageA_gate.txt ]; then
  run stageA 1 0 2
  $PY $REPO/sender_relay_exp/rss_analyze.py --gate $K/stageA > $K/stageA_gate.txt 2>&1
  log "STAGE A GATE: $(head -1 $K/stageA_gate.txt)"
fi
grep -q ^PASS $K/stageA_gate.txt || { log "Stage A gate not PASS — stop"; exit 1; }
run stageB 1 0 24
$PY $REPO/sender_relay_exp/rss_analyze.py --profile $K/stageB > $K/stageB_profile.txt 2>&1
log "STAGE B profile -> stageB_profile.txt ($(tail -1 $K/stageB_profile.txt | cut -c1-160))"
run stageC 0 0 9
$PY $REPO/sender_relay_exp/rss_analyze.py $K/stageC > $K/stageC_analysis.txt 2>&1
log "ALL DONE analysis -> stageC_analysis.txt"
touch $K/ALL_DONE
