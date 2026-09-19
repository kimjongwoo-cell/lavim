#!/bin/bash
# nav3 + C1 #14.6 NTRS + C2 11.9 Latent-Provenance Visual Handoff (memory/lpvh.py, VLMAS_LPVH=1, support=obs, prov=native,
#   rows=gen, layers=all), 5 full sets, VLMAS_EFF_DUMP=1, PANDA with the nav3 x5 clamp. Base for Δ = runs/ntrs_nav3 (same
#   questions, NTRS without LPVH); PANDA NTRS base = runs/ntrs_nav3/panda (clamp).
# Import hooks (tmp_hj/lpvh_ntrs_hooks/sitecustomize.py, originals untouched):
#   VLMAS_C1_META_BRIDGE=1  NTRS drops visual tokens before the prefill, so backbone _record_visual_meta sees 768 columns vs
#                           3072 grid tokens -> _visual_meta None -> LPVH SKIPs every case. The bridge hands the C1 keep mask
#                           to _record_visual_meta (CPU tests 19/19).
#   VLMAS_NAV3_X5_CLAMP=1   PANDA only.
# Smoke gate (once, shared lock; the other shards wait for it BEFORE claiming a GPU): gtex2 cases 0-1,
#   native NTRS reference, identity (Answerer text == native for every case, maxdiff < 1.0 = one bf16 step at |x|<256) then LPVH=1; requires [C1META] used=True meta=ok, [LPVH] case lines, no [LPVH] SKIP, tb 0.
# usage: lpvh_ntrs_eff.sh <shard> <nshard> "<gpu list>"
set -u
SHARD=${1:?shard}; NSHARD=${2:?nshard}; GPUS=${3:?gpus}
REPO=/home/users/whddn12316/wsi_latent_0915_decode_hj
DATA=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
MODEL=/home/users/whddn12316/models/Qwen3-VL-4B-Thinking
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
K=$REPO/sender_relay_exp/runs/lpvh_ntrs_nav3
HOOKS=$REPO/tmp_hj/lpvh_ntrs_hooks
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
log "CLAIM gpu$G shard=$SHARD lpvh_md5=$(md5sum $REPO/memory/lpvh.py | cut -c1-8) engine_md5=$(md5sum $REPO/vision_text_mas/latent_qwen_engine.py | cut -c1-8) backbone_md5=$(md5sum $REPO/backbone/qwen3vl.py | cut -c1-8) qasc_md5=$(md5sum $REPO/memory/qasc_select.py | cut -c1-8) ssda_md5=$(md5sum $REPO/memory/ssda_select.py | cut -c1-8) bridge_md5=$(md5sum $HOOKS/c1_meta_bridge.py | cut -c1-8) nav3_md5=$(md5sum $REPO/vision_text_mas/onepass_navigator.py | cut -c1-8) roots_md5=$(md5sum $REPO/vision_text_mas/onepass_navigation_roots.py | cut -c1-8)"

# run <tag> <lpvh 1|identity> <clamp 0|1> <dataset-json> <dest> <ids...>
run() {
  local tag=$1 lmode=$2 clamp=$3 dsjson=$4 dest=$5; shift 5
  local out=$dest/gpu${G}_attempt_$(date +%Y%m%d_%H%M%S)
  mkdir -p $dest
  log "START $tag lpvh=$lmode clamp=$clamp pending=$(( $# / 2 )) -> ${out#$K/}"
  cd $REPO && env \
    CUDA_VISIBLE_DEVICES=$G PYTHONPATH=$HOOKS:$REPO PYTHONUNBUFFERED=1 \
    VLMAS_PROMPT_SET=prompt1 VLMAS_ATTN_IMPLEMENTATION=sdpa VLMAS_NAV2=1 VLMAS_NAV3_INDEPENDENT=1 \
    VLMAS_CROSS_SCALE_ROUTER=0 VLMAS_C1_QASC=1 VLMAS_C1_SELECT=ntrs VLMAS_EFF_DUMP=1 \
    VLMAS_C1_META_BRIDGE=1 VLMAS_NAV3_X5_CLAMP=$clamp \
    VLMAS_LPVH=$lmode VLMAS_LPVH_SUPPORT=obs VLMAS_LPVH_PROV=native VLMAS_LPVH_ROWS=gen VLMAS_LPVH_LAYERS=all \
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
  log "DONE $tag lpvh=$lmode clamp=$clamp rc=$rc results=$(find $out -name result.json | wc -l) ntrs=$(grep -c '^\[NTRS\] kept' $out.log) c1meta_ok=$(grep -c '^\[C1META\] keep .* used=True meta=ok' $out.log) lpvh_case=$(grep -c '^\[LPVH\] case' $out.log) lpvh_skip=$(grep -c '^\[LPVH\] SKIP' $out.log) clamp_active=$(grep -c '^\[NAV3-X5CLAMP\] active' $out.log) tb=$(grep -c Traceback $out.log) oom=$(grep -ci 'out of memory' $out.log) eff=$(find $out -name efficiency.json | wc -l)"
}

same_outputs() {  # same_outputs <dir A> <dir B> -> prints "n_same/n_total"
  $PY - "$1" "$2" <<'PY'
import glob, json, sys
def outs(root):
    o = {}
    for rp in glob.glob(f"{root}/**/result.json", recursive=True):
        d = rp[:-len("/result.json")]
        o[int(json.load(open(rp))["dataset_index"])] = (json.load(open(d + "/answerer_call.json")).get("final_outputs") or [""])[0]
    return o
a, b = outs(sys.argv[1]), outs(sys.argv[2])
print(f"{sum(1 for k in a if k in b and a[k] == b[k])}/{max(len(a), len(b))}")
PY
}
if [ $SMOKER = 1 ]; then
  SM=$REPO/sender_relay_exp/smoke/gtex2.json
  run smoke_native '' 0 $SM $K/smoke/native --dataset-index 0 --dataset-index 1
  NAT=$LAST_OUT
  run smoke_identity identity 0 $SM $K/smoke/identity --dataset-index 0 --dataset-index 1
  IDN=$LAST_OUT; SAME=$(same_outputs $NAT $IDN)
  o=$LAST_OUT.log
  MD=$(grep -o 'maxdiff_identity=[0-9.e+-]*' $o | sed 's/.*=//' | sort -g | tail -1)
  i_ok=$(grep -c '^\[C1META\] keep .* used=True meta=ok' $o); i_case=$(grep -c '^\[LPVH\] case' $o); i_skip=$(grep -c '^\[LPVH\] SKIP' $o); i_tb=$(grep -c Traceback $o)
  run smoke_lpvh 1 0 $SM $K/smoke/lpvh --dataset-index 0 --dataset-index 1
  o=$LAST_OUT.log
  r=$(find $LAST_OUT -name result.json | wc -l); ok=$(grep -c '^\[C1META\] keep .* used=True meta=ok' $o); lc=$(grep -c '^\[LPVH\] case' $o)
  sk=$(grep -c '^\[LPVH\] SKIP' $o); tb=$(grep -c Traceback $o); ef=$(find $LAST_OUT -name efficiency.json | wc -l)
  line="identity_text_same_as_native=$SAME identity_maxdiff=$MD identity(c1meta=$i_ok case=$i_case skip=$i_skip tb=$i_tb) lpvh(results=$r c1meta=$ok case=$lc skip=$sk tb=$tb eff=$ef)"
  if [ -n "$MD" ] && awk -v m="$MD" 'BEGIN{exit !(m < 1.0)}' && [ "$SAME" = 2/2 ] && [ "$i_ok" -ge 2 ] && [ "$i_case" -ge 2 ] && [ "$i_skip" -eq 0 ] \
     && [ "$i_tb" -eq 0 ] && [ "$r" -eq 2 ] && [ "$ok" -ge 2 ] && [ "$lc" -ge 2 ] && [ "$sk" -eq 0 ] && [ "$tb" -eq 0 ] && [ "$ef" -eq 2 ]; then
    echo "PASS $line" > $GATE
  else
    echo "FAIL $line" > $GATE
  fi
  grep -h '^\[LPVH\]\|^\[C1META\]' $K/smoke/*/gpu*_attempt_*.log | cut -c1-600 >> $LOG
  log "SMOKE $(cat $GATE)"
  grep -q ^PASS $GATE || exit 1
fi

pending() {  # pending <result root> <n> -> fills ids
  local root=$1 n=$2 done_ids
  done_ids=$(find $root -name result.json 2>/dev/null -exec $PY -c \
    'import json,sys; [print(json.load(open(f))["dataset_index"]) for f in sys.argv[1:]]' {} + | sort -n -u)
  ids=()
  for ((i=0; i<n; i++)); do
    [ $((i % NSHARD)) -eq "$SHARD" ] || continue
    grep -qx "$i" <<< "$done_ids" || ids+=(--dataset-index "$i")
  done
}

for spec in tcga_expert_vqa:128:0 tcga_slidebench:197:0 gtex:190:0 tcga:221:0 panda:196:1; do
  IFS=: read ds n clamp <<< "$spec"
  pending $K/$ds $n
  if [ ${#ids[@]} -eq 0 ]; then log "SKIP $ds shard $SHARD complete"; continue; fi
  run $ds 1 $clamp $DATA/$ds.json $K/$ds "${ids[@]}"
done
log "ALL DONE shard $SHARD"
touch $K/SHARD${SHARD}_DONE
