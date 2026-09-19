#!/bin/bash
# ntrs_nav3_eff.sh + PANDA with the opt-in nav3 x5 clamp (tmp_hj/nav3_x5_clamp: x5 side = min(4096, root w, root h);
#   original vision_text_mas/onepass_navigation_roots.py untouched, md5-guarded sitecustomize hook).
#   Chain per shard: tcga_expert_vqa, tcga_slidebench, gtex, tcga (NTRS, no clamp — clamp only changes slides with a side
#   < 4096 px, which only PANDA has), then PANDA NTRS + clamp -> runs/ntrs_nav3/panda and PANDA nav3 base + clamp
#   -> runs/nav3clamp_base/panda (all 196; old base lacks 11 Navigator-failure slides).
#   Clamp GPU smoke gate (2 failing slides, NTRS + clamp) runs once before any PANDA step (shared lock), and shard 2 runs it
#   right after its tcga_expert_vqa so a failure is known early. VLMAS_EFF_DUMP=1 everywhere.
# usage: ntrs_nav3_eff2.sh <shard> <nshard> "<gpu list>" [handover_gpu handover_pid]
set -u
SHARD=${1:?shard}; NSHARD=${2:?nshard}; GPUS=${3:?gpus}; HGPU=${4:-}; HPID=${5:-}
REPO=/home/users/whddn12316/wsi_latent_0915_decode_hj
DATA=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
MODEL=/home/users/whddn12316/models/Qwen3-VL-4B-Thinking
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
K=$REPO/sender_relay_exp/runs/ntrs_nav3
KB=$REPO/sender_relay_exp/runs/nav3clamp_base
CLAMP=$REPO/tmp_hj/nav3_x5_clamp
GATE=$K/smoke_clamp/gate.txt
CLAIM=$REPO/sender_relay_exp/runs/vgain
LOG=$K/chain_shard$SHARD.log
mkdir -p $K $KB $CLAIM
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG; }
mem() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' -v g=$1 '$1==g{print $2}'; }
G=""
if [ -n "$HGPU" ] && [ -n "$HPID" ]; then
  log "HANDOVER(eff2) wait for pid $HPID on gpu$HGPU (previous launcher, same shard) then take its claim"
  while kill -0 "$HPID" 2>/dev/null; do sleep 5; done
  rmdir $CLAIM/claim_gpu$HGPU 2>/dev/null
  if mkdir $CLAIM/claim_gpu$HGPU 2>/dev/null; then G=$HGPU; fi
fi
log "WAIT(eff2) one of gpus [$GPUS] shard $SHARD/$NSHARD (handover=${G:-none})"
while [ -z "$G" ]; do
  for g in $GPUS; do
    u=$(mem $g)
    if [ -n "$u" ] && [ "$u" -lt 2000 ] && mkdir $CLAIM/claim_gpu$g 2>/dev/null; then G=$g; break; fi
  done
  [ -z "$G" ] && sleep 60
done
trap 'rmdir $CLAIM/claim_gpu$G 2>/dev/null' EXIT
log "CLAIM(eff2) gpu$G shard=$SHARD nav3_md5=$(md5sum $REPO/vision_text_mas/onepass_navigator.py | cut -c1-8) roots_md5=$(md5sum $REPO/vision_text_mas/onepass_navigation_roots.py | cut -c1-8) clamp_md5=$(md5sum $CLAMP/nav3_x5_clamp.py | cut -c1-8) qasc_md5=$(md5sum $REPO/memory/qasc_select.py | cut -c1-8) ssda_md5=$(md5sum $REPO/memory/ssda_select.py | cut -c1-8) backbone_md5=$(md5sum $REPO/backbone/qwen3vl.py | cut -c1-8) engine_md5=$(md5sum $REPO/vision_text_mas/latent_qwen_engine.py | cut -c1-8)"

# run <tag> <arm ntrs|base> <clamp 0|1> <dataset-json> <dest> <ids...>
run() {
  local tag=$1 arm=$2 clamp=$3 dsjson=$4 dest=$5; shift 5
  local out=$dest/gpu${G}_attempt_$(date +%Y%m%d_%H%M%S)
  local pp=$REPO sel="" cl=""
  [ "$clamp" = 1 ] && { pp=$CLAMP:$REPO; cl="VLMAS_NAV3_X5_CLAMP=1"; }
  [ "$arm" = ntrs ] && sel="VLMAS_C1_QASC=1 VLMAS_C1_SELECT=ntrs"
  mkdir -p $dest
  log "START $tag arm=$arm clamp=$clamp pending=$(( $# / 2 )) -> ${out#$REPO/sender_relay_exp/runs/}"
  cd $REPO && env \
    CUDA_VISIBLE_DEVICES=$G PYTHONPATH=$pp PYTHONUNBUFFERED=1 \
    VLMAS_PROMPT_SET=prompt1 VLMAS_ATTN_IMPLEMENTATION=sdpa VLMAS_NAV2=1 VLMAS_NAV3_INDEPENDENT=1 \
    VLMAS_CROSS_SCALE_ROUTER=0 VLMAS_EFF_DUMP=1 $sel $cl \
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
  log "DONE $tag arm=$arm clamp=$clamp rc=$rc results=$(find $out -name result.json | wc -l) ntrs=$(grep -c '^\[NTRS\] kept' $out.log) clamp_active=$(grep -c '^\[NAV3-X5CLAMP\] active' $out.log) clamp_roots=$(grep -c '^\[NAV3-X5CLAMP\] root' $out.log) tb=$(grep -c Traceback $out.log) oom=$(grep -ci 'out of memory' $out.log) eff=$(find $out -name efficiency.json | wc -l)"
}

# clamp smoke gate: PANDA idx 9 and 11 (both failed Navigator in nav3 base: short side < 4096 px), NTRS + clamp
smoke_gate() {
  if [ ! -f $GATE ]; then
    if mkdir $K/smoke_clamp.lock 2>/dev/null; then
      run smoke_clamp ntrs 1 $DATA/panda.json $K/smoke_clamp --dataset-index 9 --dataset-index 11
      local o=$LAST_OUT
      local r=$(find $o -name result.json | wc -l) a=$(grep -c '^\[NAV3-X5CLAMP\] active' $o.log) \
            c=$(grep -c '^\[NAV3-X5CLAMP\] root' $o.log) n=$(grep -c '^\[NTRS\] kept' $o.log) \
            t=$(grep -c Traceback $o.log) e=$(find $o -name efficiency.json | wc -l)
      if [ "$r" -eq 2 ] && [ "$a" -ge 1 ] && [ "$c" -ge 1 ] && [ "$n" -ge 2 ] && [ "$t" -eq 0 ] && [ "$e" -eq 2 ]; then
        echo "PASS results=$r clamp_active=$a clamp_roots=$c ntrs=$n tb=$t eff=$e" > $GATE
      else
        echo "FAIL results=$r clamp_active=$a clamp_roots=$c ntrs=$n tb=$t eff=$e" > $GATE
      fi
      log "SMOKE_CLAMP $(cat $GATE)"
    else
      log "WAIT smoke_clamp gate (another shard runs it)"
      until [ -f $GATE ]; do sleep 30; done
    fi
  fi
  grep -q ^PASS $GATE
}

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

for spec in tcga_expert_vqa:128 tcga_slidebench:197 gtex:190 tcga:221; do
  ds=${spec%%:*}; n=${spec##*:}
  pending $K/$ds $n
  if [ ${#ids[@]} -eq 0 ]; then log "SKIP $ds shard $SHARD complete"
  else run $ds ntrs 0 $DATA/$ds.json $K/$ds "${ids[@]}"; fi
  if [ "$ds" = tcga_expert_vqa ] && [ "$SHARD" = 2 ]; then smoke_gate || log "smoke_clamp not PASS (PANDA steps will be skipped)"; fi
done

if smoke_gate; then
  pending $K/panda 196
  if [ ${#ids[@]} -eq 0 ]; then log "SKIP panda ntrs+clamp shard $SHARD complete"
  else run panda ntrs 1 $DATA/panda.json $K/panda "${ids[@]}"; fi
  pending $KB/panda 196
  if [ ${#ids[@]} -eq 0 ]; then log "SKIP panda base+clamp shard $SHARD complete"
  else run panda_base base 1 $DATA/panda.json $KB/panda "${ids[@]}"; fi
else
  log "SKIP panda steps: smoke_clamp gate $(cat $GATE)"
fi
log "ALL DONE(eff2) shard $SHARD"
touch $K/SHARD${SHARD}_DONE
