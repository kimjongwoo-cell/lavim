#!/bin/bash
# Efficient Role-Support Sensitivity on nav3 + Pruning-B (memory/rss_diag.py, env VLMAS_RSS).
# Stage A: 1 case per dataset (first dcs20 case) with proxy identity checks + full LOO; gate.
# Stage B: the next 4 dcs20 cases per dataset (N = 25 with Stage A). Same code, same flags.
# usage: rss_nav3.sh <gpu|auto>   (auto = first of GPU 6/7/8 that is idle)
set -u
G=${1:?gpu or auto}
REPO=/home/users/whddn12316/wsi_latent_0915_decode_hj
DATA=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
MODEL=/home/users/whddn12316/models/Qwen3-VL-4B-Thinking
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
K=$REPO/sender_relay_exp/runs/rss_nav3
CLAIM=$REPO/sender_relay_exp/runs/vgain
MANIFEST=$REPO/sender_relay_exp/runs/vit_sal_audit/full/manifest.json
LOG=$K/chain.log
mkdir -p $K $CLAIM
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG; }
mem() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' -v g=$1 '$1==g{print $2}'; }

# dataset indices: dcs20 sub_index 0..4 -> full-dataset index (same mapping as the a_i audit)
ids_for() {  # ids_for <ds> <first_sub> <last_sub>
  $PY - "$MANIFEST" "$1" "$2" "$3" <<'EOF'
import json, sys
man, ds, lo, hi = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
rows = sorted((r for r in json.load(open(man)) if r["ds"] == ds and r.get("status") == "ok"), key=lambda r: r["sub_index"])
print(" ".join(str(r["dataset_index"]) for r in rows[lo:hi + 1]))
EOF
}

CANDS=$G; [ "$G" = auto ] && CANDS="6 7 8"
log "WAIT gpu in [$CANDS]"
G=""
while [ -z "$G" ]; do
  for c in $CANDS; do
    u=$(mem $c)
    if [ -n "$u" ] && [ "$u" -lt 2000 ] && mkdir $CLAIM/claim_gpu$c 2>/dev/null; then G=$c; break; fi
  done
  [ -z "$G" ] && sleep 60
done
trap 'rmdir $CLAIM/claim_gpu$G 2>/dev/null' EXIT
log "CLAIM gpu$G backbone_md5=$(md5sum $REPO/backbone/qwen3vl.py | cut -c1-8) engine_md5=$(md5sum $REPO/vision_text_mas/latent_qwen_engine.py | cut -c1-8) rss_md5=$(md5sum $REPO/memory/rss_diag.py | cut -c1-8) nav3_md5=$(md5sum $REPO/vision_text_mas/onepass_navigator.py | cut -c1-8)"

# run <stage> <ds> <ids...>
run() {
  local stage=$1 ds=$2; shift 2
  local stamp; stamp=$(date +%Y%m%d_%H%M%S)
  local dest=$K/$stage/$ds
  local out=$dest/gpu${G}_attempt_$stamp
  local jl=$K/$stage/rss_$ds.jsonl
  mkdir -p $dest
  local args=()
  for i in "$@"; do args+=(--dataset-index "$i"); done
  log "START $stage $ds ids=[$*] -> ${out#$K/}"
  cd $REPO && env \
    CUDA_VISIBLE_DEVICES=$G PYTHONPATH=$REPO PYTHONUNBUFFERED=1 \
    VLMAS_PROMPT_SET=prompt1 VLMAS_ATTN_IMPLEMENTATION=sdpa VLMAS_NAV2=1 VLMAS_NAV3_INDEPENDENT=1 \
    VLMAS_CROSS_SCALE_ROUTER=0 \
    VLMAS_RSS=$jl \
    $PY -m wsi_latentmas.pipeline.latent_mas \
    --variant pruning_b --backbone qwen3-vl --dataset "$DATA/$ds.json" --slide-root "$DATA/slides" \
    --model "$MODEL" --output-root "$out" --device cuda:0 \
    --latent-steps 10 --patch-budget 12 --max-model-len 12288 --navigator-control-tokens 512 \
    --temperature 0.0 --top-p 1.0 --seed 42 --deterministic --answerer-greedy --no-answerer-thinking \
    --answerer-max-new-tokens 512 --answerer-rationale --answerer-protocol structured_json \
    --canonical-open-options --navigator-kv --io-pipeline --no-save-navigation-pngs --case-retries 0 \
    "${args[@]}" > "$out.log" 2>&1 < /dev/null
  local rc=$?
  log "DONE $stage $ds rc=$rc results=$(find $out -name result.json | wc -l) rss_lines=$(grep -c '^\[RSS\] case' $out.log) consolidate=$(grep -c 'PruneVision/v4' $out.log) tb=$(grep -c Traceback $out.log) oom=$(grep -ci 'out of memory' $out.log)"
}

DSETS="tcga_expert_vqa gtex tcga_slidebench tcga panda"
if [ ! -f $K/stageA_gate.txt ]; then
  for ds in $DSETS; do run stageA $ds $(ids_for $ds 0 0); done
  $PY $REPO/sender_relay_exp/rss_analyze.py --gate $K/stageA > $K/stageA_gate.txt 2>&1
  log "STAGE A GATE: $(head -1 $K/stageA_gate.txt)"
fi
grep -q ^PASS $K/stageA_gate.txt || { log "Stage A gate not PASS — stop"; exit 1; }
for ds in $DSETS; do run stageB $ds $(ids_for $ds 1 4); done
$PY $REPO/sender_relay_exp/rss_analyze.py $K/stageA $K/stageB > $K/analysis_N25.txt 2>&1
log "ALL DONE analysis -> analysis_N25.txt"
