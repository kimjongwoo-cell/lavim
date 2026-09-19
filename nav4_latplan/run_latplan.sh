#!/bin/bash
# nav4 + decode fix + latplan (Planner text path removed) -- self-contained runner.
# Every setting the run depends on is in this file; to stack a method on top, pass
# NAME / HOOKS / EXTRA_ENV (see README.md in this folder) instead of editing the base.
#
#   bash run_latplan.sh smoke        # 2 questions on one GPU, writes <OUTROOT>/<NAME>/SMOKE_GATE (PASS/FAIL)
#   bash run_latplan.sh run          # needs a PASS gate; NSHARD shards in the background, waits, reports
#   bash run_latplan.sh status       # results so far per dataset
#   DRY_RUN=1 bash run_latplan.sh run   # print the commands only
#
# env (all optional):
#   NAME       run name, results go to <OUTROOT>/<NAME>         default latplan
#   OUTROOT    results root                                       default <this folder>/runs
#   GPUS       candidate GPUs                                      default "6 7 8"
#   NSHARD     parallel shards (one GPU each)                      default 3
#   MAXUSED    a GPU is free when its used memory (MiB) is below   default 20000
#   SPECS      "dataset:n_questions ..."                           default "tcga_expert_vqa:128 tcga_slidebench:197"
#   SLIDE_ROOT slide folder                                        default <DATA>/slides
#   HOOKS      sitecustomize folder put first on PYTHONPATH        default $REPO/tmp_hj/latplan_hooks
#   EXTRA_ENV  extra "KEY=VALUE ..." for the method stacked on top default empty
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=/home/users/whddn12316/wsi_latent_0915_decode_hj
DATA=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
MODEL=/home/users/whddn12316/models/Qwen3-VL-4B-Thinking
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python

NAME=${NAME:-latplan}
OUTROOT=${OUTROOT:-$HERE/runs}
GPUS=${GPUS:-"6 7 8"}
NSHARD=${NSHARD:-3}
MAXUSED=${MAXUSED:-20000}
SPECS=${SPECS:-"tcga_expert_vqa:128 tcga_slidebench:197"}
SLIDE_ROOT=${SLIDE_ROOT:-$DATA/slides}
HOOKS=${HOOKS:-$REPO/tmp_hj/latplan_hooks}
EXTRA_ENV=${EXTRA_ENV:-}
SMOKE_JSON=${SMOKE_JSON:-$REPO/sender_relay_exp/smoke/glvr_rho2.json}   # ExpertVQA #4 + SlideBench #2339
DRY=${DRY_RUN:-0}

K=$OUTROOT/$NAME
CLAIM=$OUTROOT/claims          # shared by every run under OUTROOT: one run per GPU at a time
mkdir -p "$K" "$CLAIM"
log() { echo "$(date '+%m-%d %H:%M:%S') [$NAME] $*" | tee -a "$K/run.log"; }

# ---- the base configuration: nav4 + decode fix (tree default) + latplan ------------------------
# decode fix = this tree's latent_answerer.py: no forced digit re-decode on an empty answer
# (stored as <no-answer>, scored wrong), 512-token answer budget, JSON opener teacher-forced.
BASE_ENV="VLMAS_PLAN_LATENT_ONLY=1 \
VLMAS_PROMPT_SET=prompt1 VLMAS_ATTN_IMPLEMENTATION=sdpa \
VLMAS_NAV4=1 VLMAS_NAV2=1 VLMAS_CROSS_SCALE_ROUTER=0 \
VLMAS_EFF_DUMP=1 VLMAS_ANSWERER_GRADE_RULE=1"
RTASK=$'For each patch, examine its image quality, architecture, cellularity, cytology, stroma, necrosis, tissue boundary, artifacts, and relevance to the question. Focus only on visually observable findings and distinguish true tissue features from artifacts. State your best morphological read for every feature even when uncertain; do not report a feature as unresolvable.\n\nFormat your response as follows:\n- One bullet point per patch, starting with its patch ID, using terse 2-4 word phrases per feature.\n- Consistent findings: findings shared across patches and the patch IDs that support them.\n- Contradictions: disagreements between patches.\n- Unresolved evidence: what remains uncertain.\n- Diagnostically usable patches: patch IDs.\n\nDo not diagnose, choose an answer, or navigate.\nNow, output your response below:'
ARGS=(--variant base --backbone qwen3-vl --model "$MODEL" --slide-root "$SLIDE_ROOT" --device cuda:0
      --latent-steps 10 --patch-budget 12 --max-model-len 12288 --navigator-control-tokens 512
      --temperature 0.0 --top-p 1.0 --seed 42 --deterministic --answerer-greedy --no-answerer-thinking
      --answerer-max-new-tokens 512 --answerer-rationale --answerer-protocol structured_json
      --canonical-open-options --navigator-kv --io-pipeline --no-save-navigation-pngs --case-retries 0)

mem() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' '; }

claim_gpu() {                  # prints the claimed GPU; waits until one is free
  local g u
  while :; do
    for g in $GPUS; do
      u=$(mem "$g")
      if [ -n "$u" ] && [ "$u" -lt "$MAXUSED" ] && mkdir "$CLAIM/gpu$g" 2>/dev/null; then echo "$g"; return; fi
    done
    sleep 60
  done
}

run_one() {                    # run_one <gpu> <dataset json> <out dir> [--dataset-index i ...]
  local g=$1 js=$2 dest=$3; shift 3
  local out=$dest/gpu${g}_attempt_$(date +%Y%m%d_%H%M%S)
  mkdir -p "$dest"
  log "START gpu$g $(basename "$js") n=$(( $# / 2 )) -> ${out#$OUTROOT/} env=[$EXTRA_ENV]"
  if [ "$DRY" = 1 ]; then
    echo "  + cd $REPO && env CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$HOOKS:$REPO $BASE_ENV $EXTRA_ENV VLMAS_REASONER_TASK=<RTASK> $PY -m wsi_latentmas.pipeline.latent_mas --dataset $js --output-root $out ${ARGS[*]} $*"
    return 0
  fi
  ( cd "$REPO" && env CUDA_VISIBLE_DEVICES="$g" PYTHONPATH="$HOOKS:$REPO" PYTHONUNBUFFERED=1 \
      $BASE_ENV $EXTRA_ENV VLMAS_REASONER_TASK="$RTASK" \
      "$PY" -m wsi_latentmas.pipeline.latent_mas --dataset "$js" --output-root "$out" "${ARGS[@]}" "$@" \
      > "$out.log" 2>&1 < /dev/null )
  local rc=$?
  log "DONE gpu$g rc=$rc results=$(find "$out" -name result.json 2>/dev/null | wc -l) traceback=$(grep -c Traceback "$out.log") oom=$(grep -ci 'out of memory' "$out.log") empty=$(grep -c 'EMPTY terminal answer' "$out.log")"
}

done_ids() { find "$K/$1" -name result.json 2>/dev/null -exec "$PY" -c \
  'import json,sys; [print(json.load(open(f))["dataset_index"]) for f in sys.argv[1:]]' {} + | sort -n -u; }

shard() {                      # shard <index>: the questions i with i % NSHARD == index that are not done yet
  local s=$1 spec ds n i ids dn g
  [ "$DRY" = 1 ] && g=${GPUS%% *} || g=$(claim_gpu)
  trap "rmdir $CLAIM/gpu$g 2>/dev/null" EXIT
  log "CLAIM gpu$g shard $s/$NSHARD md5 nav4=$(md5sum $REPO/vision_text_mas/onepass_navigation_nav4.py | cut -c1-8) answerer=$(md5sum $REPO/vision_text_mas/latent_answerer.py | cut -c1-8) backbone=$(md5sum $REPO/backbone/qwen3vl.py | cut -c1-8) engine=$(md5sum $REPO/vision_text_mas/latent_qwen_engine.py | cut -c1-8) hooks=$(md5sum $HOOKS/sitecustomize.py | cut -c1-8)"
  for spec in $SPECS; do
    ds=${spec%%:*}; n=${spec##*:}
    dn=$(done_ids "$ds")
    ids=()
    for ((i = 0; i < n; i++)); do
      [ $((i % NSHARD)) -eq "$s" ] || continue
      grep -qx "$i" <<< "$dn" || ids+=(--dataset-index "$i")
    done
    if [ ${#ids[@]} -eq 0 ]; then log "SKIP $ds shard $s (complete)"; continue; fi
    run_one "$g" "$DATA/$ds.json" "$K/$ds" "${ids[@]}"
  done
  log "SHARD $s DONE"
}

case ${1:-} in
  smoke)
    g=$( [ "$DRY" = 1 ] && echo "${GPUS%% *}" || claim_gpu )
    trap "rmdir $CLAIM/gpu$g 2>/dev/null" EXIT
    rm -f "$K/SMOKE_GATE"
    run_one "$g" "$SMOKE_JSON" "$K/smoke" --dataset-index 0 --dataset-index 1
    [ "$DRY" = 1 ] && exit 0
    o=$(ls -d "$K"/smoke/gpu*_attempt_* | grep -v 'log$' | tail -1)
    r=$(find "$o" -name result.json | wc -l); t=$(grep -c Traceback "$o.log"); e=$(grep -c 'EMPTY terminal answer' "$o.log")
    lp=$(grep -c '^\[LATPLAN\] patched' "$o.log"); pt=$(grep -c '^\[PlanTargets\]' "$o.log")
    if [ "$r" -eq 2 ] && [ "$t" -eq 0 ] && [ "$e" -eq 0 ] && [ "$lp" -ge 2 ] && [ "$pt" -eq 0 ]; then v=PASS; else v=FAIL; fi
    echo "$v results=$r traceback=$t empty=$e latplan_patched=$lp plan_targets=$pt log=$o.log" | tee "$K/SMOKE_GATE"
    ;;
  run)
    if [ "$DRY" != 1 ] && ! grep -q ^PASS "$K/SMOKE_GATE" 2>/dev/null; then
      echo "no PASS smoke gate at $K/SMOKE_GATE -- run: bash $0 smoke"; exit 1
    fi
    pids=""
    for ((s = 0; s < NSHARD; s++)); do
      if [ "$DRY" = 1 ]; then shard "$s"; else shard "$s" & pids="$pids $!"; sleep 3; fi
    done
    [ -n "$pids" ] && wait $pids
    for spec in $SPECS; do log "TOTAL ${spec%%:*}: $(done_ids "${spec%%:*}" | wc -l)/${spec##*:}"; done
    ;;
  status)
    for spec in $SPECS; do echo "${spec%%:*}: $(done_ids "${spec%%:*}" | wc -l)/${spec##*:}"; done
    ;;
  *) sed -n '2,20p' "$0"; exit 2 ;;
esac
