#!/bin/bash
# SlideBench Latent base — 기준선으로 채택한 ExpertVQA·GTEx·TCGA·PANDA 와 같은 세팅으로 재실행.
#
# 기존 SlideBench Base(0912/runs/tcga_slidebench_base_12patch_*)는 grid navigator 였고
# 그 폴더를 만든 스크립트가 없다. 나머지 네 셋은 sdpa · VLMAS_NAV2=1 · patch 12 이므로
# 이 스크립트는 그 플래그를 그대로 쓴다(sender_relay_exp/nav2v2_mine_run.sh 와 25개 플래그 동일).
# VLMAS_PROMPT_SET 은 설정하지 않는다 — 코드(latent_agent_prompts.prompt2_enabled)는 "prompt2"
# 일 때만 달라지므로 네 셋이 쓴 prompt1/미설정과 같은 프롬프트다.
#
# attention 설정은 로그·run_manifest 어디에도 남지 않는다. 그래서 이 스크립트는 실행 시점의
# env 를 run_meta.json 으로 따로 남긴다.
#
# usage: sb_sdpa_base.sh <gpu> [first last]
#   first/last : dataset-index 범위(포함). 생략하면 197 전체.
#   같은 범위를 다시 실행하면 이미 result.json 이 있는 index 는 건너뛴다(이어 돌리기).
#   CLI 가 --output-root 로 새 폴더만 받으므로 실행마다 DEST/attempt_<시각> 을 새로 쓰고,
#   run_meta 는 그 옆 DEST/attempt_<시각>.meta.json 에 둔다 (09-14 01:31 첫 실행이 폴더에 meta 를 먼저 써서 rc=2).
set -u
GPU=${1:?usage: sb_sdpa_base.sh GPU [FIRST LAST]}
FIRST=${2:-0}; LAST=${3:-196}

REPO=/home/users/whddn12316/wsi_latent_0915_decode_hj
DATA=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
MODEL=/home/users/whddn12316/models/Qwen3-VL-4B-Thinking
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
ROOT=$REPO/sender_relay_exp/runs/sb_sdpa_base
DEST=$ROOT/part$(printf '%03d' $FIRST)_$(printf '%03d' $LAST)
STAMP=$(date +%Y%m%d_%H%M%S)
ATT=$DEST/attempt_$STAMP
mkdir -p "$DEST"

export VLMAS_ATTN_IMPLEMENTATION=sdpa
export VLMAS_NAV2=1

# 이어 돌리기: 이미 끝난 index 는 빼고 넘긴다
DONE_IDS=$(find "$DEST" -name result.json 2>/dev/null -exec "$PY" -c \
  'import json,sys; [print(json.load(open(f))["dataset_index"]) for f in sys.argv[1:]]' {} + | sort -n -u)
IDS=()
for ((i=FIRST; i<=LAST; i++)); do
  grep -qx "$i" <<< "$DONE_IDS" || IDS+=(--dataset-index "$i")
done
if [ ${#IDS[@]} -eq 0 ]; then
  echo "$(date +%H:%M) part$FIRST-$LAST already complete" >> $ROOT/chain.log; exit 0
fi

"$PY" - "$ATT.meta.json" <<PYMETA
import json, os, sys, datetime
json.dump({
  "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
  "script": "sender_relay_exp/sb_sdpa_base.sh", "gpu": "$GPU",
  "range": [$FIRST, $LAST], "pending_cases": ${#IDS[@]} // 2,
  "env": {k: os.environ.get(k) for k in ("VLMAS_ATTN_IMPLEMENTATION", "VLMAS_NAV2", "VLMAS_PROMPT_SET")},
  "flags": "--variant base --patch-budget 12 --max-model-len 12288 --latent-steps 10 "
           "--answerer-protocol structured_json --navigator-kv --io-pipeline --deterministic --seed 42",
}, open(sys.argv[1], "w"), indent=1)
PYMETA

cd $REPO && env \
  CUDA_VISIBLE_DEVICES="$GPU" \
  PYTHONPATH="$REPO" \
  PYTHONUNBUFFERED=1 \
  "$PY" -m wsi_latentmas.pipeline.latent_mas \
  --variant base \
  --backbone qwen3-vl \
  --dataset "$DATA/tcga_slidebench.json" \
  --slide-root "$DATA/slides" \
  --model "$MODEL" \
  --output-root "$ATT" \
  --device cuda:0 \
  --latent-steps 10 \
  --patch-budget 12 \
  --max-model-len 12288 \
  --navigator-control-tokens 512 \
  --temperature 0.0 \
  --top-p 1.0 \
  --seed 42 \
  --deterministic \
  --answerer-greedy \
  --no-answerer-thinking \
  --answerer-max-new-tokens 512 \
  --answerer-rationale \
  --answerer-protocol structured_json \
  --canonical-open-options \
  --navigator-kv \
  --io-pipeline \
  --no-save-navigation-pngs \
  --case-retries 0 \
  "${IDS[@]}" >> "$ATT.log" 2>&1
rc=$?
echo "$(date +%H:%M) DONE part$FIRST-$LAST gpu=$GPU rc=$rc results=$(find $DEST -name result.json | wc -l)/$((LAST-FIRST+1)) tb=$(grep -c Traceback $ATT.log) oom=$(grep -c 'out of memory' $ATT.log)" >> $ROOT/chain.log
