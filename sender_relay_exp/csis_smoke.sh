#!/bin/bash
# CSIS 2케이스 스모크 — 게이트: [WSIConsol] csis 마커 2 · result.json 2 · traceback 0.
# 플래그는 PCRIS v2 전체 케이스 런과 동일(eager · step 10 · budget 8 · mlen 8192 · B=512).
# usage: csis_smoke.sh <gpu> [sigma]
set -u
GPU=$1; SIGMA=${2:-}
TREE=/home/users/whddn12316/wsi_latent_0915_decode_hj
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
MP=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
K=$TREE/sender_relay_exp/runs/csis
NAME=smoke${SIGMA:+_sig$SIGMA}
mkdir -p $K; rm -rf $K/$NAME $K/${NAME}_dump

cd $TREE && env VLMAS_EFF_DUMP=1 VLMAS_ATTN_IMPLEMENTATION=eager \
  VLMAS_WSI_CONSOL=1 VLMAS_WSI_CONSOL_MODE=csis VLMAS_WSI_CONSOL_BUDGET=512 \
  ${SIGMA:+VLMAS_CSIS_SIGMA=$SIGMA} VLMAS_CSIS_DUMP=$K/${NAME}_dump \
  CUDA_LAUNCH_BLOCKING=1 \
  PYTHONPATH=$TREE PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$GPU \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout 3600 \
  $PY -m wsi_latentmas.pipeline.latent_mas --variant base --backbone qwen3-vl \
  --dataset $TREE/sender_relay_exp/smoke/gtex2.json --slide-root $MP/slides \
  --model /home/users/whddn12316/models/Qwen3-VL-4B-Thinking --device cuda:0 \
  --latent-steps 10 --patch-budget 8 --max-model-len 8192 --navigator-control-tokens 512 \
  --temperature 0.0 --top-p 1.0 --seed 42 --answerer-max-new-tokens 512 \
  --terminal-no-repeat-ngram-size 0 --terminal-repetition-penalty 1.0 --terminal-antiloop-method none \
  --answerer-protocol structured_json --transport-mode cumulative --realign-method wa --case-retries 0 \
  --deterministic --answerer-greedy --no-answerer-thinking --canonical-open-options --navigator-kv \
  --io-pipeline --answerer-rationale --no-save-navigation-pngs \
  --output-root $K/$NAME > $K/$NAME.log 2>&1 < /dev/null
rc=$?

marks=$(grep -c 'WSIConsol. csis' $K/$NAME.log)
res=$(find $K/$NAME -name result.json 2>/dev/null | wc -l)
tb=$(grep -c Traceback $K/$NAME.log)
oom=$(grep -c 'out of memory' $K/$NAME.log)
echo "=== CSIS smoke rc=$rc gpu=$GPU sigma=${SIGMA:-default}"
echo "  marker    : $marks (>=2 필요)"
echo "  result    : $res (==2 필요)"
echo "  traceback : $tb (==0 필요)   oom=$oom"
grep -o '\[WSIConsol\] csis .*' $K/$NAME.log | head -2
echo "--- 답 ---"
for f in $(find $K/$NAME -name result.json 2>/dev/null | sort); do
  $PY -c "
import json; d=json.load(open('$f'))
a=d.get('answer'); a=a.get('answer') if isinstance(a,dict) else a
print('   ', d.get('slide_id'), '| pred:', repr(a)[:40], '| gold:', repr(d.get('gold_answer'))[:30])"
done
if [ "$marks" -ge 2 ] && [ "$res" -eq 2 ] && [ "$tb" -eq 0 ]; then echo "GATE PASS"; else echo "GATE FAIL"; fi
