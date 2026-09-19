#!/bin/bash
# Hypothesis test for the canon_diag finding "post-hoc re-rotation of cached visual keys changes 1/4-1/3 of answers":
# is the damage from the INCONSISTENCY (keys re-rotated, values / deeper hidden states / later tokens computed at the
# old positions), or is the model simply sensitive to where the visual block sits?
# Same geometry, two ways (memory/canon_diag.py gap<G> vs backbone VLMAS_POS_GAP=<G>):
#   posthoc      native nav3 run; at the Answerer boundary the cache copy has columns 0..last-visual re-rotated by -G
#                (arms native, gap512, gap2048, shift)  -> runs/posgap/posthoc/<ds>
#   consistent   VLMAS_POS_GAP=2048: G empty MRoPE positions inserted right after the Reasoner's last visual token at
#                prefill, so the Reasoner tail, the latent steps and the Answerer are computed with that geometry
#                (arm native only)                         -> runs/posgap/consistent2048/<ds>
# nav3 · sdpa · patch 12 · mlen 12288 · step 10; 5 sets x 20 (dcs20 subsets mapped to full-set indices, ids_<ds>.txt),
# split by list position % 3 over GPU 6/7/8. Waits for a free GPU (claim dir shared with the other chains).
# usage: posgap_nav3.sh <gpu> <shard 0|1|2> <smoke|wait>
set -u
G=${1:?gpu}; SHARD=${2:?shard}; ROLE=${3:?smoke|wait}
REPO=/home/users/whddn12316/wsi_latent_0915_decode_hj
DATA=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
MODEL=/home/users/whddn12316/models/Qwen3-VL-4B-Thinking
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
K=$REPO/sender_relay_exp/runs/posgap
CLAIM=$REPO/sender_relay_exp/runs/vgain
LOG=$K/chain_gpu$G.log
GAP=2048
POSTHOC_ARMS=native,gap512,gap2048,shift
mkdir -p $K $CLAIM
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG; }
mem() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' -v g=$G '$1==g{print $2}'; }

if [ "$ROLE" = wait ]; then
  log "WAIT smoke gate"
  until [ -f $K/smoke_gate.txt ]; do sleep 30; done
  grep -q ^PASS $K/smoke_gate.txt || { log "smoke gate not PASS — exit"; exit 1; }
fi
log "WAIT gpu$G"
until u=$(mem); [ -n "$u" ] && [ "$u" -lt 2000 ] && mkdir $CLAIM/claim_gpu$G 2>/dev/null; do sleep 60; done
trap 'rmdir $CLAIM/claim_gpu$G 2>/dev/null' EXIT
log "CLAIM gpu$G shard=$SHARD backbone_md5=$(md5sum $REPO/backbone/qwen3vl.py | cut -c1-8) diag_md5=$(md5sum $REPO/memory/canon_diag.py | cut -c1-8) engine_md5=$(md5sum $REPO/vision_text_mas/latent_qwen_engine.py | cut -c1-8) nav3_md5=$(md5sum $REPO/vision_text_mas/onepass_navigator.py | cut -c1-8)"

# run <tag> <mode posthoc|consistent> <dataset-json> <outdir> <ids...>
run() {
  local tag=$1 mode=$2 dsjson=$3 dest=$4; shift 4
  local stamp; stamp=$(date +%Y%m%d_%H%M%S)
  local out=$dest/gpu${G}_attempt_$stamp
  local gapenv="" arms=$POSTHOC_ARMS
  if [ "$mode" = consistent ]; then gapenv="VLMAS_POS_GAP=$GAP"; arms=native; fi
  mkdir -p $dest
  log "START $tag mode=$mode pending=$(( $# / 2 )) -> ${out#$K/}"
  cd $REPO && env \
    CUDA_VISIBLE_DEVICES=$G PYTHONPATH=$REPO PYTHONUNBUFFERED=1 \
    VLMAS_PROMPT_SET=prompt1 VLMAS_ATTN_IMPLEMENTATION=sdpa VLMAS_NAV2=1 VLMAS_NAV3_INDEPENDENT=1 \
    VLMAS_CROSS_SCALE_ROUTER=0 \
    VLMAS_KV_ROUTE=1 VLMAS_KV_ROUTE_MODE=bookkeep \
    VLMAS_CANON_DIAG=$dest/diag_gpu$G.jsonl VLMAS_CANON_DIAG_ARMS=$arms $gapenv \
    $PY -m wsi_latentmas.pipeline.latent_mas \
    --variant base --backbone qwen3-vl --dataset "$dsjson" --slide-root "$DATA/slides" \
    --model "$MODEL" --output-root "$out" --device cuda:0 \
    --latent-steps 10 --patch-budget 12 --max-model-len 12288 --navigator-control-tokens 512 \
    --temperature 0.0 --top-p 1.0 --seed 42 --deterministic --answerer-greedy --no-answerer-thinking \
    --answerer-max-new-tokens 512 --answerer-rationale --answerer-protocol structured_json \
    --canonical-open-options --navigator-kv --io-pipeline --no-save-navigation-pngs --case-retries 0 \
    "$@" > "$out.log" 2>&1 < /dev/null
  local rc=$?
  local diag; diag=$(grep -c '^\[CanonDiag\] case' $out.log)
  local gapl; gapl=$(grep -c '^\[PosGap\] reasoner' $out.log)
  local res; res=$(find $out -name result.json | wc -l)
  log "DONE $tag mode=$mode rc=$rc results=$res diag=$diag posgap_lines=$gapl tb=$(grep -c Traceback $out.log) oom=$(grep -ci 'out of memory' $out.log)"
  echo "$res $diag $gapl $(grep -c Traceback $out.log)"
}

if [ "$ROLE" = smoke ]; then
  SM=$REPO/sender_relay_exp/smoke/gtex2.json
  read r1 d1 g1 t1 <<< "$(run smoke_posthoc posthoc $SM $K/smoke/posthoc --dataset-index 0 --dataset-index 1)"
  read r2 d2 g2 t2 <<< "$(run smoke_rerun posthoc $SM $K/smoke/rerun --dataset-index 0 --dataset-index 1)"
  read r3 d3 g3 t3 <<< "$(run smoke_consistent consistent $SM $K/smoke/consistent --dataset-index 0 --dataset-index 1)"
  # determinism floor: two separate identical runs must give the same native choice log-probs and answers
  det=$($PY - $K/smoke <<'PY'
import json, sys, glob
root = sys.argv[1]
def load(sub):
    rows = {}
    for f in glob.glob(f"{root}/{sub}/diag_gpu*.jsonl"):
        for line in open(f):
            r = json.loads(line); rows[r["case"]] = r
    return rows
a, b, c = load("posthoc"), load("rerun"), load("consistent")
ok = len(a) == 2 and len(b) == 2 and len(c) == 2
dmax = 0.0; same = 0
for k in a:
    if k not in b: ok = False; continue
    la, lb = a[k]["arms"]["native"]["logp"], b[k]["arms"]["native"]["logp"]
    dmax = max([dmax] + [abs(x - y) for x, y in zip(la, lb) if x is not None and y is not None])
    same += a[k]["arms"]["native"]["answer"] == b[k]["arms"]["native"]["answer"]
gaps = all(r.get("pos_gap") == 2048 for r in c.values()) and all(r.get("pos_gap") == 0 for r in a.values())
arms = all(set(r["arms"]) == {"native", "gap512", "gap2048", "shift"} for r in a.values())
print(("PASS" if ok and same == 2 and dmax < 1e-3 and gaps and arms else "FAIL") + f" rerun_logp_maxdiff={dmax:.2e} same_answer={same}/2 pos_gap_rec={gaps} arms={arms}")
PY
)
  if [ "$r1" -eq 2 ] && [ "$d1" -eq 2 ] && [ "$g1" -eq 0 ] && [ "$t1" -eq 0 ] && [ "$r2" -eq 2 ] && [ "$t2" -eq 0 ] \
     && [ "$r3" -eq 2 ] && [ "$d3" -eq 2 ] && [ "$g3" -eq 2 ] && [ "$t3" -eq 0 ] && [[ "$det" == PASS* ]]; then
    echo "PASS $det" > $K/smoke_gate.txt; log "SMOKE PASS $det"
  else
    echo "FAIL posthoc=$r1/$d1/$g1/$t1 rerun=$r2/$t2 consistent=$r3/$d3/$g3/$t3 $det" > $K/smoke_gate.txt
    log "SMOKE FAIL posthoc=$r1/$d1/$g1/$t1 rerun=$r2/$t2 consistent=$r3/$d3/$g3/$t3 $det"; exit 1
  fi
fi

for ds in tcga_expert_vqa tcga_slidebench gtex tcga panda; do
  mapfile -t all < $K/ids_$ds.txt
  for mode in posthoc consistent; do
    sub=$([ $mode = posthoc ] && echo posthoc || echo consistent$GAP)
    done_ids=$(find $K/$sub/$ds -name result.json 2>/dev/null -exec $PY -c \
      'import json,sys; [print(json.load(open(f))["dataset_index"]) for f in sys.argv[1:]]' {} + | sort -n -u)
    ids=()
    for j in "${!all[@]}"; do
      [ $((j % 3)) -eq "$SHARD" ] || continue
      grep -qx "${all[$j]}" <<< "$done_ids" || ids+=(--dataset-index "${all[$j]}")
    done
    if [ ${#ids[@]} -eq 0 ]; then log "SKIP $ds $mode shard $SHARD complete"; continue; fi
    run $ds $mode $DATA/$ds.json $K/$sub/$ds "${ids[@]}" > /dev/null
  done
done
log "ALL DONE gpu$G shard $SHARD"
