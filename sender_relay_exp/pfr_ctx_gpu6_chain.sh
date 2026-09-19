#!/bin/bash
# [C2 구조 검증] Context-Updated Answerer Query Visual Re-Read — matched-context specificity.
# Notion Experiment Log 3dbd771b-c9e2-815a-97ba-d6ee42ced077. GPU 6 only.
#
# Setting = patch-12 baseline (sb_sdpa_base.sh flags: sdpa · VLMAS_NAV2=1 · patch 12 · mlen 12288 · step 10).
# Receiver site l* = 27 (VLMAS_ANSWERER_PFR_LAYER=27). Arms (memory/pfr_attention.py):
#   A record   native output, r_C + JS stats dumped (donor bank for D)
#   B identity same-query re-read (implementation identity control)
#   C 1        matched context-updated query (main)
#   D shuffled donor r_C from a FIXED in-dataset derangement (runs/pfr_ctx/perm_<ds>.json, seed 0,
#              written before any run), relative-progress call alignment
# 1) smoke on smoke/gtex2.json (A→B→C→D) with gates; 2) dcs20 5 datasets × 20, each A→B→C→D.
# Re-running resumes: indices that already have result.json under the arm are skipped.
set -u
G=6
REPO=/home/users/whddn12316/wsi_latent_0915_decode_hj
DATA=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
MODEL=/home/users/whddn12316/models/Qwen3-VL-4B-Thinking
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
K=$REPO/sender_relay_exp/runs/pfr_ctx
CLAIM=$REPO/sender_relay_exp/runs/vgain
LOG=$K/chain.log
mkdir -p $K $CLAIM
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG; }
mem() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' -v g=$G '$1==g{print $2}'; }

log "WAIT gpu$G"
while true; do
  u=$(mem); if [ -n "$u" ] && [ "$u" -lt 2000 ] && mkdir $CLAIM/claim_gpu$G 2>/dev/null; then break; fi
  sleep 60
done
trap 'rmdir $CLAIM/claim_gpu$G 2>/dev/null' EXIT
log "CLAIM gpu$G pfr_md5=$(md5sum $REPO/memory/pfr_attention.py | cut -c1-8) engine_md5=$(md5sum $REPO/vision_text_mas/latent_qwen_engine.py | cut -c1-8)"

# run_arm <tag> <dataset-json> <arm-letter> <perm-json>
run_arm() {
  local tag=$1 dsjson=$2 arm=$3 perm=$4
  local base=$K/$tag/$arm
  mkdir -p $K/$tag
  local done_ids pending ids=()
  done_ids=$(find $K/$tag -path "*/${arm}_attempt_*" -name result.json 2>/dev/null -exec $PY -c \
    'import json,sys; [print(json.load(open(f))["dataset_index"]) for f in sys.argv[1:]]' {} + | sort -n -u)
  local n; n=$($PY -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' $dsjson)
  for ((i=0; i<n; i++)); do grep -qx "$i" <<< "$done_ids" || ids+=(--dataset-index "$i"); done
  if [ ${#ids[@]} -eq 0 ]; then log "SKIP $tag/$arm complete"; return 0; fi
  local mode
  case $arm in A) mode=record;; B) mode=identity;; C) mode=1;; D) mode=shuffled;; esac
  local stamp; stamp=$(date +%Y%m%d_%H%M%S)
  local out=$K/$tag/${arm}_attempt_$stamp
  log "START $tag/$arm mode=$mode pending=$(( ${#ids[@]} / 2 ))/$n -> $(basename $out)"
  cd $REPO && env \
    CUDA_VISIBLE_DEVICES=$G PYTHONPATH=$REPO PYTHONUNBUFFERED=1 \
    VLMAS_ATTN_IMPLEMENTATION=sdpa VLMAS_NAV2=1 \
    VLMAS_ANSWERER_PFR=$mode VLMAS_ANSWERER_PFR_LAYER=27 \
    VLMAS_ANSWERER_PFR_DUMP=$base.dump \
    VLMAS_ANSWERER_PFR_DONOR_DIR=$K/$tag/A.dump VLMAS_ANSWERER_PFR_PERM=$perm \
    timeout 21600 $PY -m wsi_latentmas.pipeline.latent_mas \
    --variant base --backbone qwen3-vl --dataset "$dsjson" --slide-root "$DATA/slides" \
    --model "$MODEL" --output-root "$out" --device cuda:0 \
    --latent-steps 10 --patch-budget 12 --max-model-len 12288 --navigator-control-tokens 512 \
    --temperature 0.0 --top-p 1.0 --seed 42 --deterministic --answerer-greedy --no-answerer-thinking \
    --answerer-max-new-tokens 512 --answerer-rationale --answerer-protocol structured_json \
    --canonical-open-options --navigator-kv --io-pipeline --no-save-navigation-pngs --case-retries 0 \
    "${ids[@]}" > "$out.log" 2>&1 < /dev/null
  local rc=$?
  local res; res=$(find $K/$tag -path "*/${arm}_attempt_*" -name result.json | wc -l)
  log "DONE $tag/$arm rc=$rc results=$res/$n pfrctx=$(grep -c '^\[PFRctx\]' $out.log) dumps=$(ls $base.dump/case_*.pt 2>/dev/null | wc -l) tb=$(grep -c Traceback $out.log) oom=$(grep -ci 'out of memory' $out.log)"
  [ "$(grep -c Traceback $out.log)" -eq 0 ] && [ "$res" -eq "$n" ]
}

# ---------------------------------------------------------------- smoke
SM=$REPO/sender_relay_exp/smoke/gtex2.json
ok=1
for arm in A B C D; do run_arm smoke $SM $arm $K/perm_smoke.json || { ok=0; break; }; done
GATE=$($PY - $K/smoke <<'PYG'
import json, glob, sys, os
root = sys.argv[1]
def rows(arm):
    out = {}
    for f in glob.glob(f"{root}/{arm}_attempt_*.log"):
        for line in open(f, errors="ignore"):
            if line.startswith("[PFRctx] "):
                r = json.loads(line[9:]); out[r["case"]] = r
    return out
def answers(arm):
    out = {}
    for f in glob.glob(f"{root}/{arm}_attempt_*/*/result.json"):
        d = json.load(open(f)); out[d["dataset_index"]] = str(d["answer"].get("answer", ""))
    return out
A, B, C, D = rows("A"), rows("B"), rows("C"), rows("D")
msgs = []
for name, r in (("A", A), ("B", B), ("C", C), ("D", D)):
    if len(r) != 2: msgs.append(f"{name} PFRctx rows {len(r)}!=2")
    if any((x.get("n_vis") or 0) <= 0 for x in r.values()): msgs.append(f"{name} n_vis<=0")
if B and max(x["js_nat_id_max"] for x in B.values()) > 1e-4: msgs.append(f"B js_nat_id_max {max(x['js_nat_id_max'] for x in B.values()):.2e} > 1e-4")
if A and B and answers("A") != answers("B"): msgs.append(f"B answers {answers('B')} != A {answers('A')}")
if C and not all(x["js_nat_m_mean"] > x["js_nat_id_max"] for x in C.values()): msgs.append("C js_nat_m_mean <= js_nat_id_max")
if D and not all(x.get("donor") is not None and x.get("js_m_s_mean") is not None for x in D.values()): msgs.append("D donor/js_m_s missing")
detail = " | ".join(f"{n}:" + ",".join(f"c{c} nvis={x['n_vis']} idmax={x['js_nat_id_max']:.1e} m={x['js_nat_m_mean']:.3e} s={x.get('js_nat_s_mean')} ms={x.get('js_m_s_mean')} ratio={x.get('rnorm_ratio_mean')} maxdiff={x.get('maxdiff_max')}" for c, x in sorted(r.items())) for n, r in (("A", A), ("B", B), ("C", C), ("D", D)))
print(("PASS " if not msgs else "FAIL " + "; ".join(msgs) + " ") + "|| " + detail + " || answers A=" + str(answers("A")) + " B=" + str(answers("B")) + " C=" + str(answers("C")) + " D=" + str(answers("D")))
PYG
)
log "SMOKE $GATE"
if [ $ok -ne 1 ] || [[ "$GATE" != PASS* ]]; then log "smoke gate failed — full run not launched"; exit 1; fi

# ---------------------------------------------------------------- full 5 × 20
for ds in gtex tcga_expert_vqa tcga_slidebench tcga panda; do
  for arm in A B C D; do
    run_arm $ds $REPO/sender_relay_exp/smoke/dcs20_$ds.json $arm $K/perm_$ds.json || log "WARN $ds/$arm incomplete (continuing)"
  done
done
log "ALL DONE"
