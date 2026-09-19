#!/bin/bash
# [C2 causal test] Answerer Visual-Path Gain Intervention on the nav3 setting, GPU 6 only.
# Notion Experiment Log 3dad771b-c9e2-81e7-aaee-d38619006f85. Probe: memory/vgain_diag.py (VLMAS_VGAIN).
# Env = nav3 base / nav3+PSAR runs of 09-14 (VLMAS_NAV3_INDEPENDENT=1 VLMAS_NAV2=1 VLMAS_PROMPT_SET=prompt1 sdpa
# VLMAS_CROSS_SCALE_ROUTER=0), 4B · patch 12 · step 10 · max-model-len 12288, no C1/PSAR.
# 1) claim GPU 6 (runs/vgain/claim_gpu6)  2) gtex2 smoke + gate (tb 0, results 2, markers 2, rows 2,
#    native parse == result.json answer, candidate tokens == native decode tokens (lead_tok_ok), native candidate
#    log-prob top == native answer, split vs joint pass argmax same and gap <= 1.0 logit (bf16 lm_head: |z| 16-32
#    quantum 0.125; 0.05 was below the bf16 floor), native vs ident argmax same, V0 vs visual_cut zero <= 0.01,
#    W arms ran on case 2)
# 3) dcs20 five datasets (5x20) one after another. Output runs/vgain_nav3/.
set -u
TREE=/home/users/whddn12316/wsi_latent_0915_decode_hj
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
MP=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
SM=$TREE/sender_relay_exp/smoke
K=$TREE/sender_relay_exp/runs/vgain_nav3
CLAIM=$TREE/sender_relay_exp/runs/vgain
G=6
mkdir -p $K $CLAIM
LOG=$K/chain.log
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG; }
mem() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' -v g=$1 '$1==g{print $2}'; }

if ! mkdir $CLAIM/claim_gpu$G 2>/dev/null; then log "ABORT: claim_gpu$G exists"; exit 1; fi
trap 'rmdir $CLAIM/claim_gpu$G 2>/dev/null' EXIT
while true; do u=$(mem $G); [ -n "$u" ] && [ "$u" -lt 2000 ] && break; log "gpu$G busy (${u}MiB)"; sleep 60; done
log "CLAIM gpu$G nav3_md5=$(md5sum $TREE/vision_text_mas/onepass_navigation_nav3.py | cut -c1-8) vgain_md5=$(md5sum $TREE/memory/vgain_diag.py | cut -c1-8)"

ENVS="VLMAS_NAV3_INDEPENDENT=1 VLMAS_NAV2=1 VLMAS_PROMPT_SET=prompt1 VLMAS_ATTN_IMPLEMENTATION=sdpa VLMAS_CROSS_SCALE_ROUTER=0"

run () {   # $1 name  $2 dataset json (relative to smoke/)
  local NAME=$1 DSJ=$2
  local STAMP=$(date +%Y%m%d_%H%M%S)
  local OUT=$K/${NAME}_$STAMP JL=$K/vgain_${NAME}_$STAMP.jsonl
  env $ENVS "$PY" - "$OUT.meta.json" "$SM/$DSJ" "$JL" "$(md5sum $TREE/vision_text_mas/onepass_navigation_nav3.py | cut -d' ' -f1)" <<'PYMETA'
import datetime, json, os, sys
json.dump({"started_at": datetime.datetime.now().isoformat(timespec="seconds"), "script": "sender_relay_exp/vgain_nav3_gpu6.sh",
           "gpu": "6", "dataset": sys.argv[2], "jsonl": sys.argv[3], "nav3_md5": sys.argv[4],
           "env": {k: os.environ[k] for k in sorted(os.environ) if k.startswith("VLMAS_")}}, open(sys.argv[1], "w"), indent=1)
PYMETA
  log "START $NAME -> $(basename $OUT)"
  cd $TREE && env CUDA_VISIBLE_DEVICES=$G PYTHONPATH=$TREE PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $ENVS VLMAS_VGAIN=$JL timeout 43200 \
    $PY -m wsi_latentmas.pipeline.latent_mas --variant base --backbone qwen3-vl --dataset $SM/$DSJ \
    --slide-root $MP/slides --model /home/users/whddn12316/models/Qwen3-VL-4B-Thinking --device cuda:0 \
    --latent-steps 10 --patch-budget 12 --max-model-len 12288 --navigator-control-tokens 512 \
    --temperature 0.0 --top-p 1.0 --seed 42 --deterministic --answerer-greedy --no-answerer-thinking \
    --answerer-max-new-tokens 512 --answerer-rationale --answerer-protocol structured_json \
    --canonical-open-options --navigator-kv --io-pipeline --no-save-navigation-pngs --case-retries 0 \
    --output-root $OUT > $OUT.log 2>&1 < /dev/null
  local rc=$?
  LAST_OUT=$OUT; LAST_JL=$JL
  log "DONE $NAME rc=$rc rows=$(wc -l < $JL 2>/dev/null || echo 0) markers=$(grep -c '^\[VGain\] case [0-9]* slide' $OUT.log) skips=$(grep -c '^\[VGain\] case .*SKIP' $OUT.log) tb=$(grep -c Traceback $OUT.log) oom=$(grep -ci 'out of memory' $OUT.log) results=$(find $OUT -name result.json 2>/dev/null | wc -l)"
}

# ---- smoke + gate
run smoke gtex2.json
"$PY" - "$LAST_OUT" "$LAST_JL" >> $LOG 2>&1 <<'PYGATE'
import glob, json, os, sys
out, jl = sys.argv[1], sys.argv[2]
log = open(out + ".log").read() if os.path.exists(out + ".log") else ""
rows = [json.loads(l) for l in open(jl)] if os.path.exists(jl) else []
res = {}
for f in glob.glob(os.path.join(out, "*", "result.json")):
    d = json.load(open(f)); a = d.get("answer")
    res[int(d["dataset_index"])] = a.get("answer") if isinstance(a, dict) else a
fails = []
tb = log.count("Traceback"); markers = sum(1 for l in log.splitlines() if l.startswith("[VGain] case") and " slide=" in l)
if tb: fails.append(f"traceback={tb}")
if len(res) != 2: fails.append(f"results={len(res)}")
if markers < 2: fails.append(f"markers={markers}")
if len(rows) < 2: fails.append(f"rows={len(rows)}")
for r in rows:
    c = r["case"]; arms = r.get("arms", {})
    parse_ok = res.get(c) == r.get("native_answer")
    print(f"  case {c}: parse_ok={parse_ok} gate_A={r.get('gate_A')} gate_split={r.get('gate_split')} gate_B={r.get('gate_B')} "
          f"lead_tok_ok={r.get('lead_tok_ok')} n_vis={r.get('n_vis')} donor={r.get('donor_note') or r.get('donor')} sec={r.get('sec_total')}")
    print("    answers: " + " ".join(f"{k}={v.get('answer')!r}" for k, v in arms.items() if "answer" in v))
    nat = arms.get("native", {})
    top = r["cands"][max(range(len(nat["logp"])), key=lambda i: nat["logp"][i])] if nat.get("logp") else None
    print(f"    tail={r.get('tail')!r} native_cand_top={top!r} split_argmax_same={r.get('gate_split_argmax_same')} "
          f"A_argmax_same={r.get('gate_A_argmax_same')} native_logp_top3={sorted(zip(nat.get('logp', []), r['cands']), reverse=True)[:3]}")
    if not parse_ok: fails.append(f"parse_mismatch case{c}")
    if r.get("lead_tok_ok") is not True: fails.append(f"lead_tok_ok case{c}={r.get('lead_tok_ok')}")
    if top != r.get("native_answer"): fails.append(f"native_cand_top case{c}={top!r}!={r.get('native_answer')!r}")
    if r.get("gate_split") is None or r["gate_split"] > 1.0 or not r.get("gate_split_argmax_same"):
        fails.append(f"gate_split case{c}={r.get('gate_split')} argmax_same={r.get('gate_split_argmax_same')}")
    if not r.get("gate_A_argmax_same"): fails.append(f"gate_A argmax case{c}")
    if r.get("gate_B") is None or r["gate_B"] > 0.01: fails.append(f"gate_B case{c}={r.get('gate_B')}")
if rows and all("skip" in v for k, v in rows[-1]["arms"].items() if k.startswith("W")):
    fails.append("W arms never ran (donor)")
print("GATE " + ("PASS" if not fails else "FAIL " + "; ".join(fails)))
PYGATE
if ! tail -1 $LOG | grep -q "GATE PASS"; then log "smoke gate failed — full run not launched"; exit 1; fi

# ---- 5 x 20
for ds in gtex tcga_expert_vqa tcga_slidebench tcga panda; do
  run $ds dcs20_$ds.json
done
log "ALL DONE"
