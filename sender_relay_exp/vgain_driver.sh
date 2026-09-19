#!/bin/bash
# VGain driver: wait for a free GPU among 6/7/8 (<2GB twice, 60 s apart, not claimed by a VGain
# worker), run the mandatory 2-case smoke, and only if the gate passes start dataset workers.
# Each worker claims datasets one at a time (mkdir lock), so a second GPU that frees up later
# takes the remaining datasets. Never stacks on a busy GPU. Log: runs/vgain/driver.log
set -u
TREE=/home/users/whddn12316/wsi_latent_0915_decode_hj
R=$TREE/sender_relay_exp
K=$R/runs/vgain
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
mkdir -p $K/queue
LOG=$K/driver.log
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG; }

free_gpu() {   # prints one free, unclaimed GPU or nothing
  for g in 6 7 8; do
    [ -d $K/claim_gpu$g ] && continue
    u=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' -v g=$g '$1==g{print $2}')
    [ -n "$u" ] && [ "$u" -lt 2000 ] || continue
    sleep 60
    u2=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' -v g=$g '$1==g{print $2}')
    [ -d $K/claim_gpu$g ] && continue
    if [ -n "$u2" ] && [ "$u2" -lt 2000 ]; then echo $g; return; fi
  done
}

log "WAIT smoke gpu"
while true; do
  G=$(free_gpu)
  [ -n "$G" ] && break
  sleep 120
done
mkdir $K/claim_gpu$G
log "CLAIM gpu$G for smoke"
bash $R/vgain_chain.sh $G smoke:gtex2.json
OUT=$(ls -d $K/smoke_2* 2>/dev/null | grep -v '\.' | tail -1)
JL=$(ls $K/vgain_smoke_*.jsonl 2>/dev/null | tail -1)
"$PY" - "$OUT" "$JL" >> $LOG 2>&1 <<'PYGATE'
import glob, json, os, sys
out, jl = sys.argv[1], sys.argv[2]
log = open(out + ".log").read() if os.path.exists(out + ".log") else ""
rows = [json.loads(l) for l in open(jl)] if jl and os.path.exists(jl) else []
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
    c = r["case"]
    parse_ok = res.get(c) == r.get("native_answer")
    arms = r.get("arms", {})
    ran = [k for k, v in arms.items() if "skip" not in v]
    skipped = [k for k, v in arms.items() if "skip" in v]
    print(f"  case {c}: parse_ok={parse_ok} (result={res.get(c)!r} native={r.get('native_answer')!r}) "
          f"gate_A={r.get('gate_A')} argmax_same={r.get('gate_A_argmax_same')} gate_split={r.get('gate_split')} "
          f"gate_B={r.get('gate_B')} lead_tok_ok={r.get('lead_tok_ok')} n_vis={r.get('n_vis')} "
          f"donor={r.get('donor_note') or r.get('donor')} ran={len(ran)} skipped={skipped} sec={r.get('sec_total')}")
    print("    answers: " + " ".join(f"{k}={v.get('answer')!r}" for k, v in arms.items() if "answer" in v))
    print("    arm sec: " + " ".join(f"{k}={v.get('sec')}" for k, v in arms.items() if "sec" in v))
    if not parse_ok: fails.append(f"parse_mismatch case{c}")
    if r.get("gate_split") is None or r["gate_split"] > 0.05: fails.append(f"gate_split case{c}={r.get('gate_split')}")
    if r.get("gate_B") is None or r["gate_B"] > 0.01: fails.append(f"gate_B case{c}={r.get('gate_B')}")
    if any(k not in arms for k in ("ident", "V0", "V4", "N4")): fails.append(f"arms_missing case{c}")
if rows and all("skip" in v for k, v in rows[-1]["arms"].items() if k.startswith("W")):
    fails.append("W arms never ran (donor)")
print("GATE " + ("PASS" if not fails else "FAIL " + "; ".join(fails)))
PYGATE
if ! tail -1 $LOG | grep -q "GATE PASS"; then
  log "smoke gate failed — not launching (see above)"; rmdir $K/claim_gpu$G; exit 1
fi

cat > $K/worker.sh <<WORKER
#!/bin/bash
# one GPU: take unclaimed datasets until none are left
set -u
G=\$1
K=$K
trap 'rmdir \$K/claim_gpu\$G 2>/dev/null' EXIT
mkdir -p \$K/claim_gpu\$G
for ds in gtex tcga_expert_vqa tcga_slidebench tcga panda; do
  mkdir \$K/queue/\$ds.lock 2>/dev/null || continue
  echo "\$(date '+%m-%d %H:%M:%S') gpu\$G takes \$ds" >> $LOG
  bash $R/vgain_chain.sh \$G \$ds:dcs20_\$ds.json
done
echo "\$(date '+%m-%d %H:%M:%S') worker gpu\$G finished" >> $LOG
WORKER
rmdir $K/claim_gpu$G
mkdir $K/claim_gpu$G   # hand the claim straight to the worker below
setsid nohup bash $K/worker.sh $G > $K/worker_gpu$G.out 2>&1 < /dev/null &
log "worker started on gpu$G"

# extra GPUs while datasets remain unclaimed
while [ $(ls -d $K/queue/*.lock 2>/dev/null | wc -l) -lt 5 ]; do
  G2=$(free_gpu)
  if [ -n "$G2" ] && [ $(ls -d $K/queue/*.lock 2>/dev/null | wc -l) -lt 5 ]; then
    mkdir $K/claim_gpu$G2 && setsid nohup bash $K/worker.sh $G2 > $K/worker_gpu$G2.out 2>&1 < /dev/null &
    log "worker started on gpu$G2"
  fi
  sleep 120
done
log "all 5 datasets claimed; driver exits"
