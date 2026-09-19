"""Make a 3-shard, full-196, resume-aware PANDA grade-rule launcher from the 20-case one."""
from pathlib import Path

D = Path("/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp")
s = (D / "rtask_panda_graderule.sh").read_text()

s = s.replace(
    "SHARD=0; NSHARD=1; GPUS=${1:?gpus}; HGPU=; HPID=",
    "SHARD=${1:?shard}; NSHARD=${2:?nshard}; GPUS=${3:?gpus}; HGPU=; HPID=", 1)
s = s.replace(
    "CLAIM=/home/users/whddn12316/wsi_latent_0902_2155_hj/sender_relay_exp/runs/vgain",
    "CLAIM=${CLAIMDIR:-/home/users/whddn12316/wsi_latent_0902_2155_hj/sender_relay_exp/runs/vgain}", 1)
s = s.replace("LOG=$K/chain_shard$SHARD.log", "LOG=$K/chain_gr_shard$SHARD.log", 1)

old = 'ids=(); for ((i=0; i<196; i+=10)); do ids+=(--dataset-index "$i"); done\nrun panda $DATA/panda.json $K/panda "${ids[@]}"'
new = '''done_ids=$(find $K/panda -name result.json 2>/dev/null -exec $PY -c \\
  'import json,sys; [print(json.load(open(f))["dataset_index"]) for f in sys.argv[1:]]' {} + | sort -n -u)
ids=()
for ((i=0; i<196; i++)); do
  [ $((i % NSHARD)) -eq "$SHARD" ] || continue
  grep -qx "$i" <<< "$done_ids" || ids+=(--dataset-index "$i")
done
log "RESUME shard $SHARD/$NSHARD pending=${#ids[@]} already_done=$(wc -w <<< "$done_ids")"
if [ ${#ids[@]} -eq 0 ]; then log "SKIP shard $SHARD complete"; exit 0; fi
run panda $DATA/panda.json $K/panda "${ids[@]}"
log "ALL DONE shard $SHARD"'''
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new, 1)
s = s.replace("PANDA 20 cases (index%10)", "PANDA 196 cases, 3 shards, resume-aware", 1)

out = D / "rtask_panda_gr_full.sh"
out.write_text(s)
print("written", out)
