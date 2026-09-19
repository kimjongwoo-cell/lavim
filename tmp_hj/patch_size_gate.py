"""Strengthen the nav4 size-launcher smoke gate: the Navigator selection must actually parse.

Before: PASS only required that the nav4 prompt was sent. A Navigator that failed every attempt and
fell back to candidate IDs [1..n] still passed. Now every smoke case must have format_repairs == 0
in round_1/navigator_calls.json (4B has 0 on all 207 cases so far).
"""
from pathlib import Path

P = Path("/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/rtask_nav4_size_arm.sh")
s = P.read_text()

helper = r'''navrep() {  # cases whose Navigator needed a format repair (0 = every selection parsed first time)
  $PY - "$1" <<'PYEOF'
import glob, json, os, sys
bad = 0
for f in glob.glob(os.path.join(sys.argv[1], "**", "round_1", "navigator_calls.json"), recursive=True):
    j = json.load(open(f)); it = j[0] if isinstance(j, list) else j
    bad += int((it.get("format_repairs") or 0) > 0)
print(bad)
PYEOF
}
'''
anchor = "mem() { nvidia-smi"
assert s.count(anchor) == 1
s = s.replace(anchor, helper + anchor, 1)

old = '''  if [ "$r" -eq 2 ] && [ "$n4" -gt 0 ] && [ "$t" -eq 0 ] && [ "$em" -eq 0 ] \\'''
new = '''  nr=$(navrep $o)
  if [ "$r" -eq 2 ] && [ "$n4" -gt 0 ] && [ "$nr" -eq 0 ] && [ "$t" -eq 0 ] && [ "$em" -eq 0 ] \\'''
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new, 1)
s = s.replace('echo "PASS results=$r nav4=$n4 ', 'echo "PASS results=$r nav4=$n4 nav_repaired=$nr ', 1)
s = s.replace('echo "FAIL results=$r nav4=$n4 ', 'echo "FAIL results=$r nav4=$n4 nav_repaired=$nr ', 1)
s = s.replace(' empty=$(grep -c \'EMPTY terminal answer\' $out.log) tb=',
              ' nav_repaired=$(navrep $out) empty=$(grep -c \'EMPTY terminal answer\' $out.log) tb=', 1)
s = s.replace('model=$(basename $MODEL)', 'model=$(basename $MODEL) planner_task=${VLMAS_PLANNER_TASK:-default}', 1)
P.write_text(s)
print("gate patched")
