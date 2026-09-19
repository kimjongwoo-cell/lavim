"""C2 kill test readout: wrong-visual flip rate per latent allocation.
usage: c2ra_flip.py <tag>   (tag in A10, R5A5)  -- true = c2ra_<tag>_gtex10.s*, wrong = rpath_w_c2ra_<tag>"""
import glob, json, sys
from pathlib import Path
Q = Path("/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs/expq")
def answers(pattern):
    out = {}
    for f in glob.glob(str(Q / pattern / "*" / "result.json")):
        ci = int(Path(f).parent.name.split("_")[0]); a = json.load(open(f)).get("answer")
        out[ci] = ((a.get("answer") if isinstance(a, dict) else a) or "").strip()
    return out
def corr(pattern):
    out = {}
    for f in glob.glob(str(Q / pattern / "*" / "result.json")):
        ci = int(Path(f).parent.name.split("_")[0]); d = json.load(open(f))
        out[ci] = d.get("correct", d.get("is_correct"))
    return out
tag = sys.argv[1]
true = answers(f"c2ra_{tag}_gtex10.s*"); wrong = answers(f"rpath_w_c2ra_{tag}")
common = sorted(c for c in wrong if c in true and c != 0)     # case 0 = donor (own visual)
flips = [c for c in common if true[c] != wrong[c]]
print(f"[{tag}] wrong-visual flips {len(flips)}/{len(common)} = {100*len(flips)/max(1,len(common)):.0f}%  (baseline R10/A0: 17/37 = 46%)")
ct = corr(f"c2ra_{tag}_gtex10.s*"); cw = corr(f"rpath_w_c2ra_{tag}")
print(f"   acc on the 0/5 subset: true {sum(1 for c in common if ct.get(c))}/{len(common)}  wrong {sum(1 for c in common if cw.get(c))}/{len(common)}")
