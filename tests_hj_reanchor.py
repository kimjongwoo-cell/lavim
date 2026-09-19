"""MRoPE re-anchor: off must be byte-identical, on must preserve grid layout."""
import os, sys
sys.path.insert(0, "/home/users/whddn12316/wsi_latent_0915_decode_hj")
import torch
from backbone.qwen3vl import Qwen3VLBackbone as B

ok = 0
def check(name, cond):
    global ok
    assert cond, f"FAIL {name}"
    ok += 1

# a 2-image block: image A 2x3 grid, image B 2x2 grid, then anchored at 100
t = torch.tensor
old = torch.stack([
    t([100,100,100,100,100,100, 103,103,103,103]),          # temporal
    t([100,100,100,101,101,101, 103,103,104,104]),          # height
    t([100,101,102,100,101,102, 103,104,103,104]),          # width
])
n = old.shape[-1]
dev = torch.device("cpu")

os.environ.pop("VLMAS_KV_RESTAGE_MROPE", None)
off, off_cur = B._reanchor_positions(old, 500, dev)
check("off is 1-D run", torch.equal(off[0], torch.arange(500, 500+n)))
check("off all axes equal", torch.equal(off[0], off[1]) and torch.equal(off[1], off[2]))
check("off cursor = +n", off_cur == 500 + n)

os.environ["VLMAS_KV_RESTAGE_MROPE"] = "1"
on, on_cur = B._reanchor_positions(old, 500, dev)
check("on min lands on cursor", int(on.min()) == 500)
check("on preserves relative structure", torch.equal(on - on.min(), old - old.min()))
check("on keeps axes distinct", not torch.equal(on[1], on[2]))
check("on cursor = compressed span", on_cur == int(on.max()) + 1)
check("on cursor << token count", on_cur - 500 < n)
# within-image width ordering intact
check("within-image w order", torch.equal(on[2][:3], t([500,501,502]) + (old[2][0]-old.min())))

os.environ["VLMAS_KV_RESTAGE_MROPE"] = "shuffle"
sh, sh_cur = B._reanchor_positions(old, 500, dev)
check("shuffle same temporal axis", torch.equal(sh[0], on[0]))
check("shuffle same h multiset", torch.equal(sh[1].sort().values, on[1].sort().values))
check("shuffle breaks layout", not torch.equal(sh[1], on[1]) or not torch.equal(sh[2], on[2]))
check("shuffle deterministic", torch.equal(B._reanchor_positions(old, 500, dev)[0], sh))

os.environ.pop("VLMAS_KV_RESTAGE_MROPE", None)
print(f"REANCHOR UNIT PASS: {ok}/{ok} — off byte-identical, on preserves (t,h,w) grid, "
      f"cursor {n} -> {on_cur-500} compressed")
