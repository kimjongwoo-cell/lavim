"""CPU tests for tmp_hj/lpvh_ntrs_hooks. Run from the repo root: python tmp_hj/lpvh_ntrs_hooks/test_lpvh_ntrs_hooks.py"""
import os
import subprocess
import sys
from pathlib import Path

REPO = Path("/home/users/whddn12316/wsi_latent_0915_decode_hj")
HERE = Path(__file__).resolve().parent
PY = sys.executable
PASS = 0


def check(name, cond, detail=""):
    global PASS
    if not cond:
        raise AssertionError(f"{name} {detail}")
    PASS += 1
    print("ok", name, detail)


def run(code, env_extra):
    env = {k: v for k, v in os.environ.items() if k not in ("VLMAS_NAV3_X5_CLAMP", "VLMAS_C1_META_BRIDGE")}
    env.update({"PYTHONPATH": f"{HERE}:{REPO}", "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", **env_extra})
    r = subprocess.run([PY, "-c", code], cwd=REPO, env=env, capture_output=True, text=True, timeout=900)
    return r.returncode, r.stdout, r.stderr


IMPORT = ("import memory.qasc_select as q, vision_text_mas.onepass_navigation_roots as r, nav_probe_dummy\n"
          .replace(", nav_probe_dummy", ""))
PROBE = IMPORT + ("print('BRIDGE', getattr(q.select_vision_features, '_c1_meta_bridge', False))\n"
                  "print('CLAMP', r._best_x5_child.__name__ if hasattr(r, '_best_x5_child') else None)\n")

# 1. hooks off: nothing patched
rc, out, err = run(PROBE, {})
check("off: import ok", rc == 0, err[-400:])
check("off: qasc not wrapped", "BRIDGE False" in out, out)
check("off: no [C1META]/[NAV3-X5CLAMP] lines", "[C1META]" not in out and "[NAV3-X5CLAMP]" not in out, out)

# 2. bridge only
rc, out, err = run(PROBE, {"VLMAS_C1_META_BRIDGE": "1"})
check("bridge: import ok", rc == 0, err[-400:])
check("bridge: qasc wrapped", "BRIDGE True" in out and "[C1META] active" in out, out)
check("bridge: clamp not active", "[NAV3-X5CLAMP] active" not in out, out)

# 3. both
rc, out, err = run(PROBE, {"VLMAS_C1_META_BRIDGE": "1", "VLMAS_NAV3_X5_CLAMP": "1"})
check("both: import ok", rc == 0, err[-400:])
check("both: bridge + clamp active", "BRIDGE True" in out and "[NAV3-X5CLAMP] active" in out, out)

# 4. bridge logic against the real backbone _record_visual_meta on a stub backbone
LOGIC = r'''
import types, torch
from backbone import qwen3vl as qv
import c1_meta_bridge as cb
cls = [c for c in vars(qv).values() if isinstance(c, type) and hasattr(c, "_record_visual_meta")][0]
def stub():
    s = types.SimpleNamespace(_prune_morphology_enabled=True,
        model=types.SimpleNamespace(config=types.SimpleNamespace(vision_config=types.SimpleNamespace(spatial_merge_size=2))))
    s._record_visual_meta = types.MethodType(cls._record_visual_meta, s)
    return s
grids = torch.tensor([[1, 32, 32]] * 12)
g = torch.Generator().manual_seed(0)
keep = torch.zeros(3072, dtype=torch.bool); keep[torch.randperm(3072, generator=g)[:768]] = True
cols = torch.arange(500, 1268)
feats = torch.randn(768, 8)
fake = types.SimpleNamespace(select_vision_features=lambda bb, pv, thw: (feats, keep))
# native (no bridge): mismatch
s0 = stub(); s0._record_visual_meta(grids, None, cols)
print("NATIVE", s0._visual_meta is None, s0._visual_meta_note)
# bridged
print("PATCH", cb.patch(fake), cb.patch(fake))
s = stub()
f2, k2 = fake.select_vision_features(s, None, grids)
print("PASSTHRU", f2 is feats, bool(torch.equal(k2, keep)))
s._record_visual_meta(grids, None, cols)
m = s._visual_meta
sel = keep.nonzero(as_tuple=True)[0]
obs = torch.repeat_interleave(torch.arange(12), torch.tensor([256] * 12))
print("META", m is not None and bool(torch.equal(m["abs_cols"], cols)) and bool(torch.equal(m["orig_index"], sel))
      and bool(torch.equal(m["obs_id"], obs[sel])) and m["c1"] is True and sum(m["kept_counts"]) == 768)
print("CONSUMED", s._c1_prefill_keep is None)
# next call without a new selection: bridge must not reuse the old mask (no-C1 3072 cols -> ok, c1 False)
s._record_visual_meta(grids, None, torch.arange(10, 3082))
print("NOREUSE", s._visual_meta is not None and s._visual_meta["c1"] is False)
# count mismatch: mask not applied, meta none, still consumed
fake.select_vision_features(s, None, grids)
s._record_visual_meta(grids, None, torch.arange(500, 1200))
print("MISMATCH", s._visual_meta is None and s._c1_prefill_keep is None)
# explicit keep from consolidation wins over the pending mask
fake.select_vision_features(s, None, grids)
keep2 = torch.zeros(3072, dtype=torch.bool); keep2[:768] = True
s._record_visual_meta(grids, keep2, cols)
print("EXPLICIT", s._visual_meta is not None and bool(torch.equal(s._visual_meta["orig_index"], torch.arange(768))))
# bridge installed once on the instance
print("ONCE", s._c1_meta_bridge_installed is True and s._record_visual_meta.__name__ == "_record_visual_meta")
'''
rc, out, err = run(LOGIC, {})
check("logic: ran", rc == 0, err[-800:])
check("native path without bridge = mismatch (the bug)", "NATIVE True token/grid mismatch (768 vs 3072)" in out, out)
check("patch idempotent", "PATCH True False" in out, out)
check("wrapper passes features/mask through", "PASSTHRU True True" in out, out)
check("bridged meta = keep.nonzero order, obs ids, c1", "META True" in out, out)
check("mask consumed", "CONSUMED True" in out, out)
check("no reuse on next call", "NOREUSE True" in out, out)
check("count mismatch -> not applied", "MISMATCH True" in out, out)
check("explicit keep wins", "EXPLICIT True" in out, out)
check("installed once", "ONCE True" in out, out)
check("[C1META] log line", "[C1META] keep 768/3072 cols=768 used=True meta=ok" in out, out)
print(f"{PASS}/{PASS} passed")
