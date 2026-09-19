"""Add the two OFH blocks to latent_qwen_engine.py (decode-time arm + density probe).

Guarded: applies only when the file still hashes to the expected md5 (shared file), is a
no-op if the blocks are already there, backs the original up and writes atomically.
usage: python patch_ofh_engine.py <expected_md5>
"""
import hashlib
import os
import shutil
import sys

ROOT = "/home/users/whddn12316/wsi_latent_0915_decode_hj"
PATH = os.path.join(ROOT, "vision_text_mas/latent_qwen_engine.py")

ANCHOR_ARM = '''            else:
                print("[GroupC2] SKIP: visual bookkeeping missing/mismatch", flush=True)
'''

BLOCK_ARM = '''        if is_terminal and os.environ.get("VLMAS_OFH", "").strip():
            # C2 structure test (memory/ofh.py): Observation-Factorized Latent Handoff —
            # the same LME competition as GroupC2, but the partition is the Navigator
            # observation (crop) with shuffled / random-group falsifiers. Native total
            # visual mass is preserved by the "c2" rule. Off = unchanged.
            from memory import group_attn as _ga4
            from memory import ofh as _ofh
            _om = os.environ.get("VLMAS_OFH", "true").strip()
            _og = _ofh.observation_groups(self, _om)
            if _og is not None:
                _dev = self._backbone.device
                self._gc2_groups = (_og[0].to(_dev), [g.to(_dev) for g in _og[1]])
                _gc2_ctx = _ga4.install_group_attention(
                    self._backbone, self._gc2_groups[1], self._gc2_groups[0], "c2")
                print(f"[OFH] mode={_om} obs={[int(g.numel()) for g in self._gc2_groups[1]]} "
                      f"n_vis={int(self._gc2_groups[0].numel())}", flush=True)
            else:
                print(f"[OFH] mode={_om} SKIP: visual bookkeeping missing/mismatch", flush=True)
'''

ANCHOR_DIAG = '''            if _rm_cache is not cache:
                del _rm_cache
'''

BLOCK_DIAG = '''        if is_terminal and os.environ.get("VLMAS_OFH_DIAG", "").strip():
            # C2 structure test (memory/ofh.py): within-observation token-density
            # sensitivity of the native readout vs the observation-factorized one
            # (duplicate / subsample falsifiers). Off = unchanged.
            from memory import ofh as _ofh2
            _of_cache = cache
            if int(self._backbone._kv_len(cache)) > probe_base_len:
                from copy import deepcopy as _dc6
                _of_cache = _dc6(cache)
                _of_cache.crop(probe_base_len)
            _ofh2.run(self, cache=_of_cache, position_cursor=position_cursor,
                      system_prompt=system_prompt, user_prompt=user_prompt,
                      json_prefix=json_prefix,
                      case_index=getattr(self, "_rpath_case_index", -1),
                      max_new_tokens=max_new_tokens)
            if _of_cache is not cache:
                del _of_cache
'''


def main(expected):
    src = open(PATH).read()
    md5 = hashlib.md5(src.encode()).hexdigest()
    if "VLMAS_OFH" in src:
        print(f"ALREADY_PRESENT md5={md5}")
        return 0
    if md5 != expected:
        print(f"GUARD_FAIL md5={md5} expected={expected}")
        return 3
    for name, anchor in (("arm", ANCHOR_ARM), ("diag", ANCHOR_DIAG)):
        if src.count(anchor) != 1:
            print(f"ANCHOR_FAIL {name} count={src.count(anchor)}")
            return 4
    out = src.replace(ANCHOR_ARM, ANCHOR_ARM + BLOCK_ARM).replace(ANCHOR_DIAG, ANCHOR_DIAG + BLOCK_DIAG)
    shutil.copyfile(PATH, os.path.join(ROOT, f"tmp_hj/latent_qwen_engine.py.orig_{md5[:8]}"))
    tmp = PATH + ".new"
    with open(tmp, "w") as fh:
        fh.write(out)
    os.replace(tmp, PATH)
    print(f"PATCHED {md5} -> {hashlib.md5(out.encode()).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
