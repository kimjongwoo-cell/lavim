"""Add the VLMAS_RMVR block to latent_qwen_engine.py, right after the VCCA block.

Guarded: applies only when the file still hashes to the expected md5 (the engine is a
shared file that other sessions edit too), is a no-op if the block is already there, and
writes atomically after backing the original up under tmp_hj/.
usage: python patch_rmvr_engine.py <expected_md5>
"""
import hashlib
import os
import shutil
import sys

ROOT = "/home/users/whddn12316/wsi_latent_0915_decode_hj"
PATH = os.path.join(ROOT, "vision_text_mas/latent_qwen_engine.py")

ANCHOR = '''            _vcca.run(self, cache=_vc_cache, position_cursor=position_cursor,
                      system_prompt=system_prompt, user_prompt=user_prompt,
                      json_prefix=json_prefix,
                      candidates=getattr(self, "_ver_candidates", None) or (),
                      case_index=getattr(self, "_rpath_case_index", -1))
            if _vc_cache is not cache:
                del _vc_cache
'''

BLOCK = '''        if is_terminal and os.environ.get("VLMAS_RMVR", "").strip():
            # C2 structural diagnosis (memory/rmvr_diag.py): at every generated answer
            # step, the receiver competition margin of the zeroA arm vs the differential
            # visual push of full-vs-zeroA, over the whole vocabulary (candidate trie
            # kept as a closed-set control). Off = unchanged.
            from memory import rmvr_diag as _rmvr
            _rm_cache = cache
            if int(self._backbone._kv_len(cache)) > probe_base_len:
                from copy import deepcopy as _dc5
                _rm_cache = _dc5(cache)
                _rm_cache.crop(probe_base_len)
            _rmvr.run(self, cache=_rm_cache, position_cursor=position_cursor,
                      system_prompt=system_prompt, user_prompt=user_prompt,
                      json_prefix=json_prefix,
                      candidates=getattr(self, "_ver_candidates", None) or (),
                      case_index=getattr(self, "_rpath_case_index", -1),
                      normal_output=output, max_new_tokens=max_new_tokens)
            if _rm_cache is not cache:
                del _rm_cache
'''


def main(expected):
    src = open(PATH).read()
    md5 = hashlib.md5(src.encode()).hexdigest()
    if "VLMAS_RMVR" in src:
        print(f"ALREADY_PRESENT md5={md5}")
        return 0
    if md5 != expected:
        print(f"GUARD_FAIL md5={md5} expected={expected}")
        return 3
    if src.count(ANCHOR) != 1:
        print(f"ANCHOR_FAIL count={src.count(ANCHOR)}")
        return 4
    out = src.replace(ANCHOR, ANCHOR + BLOCK)
    shutil.copyfile(PATH, os.path.join(ROOT, f"tmp_hj/latent_qwen_engine.py.orig_{md5[:8]}"))
    tmp = PATH + ".new"
    with open(tmp, "w") as fh:
        fh.write(out)
    os.replace(tmp, PATH)
    print(f"PATCHED {md5} -> {hashlib.md5(out.encode()).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
