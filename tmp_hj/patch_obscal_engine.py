"""Add the VLMAS_OBSCAL block to latent_qwen_engine.py (observation-level calibration probe).

Guarded by md5 (shared file), no-op if already present, atomic write with a backup.
usage: python patch_obscal_engine.py <expected_md5>
"""
import hashlib
import os
import shutil
import sys

ROOT = "/home/users/whddn12316/wsi_latent_0915_decode_hj"
PATH = os.path.join(ROOT, "vision_text_mas/latent_qwen_engine.py")

ANCHOR = '''            if _of_cache is not cache:
                del _of_cache
'''

BLOCK = '''        if is_terminal and os.environ.get("VLMAS_OBSCAL", "").strip():
            # C2 feasibility probe (memory/obscal_diag.py): per-observation visual write
            # at a fixed receiver layer, its local logit influence, and the receiver
            # competition it would have to clear. Measurement only. Off = unchanged.
            from memory import obscal_diag as _obscal
            _oc_cache = cache
            if int(self._backbone._kv_len(cache)) > probe_base_len:
                from copy import deepcopy as _dc7
                _oc_cache = _dc7(cache)
                _oc_cache.crop(probe_base_len)
            _obscal.run(self, cache=_oc_cache, position_cursor=position_cursor,
                        system_prompt=system_prompt, user_prompt=user_prompt,
                        json_prefix=json_prefix,
                        case_index=getattr(self, "_rpath_case_index", -1),
                        max_new_tokens=max_new_tokens)
            if _oc_cache is not cache:
                del _oc_cache
'''


def main(expected):
    src = open(PATH).read()
    md5 = hashlib.md5(src.encode()).hexdigest()
    if "VLMAS_OBSCAL" in src:
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
