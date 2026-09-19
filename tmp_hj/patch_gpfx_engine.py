"""Add the VLMAS_GPFX block to latent_qwen_engine.py (prefix share of the answer margin).
md5-guarded, no-op if already present, atomic write with a backup into tmp_hj/.
usage: python patch_gpfx_engine.py <expected_md5>
"""
import hashlib
import os
import shutil
import sys

ROOT = "/home/users/whddn12316/wsi_latent_0915_decode_hj"
PATH = os.path.join(ROOT, "vision_text_mas/latent_qwen_engine.py")

ANCHOR = '''            if _ln_cache is not cache:
                del _ln_cache
'''

BLOCK = '''        if is_terminal and os.environ.get("VLMAS_GPFX", "").strip():
            # How much of the answer margin g is the JSON prefill worth?
            # (memory/gpfx_diag.py) Same generated tokens, teacher-forced under four
            # prompt conditions: full / no-prefix / prefix-only / cache-only.
            from memory import gpfx_diag as _gpfx
            _gp_cache = cache
            if int(self._backbone._kv_len(cache)) > probe_base_len:
                from copy import deepcopy as _dc10
                _gp_cache = _dc10(cache)
                _gp_cache.crop(probe_base_len)
            _gpfx.run(self, cache=_gp_cache, position_cursor=position_cursor,
                      system_prompt=system_prompt, user_prompt=user_prompt,
                      json_prefix=json_prefix,
                      case_index=getattr(self, "_rpath_case_index", -1),
                      max_new_tokens=max_new_tokens)
            if _gp_cache is not cache:
                del _gp_cache
'''


def main(expected):
    src = open(PATH).read()
    md5 = hashlib.md5(src.encode()).hexdigest()
    if "VLMAS_GPFX" in src:
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
