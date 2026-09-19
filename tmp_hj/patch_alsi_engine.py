"""Add the VLMAS_ALSI block to latent_qwen_engine.py (Answerer latent source interference).

md5-guarded, no-op if already present, atomic write with a backup into tmp_hj/.
usage: python patch_alsi_engine.py <expected_md5>
"""
import hashlib
import os
import shutil
import sys

ROOT = "/home/users/whddn12316/wsi_latent_0915_decode_hj"
PATH = os.path.join(ROOT, "vision_text_mas/latent_qwen_engine.py")

ANCHOR = '''            if _oc_cache is not cache:
                del _oc_cache
'''

BLOCK = '''        if is_terminal and os.environ.get("VLMAS_ALSI", "").strip():
            # C2 structural diagnosis (memory/alsi_diag.py): decompose the Answerer
            # attention output into persistent visual KV / Reasoner latent / static text
            # / generated prefix, propagate each source through the native suffix, and
            # measure visual-vs-nonvisual latent interference. Measurement only.
            from memory import alsi_diag as _alsi
            _al_cache = cache
            if int(self._backbone._kv_len(cache)) > probe_base_len:
                from copy import deepcopy as _dc8
                _al_cache = _dc8(cache)
                _al_cache.crop(probe_base_len)
            _alsi.run(self, cache=_al_cache, position_cursor=position_cursor,
                      system_prompt=system_prompt, user_prompt=user_prompt,
                      json_prefix=json_prefix,
                      case_index=getattr(self, "_rpath_case_index", -1),
                      max_new_tokens=max_new_tokens)
            if _al_cache is not cache:
                del _al_cache
'''


def main(expected):
    src = open(PATH).read()
    md5 = hashlib.md5(src.encode()).hexdigest()
    if "VLMAS_ALSI" in src:
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
