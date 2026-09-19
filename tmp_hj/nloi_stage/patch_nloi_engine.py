"""Insert the VLMAS_NLOI block into latent_qwen_engine.py (md5-guarded, backup kept)."""
import hashlib, shutil, sys, time
P = "/home/users/whddn12316/wsi_latent_0915_decode_hj/vision_text_mas/latent_qwen_engine.py"
EXPECT = "71db1738ca8932e978bf4a2a756c2cbc"
src = open(P, "rb").read()
md5 = hashlib.md5(src).hexdigest()
if md5 != EXPECT:
    sys.exit(f"md5 mismatch {md5} != {EXPECT}; engine changed, not patching")
s = src.decode()
anchor = '''        if is_terminal and os.environ.get("VLMAS_RPATH_PATCH", "").strip():
            from memory import rpath_patch as _rpatch'''
assert s.count(anchor) == 1
block = '''        if is_terminal and os.environ.get("VLMAS_NLOI", "").strip():
            # C2 structural diagnosis (memory/nloi_diag.py): exact single / pair
            # observation value-cut forwards at the Answerer, final-state interaction
            # Psi_ij and complementarity C_ij, true vs token-shuffled grouping.
            from memory import nloi_diag as _nloi
            _nl_cache = cache
            if int(self._backbone._kv_len(cache)) > probe_base_len:
                from copy import deepcopy as _dc11
                _nl_cache = _dc11(cache)
                _nl_cache.crop(probe_base_len)
            _nloi.run(self, cache=_nl_cache, position_cursor=position_cursor,
                      system_prompt=system_prompt, user_prompt=user_prompt,
                      json_prefix=json_prefix,
                      case_index=getattr(self, "_rpath_case_index", -1),
                      max_new_tokens=max_new_tokens)
            if _nl_cache is not cache:
                del _nl_cache
'''
s = s.replace(anchor, block + anchor)
bk = f"/home/users/whddn12316/wsi_latent_0915_decode_hj/tmp_hj/nloi_stage/latent_qwen_engine.py.orig_{EXPECT[:8]}_{time.strftime('%m%d_%H%M')}"
shutil.copy2(P, bk)
open(P, "w").write(s)
print("patched", hashlib.md5(open(P, "rb").read()).hexdigest(), "backup", bk)
