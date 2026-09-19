"""Wire memory/qasc_select.py (C1 #14) into the Reasoner prefill of backbone/qwen3vl.py.

md5-guarded (expects the post-RSVMH file), backup *.bak_0915_qasc, py_compile before atomic replace.
Off (VLMAS_C1_QASC unset) = byte-identical behaviour.
"""
import hashlib
import os
import py_compile
import shutil
import sys

BB = "/home/users/whddn12316/wsi_latent_0915_decode_hj/backbone/qwen3vl.py"
EXPECT = "884d289e71dc5e936264f7b0ed882d67"

got = hashlib.md5(open(BB, "rb").read()).hexdigest()
if got != EXPECT:
    sys.exit(f"ABORT md5 mismatch: {got} != {EXPECT}")
src = open(BB).read()
if "qasc_select" in src:
    sys.exit("ABORT already patched")

anchor = ('        self._prefill_prune_survivor_image_ids = None\n'
          '        if hierarchy_prefill and nV > 0:\n'
          '            img_feat, keep_visual = self._hierarchy_pruned_vision_features(\n')
assert src.count(anchor) == 1, "prefill anchor"
block = ('        self._prefill_prune_survivor_image_ids = None\n'
         '        # C1 #14 Question-Agnostic Salience-Coverage Log-Det Selection (memory/qasc_select.py,\n'
         '        # env VLMAS_C1_QASC=1, Reasoner stage): full vision tower, joint x5/x20 log-det greedy on\n'
         '        # merged tokens; retained tokens alone enter the decoder prefill. Off = unchanged.\n'
         '        _qasc_prefill = bool(\n'
         '            os.environ.get("VLMAS_C1_QASC", "") == "1"\n'
         '            and getattr(self, "_prune_morphology_enabled", False)\n'
         '        )\n'
         '        if (hierarchy_prefill or _qasc_prefill) and nV > 0:\n'
         '            if _qasc_prefill:\n'
         '                from memory import qasc_select as _qasc\n'
         '            img_feat, keep_visual = _qasc.select_vision_features(self, pv, thw) if _qasc_prefill else self._hierarchy_pruned_vision_features(\n')
src = src.replace(anchor, block)
tmp = BB + ".qasc_tmp.py"
open(tmp, "w").write(src)
py_compile.compile(tmp, doraise=True)
shutil.copy2(BB, BB + ".bak_0915_qasc")
os.replace(tmp, BB)
print("OK", hashlib.md5(open(BB, "rb").read()).hexdigest())
