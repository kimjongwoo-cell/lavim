"""Wire memory/rsvmh.py into backbone/qwen3vl.py: probe gate + restage (non-park) support reordering.

md5-guarded (expects the post-RSS file), backup *.bak_0915_rsvmh, py_compile before atomic replace.
Off (VLMAS_KV_RESTAGE_ORDER not an RSVMH mode) = byte-identical behaviour.
"""
import hashlib
import os
import py_compile
import shutil
import sys

BB = "/home/users/whddn12316/wsi_latent_0915_decode_hj/backbone/qwen3vl.py"
EXPECT = "c51d14f2867aad4c346cc546a1583847"

got = hashlib.md5(open(BB, "rb").read()).hexdigest()
if got != EXPECT:
    sys.exit(f"ABORT md5 mismatch: {got} != {EXPECT}")
src = open(BB).read()
if "rsvmh" in src:
    sys.exit("ABORT already patched")

old_gate = ('        if os.environ.get("VLMAS_RSS", "").strip() and getattr(self, "_prune_morphology_enabled", False):\n'
            '            from memory import rss_diag as _rssd\n')
assert src.count(old_gate) == 1, "probe gate anchor"
new_gate = ('        if (os.environ.get("VLMAS_RSS", "").strip()\n'
            '                or os.environ.get("VLMAS_KV_RESTAGE_ORDER", "").strip()\n'
            '                in ("sensitivity", "sensitivity_rev", "mass", "size", "page_shuffle")) \\\n'
            '                and getattr(self, "_prune_morphology_enabled", False):\n'
            '            from memory import rss_diag as _rssd\n')
src = src.replace(old_gate, new_gate)

anchor = ('        old_pos = old_positions.to(device=self.device, dtype=torch.long)  # [3, n]\n'
          '        source_note = ""\n'
          '        if wrong_slide:\n')
assert src.count(anchor) == 1, "restage anchor"
block = ('        old_pos = old_positions.to(device=self.device, dtype=torch.long)  # [3, n]\n'
         '        source_note = ""\n'
         '        # RSVMH (memory/rsvmh.py, env VLMAS_KV_RESTAGE_ORDER in rsvmh.ORDERS): place the SAME\n'
         '        # retained supports in ascending order of the Reasoner\'s closed-form support-deletion\n'
         '        # sensitivity (or a control score) before the recency move. Off = unchanged.\n'
         '        _rsvmh_counts = None\n'
         '        _rsvmh_order = os.environ.get("VLMAS_KV_RESTAGE_ORDER", "").strip()\n'
         '        if _rsvmh_order and not identity and not wrong_slide:\n'
         '            from memory import rsvmh as _rsvmh\n'
         '            if _rsvmh.enabled_order(_rsvmh_order):\n'
         '                _seed = int(getattr(getattr(self, "_rss_ctx", None), "pre_len", 0) or 0)\n'
         '                _perm, _counts, _note = _rsvmh.plan(self, cols, _rsvmh_order, case_seed=_seed)\n'
         '                if _perm is not None:\n'
         '                    cols = cols.index_select(0, _perm)\n'
         '                    old_pos = old_pos.index_select(1, _perm)\n'
         '                    _rsvmh_counts = _counts\n'
         '                print(f"[RSVMH] {_note}" if _perm is not None else f"[RSVMH] SKIP {_note}", flush=True)\n'
         '        if wrong_slide:\n')
src = src.replace(anchor, block)

pos_anchor = ('        if identity:\n'
              '            new_pos, restage_next_cursor = old_pos, position_cursor + n\n'
              '        else:\n'
              '            new_pos, restage_next_cursor = self._reanchor_positions(\n'
              '                old_pos, position_cursor, self.device)\n')
assert src.count(pos_anchor) == 1, "position anchor"
pos_block = ('        if identity:\n'
             '            new_pos, restage_next_cursor = old_pos, position_cursor + n\n'
             '        elif _rsvmh_counts is not None:\n'
             '            from memory.paging import page_anchor_positions\n'
             '            new_pos, restage_next_cursor = page_anchor_positions(\n'
             '                old_pos, _rsvmh_counts, position_cursor)\n'
             '        else:\n'
             '            new_pos, restage_next_cursor = self._reanchor_positions(\n'
             '                old_pos, position_cursor, self.device)\n')
src = src.replace(pos_anchor, pos_block)

tmp = BB + ".rsvmh_tmp.py"
open(tmp, "w").write(src)
py_compile.compile(tmp, doraise=True)
shutil.copy2(BB, BB + ".bak_0915_rsvmh")
os.replace(tmp, BB)
print("OK", hashlib.md5(open(BB, "rb").read()).hexdigest())
