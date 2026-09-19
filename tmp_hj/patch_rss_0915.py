"""Wire memory/rss_diag.py into backbone/qwen3vl.py (Reasoner latent loop) and the engine (terminal decode).

md5-guarded, backups *.bak_0915_rss, py_compile before atomic replace. Env-gated (VLMAS_RSS); off = no-op.
"""
import hashlib
import os
import py_compile
import shutil
import sys

REPO = "/home/users/whddn12316/wsi_latent_0915_decode_hj/"
BB = REPO + "backbone/qwen3vl.py"
EN = REPO + "vision_text_mas/latent_qwen_engine.py"
EXPECT = {BB: "c53d10357f71900a83f27d912a2247bc", EN: "fd24ef8bd03a231be5f5173428a450c5"}


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def replace_atomic(path, text):
    tmp = path + ".rss_tmp.py"
    with open(tmp, "w") as fh:
        fh.write(text)
    py_compile.compile(tmp, doraise=True)
    shutil.copy2(path, path + ".bak_0915_rss")
    os.replace(tmp, path)


for path, want in EXPECT.items():
    got = md5(path)
    if got != want:
        sys.exit(f"ABORT md5 mismatch {path}: {got} != {want}")

# ---------------------------------------------------------------- backbone
src = open(BB).read()
if "rss_diag" in src:
    sys.exit("ABORT backbone already patched")
start_line = "        cur_vis   = (past_len + vis_idx)   # ABSOLUTE vision cols into the cache\n"
assert src.count(start_line) == 1, "cur_vis anchor not unique"
i0 = src.index(start_line)
loop_head = ("        probe_mass_steps: list[float] = []\n\n"
             "        for step in range(m):\n"
             "            le = self._apply_realign(last_hidden)\n")
i1 = src.index(loop_head, i0)
assert i1 - i0 < 1500, f"loop head too far from anchor ({i1 - i0})"
new_head = ("        probe_mass_steps: list[float] = []\n\n"
            "        # Efficient Role-Support Sensitivity (memory/rss_diag.py, env VLMAS_RSS): closed-form\n"
            "        # support-deletion proxy at the last latent step + state for the Answerer-side LOO.\n"
            "        # Off = no-op.\n"
            "        _rss = None\n"
            "        if os.environ.get(\"VLMAS_RSS\", \"\").strip() and getattr(self, \"_prune_morphology_enabled\", False):\n"
            "            from memory import rss_diag as _rssd\n"
            "            _rss = _rssd.begin(self, cur_vis=cur_vis, cursor=cursor, cur_len=cur_len,\n"
            "                               last_hidden=last_hidden, m=m, thw=thw)\n\n"
            "        for step in range(m):\n"
            "            if _rss is not None:\n"
            "                _rss.before_step(step)\n"
            "            le = self._apply_realign(last_hidden)\n")
src = src[:i1] + new_head + src[i1 + len(loop_head):]
after_anchor = ("            past_kv = o.past_key_values\n"
                "            last_hidden = o.hidden_states[-1][:, -1:, :]\n")
i2 = src.index(after_anchor, i1)
end_marker = "        latent_to_vision_attn    = {li: torch.stack(v, 0) for li, v in l2v_lists.items() if v}\n"
i3 = src.index(end_marker, i1)
assert i2 < i3, "after-step anchor not inside the loop"
src = src[:i2 + len(after_anchor)] + (
    "            if _rss is not None:\n"
    "                _rss.after_step(step, o, cur_vis)\n") + src[i2 + len(after_anchor):]
replace_atomic(BB, src)

# ---------------------------------------------------------------- engine
es = open(EN).read()
if "rss_diag" in es:
    sys.exit("ABORT engine already patched")
anchor = ("            if _cd_cache is not cache:\n"
          "                del _cd_cache\n"
          "        if is_terminal and os.environ.get(\"VLMAS_RMVR\", \"\").strip():\n")
assert es.count(anchor) == 1, "engine anchor not unique"
block = ("            if _cd_cache is not cache:\n"
         "                del _cd_cache\n"
         "        if is_terminal and os.environ.get(\"VLMAS_RSS\", \"\").strip():\n"
         "            # Efficient Role-Support Sensitivity (memory/rss_diag.py): validation-only\n"
         "            # leave-one-support-out replays of the Reasoner latent suffix on copies of the\n"
         "            # cache, after the normal decode. Off = unchanged.\n"
         "            from memory import rss_diag as _rssd\n"
         "            _rssd.run(self, cache=cache, probe_base_len=probe_base_len,\n"
         "                      position_cursor=position_cursor, system_prompt=system_prompt,\n"
         "                      user_prompt=user_prompt, json_prefix=json_prefix,\n"
         "                      case_index=getattr(self, \"_rpath_case_index\", -1), normal_output=output)\n"
         "        if is_terminal and os.environ.get(\"VLMAS_RMVR\", \"\").strip():\n")
es = es.replace(anchor, block)
replace_atomic(EN, es)
print("OK backbone", md5(BB), "engine", md5(EN))
