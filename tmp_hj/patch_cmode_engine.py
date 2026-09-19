"""Wire the C2 cross-dataset common-mode diagnosis into vision_text_mas/latent_qwen_engine.py.

Adds one opt-in block (VLMAS_CMODE=<jsonl>) right after the igsweep kill-test block, using the
same pre-Answerer cache crop. Every other path is untouched; unset env = no-op.
Atomic, md5-guarded, py_compile before replace.
usage: python patch_cmode_engine.py vision_text_mas/latent_qwen_engine.py
"""
import hashlib
import os
import py_compile
import sys
import tempfile

EXPECT = "fc564dcb2bfcb8a11fca88c40cab9929"
path = sys.argv[1]
raw = open(path, "rb").read()
before = hashlib.md5(raw).hexdigest()
text = raw.decode("utf-8")
if "VLMAS_CMODE" in text:
    sys.exit("already patched")
if before != EXPECT:
    sys.exit(f"md5 {before} != expected {EXPECT}; file changed upstream, aborted")

ANCHOR = ("            if _ig_cache is not cache:\n"
          "                del _ig_cache\n"
          '        if is_terminal and os.environ.get("VLMAS_RPATH_PATCH", "").strip():\n')
BLOCK = '''        if is_terminal and os.environ.get("VLMAS_CMODE", "").strip():
            # C2 structural diagnosis (memory/cmode_diag.py): decision-row hidden state
            # under full vs zeroA on the pre-Answerer cache, for the cross-dataset
            # common-mode / residual decomposition. Off = unchanged.
            from memory import cmode_diag as _cmode
            _cm_cache = cache
            if int(self._backbone._kv_len(cache)) > probe_base_len:
                from copy import deepcopy as _dc3
                _cm_cache = _dc3(cache)
                _cm_cache.crop(probe_base_len)
            _cmode.run(self, cache=_cm_cache, position_cursor=position_cursor,
                       system_prompt=system_prompt, user_prompt=user_prompt,
                       json_prefix=json_prefix,
                       candidates=getattr(self, "_ver_candidates", None) or (),
                       case_index=getattr(self, "_rpath_case_index", -1))
            if _cm_cache is not cache:
                del _cm_cache
'''
if text.count(ANCHOR) != 1:
    sys.exit(f"anchor count {text.count(ANCHOR)} != 1")
head, tail = ANCHOR.rsplit('        if is_terminal and os.environ.get("VLMAS_RPATH_PATCH", "").strip():\n', 1)
text = text.replace(ANCHOR, head + BLOCK + '        if is_terminal and os.environ.get("VLMAS_RPATH_PATCH", "").strip():\n', 1)
fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)), suffix=".py.tmp")
with os.fdopen(fd, "w", encoding="utf-8") as fh:
    fh.write(text)
py_compile.compile(tmp, doraise=True)
if hashlib.md5(open(path, "rb").read()).hexdigest() != before:
    os.unlink(tmp)
    sys.exit("file changed during patch; aborted")
os.chmod(tmp, os.stat(path).st_mode)
os.replace(tmp, path)
print(f"patched {path}\n  before {before}\n  after  {hashlib.md5(open(path, 'rb').read()).hexdigest()}")
