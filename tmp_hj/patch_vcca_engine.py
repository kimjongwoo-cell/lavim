"""Wire the C2 Visual-to-Candidate Contrast Alignment diagnosis into
vision_text_mas/latent_qwen_engine.py.

Adds one opt-in block (VLMAS_VCCA=<jsonl>) right after the CMODE block, using the same
pre-Answerer cache crop. Every other path is untouched; unset env = no-op.
Atomic, md5-guarded, py_compile before replace.
usage: python patch_vcca_engine.py vision_text_mas/latent_qwen_engine.py
"""
import hashlib
import os
import py_compile
import sys
import tempfile

EXPECT = "0da6f661dd7c08c11f720fb45c8a49ba"
path = sys.argv[1]
raw = open(path, "rb").read()
before = hashlib.md5(raw).hexdigest()
text = raw.decode("utf-8")
if "VLMAS_VCCA" in text:
    sys.exit("already patched")
if before != EXPECT:
    sys.exit(f"md5 {before} != expected {EXPECT}; file changed upstream, aborted")

ANCHOR = ("            if _cm_cache is not cache:\n"
          "                del _cm_cache\n"
          '        if is_terminal and os.environ.get("VLMAS_RPATH_PATCH", "").strip():\n')
TAIL = '        if is_terminal and os.environ.get("VLMAS_RPATH_PATCH", "").strip():\n'
BLOCK = '''        if is_terminal and os.environ.get("VLMAS_VCCA", "").strip():
            # C2 structural diagnosis (memory/vcca_diag.py): at every candidate-trie
            # branching node, how much of the full-vs-zeroA visual delta lands in the
            # candidate-contrast subspace, before and after the final norm. Off = unchanged.
            from memory import vcca_diag as _vcca
            _vc_cache = cache
            if int(self._backbone._kv_len(cache)) > probe_base_len:
                from copy import deepcopy as _dc4
                _vc_cache = _dc4(cache)
                _vc_cache.crop(probe_base_len)
            _vcca.run(self, cache=_vc_cache, position_cursor=position_cursor,
                      system_prompt=system_prompt, user_prompt=user_prompt,
                      json_prefix=json_prefix,
                      candidates=getattr(self, "_ver_candidates", None) or (),
                      case_index=getattr(self, "_rpath_case_index", -1))
            if _vc_cache is not cache:
                del _vc_cache
'''
if text.count(ANCHOR) != 1:
    sys.exit(f"anchor count {text.count(ANCHOR)} != 1")
head = ANCHOR[: -len(TAIL)]
text = text.replace(ANCHOR, head + BLOCK + TAIL, 1)

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
