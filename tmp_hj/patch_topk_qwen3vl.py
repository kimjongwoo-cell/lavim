"""Add the score top-k pruning arm to backbone/qwen3vl.py (VLMAS_WSI_CONSOL_MODE=topk).

Plain top-k by question->vision salience: the "score pruning" reference for the matched-budget
accuracy comparison. Every other mode untouched. Atomic, md5-guarded, py_compile before replace.
usage: python patch_topk_qwen3vl.py backbone/qwen3vl.py
"""
import hashlib
import os
import py_compile
import sys
import tempfile

EXPECT = "0de380e2bba9a02bb3bf0a61b16eb271"
path = sys.argv[1]
raw = open(path, "rb").read()
before = hashlib.md5(raw).hexdigest()
text = raw.decode("utf-8")
if 'elif mode == "topk"' in text:
    sys.exit("already patched")
if before != EXPECT:
    sys.exit(f"md5 {before} != expected {EXPECT}; file changed upstream, aborted")

HEAD = ("                + f\" attn={os.environ.get('VLMAS_ATTN_IMPLEMENTATION', 'eager')}\"\n"
        "                + f\" control={control or 'none'}\", flush=True)\n")
TAIL = '        elif mode == "strat":\n'
BRANCH = '''        elif mode == "topk":
            # Score pruning reference arm (matched-budget accuracy comparison): keep the B
            # visual tokens with the highest question->vision salience (VLMAS_WSI_CONSOL_SC=1),
            # no coverage, no provenance, no representation similarity.
            # VLMAS_TOPK_PER_CROP=1 keeps a proportional share inside every crop instead.
            from memory.topk_select import salience_topk_keep_mask
            _per_crop = os.environ.get("VLMAS_TOPK_PER_CROP", "").strip() == "1"
            keep = salience_topk_keep_mask(
                relevance, token_counts, budget, per_crop=_per_crop).to(self.device)
            print(f"[WSIConsol] topk rel={'q2v' if relevance is not None else 'uniform'} "
                  f"per_crop={int(_per_crop)} kept={int(keep.sum())}/{nV} "
                  f"attn={os.environ.get('VLMAS_ATTN_IMPLEMENTATION', 'eager')}", flush=True)
'''
if text.count(HEAD + TAIL) != 1:
    sys.exit(f"anchor count {text.count(HEAD + TAIL)} != 1")
text = text.replace(HEAD + TAIL, HEAD + BRANCH + TAIL, 1)
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
