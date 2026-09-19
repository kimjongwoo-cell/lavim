"""Wire PCRIS into backbone/qwen3vl.py (VLMAS_WSI_CONSOL_MODE=pcris; every other mode untouched).

Adds the `elif mode == "pcris"` branch right after the pcsi branch. Atomic, md5-guarded
(expects the PCSI-patched file c931ea0f), py_compile before replace.
usage: python patch_pcris_qwen3vl.py backbone/qwen3vl.py
"""
import hashlib
import os
import py_compile
import sys
import tempfile

EXPECT = "c931ea0ff62d9632ff5bc91b6dc785f1"
path = sys.argv[1]
raw = open(path, "rb").read()
before = hashlib.md5(raw).hexdigest()
text = raw.decode("utf-8")
if 'elif mode == "pcris"' in text:
    sys.exit("already patched")
if before != EXPECT:
    sys.exit(f"md5 {before} != expected {EXPECT}; file changed upstream, aborted")
HEAD = ("                + f\" q_text={(_pq[1] if _pq is not None else '')!r}\"\n"
        "                + f\" control={control or 'none'}\", flush=True)\n")
TAIL = '        elif mode == "strat":\n'
BRANCH = '''        elif mode == "pcris":
            # Provenance-Conditioned Residual Information Selection (C1 candidate,
            # Notion PCRIS section 8): r_i = (I - U_o U_o^T) h_hat_i with U_o an
            # orthonormal basis (SVD) of the parent observation's normalized states
            # (_prune_image_parent_indices), r_i = h_hat_i for roots; F(S) =
            # log det(I + sum_{i in S} r_i r_i^T), |S| <= B, greedy marginal gain.
            # No question signal. Arms: VLMAS_PCRIS_COND=true|none|wrong,
            # VLMAS_PCRIS_CONTEXT=parent|ancestors, VLMAS_PCRIS_DUMP=<dir>.
            from memory.pcris_select import pcris_keep_mask
            _rparents = tuple(getattr(self, "_prune_image_parent_indices", ()) or ())
            keep, pcris_info = pcris_keep_mask(
                vis_embeddings.to(self.device).float(), token_counts, _rparents,
                budget, magnifications=magnifications,
                cond=os.environ.get("VLMAS_PCRIS_COND", "true"),
                context=os.environ.get("VLMAS_PCRIS_CONTEXT", "parent"),
                return_info=True)
            _pcris_dump = os.environ.get("VLMAS_PCRIS_DUMP", "").strip()
            if _pcris_dump:
                os.makedirs(_pcris_dump, exist_ok=True)
                _rn = getattr(self, "_pcris_dump_n", 0)
                self._pcris_dump_n = _rn + 1
                torch.save({
                    "z": vis_embeddings.detach().cpu(),   # native dtype, exact
                    "z_dtype": str(vis_embeddings.dtype),
                    "u": (relevance.detach().float().cpu()
                          if relevance is not None else None),
                    "token_counts": token_counts, "grid_hws": grid_hws,
                    "magnifications": magnifications, "boxes": boxes,
                    "parents": _rparents, "keep": keep.detach().cpu(),
                    "info": dict(pcris_info),
                    "attn_impl": os.environ.get("VLMAS_ATTN_IMPLEMENTATION", "eager"),
                }, os.path.join(_pcris_dump, f"pcris_{_rn:04d}.pt"))
            keep = keep.to(self.device)
            print("[WSIConsol] pcris " + " ".join(
                f"{k}={v}" for k, v in pcris_info.items() if k != "order")
                + f" attn={os.environ.get('VLMAS_ATTN_IMPLEMENTATION', 'eager')}"
                + f" control={control or 'none'}", flush=True)
'''
n = text.count(HEAD + TAIL)
if n != 1:
    sys.exit(f"anchor count {n} != 1")
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
