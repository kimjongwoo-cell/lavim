"""Wire PCSI into backbone/qwen3vl.py (VLMAS_WSI_CONSOL_MODE=pcsi; every other mode untouched).

Adds (1) _pcsi_capture_query(): Q = the "Question stem:" token embeddings of the Reasoner
chunk, from the same embedding-layer output as the visual states; (2) one call to it right
before each _wsi_consolidate_boundary() call site (fresh and on-KV prefill); (3) the
`elif mode == "pcsi"` branch. Atomic, md5-guarded, py_compile before replace.
usage: python patch_pcsi_qwen3vl.py backbone/qwen3vl.py
"""
import hashlib
import os
import py_compile
import sys
import tempfile

EXPECT = "b8a55b57584438807dddc4c41b87cc12"
path = sys.argv[1]
raw = open(path, "rb").read()
before = hashlib.md5(raw).hexdigest()
text = raw.decode("utf-8")
if "_pcsi_capture_query" in text:
    sys.exit("already patched")
if before != EXPECT:
    sys.exit(f"md5 {before} != expected {EXPECT}; file changed upstream, aborted")

METHOD_ANCHOR = "    def _wsi_consolidate_boundary(self, past_kv, seq_len, vis_abs,\n"
METHOD = '''    def _pcsi_capture_query(self, ids, last, prefill_input_embeddings):
        """PCSI (env VLMAS_WSI_CONSOL_MODE=pcsi, Reasoner stage only; else no-op).

        Question set Q = the "Question stem:" tokens of this chunk's post-image
        text, read from the SAME embedding-layer output as the visual states
        handed to _wsi_consolidate_boundary (shared LLM-input space). Stores
        (embeddings [nq, d] cpu, decoded question text) in self._pcsi_query.
        """
        self._pcsi_query = None
        if (os.environ.get("VLMAS_WSI_CONSOL", "") != "1"
                or os.environ.get("VLMAS_WSI_CONSOL_MODE", "") != "pcsi"
                or not getattr(self, "_prune_morphology_enabled", False)):
            return
        from memory.pcsi_select import locate_question_rows
        tok = self.processor.tokenizer
        tail = ids[last + 1:].tolist()
        rows = locate_question_rows([tok.decode([t]) for t in tail])
        n_emb = int(prefill_input_embeddings.shape[0])
        rows = [r for r in rows if last + 1 + r < n_emb]
        if rows:
            index = torch.tensor([last + 1 + r for r in rows], dtype=torch.long)
            self._pcsi_query = (
                prefill_input_embeddings[index].detach().float().cpu(),
                tok.decode([tail[r] for r in rows]))

'''
CALL = "        self._pcsi_capture_query(ids, last, prefill_input_embeddings)\n"
CALL1 = ("        past_kv, seq_prefill, prefill_spans, _consol_keep, _consol_vis = (\n"
         "            self._wsi_consolidate_boundary(\n")
CALL2 = ("        past_kv, seq_prefill, spans, _consol_keep, _consol_vis = (\n"
         "            self._wsi_consolidate_boundary(\n")
BR_HEAD = ('                + f" q2v_layers={sorted(q2v.keys()) if q2v else []}", flush=True)\n')
BR_TAIL = '        elif mode == "strat":\n'
BRANCH = '''        elif mode == "pcsi":
            # Provenance-Conditioned Submodular Information Selection (C1 candidate,
            # Notion PCSI section 10): F(S) = sum_o I_f(S & V_o; Q | P_o), facility-
            # location CMI, P_o = V_parent(o) from the acquisition path
            # (_prune_image_parent_indices), s = [cos]_+ in the LLM-input space,
            # Q = "Question stem:" token embeddings (_pcsi_capture_query), |S| <= B,
            # greedy argmax marginal gain. Arms: VLMAS_PCSI_COND=true|none|wrong
            # (none = generic SMI, wrong = deranged parents), VLMAS_PCSI_FILL=lex|none.
            # VLMAS_WSI_CONSOL_CONTROL=shuffled_parent remaps boxes only (unused here).
            from memory.pcsi_select import pcsi_keep_mask
            _pq = getattr(self, "_pcsi_query", None)
            _pparents = tuple(getattr(self, "_prune_image_parent_indices", ()) or ())
            keep, pcsi_info = pcsi_keep_mask(
                vis_embeddings.to(self.device).float(),
                (_pq[0].to(self.device).float() if _pq is not None else None),
                token_counts, _pparents, budget, magnifications=magnifications,
                cond=os.environ.get("VLMAS_PCSI_COND", "true"),
                fill=os.environ.get("VLMAS_PCSI_FILL", "lex"),
                return_info=True)
            _pcsi_dump = os.environ.get("VLMAS_PCSI_DUMP", "").strip()
            if _pcsi_dump:
                os.makedirs(_pcsi_dump, exist_ok=True)
                _pn = getattr(self, "_pcsi_dump_n", 0)
                self._pcsi_dump_n = _pn + 1
                torch.save({
                    "z": vis_embeddings.detach().to(torch.float16).cpu(),
                    "q": (_pq[0].detach().to(torch.float16).cpu()
                          if _pq is not None else None),
                    "q_text": (_pq[1] if _pq is not None else ""),
                    "u": (relevance.detach().float().cpu()
                          if relevance is not None else None),
                    "token_counts": token_counts, "grid_hws": grid_hws,
                    "magnifications": magnifications, "boxes": boxes,
                    "parents": _pparents, "keep": keep.detach().cpu(),
                    "info": dict(pcsi_info),
                }, os.path.join(_pcsi_dump, f"pcsi_{_pn:04d}.pt"))
            keep = keep.to(self.device)
            print("[WSIConsol] pcsi " + " ".join(
                f"{k}={v}" for k, v in pcsi_info.items() if k != "order")
                + f" q_text={(_pq[1] if _pq is not None else '')!r}"
                + f" control={control or 'none'}", flush=True)
'''
reps = [(METHOD_ANCHOR, METHOD + METHOD_ANCHOR), (CALL1, CALL + CALL1), (CALL2, CALL + CALL2),
        (BR_HEAD + BR_TAIL, BR_HEAD + BRANCH + BR_TAIL)]
for old, new in reps:
    n = text.count(old)
    if n != 1:
        sys.exit(f"anchor count {n} != 1: {old[:70]!r}")
    text = text.replace(old, new, 1)
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
