"""Wire Q-PRIS into backbone/qwen3vl.py (VLMAS_WSI_CONSOL_MODE=qpris; every other mode untouched).

Two edits, both md5-guarded and atomic, with a backup into tmp_hj/:
  1. _pcsi_capture_query also fires for mode=qpris (the question states are the same object).
  2. a new `elif mode == "qpris":` branch inserted right before the topk branch.

usage: python patch_qpris_qwen3vl.py <expected_md5>
"""
import hashlib
import os
import shutil
import sys

ROOT = "/home/users/whddn12316/wsi_latent_0915_decode_hj"
PATH = os.path.join(ROOT, "backbone/qwen3vl.py")

GATE_OLD = '''                or os.environ.get("VLMAS_WSI_CONSOL_MODE", "") != "pcsi"'''
GATE_NEW = '''                or os.environ.get("VLMAS_WSI_CONSOL_MODE", "") not in ("pcsi", "qpris")'''

ANCHOR = '''        elif mode == "topk":'''

BRANCH = '''        elif mode == "qpris":
            # Question-Conditioned Provenance Residual Information Selection (C1
            # candidate, Notion Q-PRIS sections 2-5): r~_i = (I - P_pi(o)) h_hat_i / nu_i
            # with nu_i the within-case alternative-parent null (PCRIS v2 calibration),
            # q_i = ||U_Q^T r~_i||^2 against the orthonormal question-token subspace at
            # the C1 boundary (_pcsi_capture_query), and
            #   F_Q(S) = log det(I + sum_{i in S} q_i r~_i r~_i^T),  |S| <= B, greedy.
            # No q2v salience, no pooled Q-V cosine, no scale quota, no learned selector.
            # Arms: VLMAS_QPRIS_Q=true|shuffled|none  (shuffled = the PREVIOUS case's
            # question, so the true question never leaks into that arm; the first case of
            # a run has no predecessor and falls back to q=none, reported as q_src),
            # VLMAS_QPRIS_COND=true|none|wrong, VLMAS_QPRIS_CONTEXT=parent|ancestors,
            # VLMAS_QPRIS_NULLCAL=0|1, VLMAS_QPRIS_DUMP=<dir>.
            from memory.qpris_select import qpris_keep_mask
            _qparents = tuple(getattr(self, "_prune_image_parent_indices", ()) or ())
            _qq = getattr(self, "_pcsi_query", None)
            _qarm = (os.environ.get("VLMAS_QPRIS_Q", "true").strip() or "true")
            _qcpu = (_qq[0] if _qq is not None else None)
            _qtext = (_qq[1] if _qq is not None else "")
            if _qarm == "shuffled":
                _bank = getattr(self, "_qpris_prev_query", None)
                self._qpris_prev_query = (_qcpu, _qtext)
                _qsrc = "prev_case" if _bank is not None else "none_first_case"
                _qcpu, _qtext = (_bank if _bank is not None else (None, ""))
            elif _qarm == "none":
                _qcpu, _qtext, _qsrc = None, "", "none"
            else:
                _qsrc = "true" if _qcpu is not None else "none_missing"
            _qdev = (_qcpu.to(self.device).float() if _qcpu is not None else None)
            keep, qpris_info = qpris_keep_mask(
                vis_embeddings.to(self.device).float(), _qdev,
                token_counts, _qparents, budget, magnifications=magnifications,
                cond=os.environ.get("VLMAS_QPRIS_COND", "true"),
                context=os.environ.get("VLMAS_QPRIS_CONTEXT", "parent"),
                q_mode=("true" if _qdev is not None else "none"),
                nullcal=(os.environ.get("VLMAS_QPRIS_NULLCAL", "1").strip() != "0"),
                return_info=True)
            _qpris_dump = os.environ.get("VLMAS_QPRIS_DUMP", "").strip()
            if _qpris_dump:
                os.makedirs(_qpris_dump, exist_ok=True)
                _qn = getattr(self, "_qpris_dump_n", 0)
                self._qpris_dump_n = _qn + 1
                torch.save({
                    "z": vis_embeddings.detach().cpu(),   # native dtype, exact
                    "z_dtype": str(vis_embeddings.dtype),
                    "q": (_qcpu.detach().float().cpu() if _qcpu is not None else None),
                    "q_text": _qtext, "q_arm": _qarm, "q_src": _qsrc,
                    "u": (relevance.detach().float().cpu()
                          if relevance is not None else None),
                    "token_counts": token_counts, "grid_hws": grid_hws,
                    "magnifications": magnifications, "boxes": boxes,
                    "parents": _qparents, "keep": keep.detach().cpu(),
                    "info": {k: v for k, v in qpris_info.items() if k != "nu"},
                    "attn_impl": os.environ.get("VLMAS_ATTN_IMPLEMENTATION", "eager"),
                }, os.path.join(_qpris_dump, f"qpris_{_qn:04d}.pt"))
            keep = keep.to(self.device)
            print("[WSIConsol] qpris " + " ".join(
                f"{k}={v}" for k, v in qpris_info.items() if k not in ("order", "parents"))
                + f" q_arm={_qarm} q_src={_qsrc} q_text={_qtext!r}"
                + f" attn={os.environ.get('VLMAS_ATTN_IMPLEMENTATION', 'eager')}"
                + f" control={control or 'none'}", flush=True)
'''


def main(expected):
    src = open(PATH).read()
    md5 = hashlib.md5(src.encode()).hexdigest()
    if 'elif mode == "qpris"' in src:
        print(f"ALREADY_PRESENT md5={md5}")
        return 0
    if md5 != expected:
        print(f"GUARD_FAIL md5={md5} expected={expected}")
        return 3
    for name, frag, n in (("GATE", GATE_OLD, 1), ("ANCHOR", ANCHOR, 1)):
        if src.count(frag) != n:
            print(f"{name}_FAIL count={src.count(frag)}")
            return 4
    out = src.replace(GATE_OLD, GATE_NEW).replace(ANCHOR, BRANCH + ANCHOR)
    shutil.copyfile(PATH, os.path.join(ROOT, f"tmp_hj/qwen3vl.py.orig_{md5[:8]}"))
    tmp = PATH + ".new"
    with open(tmp, "w") as fh:
        fh.write(out)
    os.replace(tmp, PATH)
    print(f"PATCHED {md5} -> {hashlib.md5(out.encode()).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
