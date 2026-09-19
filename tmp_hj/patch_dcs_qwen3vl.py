"""Insert the opt-in `dcs` branch into backbone/qwen3vl.py::_wsi_consolidate_boundary.

Atomic: anchor must appear exactly once, file md5 is re-checked right before
os.replace, the patched text is py_compiled before replacing. Off (mode != dcs)
= byte-identical behaviour (new elif only).
"""
import hashlib
import os
import py_compile
import sys
import tempfile

path = sys.argv[1]
ANCHOR = '        elif mode == "strat":\n'
BRANCH = '''        elif mode == "dcs":
            # Directed Cross-Scale Submodular Visual KV Selection (C1 candidate,
            # Notion 0912): F(S) = sum_i s_i max_{j in S} a_ij, |S| <= B, greedy.
            # s = cross-observation debiased q2v salience (needs
            # VLMAS_WSI_CONSOL_SC=1; VLMAS_DCS_SAL=raw|uniform), a_ij = k_sem
            # (same scale) or k_sem * |Om_i & Om_j| / |Om_i| (cross scale;
            # VLMAS_DCS_AFF=symmetric -> IoU; VLMAS_DCS_Z=none -> spatial-only).
            # No hard 5x->20x precedence. Wrong-parent falsifier =
            # VLMAS_WSI_CONSOL_CONTROL=shuffled_parent (boxes remapped above).
            from memory.dcs_select import dcs_keep_mask
            keep, dcs_info = dcs_keep_mask(
                vis_embeddings.to(self.device).float(), relevance,
                token_counts, grid_hws, magnifications, boxes, budget,
                aff=os.environ.get("VLMAS_DCS_AFF", "directed"),
                z_mode=os.environ.get("VLMAS_DCS_Z", "merger"),
                sal=os.environ.get("VLMAS_DCS_SAL", "debiased"),
                return_info=True)
            keep = keep.to(self.device)
            print("[WSIConsol] dcs " + " ".join(
                f"{k}={v}" for k, v in dcs_info.items())
                + f" control={control or 'none'}"
                + f" q2v_layers={sorted(q2v.keys()) if q2v else []}", flush=True)
'''


def md5(b):
    return hashlib.md5(b).hexdigest()


with open(path, "rb") as fh:
    raw = fh.read()
before = md5(raw)
text = raw.decode("utf-8")
if 'elif mode == "dcs":' in text:
    sys.exit("already patched")
if text.count(ANCHOR) != 1:
    sys.exit(f"anchor count {text.count(ANCHOR)} != 1")
new = text.replace(ANCHOR, BRANCH + ANCHOR, 1)
fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".py.tmp")
with os.fdopen(fd, "w", encoding="utf-8") as fh:
    fh.write(new)
py_compile.compile(tmp, doraise=True)
with open(path, "rb") as fh:
    if md5(fh.read()) != before:
        os.unlink(tmp)
        sys.exit("file changed during patch; aborted")
os.chmod(tmp, os.stat(path).st_mode)
os.replace(tmp, path)
with open(path, "rb") as fh:
    after = md5(fh.read())
print(f"patched {path}\n  before {before}\n  after  {after}")
