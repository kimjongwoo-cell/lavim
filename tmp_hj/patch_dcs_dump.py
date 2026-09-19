"""Add an opt-in input dump to the dcs branch (VLMAS_DCS_DUMP=<dir>; unset = no-op).
Saves z (fp16), u (pipeline q2v relevance), grids, mags, boxes, parents per Reasoner
boundary so affinity formulas can be compared offline. Atomic, md5-guarded."""
import hashlib, os, py_compile, sys, tempfile

path = sys.argv[1]
ANCHOR = ('            keep = keep.to(self.device)\n'
          '            print("[WSIConsol] dcs " + " ".join(\n')
DUMP = '''            _dcs_dump = os.environ.get("VLMAS_DCS_DUMP", "").strip()
            if _dcs_dump:
                os.makedirs(_dcs_dump, exist_ok=True)
                _dn = getattr(self, "_dcs_dump_n", 0)
                self._dcs_dump_n = _dn + 1
                torch.save({
                    "z": vis_embeddings.detach().to(torch.float16).cpu(),
                    "u": (relevance.detach().float().cpu()
                          if relevance is not None else None),
                    "token_counts": token_counts, "grid_hws": grid_hws,
                    "magnifications": magnifications, "boxes": boxes,
                    "parents": tuple(getattr(self, "_prune_image_parent_indices", ()) or ()),
                    "q2v_layers": sorted(q2v.keys()) if q2v else [],
                }, os.path.join(_dcs_dump, f"dcs_{_dn:04d}.pt"))
'''
raw = open(path, "rb").read()
before = hashlib.md5(raw).hexdigest()
text = raw.decode("utf-8")
if "VLMAS_DCS_DUMP" in text:
    sys.exit("already patched")
if text.count(ANCHOR) != 1:
    sys.exit(f"anchor count {text.count(ANCHOR)} != 1")
new = text.replace(ANCHOR, DUMP + ANCHOR, 1)
fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".py.tmp")
with os.fdopen(fd, "w", encoding="utf-8") as fh:
    fh.write(new)
py_compile.compile(tmp, doraise=True)
if hashlib.md5(open(path, "rb").read()).hexdigest() != before:
    os.unlink(tmp)
    sys.exit("file changed during patch; aborted")
os.chmod(tmp, os.stat(path).st_mode)
os.replace(tmp, path)
print(f"patched {path}\n  before {before}\n  after  {hashlib.md5(open(path, 'rb').read()).hexdigest()}")
