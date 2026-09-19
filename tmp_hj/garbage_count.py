import importlib.util, sys
from pathlib import Path
T = Path("/home/users/whddn12316/wsi_latent_0915_decode_hj"); R = T / "sender_relay_exp/runs"
s = importlib.util.spec_from_file_location("rescore", T / "scripts/rescore.py"); rs = importlib.util.module_from_spec(s)
sys.modules["rescore"] = rs; s.loader.exec_module(rs)
nrm = rs.am.normalize
for ds in ["tcga_expert_vqa", "tcga_slidebench", "gtex", "tcga", "panda"]:
    recs = rs.load_records(ds)
    row = []
    for name, roots in (("base", sorted(R.glob(f"nav3_prompt1_base_{ds}_gpu*"))), ("qasc", [R / "qasc_nav3" / ds]), ("ntrs", [R / "ntrs_nav3" / ds])):
        c, _ = rs.collect(rs.run_dirs(roots, None, []), {ds}); c = c.get(ds, {})
        if not c:
            row.append(f"{name} -"); continue
        bad = [i for i in c if not any(nrm(ch) and nrm(ch) == nrm(c[i][1]) for ch in (recs[i].get("Choice") or []))]
        text = sum(1 for i in bad if nrm(c[i][1]) == "text")
        row.append(f"{name} {len(bad)}/{len(c)} (text {text})")
    print(f"{ds:16s} " + " | ".join(row))
