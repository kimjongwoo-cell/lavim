#!/usr/bin/env python3
"""What a top-25% a_i selection inside each crop would keep: border share, tissue, fixed hot cells.

usage: vit_sal_topk.py <out_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vit_sal_stats import crop_map, crop_spans, summarize  # noqa: E402

out = Path(sys.argv[1])
K = 64                                   # 25% of 256
ring = np.ones((16, 16), bool); ring[1:-1, 1:-1] = False
ring = ring.ravel()                      # 60 tokens = 23.4%
crops = []
for f in sorted((out / "cases").glob("*.npz")):
    z = np.load(f)
    pb = z["per_block"]
    for sp, mag in zip(crop_spans(z["thw"]), z["mags"]):
        blocks = np.stack([crop_map(pb[b], sp) for b in range(pb.shape[0])])
        crops.append({"ds": f.stem.rsplit("_", 1)[0], "mag": int(mag), "case": f.stem,
                      "maps": np.vstack([blocks, blocks.mean(0, keepdims=True)]),
                      "tissue": z["tissue"][sp[0]:sp[1]]})
LAY = {"mean": -1, "b3": 3, "b22": 22, "b23": 23}
res = {}
for lname, li in LAY.items():
    allmaps = np.stack([c["maps"][li] for c in crops])
    hot8 = np.argsort(-allmaps.mean(0))[:8]
    for mag in (5, 20):
        rows = {"ring_share": [], "tissue_sel_minus_crop": [], "tissue_sel_minus_bottom": [], "hot8_mass": [],
                "hot8_in_topk": []}
        for c in crops:
            if c["mag"] != mag:
                continue
            v = c["maps"][li]; t = c["tissue"]
            order = np.argsort(-v)
            top, bot = order[:K], order[-K:]
            rows["ring_share"].append(ring[top].mean())
            rows["tissue_sel_minus_crop"].append(t[top].mean() - t.mean())
            rows["tissue_sel_minus_bottom"].append(t[top].mean() - t[bot].mean())
            rows["hot8_mass"].append(v[hot8].sum() / v.sum())
            rows["hot8_in_topk"].append(np.isin(hot8, top).mean())
        res[f"{lname}|{mag}"] = {k: summarize(v) for k, v in rows.items()}
        res[f"{lname}|{mag}"]["hot8_rc"] = [list(divmod(int(i), 16)) for i in hot8]
(out / "topk_summary.json").write_text(json.dumps(res, indent=1))
print("uniform: ring 60/256 = 0.234, hot8 mass 8/256 = 0.031")
for k, r in res.items():
    print(f"{k:>9}: ring_share med {r['ring_share']['median']:.3f} | tissue top-crop {r['tissue_sel_minus_crop']['median']:+.3f}"
          f" top-bottom {r['tissue_sel_minus_bottom']['median']:+.3f} | hot8 mass {r['hot8_mass']['median']:.3f}"
          f" hot8 in top25% {r['hot8_in_topk']['median']:.2f} | hot8 rc {r['hot8_rc']}")
# tissue coverage of crops, for context
tis = np.array([c["tissue"].mean() for c in crops])
print("crop tissue fraction median", round(float(np.median(tis)), 3), "p25", round(float(np.percentile(tis, 25)), 3))
