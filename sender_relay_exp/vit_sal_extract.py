#!/usr/bin/env python3
"""Extract vision-encoder salience a_i for the crops nav3 actually selected (5 sets x dcs20).

No LLM run: the crop boxes come from existing nav3 result.json files (Navigator decode is
native in all three sources — canonical addressing is Answerer-only there, re-read touches
only the Answerer), crops are re-rendered with the pipeline's own render_box(max_side=512),
and only the vision tower is run (CPU float32; GPU 6/7/8 are busy). a_i is the per-block
received attention from vit_indeg.collect (sum over the 2x2 pre-merge patches, which ranks
identically to the Notion definition's mean).

Reproduction check per crop: tissue_fraction of the re-rendered 128-px preview must match
the value the Navigator recorded in result.json.

usage: vit_sal_extract.py <out_dir> [--datasets a,b] [--limit N] [--threads T]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path("/home/users/whddn12316/wsi_latent_0915_decode_hj")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "sender_relay_exp"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from vision_text_mas.cli import _open_slide  # noqa: E402
from vision_text_mas.dataset import load_cases  # noqa: E402
from vision_text_mas.geometry import Box, tissue_fraction  # noqa: E402
from vision_text_mas.navigation_render import render_box  # noqa: E402
from vit_indeg import collect  # noqa: E402
from vit_sal_stats import crop_spans, tissue_blocks  # noqa: E402

DATA = Path("/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda")
MODEL = Path("/home/users/whddn12316/models/Qwen3-VL-4B-Thinking")
RUNS = REPO / "sender_relay_exp" / "runs"
SMOKE = REPO / "sender_relay_exp" / "smoke"
SOURCES = ("canon_reread_nav3", "canon_only_nav3", "canon_diag")
ALL_DS = ("tcga_expert_vqa", "gtex", "tcga", "tcga_slidebench", "panda")


def result_index(ds: str) -> dict[int, list[tuple[str, Path]]]:
    found: dict[int, list[tuple[str, Path]]] = {}
    for src in SOURCES:
        paths = sorted((RUNS / src / ds).glob("*/*/result.json"), key=lambda p: p.stat().st_mtime,
                       reverse=True)
        for p in paths:
            try:
                idx = int(json.loads(p.read_text())["dataset_index"])
            except (OSError, ValueError, KeyError):
                continue
            found.setdefault(idx, []).append((src, p))
    return found


def boxes_of(result: dict) -> list[tuple]:
    return [(p["magnification"], p["box"]["x"], p["box"]["y"], p["box"]["width"], p["box"]["height"])
            for p in result["patches"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--datasets", default=",".join(ALL_DS))
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "cases").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)

    from transformers import AutoModelForImageTextToText, AutoProcessor
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.float32).eval()
    print(f"[load] {time.time() - t0:.0f}s", flush=True)

    manifest = []
    for ds in args.datasets.split(","):
        full = json.loads((DATA / f"{ds}.json").read_text())
        sub = json.loads((SMOKE / f"dcs20_{ds}.json").read_text())
        results = result_index(ds)
        for k, item in enumerate(sub[: args.limit]):
            key = f"{ds}_{k:02d}"
            dst = out / "cases" / f"{key}.npz"
            matches = [i for i, f in enumerate(full)
                       if f["Id"] == item["Id"] and f["Question"] == item["Question"]
                       and f.get("Choice") == item.get("Choice")]
            hits = [i for i in matches if i in results]
            row = {"key": key, "ds": ds, "sub_index": k, "id": item["Id"], "full_matches": matches}
            if not hits:
                row["status"] = "no_nav3_result"
                manifest.append(row)
                print(f"[{key}] SKIP no nav3 result (full idx {matches})", flush=True)
                continue
            idx = hits[0]
            src, rpath = results[idx][0]
            result = json.loads(rpath.read_text())
            # other sources for the same case must have picked the same crops
            agree = [boxes_of(json.loads(p.read_text())) == boxes_of(result) for _, p in results[idx][1:]]
            row.update(dataset_index=idx, source=src, result=str(rpath),
                       other_sources_agree=f"{sum(agree)}/{len(agree)}")
            if dst.exists():
                row["status"] = "cached"
                manifest.append(row)
                continue
            t1 = time.time()
            case = load_cases(DATA / f"{ds}.json", slide_root=DATA / "slides", indices=(idx,))[0]
            slide = _open_slide(case)
            patches = result["patches"]
            imgs, tf_diff = [], []
            for p in patches:
                box = Box(**p["box"])
                imgs.append(render_box(slide, box, max_side=512))
                tf_diff.append(abs(tissue_fraction(render_box(slide, box, max_side=128))
                                   - float(p["tissue_fraction"])))
            enc = proc.image_processor(images=imgs, return_tensors="pt")
            pv, thw = enc["pixel_values"].float(), enc["image_grid_thw"]
            _, per_block = collect(model, pv, thw, merge_unit=4)
            per_block = per_block.float().numpy()
            spans = crop_spans(thw.numpy())
            tissue = np.concatenate([tissue_blocks(im, h, w) for im, (_, _, h, w) in zip(imgs, spans)])
            crop_sums = np.array([[per_block[b, a:e].sum() for (a, e, _, _) in spans]
                                  for b in range(per_block.shape[0])])
            np.savez_compressed(
                dst, per_block=per_block, thw=thw.numpy(), tissue=tissue,
                mags=np.array([int(p["magnification"]) for p in patches]),
                boxes=np.array([[p["box"]["x"], p["box"]["y"], p["box"]["width"], p["box"]["height"]]
                                for p in patches]),
                img_sizes=np.array([im.size for im in imgs]), tf_diff=np.array(tf_diff))
            row.update(status="ok", n_crops=len(patches), n_tokens=int(per_block.shape[1]),
                       grids=sorted({f"{h}x{w}" for _, _, h, w in spans}),
                       mags={m: sum(int(p["magnification"]) == m for p in patches) for m in (5, 20)},
                       tf_diff_max=float(max(tf_diff)), crop_sum_dev=float(np.abs(crop_sums - 1).max()),
                       sec=round(time.time() - t1, 1))
            manifest.append(row)
            print(f"[{key}] idx={idx} src={src} crops={len(patches)} tok={per_block.shape[1]} "
                  f"grids={row['grids']} mags={row['mags']} tf_diff_max={row['tf_diff_max']:.4f} "
                  f"sum_dev={row['crop_sum_dev']:.2e} agree={row['other_sources_agree']} {row['sec']}s",
                  flush=True)
            (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("[done]", {s: sum(r["status"] == s for r in manifest) for s in {r["status"] for r in manifest}},
          flush=True)


if __name__ == "__main__":
    main()
