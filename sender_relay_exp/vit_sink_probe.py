#!/usr/bin/env python3
"""Why does the vision-encoder salience a_i peak at fixed border cells? Four cheap checks (CPU, vision tower only).

Context: the a_i audit (runs/vit_sal_audit/full, 1182 nav3 crops) found the same hot merged cells
(2,15)(2,0)(5,15)(13,15)(13,0)(4,15)(15,7)(15,8) in every dataset and magnification, unrelated to tissue.
Candidate causes (unverified): position-bound attention sinks / registers, absolute position-embedding
interpolation edge effects, pretraining border statistics. This script separates them:

  1 content-free   solid white / black / gray / H&E pink and seeded noise images at 512 px
                   -> does the template appear without tissue content?
  2 translation    real crops re-rendered with the box shifted by k merged cells (k = 1, 3; x and y)
                   -> do hot cells stay at grid coordinates (positional) or move with content?
  3 resolution     the same boxes rendered at 384 / 512 / 768 px (12x12, 16x16, 24x24 merged grids)
                   -> do hot cells follow relative position (pos-embed interpolation) or absolute index?
  4 hidden norm    per-token output norm of every vision block for the real crops
                   -> are hot tokens high-norm (register-like)?

a_i is computed exactly like the audit (vit_indeg.indegree: received attention, mean over heads and query
rows inside the crop, summed over the 2x2 pre-merge patches; maps rescaled so uniform = 1).

usage: vit_sink_probe.py <out_dir> [--cases N] [--res-cases N] [--threads T]
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
from PIL import Image  # noqa: E402

from vit_indeg import chunk_bounds, indegree, pool_to_merged  # noqa: E402
from vit_sal_stats import crop_map, crop_spans, spearman  # noqa: E402

DATA = Path("/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda")
MODEL = Path("/home/users/whddn12316/models/Qwen3-VL-4B-Thinking")
AUDIT = REPO / "sender_relay_exp" / "runs" / "vit_sal_audit" / "full"
BLOCKS_REPORT = ("mean", 0, 3, 12, 23)


# --------------------------------------------------------------------------- pure helpers


def template_maps(audit_dir: Path, grid: int = 16) -> np.ndarray:
    """Mean per-block map over all audit crops with a grid x grid merged layout -> [n_blocks, grid*grid]."""
    acc, n = None, 0
    for f in sorted((audit_dir / "cases").glob("*.npz")):
        z = np.load(f)
        pb = z["per_block"]
        for sp in crop_spans(z["thw"]):
            if sp[2] != grid or sp[3] != grid:
                continue
            m = np.stack([crop_map(pb[b], sp) for b in range(pb.shape[0])])
            acc = m if acc is None else acc + m
            n += 1
    if acc is None:
        raise RuntimeError("no audit crops with the requested grid")
    return acc / n


def hot_cells(template_block: np.ndarray, k: int = 8) -> np.ndarray:
    return np.argsort(-template_block)[:k]


def with_mean_block(maps: np.ndarray) -> dict:
    """[n_blocks, n] -> {name: map} for the reported blocks ('mean' = block average)."""
    out = {}
    for b in BLOCKS_REPORT:
        out[str(b)] = maps.mean(axis=0) if b == "mean" else maps[int(b)]
    return out


def shift_alignment(m0: np.ndarray, mk: np.ndarray, grid: int, k: int, axis: str):
    """Maps of the original (m0) and the box shifted by +k cells along axis ('x' or 'y').

    Shifting the box by +k cells moves the tissue content by -k cells in the image, so content cell
    (r, c) of the original appears at (r, c-k) in the shifted crop (x shift). Returns
    (rho_same_grid_coords, rho_content_aligned) over the region both crops cover.
    """
    a = m0.reshape(grid, grid)
    b = mk.reshape(grid, grid)
    if axis == "x":
        fixed = spearman(a[:, k:], b[:, k:])
        content = spearman(a[:, k:], b[:, :grid - k])
    else:
        fixed = spearman(a[k:, :], b[k:, :])
        content = spearman(a[k:, :], b[:grid - k, :])
    return fixed, content


def hot_fixed_vs_moved(m0: np.ndarray, mk: np.ndarray, hot: np.ndarray, grid: int, k: int, axis: str) -> dict:
    """Track the hot cells themselves (uniform-normalised maps).

    fixed = mean value of the shifted map at the SAME grid cells as the hot cells;
    moved = mean value at the cells where the original content of those hot cells went (-k along axis).
    Positional sinks: fixed >> moved. Content-driven peaks: moved >= fixed.
    Also the original crop's own top-1 cell: value at the same coordinate vs at the content-moved coordinate.
    """
    a = m0.reshape(grid, grid)
    b = mk.reshape(grid, grid)
    fixed, moved = [], []
    hot_set = {int(x) for x in hot.tolist()}
    for cell in hot.tolist():
        r, c = divmod(int(cell), grid)
        r2, c2 = (r, c - k) if axis == "x" else (r - k, c)
        # skip pairs whose content-moved cell is itself a hot cell (e.g. (5,15)->(4,15) for a y shift of 1)
        if 0 <= r2 < grid and 0 <= c2 < grid and (r2 * grid + c2) not in hot_set:
            fixed.append(b[r, c])
            moved.append(b[r2, c2])
    r, c = divmod(int(np.argmax(a)), grid)
    r2, c2 = (r, c - k) if axis == "x" else (r - k, c)
    top_ok = 0 <= r2 < grid and 0 <= c2 < grid
    return {"hot_fixed_mean": float(np.mean(fixed)) if fixed else None,
            "hot_moved_mean": float(np.mean(moved)) if moved else None,
            "top1_fixed": float(b[r, c]), "top1_moved": float(b[r2, c2]) if top_ok else None}


def resample_relative(template16: np.ndarray, g: int) -> np.ndarray:
    """Nearest-neighbour resample of a 16x16 map to g x g by relative position."""
    t = template16.reshape(16, 16)
    idx = np.minimum((np.arange(g) + 0.5) / g * 16, 15).astype(int)
    return t[np.ix_(idx, idx)].ravel()


def absolute_overlap(template16: np.ndarray, m: np.ndarray, g: int):
    """Spearman of a g x g map with the 16x16 template on shared ABSOLUTE indices (top-left aligned)."""
    s = min(g, 16)
    return spearman(template16.reshape(16, 16)[:s, :s], m.reshape(g, g)[:s, :s])


# --------------------------------------------------------------------------- model runtime


def collect_maps_and_norms(model, pixel_values, grid_thw, merge_unit: int = 4):
    """Per-block received attention and per-block output norm, both pooled to merged tokens.

    Returns (per_block_attn [B, n_merged] summed over 2x2, per_block_norm [B, n_merged] mean over 2x2).
    """
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb_vision

    visual = model.model.visual
    attn_grab, norm_grab, handles = {}, {}, []

    def attn_hook(idx):
        def hook(module, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            cu = kwargs.get("cu_seqlens")
            pos = kwargs.get("position_embeddings")
            if hidden is None or cu is None or pos is None:
                return None
            seq = hidden.shape[0]
            qkv = module.qkv(hidden).reshape(seq, 3, module.num_heads, -1).permute(1, 0, 2, 3)
            q, k, _ = qkv.unbind(0)
            cos, sin = pos
            q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
            q, k = q.transpose(0, 1), k.transpose(0, 1)
            out = torch.zeros(seq, dtype=torch.float32)
            for a, b in chunk_bounds(cu):
                out[a:b] = indegree(q[:, a:b], k[:, a:b], module.scaling).float()
            attn_grab[idx] = out
            return None
        return hook

    def norm_hook(idx):
        def hook(module, args, kwargs, output):
            h = output[0] if isinstance(output, tuple) else output
            norm_grab[idx] = h.float().norm(dim=-1).detach()
            return None
        return hook

    for i, blk in enumerate(visual.blocks):
        handles.append(blk.attn.register_forward_pre_hook(attn_hook(i), with_kwargs=True))
        handles.append(blk.register_forward_hook(norm_hook(i), with_kwargs=True))
    try:
        with torch.no_grad():
            model.model.get_image_features(pixel_values, grid_thw, return_dict=True)
    finally:
        for h in handles:
            h.remove()
    order = sorted(attn_grab)
    attn = torch.stack([pool_to_merged(attn_grab[i], merge_unit) for i in order]).numpy()
    norms = torch.stack([norm_grab[i].reshape(-1, merge_unit).mean(dim=1) for i in order]).numpy()
    return attn, norms


def encode(proc, images):
    enc = proc.image_processor(images=images, return_tensors="pt")
    return enc["pixel_values"].float(), enc["image_grid_thw"]


def maps_per_crop(attn: np.ndarray, thw) -> list[tuple[tuple, np.ndarray]]:
    """-> [(span, maps [B, n_crop] rescaled uniform=1)]"""
    out = []
    for sp in crop_spans(np.asarray(thw)):
        out.append((sp, np.stack([crop_map(attn[b], sp) for b in range(attn.shape[0])])))
    return out


def block_stats(maps: np.ndarray, template: np.ndarray, hot8_by_block: dict, grid: int) -> dict:
    """maps [B, grid*grid] of one crop vs template [B, 256] (grid must be 16)."""
    res = {}
    rep_maps = with_mean_block(maps)
    rep_tmpl = with_mean_block(template)
    for name, m in rep_maps.items():
        hot = hot8_by_block[name]
        res[name] = {"rho_template": spearman(m, rep_tmpl[name]), "hot8_mass": float(m[hot].sum() / m.sum()),
                     "top1_rc": list(divmod(int(np.argmax(m)), grid)),
                     "top1_is_hot8": bool(int(np.argmax(m)) in set(hot.tolist()))}
    return res


def median(v):
    v = [x for x in v if x is not None]
    return None if not v else float(np.median(v))


# --------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--cases", type=int, default=4, help="audit cases for translation / norm checks")
    ap.add_argument("--res-cases", type=int, default=2, help="audit cases for the resolution check")
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)

    from transformers import AutoModelForImageTextToText, AutoProcessor

    from vision_text_mas.cli import _open_slide
    from vision_text_mas.dataset import load_cases
    from vision_text_mas.geometry import Box
    from vision_text_mas.navigation_render import render_box

    t0 = time.time()
    proc = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.float32).eval()
    template = template_maps(AUDIT)
    tmpl_rep = with_mean_block(template)
    hot8 = {name: hot_cells(m) for name, m in tmpl_rep.items()}
    summary = {"hot8_rc": {k: [list(divmod(int(i), 16)) for i in v] for k, v in hot8.items()}}
    print(f"[load] {time.time() - t0:.0f}s template from audit; hot8(mean) {summary['hot8_rc']['mean']}", flush=True)

    # ---------------- 1 content-free inputs
    rng = np.random.default_rng(0)
    synth = {
        "white": Image.new("RGB", (512, 512), (255, 255, 255)),
        "black": Image.new("RGB", (512, 512), (0, 0, 0)),
        "gray": Image.new("RGB", (512, 512), (128, 128, 128)),
        "he_pink": Image.new("RGB", (512, 512), (230, 180, 210)),
        "noise": Image.fromarray(rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)),
    }
    pv, thw = encode(proc, list(synth.values()))
    attn, _ = collect_maps_and_norms(model, pv, thw)
    res1 = {}
    for (name, _img), (sp, maps) in zip(synth.items(), maps_per_crop(attn, thw)):
        grid = sp[2]
        res1[name] = {"grid": [sp[2], sp[3]], **(block_stats(maps, template, hot8, grid) if grid == 16 else {})}
    summary["content_free"] = res1
    print("[1 content-free]", json.dumps({k: {b: (round(v[b]["rho_template"], 3) if v.get(b) and v[b]["rho_template"] is not None else None,
                                                  round(v[b]["hot8_mass"], 3) if v.get(b) else None) for b in ("mean", "3", "23")}
                                          for k, v in res1.items()}), flush=True)

    # real audit cases (first N ok cases in manifest order, spread over datasets)
    manifest = [r for r in json.loads((AUDIT / "manifest.json").read_text()) if r.get("status") in ("ok", "cached")]
    by_ds: dict[str, list] = {}
    for r in manifest:
        by_ds.setdefault(r["ds"], []).append(r)
    picked, i = [], 0
    while len(picked) < args.cases and any(by_ds.values()):
        for ds in list(by_ds):
            if by_ds[ds] and len(picked) < args.cases:
                picked.append(by_ds[ds].pop(0))
        i += 1

    def render_case(row, shift=(0.0, 0.0), max_side=512):
        result = json.loads(Path(row["result"]).read_text())
        case = load_cases(DATA / f"{row['ds']}.json", slide_root=DATA / "slides", indices=(int(row["dataset_index"]),))[0]
        slide = _open_slide(case)
        imgs = []
        for p in result["patches"]:
            b = Box(**p["box"])
            dx, dy = int(round(shift[0] * b.width / 16)), int(round(shift[1] * b.height / 16))
            moved = Box(**{**p["box"], "x": int(p["box"]["x"]) + dx, "y": int(p["box"]["y"]) + dy})
            imgs.append(render_box(slide, moved, max_side=max_side))
        return imgs, [int(p["magnification"]) for p in result["patches"]]

    # ---------------- 2 translation + 4 hidden norm (shift 0)
    trans_rows, norm_rows = [], []
    for row in picked:
        t1 = time.time()
        base_imgs, mags = render_case(row)
        pv0, thw0 = encode(proc, base_imgs)
        a0, n0 = collect_maps_and_norms(model, pv0, thw0)
        base_maps = maps_per_crop(a0, thw0)
        for (sp, maps), mag in zip(base_maps, mags):
            if sp[2] != 16:
                continue
            nm = np.stack([n0[b, sp[0]:sp[1]] for b in range(n0.shape[0])])
            for bname in BLOCKS_REPORT:
                m = maps.mean(0) if bname == "mean" else maps[int(bname)]
                nv = nm.mean(0) if bname == "mean" else nm[int(bname)]
                hot = hot8[str(bname)]
                norm_rows.append({"case": row["key"], "mag": mag, "block": str(bname), "rho_a_norm": spearman(m, nv),
                                  "hot8_norm_over_median": float(nv[hot].mean() / np.median(nv)),
                                  "top_a_norm_over_median": float(nv[int(np.argmax(m))] / np.median(nv))})
        for k in (1, 3):
            for axis in ("x", "y"):
                shift = (k, 0) if axis == "x" else (0, k)
                imgs, _ = render_case(row, shift=shift)
                pvk, thwk = encode(proc, imgs)
                ak, _ = collect_maps_and_norms(model, pvk, thwk)
                for (sp0, m0), (spk, mk), mag in zip(base_maps, maps_per_crop(ak, thwk), mags):
                    if sp0[2] != 16 or spk[2] != 16:
                        continue
                    for bname in BLOCKS_REPORT:
                        v0 = m0.mean(0) if bname == "mean" else m0[int(bname)]
                        vk = mk.mean(0) if bname == "mean" else mk[int(bname)]
                        fixed, content = shift_alignment(v0, vk, 16, k, axis)
                        hot = hot8[str(bname)]
                        trans_rows.append({"case": row["key"], "mag": mag, "k": k, "axis": axis, "block": str(bname),
                                           "rho_fixed_grid": fixed, "rho_content_aligned": content,
                                           "hot8_mass_shifted": float(vk[hot].sum() / vk.sum()),
                                           **hot_fixed_vs_moved(v0, vk, hot, 16, k, axis)})
        print(f"[2/4 {row['key']}] crops={len(base_imgs)} {time.time() - t1:.0f}s", flush=True)
    summary["translation"] = {}
    for bname in BLOCKS_REPORT:
        for k in (1, 3):
            rows = [r for r in trans_rows if r["block"] == str(bname) and r["k"] == k]
            summary["translation"][f"{bname}|k{k}"] = {
                "n": len(rows), "rho_fixed_grid_median": median([r["rho_fixed_grid"] for r in rows]),
                "rho_content_aligned_median": median([r["rho_content_aligned"] for r in rows]),
                "fixed_gt_content": sum(1 for r in rows if r["rho_fixed_grid"] is not None and r["rho_content_aligned"] is not None
                                        and r["rho_fixed_grid"] > r["rho_content_aligned"]),
                "hot8_mass_shifted_median": median([r["hot8_mass_shifted"] for r in rows]),
                "hot_fixed_mean_median": median([r["hot_fixed_mean"] for r in rows]),
                "hot_moved_mean_median": median([r["hot_moved_mean"] for r in rows]),
                "hot_fixed_gt_moved": sum(1 for r in rows if r["hot_fixed_mean"] is not None and r["hot_moved_mean"] is not None
                                          and r["hot_fixed_mean"] > r["hot_moved_mean"]),
                "top1_fixed_median": median([r["top1_fixed"] for r in rows]),
                "top1_moved_median": median([r["top1_moved"] for r in rows]),
                "top1_fixed_gt_moved": sum(1 for r in rows if r["top1_moved"] is not None and r["top1_fixed"] > r["top1_moved"])}
    summary["hidden_norm"] = {}
    for bname in BLOCKS_REPORT:
        rows = [r for r in norm_rows if r["block"] == str(bname)]
        summary["hidden_norm"][str(bname)] = {
            "n": len(rows), "rho_a_norm_median": median([r["rho_a_norm"] for r in rows]),
            "hot8_norm_over_median_median": median([r["hot8_norm_over_median"] for r in rows]),
            "top_a_norm_over_median_median": median([r["top_a_norm_over_median"] for r in rows])}
    print("[2 translation]", json.dumps(summary["translation"]), flush=True)
    print("[4 hidden norm]", json.dumps(summary["hidden_norm"]), flush=True)

    # ---------------- 3 resolution
    res_rows = []
    for row in picked[: args.res_cases]:
        for side in (384, 512, 768):
            t1 = time.time()
            imgs, mags = render_case(row, max_side=side)
            pvr, thwr = encode(proc, imgs)
            ar, _ = collect_maps_and_norms(model, pvr, thwr)
            for (sp, maps), mag in zip(maps_per_crop(ar, thwr), mags):
                g = sp[2]
                if sp[2] != sp[3]:
                    continue
                for bname in BLOCKS_REPORT:
                    m = maps.mean(0) if bname == "mean" else maps[int(bname)]
                    t16 = tmpl_rep[str(bname)]
                    top5 = np.argsort(-m)[:5]
                    res_rows.append({"case": row["key"], "side": side, "grid": g, "mag": mag, "block": str(bname),
                                     "rho_relative_template": spearman(m, resample_relative(t16, g)),
                                     "rho_absolute_template": absolute_overlap(t16, m, g),
                                     "top5_relative": [[round(r / g, 3), round(c / g, 3)] for r, c in (divmod(int(i), g) for i in top5)]})
            print(f"[3 {row['key']} side={side}] grids={sorted({f'{h}x{w}' for _, _, h, w in crop_spans(np.asarray(thwr))})} {time.time() - t1:.0f}s", flush=True)
    summary["resolution"] = {}
    for bname in BLOCKS_REPORT:
        for side in (384, 512, 768):
            rows = [r for r in res_rows if r["block"] == str(bname) and r["side"] == side]
            summary["resolution"][f"{bname}|{side}"] = {
                "n": len(rows), "grids": sorted({r["grid"] for r in rows}),
                "rho_relative_median": median([r["rho_relative_template"] for r in rows]),
                "rho_absolute_median": median([r["rho_absolute_template"] for r in rows])}
    print("[3 resolution]", json.dumps(summary["resolution"]), flush=True)

    summary["picked_cases"] = [r["key"] for r in picked]
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "rows.json").write_text(json.dumps({"translation": trans_rows, "hidden_norm": norm_rows, "resolution": res_rows}))
    print(f"[done] {time.time() - t0:.0f}s -> {out / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
