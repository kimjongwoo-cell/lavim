#!/usr/bin/env python3
"""Retained white-token fraction: Pruning-B vs EOVC, on the same early-observation states.

Answers "how much white does Pruning-B actually keep" with a direct measurement instead of a
quoted number, and gets EOVC's value on the same cases and the same label (page §12 asks for
retained white fraction).

White label (identical rule to scratchpad/vistok_viz/white_stat.py): a merged token is white when
the mean gray of its 32x32 px cell is > 215. Evaluation only; no selector sees it.

Pipeline reproduced exactly as backbone/qwen3vl.py::_hierarchy_pruned_vision_features does it:
  patch_embed + fast_pos_embed_interpolate -> first `observe_blocks` vision blocks
  grouped = hidden.reshape(-1, 4, d).float();  pooled = grouped.mean(1)
  per image: novelty = 1 - cos(pooled, image mean), texture = var over the 4 patches,
             each min-max normalised, score = novelty + 0.25 * texture
  keep = memory.prune.hierarchy_prefill_keep_mask(score, quota=min_tokens_per_image, keep_ratio)
EOVC keep = memory.eovc.select(grouped, token_counts, budget) on the SAME grouped states.

usage: white_pruneb_eovc.py <out_dir> [--ds gtex] [--keeps 0.25,0.5] [--quotas 16,8]
                            [--cases N] [--threads 16]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path("/home/users/whddn12316/wsi_latent_0915_decode_hj")
SRC = Path("/home/users/whddn12316/wsi_latent_0902_2155_hj")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SRC / "sender_relay_exp"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

DATA = Path("/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda")
MODEL = Path("/home/users/whddn12316/models/Qwen3-VL-4B-Thinking")
AUDIT = SRC / "sender_relay_exp" / "runs" / "vit_sal_audit" / "full"
WHITE_GRAY = 215.0


def early_states(visual, pv, thw, observe_blocks: int) -> torch.Tensor:
    """hidden after the first observe_blocks vision blocks, exactly as the backbone does it."""
    hidden = visual.patch_embed(pv.type(visual.dtype))
    hidden = hidden + visual.fast_pos_embed_interpolate(thw)
    rotary = visual.rot_pos_emb(thw)
    seq_len = int(hidden.shape[0])
    rotary = rotary.reshape(seq_len, -1)
    emb = torch.cat((rotary, rotary), dim=-1)
    pe = (emb.cos(), emb.sin())
    raw_counts = tuple(int(t * h * w) for t, h, w in thw.tolist())
    cu = torch.tensor((0, *torch.tensor(raw_counts).cumsum(0).tolist()), dtype=torch.int32)
    split_at = min(max(1, observe_blocks), len(visual.blocks) - 1)
    for block in visual.blocks[:split_at]:
        hidden = block(hidden, cu_seqlens=cu, position_embeddings=pe)
    return hidden


def pruneb_scores(grouped: torch.Tensor, token_counts) -> torch.Tensor:
    import torch.nn.functional as F
    pooled = grouped.mean(dim=1)
    scores = torch.empty(pooled.shape[0], dtype=torch.float32)
    off = 0
    for count in token_counts:
        pf = pooled[off:off + count]
        center = pf.mean(dim=0, keepdim=True)
        novelty = 1.0 - F.cosine_similarity(pf, center, dim=-1)
        texture = grouped[off:off + count].var(dim=1, unbiased=False).mean(dim=-1)
        novelty = (novelty - novelty.min()) / (novelty.max() - novelty.min() + 1e-6)
        texture = (texture - texture.min()) / (texture.max() - texture.min() + 1e-6)
        scores[off:off + count] = novelty + 0.25 * texture
        off += count
    return scores


def white_mask(images, grids) -> np.ndarray:
    """cell mean gray > 215 per merged token, in the backbone's token order."""
    out = []
    for im, (gh, gw) in zip(images, grids):
        g = np.asarray(im.convert("L"), dtype=np.float32)
        ch, cw = g.shape[0] // gh, g.shape[1] // gw
        cell = g[:gh * ch, :gw * cw].reshape(gh, ch, gw, cw).mean(axis=(1, 3)).ravel()
        out.append(cell > WHITE_GRAY)
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--ds", default="gtex")
    ap.add_argument("--keeps", default="0.25,0.5")
    ap.add_argument("--quotas", default="16,8")
    ap.add_argument("--observe-blocks", type=int, default=4)
    ap.add_argument("--cases", type=int, default=0)
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    keeps = [float(x) for x in args.keeps.split(",")]
    quotas = [int(x) for x in args.quotas.split(",")]

    from transformers import AutoModelForImageTextToText, AutoProcessor
    from memory import eovc
    from memory.prune import hierarchy_prefill_keep_mask
    from vision_text_mas.cli import _open_slide
    from vision_text_mas.dataset import load_cases
    from vision_text_mas.geometry import Box
    from vision_text_mas.navigation_render import render_box
    from vit_sal_stats import crop_spans

    proc = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.float32).eval()
    visual = model.model.visual

    manifest = [r for r in json.loads((AUDIT / "manifest.json").read_text())
                if r["ds"] == args.ds and r.get("status") in ("ok", "cached")]
    if args.cases:
        manifest = manifest[:args.cases]
    rows, t0 = [], time.time()
    for row in manifest:
        tc = time.time()
        z = np.load(AUDIT / "cases" / f"{row['key']}.npz")
        thw = torch.from_numpy(z["thw"])
        result = json.loads(Path(row["result"]).read_text())
        case = load_cases(DATA / f"{args.ds}.json", slide_root=DATA / "slides",
                          indices=(int(row["dataset_index"]),))[0]
        slide = _open_slide(case)
        images = [render_box(slide, Box(**p["box"]), max_side=512) for p in result["patches"]]
        enc = proc.image_processor(images=images, return_tensors="pt")
        assert torch.equal(enc["image_grid_thw"], thw), (enc["image_grid_thw"], thw)
        with torch.no_grad():
            hidden = early_states(visual, enc["pixel_values"].float(), thw, args.observe_blocks)
        grouped = hidden.reshape(-1, int(visual.spatial_merge_unit), hidden.shape[-1]).float()
        spans = crop_spans(thw.numpy())
        grids = [(sp[2], sp[3]) for sp in spans]
        counts = tuple(sp[1] - sp[0] for sp in spans)
        N = int(sum(counts))
        assert grouped.shape[0] == N, (grouped.shape, N)
        wm = white_mask(images, grids)
        assert wm.size == N, (wm.size, N)
        parents = tuple(int(p.get("parent_index", -1)) if isinstance(p, dict) else -1
                        for p in result["patches"])
        scores = pruneb_scores(grouped, counts)
        rec = {"key": row["key"], "n": N, "white_all": float(wm.mean()),
                "counts": list(counts), "arms": {}, "overlap": {}}
        masks = {}
        for keep in keeps:
            B = int(round(keep * N))
            info = eovc.select(grouped, counts, B)
            k_eovc = info["keep"].numpy()
            rec["arms"][f"EOVC keep{keep}"] = {
                "white": float(wm[k_eovc].mean()), "n_keep": int(k_eovc.sum()),
                "per_support": info["per_support"], "explained": info["explained_frac"]}
            masks[f"EOVC keep{keep}"] = k_eovc
            top = np.zeros(N, dtype=bool)
            top[np.argsort(-scores.numpy())[:B]] = True
            rec["arms"][f"Pruning-B score top-k keep{keep}"] = {
                "white": float(wm[top].mean()), "n_keep": int(top.sum())}
            masks[f"Pruning-B score top-k keep{keep}"] = top
            for q in quotas:
                km = hierarchy_prefill_keep_mask(
                    scores, token_counts=counts, parent_indices=parents,
                    keep_ratio=keep, min_tokens_per_image=q).numpy()
                rec["arms"][f"Pruning-B keep{keep} quota{q}"] = {
                    "white": float(wm[km].mean()), "n_keep": int(km.sum()),
                    "per_support": [int(km[int(a):int(b)].sum()) for a, b, *_ in spans]}
                masks[f"Pruning-B keep{keep} quota{q}"] = km
            rnd = np.zeros(N, dtype=bool)
            rnd[np.random.default_rng(0).choice(N, B, replace=False)] = True
            rec["arms"][f"Random keep{keep}"] = {"white": float(wm[rnd].mean()),
                                                 "n_keep": int(rnd.sum())}
            masks[f"Random keep{keep}"] = rnd
            # §12 falsifier 1: does EOVC pick a different subset at all?
            e = masks[f"EOVC keep{keep}"]
            for other, m in masks.items():
                if other.startswith("EOVC"):
                    continue
                rec["overlap"][f"EOVC vs {other}"] = float((e & m).sum() / max(e.sum(), 1))
        rows.append(rec)
        a = rec["arms"]
        print(f"[{row['key']}] {time.time()-tc:.0f}s N={N} white_all={rec['white_all']:.3f} "
              f"| PB q16 {a[f'Pruning-B keep{keeps[0]} quota16']['white']:.3f} "
              f"| PB topk {a[f'Pruning-B score top-k keep{keeps[0]}']['white']:.3f} "
              f"| EOVC {a[f'EOVC keep{keeps[0]}']['white']:.3f} "
              f"| rand {a[f'Random keep{keeps[0]}']['white']:.3f}", flush=True)

    names = list(rows[0]["arms"]) if rows else []
    onames = list(rows[0]["overlap"]) if rows else []
    summary = {"ds": args.ds, "cases": len(rows), "observe_blocks": args.observe_blocks,
               "white_rule": f"cell mean gray > {WHITE_GRAY}",
               "white_all": float(np.mean([r["white_all"] for r in rows])),
               "arms": {n: float(np.mean([r["arms"][n]["white"] for r in rows])) for n in names},
               "overlap": {n: float(np.mean([r["overlap"][n] for r in rows])) for n in onames},
               "per_case": rows}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"\n== retained white fraction, {len(rows)} {args.ds} cases, "
          f"rule = cell mean gray > {WHITE_GRAY:.0f}")
    print(f"{'all tokens':>34} {summary['white_all']:.3f}")
    for n in names:
        print(f"{n:>34} {summary['arms'][n]:.3f}")
    print(f"\n== subset overlap with EOVC (chance = keep ratio; falsifier 1)")
    for n in onames:
        print(f"{n:>44} {summary['overlap'][n]:.3f}")
    print(f"\n[done] {time.time()-t0:.0f}s -> {out/'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
