"""Bridge: prefill-site C1 keep mask -> backbone visual provenance (_visual_meta).

Why: with VLMAS_C1_QASC=1 (QASC, and VLMAS_C1_SELECT=ntrs|ssda through the same entry point) the
visual tokens are dropped BEFORE the decoder prefill (backbone/qwen3vl.py, on_kv path), so
_wsi_consolidate_boundary returns keep=None and _record_visual_meta(thw, None, cols) sees 768
retained columns against 3072 grid tokens -> "token/grid mismatch" -> _visual_meta = None.
Every consumer of _visual_meta (memory/lpvh.py, memory/psar.py) then SKIPs the whole case.

Fix without touching backbone/qwen3vl.py or memory/qasc_select.py: wrap
memory.qasc_select.select_vision_features so that, right after C1 selects, the pre-C1 keep mask
[N] is kept on the backbone, and wrap this backbone instance's _record_visual_meta once so that a
call with keep=None whose retained column count equals keep.sum() receives that mask. The
retained tokens enter the prefill in ascending original order (sequence_keep), which is exactly
the order _record_visual_meta assumes for keep.nonzero(). The mask is consumed by the next
_record_visual_meta call (same Reasoner prefill) and cleared. No tensor the model uses changes.
Activated only by tmp_hj/lpvh_ntrs_hooks/sitecustomize.py with VLMAS_C1_META_BRIDGE=1.
"""
import torch


def bridge_record(bb) -> None:
    if getattr(bb, "_c1_meta_bridge_installed", False):
        return
    original = bb._record_visual_meta

    def _record_visual_meta(grids, keep, abs_cols):
        pending = getattr(bb, "_c1_prefill_keep", None)
        bb._c1_prefill_keep = None
        used = False
        if keep is None and pending is not None and isinstance(abs_cols, torch.Tensor):
            if int(pending.sum()) == int(abs_cols.numel()):
                keep, used = pending, True
        result = original(grids, keep, abs_cols)
        if pending is not None:
            meta = getattr(bb, "_visual_meta", None)
            state = "ok" if meta is not None else f"none ({getattr(bb, '_visual_meta_note', '')})"
            n_cols = int(abs_cols.numel()) if isinstance(abs_cols, torch.Tensor) else -1
            print(f"[C1META] keep {int(pending.sum())}/{int(pending.numel())} cols={n_cols} used={used} "
                  f"meta={state}", flush=True)
        return result

    bb._record_visual_meta = _record_visual_meta
    bb._c1_meta_bridge_installed = True


def patch(module) -> bool:
    original = module.select_vision_features
    if getattr(original, "_c1_meta_bridge", False):
        return False

    def select_vision_features(bb, pixel_values, grid_thw):
        feats, keep = original(bb, pixel_values, grid_thw)
        bridge_record(bb)
        bb._c1_prefill_keep = keep.detach().to(device="cpu", dtype=torch.bool).clone()
        return feats, keep

    select_vision_features._c1_meta_bridge = True
    module.select_vision_features = select_vision_features
    print("[C1META] active: prefill-site C1 keep mask -> backbone _record_visual_meta", flush=True)
    return True
