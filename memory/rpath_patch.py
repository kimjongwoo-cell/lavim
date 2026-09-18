"""Pathways-style causal patching over the Reasoner suffix (method audit 0909).

Three sets: V (visual KV), R = {R_1..R_m} (the Reasoner's m latent columns),
A (Answerer). Two forwards per case -- true slide x and a frozen wrong slide
x~ -- and then, on copies of the TRUE terminal cache, patch pieces of the
wrong run in and re-decode:

    vis        V(x) <- V(x~)                 direct Visual -> Answerer path
    R{k}:{m}   R_{k:m}(x) <- R_{k:m}(x~)     distributed Reasoner -> Answerer
    vis+R1:m   both                          maximal patch (consistency)

The suffix ladder R_m, R_{m-1:m}, ..., R_{1:m} answers the question the
attention numbers cannot: does slide-specific visual information become
CAUSALLY encoded across the Reasoner suffix, even when the Answerer never
re-reads the visual columns itself?

Two passes, matched by dataset index:
    pass W  VLMAS_RPATH=wrong_visual VLMAS_RPATH_DUMP_KV=<dir>
            reasons over the donor slide's visual and dumps the K/V of its
            m latent columns (+ its visual columns) per case
    pass T  VLMAS_RPATH_PATCH=<dir>  (normal run)
            at the terminal, loads the case's wrong-run K/V and runs the ladder

Each row records the decoded answer and the teacher-forced log-prob of the
normal answer under every patch, so a graded shift is visible even when the
argmax answer holds. The wrong run's donor case (first case, unswapped) is
flagged and skipped.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch


def dump_dir() -> str:
    return os.environ.get("VLMAS_RPATH_DUMP_KV", "").strip()


def patch_dir() -> str:
    return os.environ.get("VLMAS_RPATH_PATCH", "").strip()


@torch.no_grad()
def dump_reasoner_kv(engine, cache, *, latent_cols, visual_cols, case_index, is_donor_case) -> None:
    """Pass W: persist the wrong run's Reasoner latent K/V (and visual K/V)."""
    target = dump_dir()
    if not target or not hasattr(cache, "layers"):
        return
    lat = latent_cols.to(device=engine._backbone.device, dtype=torch.long)
    vis = visual_cols.to(device=engine._backbone.device, dtype=torch.long)
    payload = {
        "case_index": int(case_index),
        "donor_case": bool(is_donor_case),
        "m": int(lat.numel()),
        "n_vis": int(vis.numel()),
        "k_lat": [layer.keys.index_select(2, lat).detach().to(torch.float16).cpu()
                  for layer in cache.layers],
        "v_lat": [layer.values.index_select(2, lat).detach().to(torch.float16).cpu()
                  for layer in cache.layers],
        "k_vis": [layer.keys.index_select(2, vis).detach().to(torch.float16).cpu()
                  for layer in cache.layers],
        "v_vis": [layer.values.index_select(2, vis).detach().to(torch.float16).cpu()
                  for layer in cache.layers],
    }
    root = Path(target)
    root.mkdir(parents=True, exist_ok=True)
    torch.save(payload, root / f"case_{int(case_index):04d}.pt")
    print(f"[RPathDump] case {int(case_index)} m={payload['m']} n_vis={payload['n_vis']} "
          f"donor_case={payload['donor_case']}", flush=True)


@torch.no_grad()
def dump_close_kv(engine, cache, *, close_cols, case_index) -> None:
    """Pass W: persist the K/V of the turn-closing columns (separate file)."""
    target = dump_dir()
    if not target or not hasattr(cache, "layers") or int(close_cols.numel()) == 0:
        return
    cols = close_cols.to(device=engine._backbone.device, dtype=torch.long)
    payload = {
        "case_index": int(case_index), "n_close": int(cols.numel()),
        "k_close": [layer.keys.index_select(2, cols).detach().to(torch.float16).cpu()
                    for layer in cache.layers],
        "v_close": [layer.values.index_select(2, cols).detach().to(torch.float16).cpu()
                    for layer in cache.layers],
    }
    torch.save(payload, Path(target) / f"case_{int(case_index):04d}_close.pt")


@torch.no_grad()
def run_patch_probe(engine, *, cache, position_cursor, system_prompt, user_prompt,
                    json_prefix, max_new_tokens, normal_output,
                    pre_answerer_len: int | None = None) -> None:
    """Pass T: suffix-patch the wrong run's Reasoner state into the true cache.

    pre_answerer_len is the cache length BEFORE the pipeline's own terminal
    decode. generate_terminal_json grows the cache in place (Answerer prompt +
    generated answer), so without cropping back every re-decode below would
    simply copy the answer already sitting in context.
    """
    from copy import deepcopy

    from vision_text_mas.latent_terminal import (
        generate_terminal_json, score_terminal_continuation,
    )

    target = patch_dir()
    case_index = getattr(engine, "_rpath_case_index", None)
    latent_cols = getattr(engine, "_rpath_latent_cols", None)
    visual_cols = getattr(engine, "_rpath_visual_cols", None)
    if not target or case_index is None or latent_cols is None or not hasattr(cache, "layers"):
        return
    path = Path(target) / f"case_{int(case_index):04d}.pt"
    if not path.is_file():
        print(f"[RPathPatch] case {case_index}: no wrong-run dump, skipped", flush=True)
        return
    wrong = torch.load(path, map_location="cpu")
    row: dict = {"case_index": int(case_index), "normal": normal_output,
                 "donor_case": bool(wrong.get("donor_case", False))}
    if row["donor_case"]:
        _append(row)
        print(f"[RPathPatch] case {case_index}: donor case, recorded only", flush=True)
        return
    backbone = engine._backbone
    device = backbone.device
    live_len = int(backbone._kv_len(cache))
    row["cache_len_live"] = live_len
    if pre_answerer_len is not None and live_len > int(pre_answerer_len):
        cache = deepcopy(cache)
        cache.crop(int(pre_answerer_len))
        row["cache_len_probe"] = int(pre_answerer_len)
        print(f"[RPathPatch] case {case_index}: cache {live_len} -> cropped to "
              f"pre-Answerer {int(pre_answerer_len)}", flush=True)
    else:
        row["cache_len_probe"] = live_len
    lat = latent_cols.to(device=device, dtype=torch.long)
    m = int(lat.numel())
    if int(wrong["m"]) != m:
        print(f"[RPathPatch] case {case_index}: m mismatch {wrong['m']} != {m}", flush=True)
        return
    vis = visual_cols.to(device=device, dtype=torch.long) if visual_cols is not None else None
    n_vis = min(int(wrong["n_vis"]), int(vis.numel())) if vis is not None else 0
    close_path = Path(target) / f"case_{int(case_index):04d}_close.pt"
    close_cols = getattr(engine, "_rpath_close_cols", None)
    wrong_close = torch.load(close_path, map_location="cpu") if close_path.is_file() else None
    if wrong_close is not None and close_cols is not None:
        close_cols = close_cols.to(device=device, dtype=torch.long)
        if int(close_cols.numel()) != int(wrong_close["n_close"]):
            wrong_close = None
    else:
        wrong_close = None
    row["has_close"] = wrong_close is not None
    # Geometry guard (review finding): the recorded column sets must tile the
    # pre-Answerer tail exactly -- visual < latent < close == cache end -- and
    # the wrong run must have dumped the same number of visual columns. Any
    # restage/relay/eviction drift would otherwise misplace the patches
    # silently.
    close_attr = getattr(engine, "_rpath_close_cols", None)
    pre_len = int(backbone._kv_len(cache))
    geom = {
        "pre_len": pre_len,
        "vis_max": int(vis.max()) if vis is not None and vis.numel() else None,
        "lat_min": int(lat.min()), "lat_max": int(lat.max()),
        "close_min": int(close_attr.min()) if close_attr is not None and close_attr.numel() else None,
        "close_max": int(close_attr.max()) if close_attr is not None and close_attr.numel() else None,
        "n_vis_true": int(vis.numel()) if vis is not None else 0,
        "n_vis_wrong": int(wrong["n_vis"]),
    }
    geom["ok"] = bool(
        geom["close_max"] is not None and geom["close_max"] + 1 == pre_len
        and geom["lat_max"] + 1 == geom["close_min"]
        and (geom["vis_max"] is None or geom["vis_max"] < geom["lat_min"])
        and geom["n_vis_true"] == geom["n_vis_wrong"])
    row["geom"] = geom
    if not geom["ok"]:
        print(f"[RPathPatch] case {case_index}: GEOMETRY MISMATCH {geom}", flush=True)

    def score(source, continuation=None):
        continuation = normal_output if continuation is None else continuation
        if not continuation:
            return None
        copy = deepcopy(source)
        try:
            total, _ = score_terminal_continuation(
                backbone=backbone, cache=copy, position_cursor=position_cursor,
                system_prompt=system_prompt, user_prompt=user_prompt,
                json_prefix=json_prefix, continuation=continuation)
            return float(total)
        except Exception as error:  # noqa: BLE001
            print(f"[RPathPatch] scoring failed: {error!r}", flush=True)
            return None
        finally:
            del copy

    def patched(cols_lat, use_vis, use_close=False):
        copy = deepcopy(cache)
        for li, layer in enumerate(copy.layers):
            if use_close and wrong_close is not None:
                layer.keys.index_copy_(2, close_cols,
                    wrong_close["k_close"][li].to(device, layer.keys.dtype))
                layer.values.index_copy_(2, close_cols,
                    wrong_close["v_close"][li].to(device, layer.values.dtype))
            if cols_lat is not None and cols_lat.numel():
                offs = cols_lat - lat[0]          # offsets into the dumped m block
                layer.keys.index_copy_(2, cols_lat,
                    wrong["k_lat"][li].index_select(2, offs.cpu()).to(device, layer.keys.dtype))
                layer.values.index_copy_(2, cols_lat,
                    wrong["v_lat"][li].index_select(2, offs.cpu()).to(device, layer.values.dtype))
            if use_vis and n_vis:
                target_cols = vis[:n_vis]
                layer.keys.index_copy_(2, target_cols,
                    wrong["k_vis"][li][:, :, :n_vis, :].to(device, layer.keys.dtype))
                layer.values.index_copy_(2, target_cols,
                    wrong["v_vis"][li][:, :, :n_vis, :].to(device, layer.values.dtype))
        return copy

    def dropped_vis():
        # Knockout control: remove the visual columns altogether (RoPE is baked
        # into the remaining keys, so positions stay valid), like the KVDrop
        # path at the terminal. If the Answerer reads V directly, this must
        # move answers broadly; if it never reads V, answers stay put.
        copy = deepcopy(cache)
        if vis is None or not n_vis:
            return copy
        total = int(backbone._kv_len(copy))
        keep = torch.ones(total, dtype=torch.bool, device=device)
        keep[vis[:n_vis]] = False
        keep_idx = keep.nonzero(as_tuple=True)[0]
        for layer in copy.layers:
            layer.keys = layer.keys.index_select(2, keep_idx)
            layer.values = layer.values.index_select(2, keep_idx)
        return copy

    def decode(copy):
        return generate_terminal_json(
            backbone=backbone, cache=copy, position_cursor=position_cursor,
            system_prompt=system_prompt, user_prompt=user_prompt,
            json_prefix=json_prefix, max_new_tokens=min(int(max_new_tokens), 160),
            temperature=engine._temperature, top_p=engine._top_p,
            do_sample=False, stop_strings=("}",))

    row["score_normal"] = score(cache)
    wrong_output = _wrong_run_answer(int(case_index))
    row["wrong_answer"] = wrong_output
    row["score_wrong_on_true"] = score(cache, wrong_output) if wrong_output else None
    # KV-level state dependence: per-step 1 - cos between the TRUE cache's
    # Reasoner latent K/V and the wrong run's, averaged over heads and layers,
    # plus a per-layer profile. Same columns, same case, only the slide differs
    # -- the KV analogue of Level 3, read straight off the cache.
    k_steps = [[] for _ in range(m)]
    v_steps = [[] for _ in range(m)]
    k_layers = []
    for li, layer in enumerate(cache.layers):
        tk = layer.keys.index_select(2, lat).float()            # [1,H,m,D]
        tv = layer.values.index_select(2, lat).float()
        wk = wrong["k_lat"][li].to(device).float()
        wv = wrong["v_lat"][li].to(device).float()
        ck = torch.nn.functional.cosine_similarity(tk, wk, dim=-1)  # [1,H,m]
        cv = torch.nn.functional.cosine_similarity(tv, wv, dim=-1)
        k_layers.append(float((1 - ck).mean()))
        for t in range(m):
            k_steps[t].append(float((1 - ck[0, :, t]).mean()))
            v_steps[t].append(float((1 - cv[0, :, t]).mean()))
    row["kv_dep_k_per_step"] = [round(sum(x) / len(x), 6) for x in k_steps]
    row["kv_dep_v_per_step"] = [round(sum(x) / len(x), 6) for x in v_steps]
    row["kv_dep_k_per_layer"] = [round(x, 6) for x in k_layers]
    if vis is not None and n_vis:
        vk = torch.nn.functional.cosine_similarity(
            cache.layers[-1].keys.index_select(2, vis[:n_vis]).float(),
            wrong["k_vis"][-1][:, :, :n_vis, :].to(device).float(), dim=-1)
        row["kv_dep_visual_lastlayer"] = round(float((1 - vk).mean()), 6)
    # "none" = cropped, unpatched copy through the probe's own decoder: the
    # identity control for the crop and for probe-vs-pipeline decode parity.
    modes = [("none", None, False, False), ("vis", None, True, False)]
    for k in range(m, 0, -1):                       # R_{k:m}: steps k..m (1-indexed)
        modes.append((f"R{k}:{m}", lat[k - 1:], False, False))
    modes.append(("vis+R1:%d" % m, lat, True, False))
    if wrong_close is not None:
        # the turn-closing columns as a carrier: alone, with the latent suffix,
        # and everything after the Reasoner prefill (should reproduce pass W)
        modes.append(("close", None, False, True))
        modes.append((f"R1:{m}+close", lat, False, True))
        modes.append((f"vis+R1:{m}+close", lat, True, True))
    # Every mode gets the graded signal (teacher-forced log-prob of the normal
    # answer AND of the wrong run's answer under the patched cache -- the
    # P(y|V_true) - P(y|V_wrong) contrast the audit asks for, two single
    # forwards, no generation). Free-running decode only on the endpoints,
    # where a literal answer flip is worth its 160-token generation.
    modes.append(("vis_drop", None, False, False))
    gen_modes = {"none", "vis", f"R1:{m}", f"vis+R1:{m}", "close", f"R1:{m}+close",
                 f"vis+R1:{m}+close", "vis_drop"}
    for name, cols_lat, use_vis, use_close in modes:
        try:
            copy = dropped_vis() if name == "vis_drop" else patched(cols_lat, use_vis, use_close)
            row[f"score_{name}"] = score(copy)
            row[f"score_{name}_wrong"] = score(copy, wrong_output) if wrong_output else None
            if name in gen_modes:
                row[f"answer_{name}"] = decode(copy)
            del copy
        except Exception as error:  # noqa: BLE001
            # one broken mode must not take down the ladder for the case
            print(f"[RPathPatch] case {case_index}: mode {name} failed: {error!r}", flush=True)
            row[f"error_{name}"] = repr(error)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _append(row)
    flips = sum(1 for n in gen_modes if row.get(f"answer_{n}") not in (None, normal_output))
    print(f"[RPathPatch] case {case_index}: endpoint flips {flips}/{len(gen_modes)}, "
          f"{len(modes)} modes scored", flush=True)


def _wrong_run_answer(case_index: int) -> str:
    """The wrong run's decoded answer for this case (pass W results dir).

    VLMAS_RPATH_WRONG_ROOT names the pass-W output root; otherwise it is
    inferred from the dump dir layout (<expq>/rpath_out[/<tag>]/kv ->
    <expq>/rpath_w_<tag>, tag=gtex when the dump dir sits directly under
    rpath_out).
    """
    import glob as _glob
    root = os.environ.get("VLMAS_RPATH_WRONG_ROOT", "").strip()
    if not root:
        kv = Path(patch_dir())
        if kv.parent.name == "rpath_out":
            tag, expq = "gtex", kv.parent.parent
        else:
            tag, expq = kv.parent.name, kv.parent.parent.parent
        root = str(expq / f"rpath_w_{tag}")
    for rj in _glob.glob(f"{root}/{case_index:03d}_*/result.json"):
        try:
            d = json.loads(Path(rj).read_text())
        except (OSError, ValueError):
            continue
        a = d.get("answer")
        text = a.get("answer") if isinstance(a, dict) else a
        return (text or "").strip()
    return ""


def _append(row: dict) -> None:
    out = os.environ.get("VLMAS_RPATH_PATCH_OUT", "").strip() or str(
        Path(patch_dir()) / "rpath_patch.jsonl")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


__all__ = ["dump_reasoner_kv", "dump_close_kv", "run_patch_probe", "dump_dir", "patch_dir"]
