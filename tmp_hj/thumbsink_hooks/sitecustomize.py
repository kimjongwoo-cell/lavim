"""0919 thumbnail-sink diagnostic on the latplan base (measurement only; answers unchanged). Original files untouched.

Chains tmp_hj/latplan_hooks. With VLMAS_THUMBSINK=<out.jsonl> (and VLMAS_ALSI=<same path> so the engine calls the
Answerer-boundary diagnostic slot; the original memory.alsi_diag.run is replaced and never runs):

  1. every role turn (LatentRoleClient.append_only) records its cache span, its prefill ids (via the processor) and
     latent_steps -> absolute columns of that role's image tokens, prompt text and latent steps;
  2. at the Answerer boundary, on a COPY of the pre-Answerer cache, the Answerer prompt is prefilled once and, for the
     decision row (last prompt row = the row that picks the first answer token), every layer's attention over the cache
     is recomputed per head (same math as tmp_hj/glvr/glvr_diag.rows_alpha_per_head) and summed per column group:
     sink (col 0), Planner thumbnail, Planner text, Planner latent, Navigator image/text/latent, Reasoner
     image/text/latent, Answerer prompt. For the thumbnail it also records the single most-attended thumbnail column,
     its mass and its value-vector norm relative to the thumbnail median (a "visual sink" = high attention, low value).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_f = os.path.join(os.path.dirname(_HERE), "latplan_hooks", "sitecustomize.py")
exec(compile(open(_f).read(), _f, "exec"), {"__file__": _f, "__name__": "sitecustomize_latplan"})

_OUT = os.environ.get("VLMAS_THUMBSINK", "").strip()
if _OUT:
    import importlib.abc
    import importlib.machinery
    import json

    _STATE = {"spans": [], "last_ids": None}

    def _wrap_client(cls):
        orig = cls.append_only

        def append_only(self, *, role, images, image_labels, system_prompt, user_prompt):
            eng = self._backend._engine
            bb = eng._backbone
            if not getattr(bb.processor, "_thumbsink_wrapped", False):
                p_orig = bb.processor.__class__.__call__

                def p_call(pself, *a, **k):
                    out = p_orig(pself, *a, **k)
                    if k.get("images") is not None:
                        try:
                            _STATE["last_ids"] = out["input_ids"][0].tolist()
                        except Exception:
                            pass
                    return out
                bb.processor.__class__.__call__ = p_call
                bb.processor._thumbsink_wrapped = True
            if role == "evidence_planner":
                _STATE["spans"] = []
            before = int(self._state.cache_length) if self._state is not None and self._state.cache_length else 0
            _STATE["last_ids"] = None
            rec = orig(self, role=role, images=images, image_labels=image_labels,
                       system_prompt=system_prompt, user_prompt=user_prompt)
            after = int(self._state.cache_length)
            ids = _STATE["last_ids"] or []
            img_id = bb.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
            img = [before + j for j, t in enumerate(ids) if t == img_id]
            m = int(self._latent_steps)
            lat0 = before + len(ids)
            _STATE["spans"].append({"role": role, "before": before, "after": after, "n_ids": len(ids),
                                    "img": img, "lat": list(range(lat0, lat0 + m)), "n_images": len(images)})
            print(f"[THUMBSINK] {role} span=[{before},{after}) ids={len(ids)} img={len(img)} lat={m} "
                  f"slack={after - before - len(ids) - m}", flush=True)
            return rec

        cls.append_only = append_only

    def _run(engine, *, cache, position_cursor, system_prompt, user_prompt, json_prefix, case_index,
             max_new_tokens=512, **_ignored):
        import torch
        from copy import deepcopy
        sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "glvr"))
        from glvr_diag import rows_alpha_per_head
        from memory.lu import _call_parts
        from vision_text_mas.latent_terminal import _assistant_prompt, generate_terminal_json
        bb = engine._backbone
        base_len = int(bb._kv_len(cache))
        tok = bb.processor.tokenizer
        prompt = _assistant_prompt(bb.processor, system_prompt=system_prompt, user_prompt=user_prompt,
                                   json_prefix=json_prefix)
        n_prompt = int(tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1])
        groups = {"sink": [0]}
        short = {"evidence_planner": "P", "navigator": "N", "reasoner": "R"}
        for sp in _STATE["spans"]:
            r = short.get(sp["role"], sp["role"])
            rng = set(range(sp["before"], min(sp["after"], base_len)))
            img, lat = set(sp["img"]) & rng, set(sp["lat"]) & rng
            groups[f"{r}_img"] = sorted(img)
            groups[f"{r}_lat"] = sorted(lat)
            groups[f"{r}_text"] = sorted(rng - img - lat - {0})
        thumb = torch.tensor(groups.get("P_img", []), dtype=torch.long)
        stats = []

        def post_for(li):
            def post(module, args, kwargs, output):
                hidden, pe, c = _call_parts(args, kwargs)
                if hidden is None or pe is None or c is None or int(hidden.shape[1]) < n_prompt:
                    return output
                with torch.no_grad():
                    keys, vals = c.layers[li].keys, c.layers[li].values
                    A = rows_alpha_per_head(module, hidden, pe, keys, n_prompt - 1, n_prompt).float()[:, 0, :]  # [H, L]
                    a = A.mean(0)                                                                  # head mean [L]
                    L = int(a.shape[0])
                    rec = {"layer": li}
                    for g, cols in groups.items():
                        idx = torch.tensor([x for x in cols if x < L], dtype=torch.long, device=a.device)
                        rec[g] = float(a[idx].sum()) if idx.numel() else 0.0
                    rec["A_prompt"] = float(a[base_len:].sum())
                    if thumb.numel():
                        t = thumb.to(a.device)
                        at = a[t]
                        j = int(torch.argmax(at))
                        V = vals[0] if vals.dim() == 4 else vals                                    # [Hkv, L, D]
                        vn = V[:, t, :].float().norm(dim=-1).mean(0)                               # [Nt]
                        rec["thumb_top_mass"] = float(at[j])
                        rec["thumb_top_col"] = int(thumb[j]) - int(thumb[0])
                        rec["thumb_top_vnorm_rel"] = float(vn[j] / vn.median().clamp_min(1e-9))
                        rec["thumb_n_gt10x"] = int((at > 10 * at.mean()).sum())
                    stats.append(rec)
                return output
            return post

        c = deepcopy(cache)
        hs = [layer.self_attn.register_forward_hook(post_for(li), with_kwargs=True)
              for li, layer in enumerate(bb.lm.layers)]
        try:
            generate_terminal_json(backbone=bb, cache=c, position_cursor=position_cursor, system_prompt=system_prompt,
                                   user_prompt=user_prompt, json_prefix=json_prefix, max_new_tokens=1,
                                   temperature=0.0, top_p=1.0, do_sample=False)
        finally:
            for h in hs:
                h.remove()
            del c
        out = {"case": int(case_index), "base_len": base_len, "n_prompt": n_prompt,
               "n_cols": {g: len(v) for g, v in groups.items()}, "spans": [
                   {k: sp[k] for k in ("role", "before", "after", "n_ids", "n_images")} for sp in _STATE["spans"]],
               "layers": stats}
        with open(_OUT, "a") as fh:
            fh.write(json.dumps(out) + "\n")
        mean = lambda k: round(sum(s.get(k, 0.0) for s in stats) / max(1, len(stats)), 4)
        print(f"[THUMBSINK] case {case_index} layers={len(stats)} sink={mean('sink')} thumb={mean('P_img')} "
              f"R_img={mean('R_img')} N_img={mean('N_img')} P_text={mean('P_text')} thumb_top={mean('thumb_top_mass')}",
              flush=True)

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname not in ("vision_text_mas.latent_onepass", "memory.alsi_diag"):
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or spec.loader is None:
                return spec
            orig = spec.loader.exec_module

            def exec_module(module):
                orig(module)
                if fullname == "vision_text_mas.latent_onepass":
                    _wrap_client(module.LatentRoleClient)
                else:
                    module.run = _run
                print(f"[THUMBSINK] patched {fullname}", flush=True)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
