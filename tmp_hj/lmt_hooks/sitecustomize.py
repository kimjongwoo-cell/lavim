"""0919 Order 92 — Native Latent Mediation Trace on the latplan base. Original files untouched.

Chains tmp_hj/latplan_hooks. The Planner turn (first role, empty cache) is run TWICE from the same empty state:
  O = the slide thumbnail (native), B = a white image of the same size (Order 91 arm B).
Both caches have identical length / positions (same image size -> same grid). The Planner cache is split by provenance:
  S = columns before the first image token (system/prefix; identical in O and B by causality)
  V = <|image_pad|> columns (thumbnail visual KV)
  L = the latent_steps latent columns
  T = every other column after the first image token (vision_end, question text, turn-closing slack)
VLMAS_LMT_ARM selects the boundary handed to Navigator -> Reasoner -> Answerer:
  O0 / B0              native O / native B
  OL OV OT             O anchor, one group (L / V / T) copied from B            (early, Planner boundary)
  BL BV BT             B anchor, one group copied from O
  OLlate / BLlate      native O (B) through Reasoner; ONLY the Planner-latent columns of the pre-Answerer cache are
                       replaced with B's (O's) Planner latent KV right before the Answerer (late transplant)
VLMAS_LMT_FIX=<crops.jsonl> (fixed evidence): the Navigator still runs natively on the hybrid cache (its KV and latent
  steps are appended as usual), but its decoded selection is replaced by the one recorded in the file for the case, so
  the Reasoner receives exactly the anchor run's 12 crops.
VLMAS_LMT_OUT=<dir>: per case writes crops.jsonl (Navigator selection), meas.jsonl and <case>.pt (Answerer first-content
  token log-probs + decision-row hidden, and Planner/Navigator/Reasoner latent K/V on a few layers). The measurement runs
  on a copy of the pre-Answerer cache through the engine's VLMAS_ALSI slot (memory.alsi_diag.run is replaced).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_f = os.path.join(os.path.dirname(_HERE), "latplan_hooks", "sitecustomize.py")
exec(compile(open(_f).read(), _f, "exec"), {"__file__": _f, "__name__": "sitecustomize_latplan"})

_ARM = os.environ.get("VLMAS_LMT_ARM", "").strip()
if _ARM:
    import importlib.abc
    import importlib.machinery
    import json

    ARMS = {"O0", "B0", "OL", "OV", "OT", "BL", "BV", "BT", "OLlate", "BLlate"}
    assert _ARM in ARMS, _ARM
    _OUT = os.environ.get("VLMAS_LMT_OUT", "").strip()
    _FIXF = os.environ.get("VLMAS_LMT_FIX", "").strip()
    _FIX = {}
    if _FIXF:
        for line in open(_FIXF):
            r = json.loads(line)
            _FIX[int(r["case"])] = r
    if _OUT:
        os.makedirs(_OUT, exist_ok=True)
    SAVE_LAYERS = (0, 8, 16, 24, 35)
    _S = {"spans": [], "last_ids": None, "late": None, "late_done": None, "groups": None, "nav_sel": None}

    def _layers(cache):
        return cache.layers

    def _copy_cols(dst, src, cols):
        import torch
        idx = torch.tensor(cols, dtype=torch.long, device=_layers(dst)[0].keys.device)
        for ld, ls in zip(_layers(dst), _layers(src)):
            ld.keys.index_copy_(2, idx, ls.keys.index_select(2, idx))
            ld.values.index_copy_(2, idx, ls.values.index_select(2, idx))

    def _grab_cols(cache, cols):
        import torch
        idx = torch.tensor(cols, dtype=torch.long, device=_layers(cache)[0].keys.device)
        return [(l.keys.index_select(2, idx).clone(), l.values.index_select(2, idx).clone()) for l in _layers(cache)]

    def _put_cols(cache, cols, kv):
        import torch
        idx = torch.tensor(cols, dtype=torch.long, device=_layers(cache)[0].keys.device)
        for l, (k, v) in zip(_layers(cache), kv):
            l.keys.index_copy_(2, idx, k)
            l.values.index_copy_(2, idx, v)

    def _wrap_client(cls):
        orig_append = cls.append_only
        orig_gen = cls.generate_json
        orig_final = cls.generate_final_text

        def _wrap_processor(bb):
            if getattr(bb.processor, "_lmt_wrapped", False):
                return
            p_orig = bb.processor.__class__.__call__

            def p_call(pself, *a, **k):
                out = p_orig(pself, *a, **k)
                if k.get("images") is not None:
                    try:
                        _S["last_ids"] = out["input_ids"][0].tolist()
                    except Exception:
                        pass
                return out
            bb.processor.__class__.__call__ = p_call
            bb.processor._lmt_wrapped = True

        def _run_role(self, role, images, image_labels, system_prompt, user_prompt, m):
            bb = self._backend._engine._backbone
            before = int(self._state.cache_length) if self._state is not None and self._state.cache_length else 0
            _S["last_ids"] = None
            rec = orig_append(self, role=role, images=images, image_labels=image_labels,
                              system_prompt=system_prompt, user_prompt=user_prompt)
            after = int(self._state.cache_length)
            ids = _S["last_ids"] or []
            img_id = bb.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
            img = [before + j for j, t in enumerate(ids) if t == img_id]
            lat0 = before + len(ids)
            sp = {"role": role, "before": before, "after": after, "n_ids": len(ids), "img": img,
                  "lat": list(range(lat0, lat0 + m))}
            return rec, sp

        def append_only(self, *, role, images, image_labels, system_prompt, user_prompt):
            bb = self._backend._engine._backbone
            _wrap_processor(bb)
            m = int(self._latent_steps)
            if role != "evidence_planner":
                rec, sp = _run_role(self, role, images, image_labels, system_prompt, user_prompt, m)
                _S["spans"].append(sp)
                print(f"[LMT] {role} span=[{sp['before']},{sp['after']}) ids={sp['n_ids']} img={len(sp['img'])}",
                      flush=True)
                return rec
            # ---- Planner: run O and B from the same (empty) state ----
            from PIL import Image
            _S.update(spans=[], late=None, late_done=None, groups=None, nav_sel=None)
            s0 = self._state
            assert not s0.cache_length, f"Planner is expected to be the first role (cache {s0.cache_length})"
            rec_o, sp_o = _run_role(self, role, images, image_labels, system_prompt, user_prompt, m)
            st_o = self._state
            self._state = s0
            blank = tuple(Image.new("RGB", im.size, (255, 255, 255)) for im in images)
            rec_b, sp_b = _run_role(self, role, blank, image_labels, system_prompt, user_prompt, m)
            st_b = self._state
            assert (st_o.cache_length, st_o.position_cursor) == (st_b.cache_length, st_b.position_cursor), \
                (st_o.cache_length, st_o.position_cursor, st_b.cache_length, st_b.position_cursor)
            assert sp_o["img"] == sp_b["img"] and sp_o["lat"] == sp_b["lat"], "O/B token layout differs"
            n = int(st_o.cache_length)
            V = sp_o["img"]
            L = [c for c in sp_o["lat"] if c < n]
            first = V[0]
            T = [c for c in range(first, n) if c not in set(V) | set(L)]
            groups = {"S": list(range(0, first)), "V": V, "L": L, "T": T}
            _S["groups"] = {k: len(v) for k, v in groups.items()}
            # identity check: S columns must already be equal in O and B
            import torch
            dS = max(float((a.keys[:, :, :first] - b.keys[:, :, :first]).abs().max())
                     for a, b in zip(_layers(st_o.kv)[:4], _layers(st_b.kv)[:4]))
            dL = float((_layers(st_o.kv)[18].keys[:, :, L] - _layers(st_b.kv)[18].keys[:, :, L]).float().norm()
                       / _layers(st_o.kv)[18].keys[:, :, L].float().norm().clamp_min(1e-9))
            k_o, k_b = _layers(st_o.kv)[18].keys, _layers(st_b.kv)[18].keys
            same_obj = (st_o.kv is st_b.kv) or (k_o.data_ptr() == k_b.data_ptr())
            dV = float((k_o[:, :, V] - k_b[:, :, V]).float().norm() / k_o[:, :, V].float().norm().clamp_min(1e-9))
            print(f"[LMT] O/B check (before swap) same_obj={same_obj} V18_relOB={dV:.3f}", flush=True)
            anchor, other = (st_o, st_b) if _ARM[0] == "O" else (st_b, st_o)
            if _ARM in ("OL", "OV", "OT", "BL", "BV", "BT"):
                _copy_cols(anchor.kv, other.kv, groups[_ARM[1]])
            if _ARM in ("OLlate", "BLlate"):
                _S["late"] = (L, _grab_cols(other.kv, L))
            self._state = anchor
            other_kv = other.kv
            del other_kv, st_o, st_b, other
            torch.cuda.empty_cache()
            sp_o["groups"] = _S["groups"]
            _S["spans"].append(sp_o)
            print(f"[LMT] planner arm={_ARM} case={self._state.dataset_index} n={n} groups={_S['groups']} "
                  f"S_maxdiff={dS:.2e} L18_relOB={dL:.3f}", flush=True)
            return rec_o

        def generate_json(self, *, role, **kw):
            call = orig_gen(self, role=role, **kw)
            if role != "navigator":
                return call
            case = int(self._state.dataset_index)
            sel = call.value
            native = {"x5_ids": list(sel.x5_ids), "x20_ids": list(sel.x20_ids)}
            if _FIX:
                f = _FIX[case]
                sel = sel.model_copy(update={"x5_ids": tuple(f["x5_ids"]) if isinstance(sel.x5_ids, tuple)
                                             else list(f["x5_ids"]),
                                             "x20_ids": tuple(f["x20_ids"]) if isinstance(sel.x20_ids, tuple)
                                             else list(f["x20_ids"])})
                call = call.__class__(value=sel, record=call.record)
            used = {"x5_ids": list(sel.x5_ids), "x20_ids": list(sel.x20_ids)}
            _S["nav_sel"] = {"native": native, "used": used}
            if _OUT:
                with open(os.path.join(_OUT, "crops.jsonl"), "a") as fh:
                    fh.write(json.dumps({"case": case, "arm": _ARM, "fixed": bool(_FIX), **used,
                                         "native": native}) + "\n")
            print(f"[LMT] navigator case={case} native={native} used={used}", flush=True)
            return call

        def generate_final_text(self, **kw):
            if _S["late"] is not None and _S["late_done"] != id(self._state):
                cols, kv = _S["late"]
                _put_cols(self._state.kv, cols, kv)
                _S["late_done"] = id(self._state)
                print(f"[LMT] late transplant: {len(cols)} Planner latent columns swapped before Answerer", flush=True)
            return orig_final(self, **kw)

        cls.append_only = append_only
        cls.generate_json = generate_json
        cls.generate_final_text = generate_final_text

    def _run(engine, *, cache, position_cursor, system_prompt, user_prompt, json_prefix, case_index,
             max_new_tokens=512, **_ignored):
        import torch
        from copy import deepcopy
        from vision_text_mas.latent_terminal import generate_terminal_json
        if not _OUT:
            return
        bb = engine._backbone
        base_len = int(bb._kv_len(cache))
        cap = {}

        def hook(module, args, output):
            if "h" not in cap:
                h = output[0] if isinstance(output, tuple) else output
                cap["h"] = h[0, -1].detach().float().clone()
        head = bb.model.lm_head
        c = deepcopy(cache)
        hh = bb.lm.norm.register_forward_hook(hook)
        try:
            generate_terminal_json(backbone=bb, cache=c, position_cursor=position_cursor, system_prompt=system_prompt,
                                   user_prompt=user_prompt, json_prefix=(json_prefix or "") + ' "',
                                   max_new_tokens=1, temperature=0.0, top_p=1.0, do_sample=False)
        finally:
            hh.remove()
            del c
        h = cap["h"]
        with torch.no_grad():
            logits = head(h.to(head.weight.dtype)[None])[0].float()
        lp = torch.log_softmax(logits, -1)
        top = torch.topk(lp, 10)
        tok = bb.processor.tokenizer
        lat = {}
        for sp in _S["spans"]:
            cols = [x for x in sp["lat"] if x < base_len]
            if not cols:
                continue
            idx = torch.tensor(cols, dtype=torch.long, device=cache.layers[0].keys.device)
            lat[sp["role"]] = torch.stack([torch.stack([cache.layers[li].keys.index_select(2, idx)[0],
                                                        cache.layers[li].values.index_select(2, idx)[0]])
                                           for li in SAVE_LAYERS]).to(torch.float16).cpu()
        torch.save({"case": int(case_index), "arm": _ARM, "logprob": lp.to(torch.float16).cpu(),
                    "hidden": h.to(torch.float16).cpu(), "lat": lat, "layers": SAVE_LAYERS},
                   os.path.join(_OUT, f"{int(case_index):04d}.pt"))
        rec = {"case": int(case_index), "arm": _ARM, "fixed": bool(_FIX), "base_len": base_len,
               "groups": _S["groups"], "nav": _S["nav_sel"],
               "spans": [{k: sp[k] for k in ("role", "before", "after", "n_ids")} for sp in _S["spans"]],
               "top10": [[tok.decode([int(i)]), round(float(v), 4)] for v, i in zip(top.values, top.indices)]}
        with open(os.path.join(_OUT, "meas.jsonl"), "a") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[LMT] meas case={case_index} top3={rec['top10'][:3]}", flush=True)

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname not in ("vision_text_mas.latent_onepass", "memory.alsi_diag"):
                return None
            # chain: ask the finders behind this one (latplan_hooks patches the same module), so their
            # exec_module wrappers still run (a plain PathFinder lookup here would silently skip them)
            spec = None
            for fnd in sys.meta_path[sys.meta_path.index(self) + 1:]:
                fs = getattr(fnd, "find_spec", None)
                spec = fs(fullname, path, target) if fs else None
                if spec is not None:
                    break
            if spec is None or spec.loader is None:
                return spec
            orig = spec.loader.exec_module

            def exec_module(module):
                orig(module)
                if fullname == "vision_text_mas.latent_onepass":
                    _wrap_client(module.LatentRoleClient)
                else:
                    module.run = _run
                print(f"[LMT] patched {fullname} (arm={_ARM}, fix={bool(_FIX)})", flush=True)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
