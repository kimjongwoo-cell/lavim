"""09-19 measurement-only probe on the latplan base (this session): answers are unchanged. Original files untouched.

Two questions:
 (a) massive activation / attention sink: at which decoder layer does column 0 (<|im_start|>) get its outlier norm,
     and what are its key / value norms per layer?
 (b) why do broken answers read the Planner / Navigator latent columns so much at layer 0?  Layer-0 attention depends
     only on the input vectors, and a latent column's layer-0 input is the realigned hidden state fed back, not a token
     embedding. So record, per latent step, the layer-0 INPUT norm and every layer's output norm, and at the Answerer
     boundary the per-layer key / value norms of every latent column.

Chains tmp_hj/latplan_hooks. With VLMAS_MASSPROBE=<out.jsonl> and VLMAS_ALSI=<same path> (so the engine calls the
Answerer-boundary slot; memory.alsi_diag.run is replaced, the original never runs). One json line per case:
  prefill0: per layer, column-0 output norm, its max |x| and the argmax dim, median token norm of that prefill
  latents:  per role, per latent step: layer-0 input norm, per-layer output norm, per-layer max |x| + its dim, and the
            value on the sink's massive dim (column 0's argmax dim at layer 6, 4 in Qwen3-VL-4B) (and the median prefill-token input
            norm of the same role turn for reference)
  cache:    per layer, key / value norms of column 0, of each role's latent columns (mean), and the median of all columns
  qk0:      layer-0 decomposition of the Answerer decision row per column group (sink, P/N/R latents, rest): attention
            mass, mean logit, mean |k|, mean cos(q,k), value-mixture norm and its share of the total, top-3 heads
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_f = os.path.join(os.path.dirname(_HERE), "latplan_hooks", "sitecustomize.py")
exec(compile(open(_f).read(), _f, "exec"), {"__file__": _f, "__name__": "sitecustomize_latplan"})

_OUT = os.environ.get("VLMAS_MASSPROBE", "").strip()
if _OUT:
    import importlib.abc
    import importlib.machinery
    import json

    _ST = {"spans": [], "last_ids": None, "role": None, "phase": None, "prefill0": None, "lat": {}, "cur": None,
           "hooked": False, "in_norm": None, "pref_in_med": {}}

    def _install_layer_hooks(bb):
        if _ST["hooked"]:
            return
        import torch
        layers = bb.lm.layers

        def pre0(mod, args, kwargs):
            h = kwargs.get("hidden_states", args[0] if args else None)
            if h is None or _ST["role"] is None:
                return
            with torch.no_grad():
                n = h[0].float().norm(dim=-1)
                if int(h.shape[1]) == 1:
                    _ST["cur"] = {"in": float(n[0]), "out": []}
                else:
                    _ST["pref_in_med"].setdefault(_ST["role"], float(n.median()))

        def post(li):
            def f(mod, args, kwargs, out):
                h = out[0] if isinstance(out, tuple) else out
                if h is None or _ST["role"] is None:
                    return
                with torch.no_grad():
                    q = int(h.shape[1])
                    if q == 1 and _ST["cur"] is not None:
                        x1 = h[0, 0].float()
                        _ST["cur"]["out"].append(round(float(x1.norm()), 3))
                        j1 = int(x1.abs().argmax())
                        _ST["cur"].setdefault("maxdim", []).append(j1)
                        _ST["cur"].setdefault("maxabs", []).append(round(float(x1.abs().max()), 2))
                        _ST["cur"].setdefault("d_sink", []).append(round(float(x1[_ST.get("sink_dim", 4)]), 2))
                        if li == len(layers) - 1:
                            _ST["lat"].setdefault(_ST["role"], []).append(_ST["cur"])
                            _ST["cur"] = None
                    elif q > 1 and _ST["role"] == "evidence_planner" and _ST["prefill0"] is not None \
                            and len(_ST["prefill0"]) == li:
                        x = h[0].float()
                        c0 = x[0]
                        j = int(c0.abs().argmax())
                        if li == 6:
                            _ST["sink_dim"] = j
                        _ST["prefill0"].append({"layer": li, "c0_norm": round(float(c0.norm()), 3),
                                                "c0_maxabs": round(float(c0.abs().max()), 3), "c0_argdim": j,
                                                "med_norm": round(float(x.norm(dim=-1).median()), 3),
                                                "p99_norm": round(float(torch.quantile(x.norm(dim=-1), 0.99)), 3)})
            return f

        layers[0].register_forward_pre_hook(pre0, with_kwargs=True)
        for li, layer in enumerate(layers):
            layer.register_forward_hook(post(li), with_kwargs=True)
        _ST["hooked"] = True
        print(f"[MASSPROBE] hooked {len(layers)} decoder layers", flush=True)

    def _wrap_client(cls):
        orig = cls.append_only

        def append_only(self, *, role, images, image_labels, system_prompt, user_prompt):
            eng = self._backend._engine
            bb = eng._backbone
            _install_layer_hooks(bb)
            if not getattr(bb.processor, "_massprobe_wrapped", False):
                p_orig = bb.processor.__class__.__call__

                def p_call(pself, *a, **k):
                    out = p_orig(pself, *a, **k)
                    if k.get("images") is not None:
                        try:
                            _ST["last_ids"] = out["input_ids"][0].tolist()
                        except Exception:
                            pass
                    return out
                bb.processor.__class__.__call__ = p_call
                bb.processor._massprobe_wrapped = True
            if role == "evidence_planner":
                _ST.update(spans=[], prefill0=[], lat={}, pref_in_med={}, cur=None)
            before = int(self._state.cache_length) if self._state is not None and self._state.cache_length else 0
            _ST["last_ids"], _ST["role"] = None, role
            try:
                rec = orig(self, role=role, images=images, image_labels=image_labels,
                           system_prompt=system_prompt, user_prompt=user_prompt)
            finally:
                _ST["role"] = None
            ids = _ST["last_ids"] or []
            m = int(self._latent_steps)
            lat0 = before + len(ids)
            _ST["spans"].append({"role": role, "lat": list(range(lat0, lat0 + m))})
            return rec

        cls.append_only = append_only

    def _qk_layer0(engine, cache, position_cursor, system_prompt, user_prompt, json_prefix):
        """Layer-0 decomposition of the Answerer decision row (last prompt row) over the cached columns, per head:
        attention mass, mean logit, |q|, mean |k|, mean cos(q,k), and the norm of the value mixture sum_j a_j v_j per
        column group (sink, P/N/R latents, rest), plus the total. Measurement on a COPY of the cache."""
        import torch
        from copy import deepcopy
        from memory.lu import _call_parts
        from vision_text_mas.latent_terminal import generate_terminal_json
        from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb, repeat_kv
        bb = engine._backbone
        base_len = int(bb._kv_len(cache))
        groups = {"sink": [0]}
        for sp in _ST["spans"]:
            r = {"evidence_planner": "P", "navigator": "N", "reasoner": "R"}.get(sp["role"], sp["role"])
            groups[f"{r}lat"] = [c for c in sp["lat"] if c < base_len]
        res = {}

        def post(module, args, kwargs, output):
            if res:
                return output
            hidden, pe, c = _call_parts(args, kwargs)
            if hidden is None or pe is None or c is None or int(hidden.shape[1]) < 2:
                return output
            with torch.no_grad():
                keys, vals = c.layers[0].keys, c.layers[0].values
                L, T, Hd = int(keys.shape[2]), int(hidden.shape[1]), module.head_dim
                G = module.num_key_value_groups
                cos, sin = pe
                row = torch.tensor([T - 1], device=hidden.device)
                h = hidden.index_select(1, row)
                q = module.q_norm(module.q_proj(h).view(1, 1, -1, Hd)).transpose(1, 2)
                q, _ = apply_rotary_pos_emb(q, torch.zeros_like(q), cos.index_select(1, row), sin.index_select(1, row))
                K = repeat_kv(keys, G).float()[0]; V = repeat_kv(vals, G).float()[0]      # [H, L, D]
                qv = q.float()[0, :, 0, :]                                                  # [H, D]
                z = (K @ qv[:, :, None])[..., 0] * module.scaling                           # [H, L]
                A = torch.softmax(z, dim=-1)
                kn = K.norm(dim=-1); qn = qv.norm(dim=-1)                                   # [H, L], [H]
                cs = z / module.scaling / (kn * qn[:, None]).clamp_min(1e-9)
                tot = (A[:, :, None] * V).sum(1).norm(dim=-1)                               # [H]
                allc = set(range(L))
                used = set()
                for g, cols in groups.items():
                    used |= set(cols)
                gg = dict(groups); gg["rest"] = sorted(allc - used)
                out = {"q_norm": round(float(qn.mean()), 3), "valmix_total": round(float(tot.mean()), 4)}
                for g, cols in gg.items():
                    if not cols:
                        continue
                    idx = torch.tensor(cols, dtype=torch.long, device=K.device)
                    mass = A[:, idx].sum(1)                                                 # [H]
                    vm = (A[:, idx, None] * V[:, idx, :]).sum(1).norm(dim=-1)               # [H]
                    out[g] = {"mass": round(float(mass.mean()), 5), "mass_maxhead": round(float(mass.max()), 4),
                              "mass_top3_heads": [int(x) for x in torch.topk(mass, 3).indices],
                              "logit": round(float(z[:, idx].mean()), 3), "k_norm": round(float(kn[:, idx].mean()), 3),
                              "cos_qk": round(float(cs[:, idx].mean()), 4), "cos_qk_max": round(float(cs[:, idx].max()), 4),
                              "valmix": round(float(vm.mean()), 4),
                              "valmix_share": round(float((vm / tot.clamp_min(1e-9)).mean()), 4)}
                res.update(out)
            return output

        c2 = deepcopy(cache)
        hk = bb.lm.layers[0].self_attn.register_forward_hook(post, with_kwargs=True)
        try:
            generate_terminal_json(backbone=bb, cache=c2, position_cursor=position_cursor, system_prompt=system_prompt,
                                   user_prompt=user_prompt, json_prefix=json_prefix, max_new_tokens=1,
                                   temperature=0.0, top_p=1.0, do_sample=False)
        finally:
            hk.remove()
            del c2
        return res

    def _run(engine, *, cache, case_index, position_cursor=None, system_prompt=None, user_prompt=None,
             json_prefix=None, **_ignored):
        import torch
        qk0 = {}
        try:
            qk0 = _qk_layer0(engine, cache, position_cursor, system_prompt, user_prompt, json_prefix)
        except Exception as e:                                                             # measurement must not kill the run
            qk0 = {"error": repr(e)[:300]}
        rows = []
        with torch.no_grad():
            for li, layer in enumerate(cache.layers):
                K, V = layer.keys, layer.values
                K = K[0] if K.dim() == 4 else K                      # [Hkv, L, D]
                V = V[0] if V.dim() == 4 else V
                kn = K.float().norm(dim=-1).mean(0)                   # head-mean per column [L]
                vn = V.float().norm(dim=-1).mean(0)
                rec = {"layer": li, "k_c0": round(float(kn[0]), 3), "v_c0": round(float(vn[0]), 3),
                       "k_med": round(float(kn.median()), 3), "v_med": round(float(vn.median()), 3)}
                for sp in _ST["spans"]:
                    r = {"evidence_planner": "P", "navigator": "N", "reasoner": "R"}.get(sp["role"], sp["role"])
                    idx = torch.tensor([c for c in sp["lat"] if c < int(kn.shape[0])], dtype=torch.long, device=kn.device)
                    if idx.numel():
                        rec[f"k_{r}lat"] = round(float(kn[idx].mean()), 3)
                        rec[f"v_{r}lat"] = round(float(vn[idx].mean()), 3)
                rows.append(rec)
        out = {"case": int(case_index), "qk0": qk0, "prefill0": _ST["prefill0"], "pref_in_med": _ST["pref_in_med"],
               "latents": _ST["lat"], "cache": rows}
        with open(_OUT, "a") as fh:
            fh.write(json.dumps(out) + "\n")
        p0 = _ST["prefill0"] or []
        jump = max(range(1, len(p0)), key=lambda i: p0[i]["c0_norm"] / max(p0[i - 1]["c0_norm"], 1e-6)) if len(p0) > 1 else -1
        pl = _ST["lat"].get("evidence_planner", [])
        print(f"[MASSPROBE] case {case_index} c0_norm_jump_at_layer={jump} "
              f"P_lat_in_norm={[round(x['in'], 1) for x in pl][:10]} pref_in_med={_ST['pref_in_med']}", flush=True)

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
                print(f"[MASSPROBE] patched {fullname}", flush=True)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
