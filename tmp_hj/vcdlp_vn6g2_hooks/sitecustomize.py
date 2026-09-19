"""[09-19 COPY of tmp_hj/nlqp_hooks/sitecustomize.py, frozen; ONLY change: chains tmp_hj/nova_lgv_latplan_hooks (NOVA Final value-norm signal via VLMAS_NOVA_SIGNAL) instead of nova_rho_latplan_hooks]
C2 NLQP — Native Latent Query Priming (Notion C1/C2 Method Evolution #30). Original files untouched.

Chains tmp_hj/nova_rho_latplan_hooks (NOVA rho + latplan; itself chains latplan_hooks -> lpvh_ntrs_hooks).
With VLMAS_NLQP=<mode>, during every Answerer call (LatentRoleClient.generate_final_text), ONCE, at the receiver row b
that forms the first answer-content token, in every decoder layer l. The Answerer prompt is teacher-forced with
'{"answer":', so the last prompt row only picks the JSON format token (' "'); row b is therefore the first 1-row
generation step whose input token contains the opening quote '"' (user decision 09-19; applying at the last prompt
row changed only that format token and broke the JSON). Prompt, prefix and every other row stay native.

  q_b      = RoPE(q_norm(W_Q LN(x_b)))                     native query of row b (x_b = raw residual into layer l)
  rho_k    = softmax_k(q_b K_{R,k}^T / sqrt(d_h))           pre-read over the m native Reasoner latent columns only
  c_R      = sum_k rho_k V_{R,k}  (per head, GQA)
  r_R      = W_O concat_h c_R
  x^_b     = x_b + r_R
  q^_b     = W_Q LN(x^_b)  -> replaces row b's q_proj output (q_norm / RoPE / full-cache attention stay native)

K/V of row b, all cache contents, the residual update and the MLP are native: r_R only changes row b's query.
No extra token, no cache column, no gain / temperature / top-k; one row of one generation step.

  VLMAS_NLQP=native    canonical NLQP
  VLMAS_NLQP=uniform   ablation: rho_k = 1/m (no receiver-latent QK selection)
  VLMAS_NLQP=direct    ablation: r_R is added to row b's attention output (payload injection); query untouched
  VLMAS_NLQP=identity  hook check: the full path with r_R = 0 (must reproduce the native answers)
  VLMAS_NLQP=vcdlp     C2 VCDLP (Method Evolution #31): after the Reasoner append, the m latent steps are replayed once
                       from the pre-latent boundary with the latent rows' attention to the Reasoner visual columns
                       masked (visual KV stays in the cache; same slots / positions / realign), giving
                       dV_k = V_R,k^full - V_R,k^noV per layer / KV head; dV is centered over k. At row b the native
                       query selects latents with the native K_R^full (alpha = softmax(q K^T/sqrt d)) and
                       r = W_O concat_h sum_k alpha_k dV~_k primes the query exactly like NLQP.
  VLMAS_NLQP=vcdlp_uncentered   ablation (#31 17.6): dV not centered
  VLMAS_NLQP=vcdlp_check        identity (#31 17.1): replay WITHOUT the mask -> dV must be ~0 -> query unchanged
  VLMAS_NLQP_LOG=<jsonl>  per Answerer call: latent cols, layers applied, mean rho entropy / max,
                          |r_R|/|x_b|, |q^-q|/|q| (or |r_R|/|attn_out| for direct)
Reasoner latent columns come from engine._rpath_latent_cols (set by the engine right after the Reasoner append:
the last m cache columns before the turn is closed).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_f = os.path.join(os.path.dirname(_HERE), "nova_lgv_latplan_hooks", "sitecustomize.py")
exec(compile(open(_f).read(), _f, "exec"), {"__file__": _f, "__name__": "sitecustomize_nova_lgv_latplan"})

_MODE = os.environ.get("VLMAS_NLQP", "").strip()
if _MODE:
    import importlib.abc
    import importlib.machinery
    import json
    import math
    import time

    VCD = ("vcdlp", "vcdlp_uncentered", "vcdlp_check")
    assert _MODE in ("native", "uniform", "direct", "identity") + VCD, _MODE
    _S = {"role": None, "dV": None, "dVinfo": None}
    _LOG = os.environ.get("VLMAS_NLQP_LOG", "").strip()

    class _NLQP:
        """Hooks for one Answerer call. Active only on row b = the first 1-row step fed the opening quote."""

        def __init__(self, bb, lat_cols):
            import torch
            self.bb = bb
            self.lat = lat_cols.to(torch.long)
            self.armed = True          # until row b has run through all layers
            self.in_prefill = False    # True while the row-b forward runs (name kept from the prefill version)
            self.prefill_seen = False
            self.ready = False         # the current 1-row step's input token carries the opening quote
            self.steps = 0             # 1-row steps after the prefill
            self.tok = bb.processor.tokenizer
            self.row_tok = None
            self.x = {}                # layer -> raw residual of row b
            self.hhat = {}             # layer -> LN(x^_b) for the q_proj hook
            self.rR = {}               # layer -> r_R (direct mode)
            self.st = {"layers": 0, "ent": 0.0, "rmax": 0.0, "r_rel": 0.0, "q_rel": 0.0}
            self.handles = [bb.lm.embed_tokens.register_forward_pre_hook(self._embed_pre)]
            layers = bb.lm.layers
            self.n_layers = len(layers)
            for li, layer in enumerate(layers):
                self.handles.append(layer.register_forward_pre_hook(self._layer_pre(li), with_kwargs=True))
                att = layer.self_attn
                self.handles.append(att.register_forward_pre_hook(self._attn_pre(li, layer), with_kwargs=True))
                if _MODE == "direct":
                    self.handles.append(att.register_forward_hook(self._attn_post(li)))
                else:
                    self.handles.append(att.q_proj.register_forward_hook(self._q_post(li)))

        def remove(self):
            for h in self.handles:
                h.remove()
            self.handles = []

        def _embed_pre(self, module, args):
            ids = args[0] if args else None
            if ids is None or not self.armed:
                return None
            if ids.dim() == 2 and ids.shape[1] > 1:
                self.prefill_seen = True
                self.ready = False
            elif self.prefill_seen and ids.numel() == 1:
                self.steps += 1
                t = self.tok.decode([int(ids.reshape(-1)[0])])
                self.ready = '"' in t and self.steps <= 3
                if self.ready:
                    self.row_tok = t
                elif self.steps > 3:
                    self.armed = False           # no opening quote within 3 steps: leave the call native
            return None

        def _layer_pre(self, li):
            def pre(module, args, kwargs):
                hs = kwargs.get("hidden_states", args[0] if args else None)
                if li == 0:
                    self.in_prefill = bool(self.armed and self.ready and hs is not None and hs.dim() == 3
                                           and hs.shape[1] == 1)
                if self.in_prefill:
                    self.x[li] = hs[0, -1].detach()
                return None
            return pre

        def _attn_pre(self, li, layer):
            def pre(module, args, kwargs):
                import torch
                from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb
                if not self.in_prefill or li not in self.x:
                    return None
                hs = kwargs.get("hidden_states", args[0] if args else None)
                pe = kwargs.get("position_embeddings", args[1] if len(args) > 1 else None)
                cache = kwargs.get("past_key_values", args[3] if len(args) > 3 else None)
                if hs is None or pe is None or cache is None:
                    return None
                with torch.no_grad():
                    keys, vals = cache.layers[li].keys, cache.layers[li].values      # past only (not yet updated)
                    lat = self.lat.to(keys.device)
                    lat = lat[lat < keys.shape[2]]
                    if lat.numel() == 0:
                        return None
                    Hd, G = module.head_dim, module.num_key_value_groups
                    h_b = hs[:, -1:, :]
                    q = module.q_norm(module.q_proj(h_b).view(1, 1, -1, Hd)).transpose(1, 2)   # [1,H,1,Hd]
                    cos, sin = pe
                    q, _ = apply_rotary_pos_emb(q, torch.zeros_like(q), cos[:, -1:], sin[:, -1:])
                    KR = keys.index_select(2, lat).repeat_interleave(G, dim=1)        # [1,H,m,Hd]
                    if _MODE in VCD:
                        dV = _S["dV"]
                        if dV is None or dV[li].shape[1] != int(lat.numel()):
                            return None
                        VR = dV[li].to(keys.device)[None].repeat_interleave(G, dim=1)   # [1,H,m,Hd] (centered)
                    else:
                        VR = vals.index_select(2, lat).repeat_interleave(G, dim=1)
                    m = int(lat.numel())
                    if _MODE == "uniform":
                        rho = torch.full((1, KR.shape[1], 1, m), 1.0 / m, device=q.device, dtype=torch.float32)
                    else:
                        s = torch.matmul(q.float(), KR.float().transpose(-2, -1)) * module.scaling
                        rho = torch.softmax(s, dim=-1)                                # [1,H,1,m]
                    c = torch.matmul(rho, VR.float())                                 # [1,H,1,Hd]
                    c = c.transpose(1, 2).reshape(1, 1, -1).to(module.o_proj.weight.dtype)
                    r = module.o_proj(c)[0, 0]                                        # [hidden]
                    x_b = self.x[li]
                    if _MODE == "identity":          # hook sanity check: same path with r_R = 0 must equal native
                        r = torch.zeros_like(r)
                    ent = float(-(rho.clamp_min(1e-12).log() * rho).sum(-1).mean() / math.log(m))
                    self.st["ent"] += ent
                    self.st["rmax"] += float(rho.max(-1).values.mean())
                    self.st["r_rel"] += float(r.float().norm() / x_b.float().norm().clamp_min(1e-9))
                    if _MODE == "direct":
                        self.rR[li] = r
                    else:
                        self.hhat[li] = layer.input_layernorm((x_b + r.to(x_b.dtype))[None, None])[0, 0]
                    self.st["layers"] += 1
                return None
            return pre

        def _q_post(self, li):
            def post(module, args, output):
                import torch
                hh = self.hhat.pop(li, None)
                if hh is None or not self.in_prefill:
                    return output
                with torch.no_grad():
                    new = module(hh[None, None].to(output.dtype))[0, 0]
                    old = output[0, -1]
                    self.st["q_rel"] += float((new.float() - old.float()).norm() / old.float().norm().clamp_min(1e-9))
                    out = output.clone()
                    out[0, -1] = new
                if li == self.n_layers - 1:
                    self._finish()
                return out
            return post

        def _attn_post(self, li):
            def post(module, args, output):
                import torch
                r = self.rR.pop(li, None)
                if r is None or not self.in_prefill:
                    return output
                a = output[0] if isinstance(output, tuple) else output
                with torch.no_grad():
                    self.st["q_rel"] += float(r.float().norm() / a[0, -1].float().norm().clamp_min(1e-9))
                    a = a.clone()
                    a[0, -1] = a[0, -1] + r.to(a.dtype)
                if li == self.n_layers - 1:
                    self._finish()
                return (a,) + tuple(output[1:]) if isinstance(output, tuple) else a
            return post

        def _finish(self):
            self.armed = False
            self.ready = False
            self.in_prefill = False
            self.x.clear()

        def summary(self):
            n = max(1, self.st["layers"])
            return {"layers": self.st["layers"], "row_tok": self.row_tok, "step": self.steps, "ent": round(self.st["ent"] / n, 4),
                    "rho_max": round(self.st["rmax"] / n, 4), "r_rel": round(self.st["r_rel"] / n, 4),
                    ("attn_rel" if _MODE == "direct" else "q_rel"): round(self.st["q_rel"] / n, 4)}

    def _mask_post(li, vis):
        """visual_cut 'mask' math: the current rows may not attend to the visual columns (renormalised softmax)."""
        def post(module, args, kwargs, output):
            import torch
            from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb, repeat_kv
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            pe = kwargs.get("position_embeddings", args[1] if len(args) > 1 else None)
            cache = kwargs.get("past_key_values", args[3] if len(args) > 3 else None)
            keys, values = cache.layers[li].keys, cache.layers[li].values
            L, T, Hd, G = int(keys.shape[2]), int(hidden.shape[1]), module.head_dim, module.num_key_value_groups
            v = vis.to(keys.device)
            v = v[v < L]
            q = module.q_norm(module.q_proj(hidden).view(1, T, -1, Hd)).transpose(1, 2)
            q, _ = apply_rotary_pos_emb(q, torch.zeros_like(q), pe[0], pe[1])
            sc = torch.matmul(q.float(), repeat_kv(keys, G).float().transpose(-2, -1)) * module.scaling
            sc[..., v] = float("-inf")
            ctx = torch.matmul(torch.softmax(sc, -1), repeat_kv(values, G).float())
            new = module.o_proj(ctx.transpose(1, 2).reshape(1, T, -1).to(module.o_proj.weight.dtype))
            a = output[0] if isinstance(output, tuple) else output
            new = new.to(a.dtype)
            return (new, *output[1:]) if isinstance(output, tuple) else new
        return post

    def _knockout(bb, result, m):
        """Replay the m Reasoner latent steps from the pre-latent boundary with visual reads masked -> centered dV."""
        import torch
        from copy import deepcopy
        t0 = time.time()
        kv = result["past_key_values"]
        L1, cur1 = int(result["past_len"]), int(result["pos_cursor"])
        L0, cur0 = L1 - m, cur1 - m
        vis = result.get("vis_cols")
        vis = vis.detach().long() if isinstance(vis, torch.Tensor) else torch.empty(0, dtype=torch.long)
        vis = vis[vis < L0]
        traj = result.get("latent_trajectory") or []
        if len(traj) != m or vis.numel() == 0 or int(bb._kv_len(kv)) != L1:
            _S["dV"], _S["dVinfo"] = None, {"skip": f"traj={len(traj)} vis={int(vis.numel())} kv={int(bb._kv_len(kv))} L1={L1}"}
            return
        c = deepcopy(kv)
        c.crop(L0)
        dev = bb.device
        le = traj[0].to(dev).view(1, 1, -1).to(bb.lm.embed_tokens.weight.dtype)
        masked = _MODE != "vcdlp_check"
        hs = [layer.self_attn.register_forward_hook(_mask_post(li, vis), with_kwargs=True)
              for li, layer in enumerate(bb.lm.layers)] if masked else []
        try:
            with torch.no_grad():
                for step in range(m):
                    if step:
                        le = bb._apply_realign(last)
                    o = bb.lm(inputs_embeds=le, position_ids=bb._text_positions(1, cur0 + step), past_key_values=c,
                              use_cache=True, output_hidden_states=True)
                    c = o.past_key_values
                    last = o.hidden_states[-1][:, -1:, :]
        finally:
            for h in hs:
                h.remove()
        with torch.no_grad():
            dV, rel = [], []
            for lf, lk in zip(kv.layers, c.layers):
                vf = lf.values[0, :, L0:L1].float()          # [Hkv, m, D]
                d = vf - lk.values[0, :, L0:L1].float()
                rel.append(float(d.norm() / vf.norm().clamp_min(1e-9)))
                if _MODE == "vcdlp":
                    d = d - d.mean(1, keepdim=True)
                dV.append(d.to(lf.values.dtype))
            kdiff = max(float((lf.keys[0, :, :L0] - lk.keys[0, :, :L0]).abs().max()) for lf, lk in
                        zip(kv.layers[:2], c.layers[:2]))
        del c
        _S["dV"] = dV
        _S["dVinfo"] = {"dV_rel_mean": round(sum(rel) / len(rel), 4), "dV_rel_max": round(max(rel), 4),
                        "prefix_kdiff": kdiff, "n_vis": int(vis.numel()), "L0": L0, "L1": L1,
                        "replay_sec": round(time.time() - t0, 2)}
        print(f"[VCDLP] knockout replay masked={masked} m={m} vis={int(vis.numel())} L=[{L0},{L1}) "
              f"|dV|/|V| mean={_S['dVinfo']['dV_rel_mean']} max={_S['dVinfo']['dV_rel_max']} "
              f"prefix_kdiff={kdiff:.1e} sec={_S['dVinfo']['replay_sec']}", flush=True)

    def _wrap_backbone(bb):
        if getattr(bb, "_vcdlp_wrapped", False):
            return
        orig = bb.grounded_prefill_and_latent_on_kv

        def wrapped(*a, **k):
            result = orig(*a, **k)
            if _S["role"] == "reasoner":
                try:
                    _knockout(bb, result, int(k.get("m", 0)))
                except Exception as exc:  # noqa: BLE001 - never break the case
                    import traceback
                    traceback.print_exc()
                    _S["dV"], _S["dVinfo"] = None, {"skip": f"error {exc!r}"}
            return result
        bb.grounded_prefill_and_latent_on_kv = wrapped
        bb._vcdlp_wrapped = True

    def _wrap_client(cls):
        orig_final = cls.generate_final_text
        if _MODE in VCD:
            orig_append = cls.append_only

            def append_only(self, *, role, **kw):
                _wrap_backbone(self._backend._engine._backbone)
                if role == "evidence_planner":
                    _S["dV"], _S["dVinfo"] = None, None
                _S["role"] = role
                try:
                    return orig_append(self, role=role, **kw)
                finally:
                    _S["role"] = None
            cls.append_only = append_only

        def generate_final_text(self, **kw):
            eng = self._backend._engine
            bb = eng._backbone
            lat = getattr(eng, "_rpath_latent_cols", None)
            case = int(getattr(self._state, "dataset_index", -1))
            if lat is None or int(lat.numel()) == 0:
                print(f"[NLQP] case {case} SKIP no Reasoner latent columns", flush=True)
                return orig_final(self, **kw)
            t0 = time.time()
            h = _NLQP(bb, lat)
            try:
                out = orig_final(self, **kw)
            finally:
                h.remove()
            rec = {"case": case, "mode": _MODE, "lat": [int(lat[0]), int(lat[-1])], "m": int(lat.numel()),
                   **({"vcdlp": _S["dVinfo"]} if _MODE in VCD else {}),
                   "cache": int(self._state.cache_length), **h.summary(), "sec": round(time.time() - t0, 2)}
            print(f"[NLQP] case {case} mode={_MODE} applied={rec['layers']}/{h.n_layers} row_tok={rec['row_tok']!r} step={rec['step']} lat={rec['lat']} "
                  f"ent={rec['ent']} rho_max={rec['rho_max']} r_rel={rec['r_rel']} "
                  f"{'attn_rel' if _MODE == 'direct' else 'q_rel'}={rec.get('attn_rel', rec.get('q_rel'))}", flush=True)
            if _LOG:
                with open(_LOG, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
            return out

        cls.generate_final_text = generate_final_text

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != "vision_text_mas.latent_onepass":
                return None
            spec = None
            for fnd in sys.meta_path[sys.meta_path.index(self) + 1:]:      # chain: keep the latplan patches
                fs = getattr(fnd, "find_spec", None)
                spec = fs(fullname, path, target) if fs else None
                if spec is not None:
                    break
            if spec is None or spec.loader is None:
                return spec
            orig = spec.loader.exec_module

            def exec_module(module):
                orig(module)
                _wrap_client(module.LatentRoleClient)
                print(f"[NLQP] patched LatentRoleClient.generate_final_text (mode={_MODE})", flush=True)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
