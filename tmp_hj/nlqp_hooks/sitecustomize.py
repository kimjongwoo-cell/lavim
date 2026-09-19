"""C2 NLQP — Native Latent Query Priming (Notion C1/C2 Method Evolution #30). Original files untouched.

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
  VLMAS_NLQP=rfnm      VCDLP §25 runtime-safe revision (Replay-Free Native-Mass Latent Visual Priming). No replay: while
                       the Reasoner's m latent steps run natively, every layer records for each query head the latent
                       row's native full-softmax visual contribution m_V,k = sum_{j in V} alpha^nat_{k,j} V_j (no
                       renormalisation inside V); m_V is centered over k. At row b, c = sum_k alpha^nat_{A,k} m~_V,k
                       with the Answerer's native FULL-cache weights on the latent columns (no latent-only softmax),
                       r = W_O concat_h c, q^ = W_Q LN(x + r).
  VLMAS_NLQP=rfnm_raw  ablation: m_V not centered
  VLMAS_NLQP=rfnm_check identity: same capture / path with r = 0 (must reproduce native answers)
  VLMAS_VCDLP_FAST=1   same replay without the cache copy and without the fp32 attention recompute: the knockout rows
                       are appended after the native latents at the same positions, a 2-D key mask hides the native
                       latent columns (+ visual columns) and native sdpa masks; the rows are cropped off afterwards
  VLMAS_NLQP_ATTNLOG=0 skip the measurement-only attention-mass log (default 1)
  VLMAS_NLQP_LOG=<jsonl>  per Answerer call: latent cols, layers applied, mean rho entropy / max,
                          |r_R|/|x_b|, |q^-q|/|q| (or |r_R|/|attn_out| for direct)
Reasoner latent columns come from engine._rpath_latent_cols (set by the engine right after the Reasoner append:
the last m cache columns before the turn is closed).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_f = os.path.join(os.path.dirname(_HERE), "nova_rho_latplan_hooks", "sitecustomize.py")
exec(compile(open(_f).read(), _f, "exec"), {"__file__": _f, "__name__": "sitecustomize_nova_rho_latplan"})

_MODE = os.environ.get("VLMAS_NLQP", "").strip()
if _MODE:
    import importlib.abc
    import importlib.machinery
    import json
    import math
    import time

    VCD = ("vcdlp", "vcdlp_uncentered", "vcdlp_check")
    RF = ("rfnm", "rfnm_raw", "rfnm_check")
    assert _MODE in ("native", "uniform", "direct", "identity") + VCD + RF, _MODE
    _S = {"role": None, "dV": None, "dVinfo": None, "mV": [], "rfinfo": None, "rf_step": -1, "rf_vis": None}
    _FAST = os.environ.get("VLMAS_VCDLP_FAST", "") == "1"
    _ATTNLOG = os.environ.get("VLMAS_NLQP_ATTNLOG", "1") == "1"
    _LOG = os.environ.get("VLMAS_NLQP_LOG", "").strip()

    class _NLQP:
        """Hooks for one Answerer call. Active only on row b = the first 1-row step fed the opening quote."""

        def __init__(self, bb, lat_cols, vis_cols=None):
            import torch
            self.bb = bb
            self.lat = lat_cols.to(torch.long)
            self.vis = vis_cols.to(torch.long) if vis_cols is not None else torch.empty(0, dtype=torch.long)
            self.am = []               # per layer: attention mass by column group, native q vs primed q^
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
                    m = int(lat.numel())
                    Hd, G = module.head_dim, module.num_key_value_groups
                    h_b = hs[:, -1:, :]
                    q = module.q_norm(module.q_proj(h_b).view(1, 1, -1, Hd)).transpose(1, 2)   # [1,H,1,Hd]
                    cos, sin = pe
                    q, _ = apply_rotary_pos_emb(q, torch.zeros_like(q), cos[:, -1:], sin[:, -1:])
                    if _MODE in RF:
                        mt = _S["dV"]
                        if mt is None or mt[li].shape[0] != int(lat.numel()):
                            return None
                        # native FULL-cache weights of row b (past + its own column), no latent-only softmax
                        k_b = module.k_norm(module.k_proj(h_b).view(1, 1, -1, Hd)).transpose(1, 2)
                        cos, sin = pe
                        _, k_b = apply_rotary_pos_emb(k_b, k_b, cos[:, -1:], sin[:, -1:])
                        Kf = torch.cat([keys, k_b.to(keys.dtype)], dim=2)[0]              # [Hkv, L+1, D]
                        Hkv = Kf.shape[0]
                        qq = q[0, :, 0].view(Hkv, G, Hd).float()
                        a = torch.softmax(torch.matmul(qq, Kf.float().transpose(-1, -2)) * module.scaling, -1)
                        aL = a[:, :, lat].reshape(Hkv * G, -1)                          # [Hq, m] native mass
                        c = torch.einsum("hk,khd->hd", aL, mt[li].to(aL.device).float())  # [Hq, D]
                        r = module.o_proj(c.reshape(1, 1, -1).to(module.o_proj.weight.dtype))[0, 0]
                        rho = aL[None, :, None, :]                                        # for stats: native lat mass
                        self.st.setdefault("rhoR", 0.0)
                        self.st["rhoR"] += float(aL.sum(-1).mean())
                    KR = keys.index_select(2, lat).repeat_interleave(G, dim=1)        # [1,H,m,Hd]
                    if _MODE in RF:
                        pass
                    elif _MODE in VCD:
                        dV = _S["dV"]
                        if dV is None or dV[li].shape[1] != int(lat.numel()):
                            return None
                        VR = dV[li].to(keys.device)[None].repeat_interleave(G, dim=1)   # [1,H,m,Hd] (centered)
                    else:
                        VR = vals.index_select(2, lat).repeat_interleave(G, dim=1)
                    if _MODE in RF:
                        pass
                    elif _MODE == "uniform":
                        rho = torch.full((1, KR.shape[1], 1, m), 1.0 / m, device=q.device, dtype=torch.float32)
                    else:
                        s = torch.matmul(q.float(), KR.float().transpose(-2, -1)) * module.scaling
                        rho = torch.softmax(s, dim=-1)                                # [1,H,1,m]
                    if _MODE not in RF:
                        c = torch.matmul(rho, VR.float())                             # [1,H,1,Hd]
                        c = c.transpose(1, 2).reshape(1, 1, -1).to(module.o_proj.weight.dtype)
                        r = module.o_proj(c)[0, 0]                                    # [hidden]
                    x_b = self.x[li]
                    if _MODE in ("identity", "rfnm_check"):          # hook sanity check: same path with r_R = 0 must equal native
                        r = torch.zeros_like(r)
                    rn = rho / rho.sum(-1, keepdim=True).clamp_min(1e-30)
                    ent = float(-(rn.clamp_min(1e-12).log() * rn).sum(-1).mean() / math.log(m))
                    self.st["ent"] += ent
                    self.st["rmax"] += float(rho.max(-1).values.mean())
                    self.st["r_rel"] += float(r.float().norm() / x_b.float().norm().clamp_min(1e-9))
                    if _MODE == "direct":
                        self.rR[li] = r
                    else:
                        hhat = layer.input_layernorm((x_b + r.to(x_b.dtype))[None, None])[0, 0]
                        if _ATTNLOG:   # before hhat is stashed: _attn_mass calls q_proj, whose hook would consume it
                            try:
                                self._attn_mass(module, li, hs, pe, keys, q, hhat)
                            except Exception as exc:  # noqa: BLE001 - measurement only
                                print(f"[NLQP] attn-mass log failed L{li}: {exc!r}", flush=True)
                        self.hhat[li] = hhat
                    self.st["layers"] += 1
                return None
            return pre

        def _attn_mass(self, module, li, hs, pe, keys, q, hhat):
            """Measurement only: row b's attention over the full cache (+ its own column) with the native q and the
            primed q^, summed per column group; answers are not affected."""
            import torch
            from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb, repeat_kv
            Hd, G = module.head_dim, module.num_key_value_groups
            cos, sin = pe[0][:, -1:], pe[1][:, -1:]
            h_b = hs[:, -1:, :]
            k_b = module.k_norm(module.k_proj(h_b).view(1, 1, -1, Hd)).transpose(1, 2)
            _, k_b = apply_rotary_pos_emb(k_b, k_b, cos, sin)
            K = torch.cat([keys, k_b.to(keys.dtype)], dim=2)                      # past + own column
            qh = module.q_norm(module.q_proj(hhat[None, None].to(h_b.dtype)).view(1, 1, -1, Hd)).transpose(1, 2)
            qh, _ = apply_rotary_pos_emb(qh, torch.zeros_like(qh), cos, sin)
            Kr = repeat_kv(K, G).float()
            a0 = torch.softmax(torch.matmul(q.float(), Kr.transpose(-2, -1)) * module.scaling, -1)[0, :, 0]   # [H,L+1]
            a1 = torch.softmax(torch.matmul(qh.float(), Kr.transpose(-2, -1)) * module.scaling, -1)[0, :, 0]
            L1 = int(K.shape[2])
            grp = torch.zeros(L1, dtype=torch.long, device=K.device)                # 0 other(prompt/history/roles)
            grp[0] = 1                                                              # 1 sink
            v = self.vis.to(K.device); v = v[v < L1]; grp[v] = 2                    # 2 Reasoner visual (C1 kept)
            l = self.lat.to(K.device); l = l[l < L1]; grp[l] = 3                    # 3 Reasoner latent
            oh = torch.nn.functional.one_hot(grp, 4).float()
            m0, m1 = (a0.mean(0) @ oh), (a1.mean(0) @ oh)
            mid = 0.5 * (a0 + a1)
            kl = lambda p: (p * (p.clamp_min(1e-30).log() - mid.clamp_min(1e-30).log())).sum(-1)
            jsd = float((0.5 * kl(a0) + 0.5 * kl(a1)).mean())
            self.am.append({"l": li, "nat": [round(float(x), 5) for x in m0], "pri": [round(float(x), 5) for x in m1],
                            "jsd": round(jsd, 6)})

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
            am = {}
            if self.am:
                k = len(self.am)
                names = ("other", "sink", "R_vis", "R_lat")
                am = {"attn_nat": {nm: round(sum(a["nat"][i] for a in self.am) / k, 5) for i, nm in enumerate(names)},
                      "attn_pri": {nm: round(sum(a["pri"][i] for a in self.am) / k, 5) for i, nm in enumerate(names)},
                      "attn_jsd": round(sum(a["jsd"] for a in self.am) / k, 6), "attn_layers": self.am}
            extra = {"rhoR_nat": round(self.st["rhoR"] / n, 6)} if "rhoR" in self.st else {}
            return {**am, **extra, "layers": self.st["layers"], "row_tok": self.row_tok, "step": self.steps, "ent": round(self.st["ent"] / n, 4),
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
        dev = bb.device
        le = traj[0].to(dev).view(1, 1, -1).to(bb.lm.embed_tokens.weight.dtype)
        masked = _MODE != "vcdlp_check"
        if _FAST:
            # Fast path (VLMAS_VCDLP_FAST=1): no cache copy, no attention recompute. The knockout rows are appended
            # to the REAL cache after L1 at the same RoPE positions cur0+k; a 2-D key mask hides the native latent
            # columns [L0, L1) (so the rows see exactly prefix [0, L0) + their own earlier rows, as in the native
            # loop) and, unless vcdlp_check, the visual columns; native sdpa does the masking. The appended rows are
            # read and then cropped off, so the cache is back to [0, L1) before anything else runs.
            with torch.no_grad():
                keep = torch.ones(1, L1 + m, dtype=torch.long, device=dev)
                keep[0, L0:L1] = 0
                if masked:
                    keep[0, vis.to(dev)] = 0
                try:
                    for step in range(m):
                        if step:
                            le = bb._apply_realign(last)
                        o = bb.lm(inputs_embeds=le, position_ids=bb._text_positions(1, cur0 + step),
                                  past_key_values=kv, attention_mask=keep[:, :L1 + step + 1], use_cache=True,
                                  output_hidden_states=True)
                        last = o.hidden_states[-1][:, -1:, :]
                    ko = [lk.values[0, :, L1:L1 + m].clone() for lk in kv.layers]
                finally:
                    kv.crop(L1)
                assert int(bb._kv_len(kv)) == L1
                dV, rel, relk = [], [], []
                for lf, vk in zip(kv.layers, ko):
                    vf = lf.values[0, :, L0:L1].float()
                    d = vf - vk.float()
                    rel.append(float(d.norm() / vf.norm().clamp_min(1e-9)))
                    relk.append(d.norm(dim=(0, 2)) / vf.norm(dim=(0, 2)).clamp_min(1e-9))
                    if _MODE == "vcdlp":
                        d = d - d.mean(1, keepdim=True)
                    dV.append(d.to(lf.values.dtype))
                kdiff = 0.0
        else:
            c = deepcopy(kv)
            c.crop(L0)
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
                dV, rel, relk = [], [], []
                for lf, lk in zip(kv.layers, c.layers):
                    vf = lf.values[0, :, L0:L1].float()          # [Hkv, m, D]
                    d = vf - lk.values[0, :, L0:L1].float()
                    rel.append(float(d.norm() / vf.norm().clamp_min(1e-9)))
                    relk.append(d.norm(dim=(0, 2)) / vf.norm(dim=(0, 2)).clamp_min(1e-9))
                    if _MODE == "vcdlp":
                        d = d - d.mean(1, keepdim=True)
                    dV.append(d.to(lf.values.dtype))
                kdiff = max(float((lf.keys[0, :, :L0] - lk.keys[0, :, :L0]).abs().max()) for lf, lk in
                            zip(kv.layers[:2], c.layers[:2]))
            del c
        _S["dV"] = dV
        import torch as _t
        _S["dVinfo"] = {"dV_rel_mean": round(sum(rel) / len(rel), 4), "dV_rel_max": round(max(rel), 4),
                        "dV_rel_step": [round(float(x), 4) for x in _t.stack(relk).mean(0)],
                        "prefix_kdiff": kdiff, "n_vis": int(vis.numel()), "L0": L0, "L1": L1,
                        "replay_sec": round(time.time() - t0, 2)}
        print(f"[VCDLP] knockout replay fast={_FAST} masked={masked} m={m} vis={int(vis.numel())} L=[{L0},{L1}) "
              f"|dV|/|V| mean={_S['dVinfo']['dV_rel_mean']} max={_S['dVinfo']['dV_rel_max']} "
              f"prefix_kdiff={kdiff:.1e} sec={_S['dVinfo']['replay_sec']} per_step={_S['dVinfo']['dV_rel_step']}", flush=True)

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

    def _rf_post(li):
        """During the Reasoner's 1-row latent steps: native full-softmax visual contribution per query head."""
        def post(module, args, kwargs, output):
            import torch
            from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            if _S["role"] != "reasoner" or hidden is None or hidden.shape[1] != 1:
                return output
            pe = kwargs.get("position_embeddings", args[1] if len(args) > 1 else None)
            cache = kwargs.get("past_key_values", args[3] if len(args) > 3 else None)
            vis = _S["rf_vis"]
            if pe is None or cache is None or vis is None:
                return output
            with torch.no_grad():
                if li == 0:
                    _S["rf_step"] += 1
                    _S["mV"].append([None] * len(_S["rf_layers"]))
                keys, vals = cache.layers[li].keys[0], cache.layers[li].values[0]      # [Hkv, L, D] incl. own row
                Hkv, L, D = keys.shape
                G = module.num_key_value_groups
                q = module.q_norm(module.q_proj(hidden).view(1, 1, -1, D)).transpose(1, 2)
                q, _ = apply_rotary_pos_emb(q, torch.zeros_like(q), pe[0], pe[1])
                q = q[0, :, 0].view(Hkv, G, D).float()
                a = torch.softmax(torch.matmul(q, keys.float().transpose(-1, -2)) * module.scaling, -1)   # [Hkv,G,L]
                v = vis[vis < L]
                mv = torch.matmul(a[:, :, v], vals[:, v].float())                   # [Hkv, G, D] (full softmax)
                _S["mV"][-1][li] = mv.reshape(Hkv * G, D).to(vals.dtype)
                if li == 0:
                    _S["rf_rhoV"].append(float(a[:, :, v].sum(-1).mean()))
            return output
        return post

    def _rf_finish(bb):
        """Stack the captured steps -> per layer [m, Hq, D]; center over m; representation-gate stats."""
        import torch
        steps = [st for st in _S["mV"] if all(x is not None for x in st)]
        if not steps:
            _S["dV"], _S["rfinfo"] = None, {"skip": "no latent steps captured"}
            return
        m = len(steps)
        raw = [torch.stack([st[li] for st in steps]) for li in range(len(steps[0]))]     # [m, Hq, D]
        cos_raw, cos_c, er_raw, er_c = [], [], [], []

        def stats(X):
            F = X.float().reshape(X.shape[0], -1)
            Fn = torch.nn.functional.normalize(F, dim=-1)
            C = Fn @ Fn.T
            off = C[~torch.eye(C.shape[0], dtype=torch.bool, device=C.device)]
            sv = torch.linalg.svdvals(F)
            er = float((sv.sum() ** 2) / (sv.pow(2).sum().clamp_min(1e-12)))              # participation ratio
            return float(off.mean()), er
        out = []
        for X in raw:
            Xc = X - X.float().mean(0, keepdim=True).to(X.dtype)
            a, b = stats(X); c, d = stats(Xc)
            cos_raw.append(a); er_raw.append(b); cos_c.append(c); er_c.append(d)
            out.append(X if _MODE == "rfnm_raw" else Xc)
        _S["dV"] = out
        n = len(raw)
        _S["rfinfo"] = {"m": m, "rhoV_L0": [round(x, 4) for x in _S["rf_rhoV"]],
                        "cos_raw": round(sum(cos_raw) / n, 4), "cos_centered": round(sum(cos_c) / n, 4),
                        "erank_raw": round(sum(er_raw) / n, 3), "erank_centered": round(sum(er_c) / n, 3),
                        "capture_sec": round(time.time() - _S["rf_t0"], 3)}
        print(f"[RFNM] captured m={m} layers={n} rhoV(L0)={_S['rfinfo']['rhoV_L0'][:3]}.. "
              f"cos raw={_S['rfinfo']['cos_raw']} centered={_S['rfinfo']['cos_centered']} "
              f"erank raw={_S['rfinfo']['erank_raw']} centered={_S['rfinfo']['erank_centered']} "
              f"sec={_S['rfinfo']['capture_sec']}", flush=True)

    def _wrap_client(cls):
        orig_final = cls.generate_final_text
        if _MODE in RF:
            orig_append_rf = cls.append_only

            def append_only(self, *, role, **kw):
                import torch
                bb = self._backend._engine._backbone
                if role == "evidence_planner":
                    _S["dV"], _S["rfinfo"] = None, None
                if role != "reasoner":
                    return orig_append_rf(self, role=role, **kw)
                _S.update(role="reasoner", mV=[], rf_step=-1, rf_rhoV=[], rf_t0=time.time(), rf_vis=None,
                          rf_layers=list(range(len(bb.lm.layers))))
                # visual columns of this Reasoner turn (after C1): set by the backbone just before the latent loop
                hs = [layer.self_attn.register_forward_hook(_rf_post(li), with_kwargs=True)
                      for li, layer in enumerate(bb.lm.layers)]

                def lm_pre(module, args, kwargs):
                    e = kwargs.get("inputs_embeds")
                    if e is not None and e.shape[1] == 1 and _S["rf_vis"] is None:
                        v = getattr(bb, "_aiva_visual_cols", None)
                        _S["rf_vis"] = v.detach().to(torch.long) if v is not None else None
                    return None
                hs.append(bb.lm.register_forward_pre_hook(lm_pre, with_kwargs=True))
                t0 = time.time()
                try:
                    rec = orig_append_rf(self, role=role, **kw)
                finally:
                    for h in hs:
                        h.remove()
                    _S["role"] = None
                _S["rf_t0"] = t0
                eng = self._backend._engine
                rv = getattr(eng, "_rpath_visual_cols", None)
                same = (rv is not None and _S["rf_vis"] is not None
                        and torch.equal(rv.to(_S["rf_vis"].device).long(), _S["rf_vis"].long()))
                _rf_finish(bb)
                if _S["rfinfo"] is not None:
                    _S["rfinfo"]["vis_match_engine"] = bool(same)
                    _S["rfinfo"]["n_vis"] = int(_S["rf_vis"].numel()) if _S["rf_vis"] is not None else 0
                return rec
            cls.append_only = append_only
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
            h = _NLQP(bb, lat, getattr(eng, "_rpath_visual_cols", None))
            try:
                out = orig_final(self, **kw)
            finally:
                h.remove()
            rec = {"case": case, "mode": _MODE, "lat": [int(lat[0]), int(lat[-1])], "m": int(lat.numel()),
                   **({"vcdlp": _S["dVinfo"]} if _MODE in VCD else {}),
                   **({"rfnm": _S["rfinfo"]} if _MODE in RF else {}),
                   "cache": int(self._state.cache_length), **h.summary(), "sec": round(time.time() - t0, 2)}
            print(f"[NLQP] case {case} mode={_MODE} applied={rec['layers']}/{h.n_layers} row_tok={rec['row_tok']!r} step={rec['step']} lat={rec['lat']} "
                  f"ent={rec['ent']} rho_max={rec['rho_max']} r_rel={rec['r_rel']} "
                  f"{'attn_rel' if _MODE == 'direct' else 'q_rel'}={rec.get('attn_rel', rec.get('q_rel'))}", flush=True)
            if rec.get("attn_nat"):
                print(f"[NLQP] case {case} attn mass (layer mean) native={rec['attn_nat']} primed={rec['attn_pri']} "
                      f"jsd={rec['attn_jsd']}", flush=True)
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
