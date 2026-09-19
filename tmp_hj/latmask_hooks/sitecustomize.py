"""09-19 Answerer latent-mask test on the latplan base (this session). Original files untouched.

Order 91 + thumbsink attention: on broken WSI-VQA answers the Answerer's first-answer-token row reads the Planner /
Navigator LATENT columns far more at layer 0 than on clean answers (Planner latent 0.160 vs 0.045, Navigator latent
0.077 vs 0.007), while the thumbnail mass is identical. The plan itself is still built from the real thumbnail;
this hook only stops the ANSWERER from attending to those latent columns (they have already steered the Navigator).

Chains tmp_hj/latplan_hooks. With VLMAS_LATMASK=all|l0:
  1. every role turn (LatentRoleClient.append_only) records the absolute cache columns of its latent steps
     (same bookkeeping as tmp_hj/thumbsink_hooks: prefill ids via the processor, latents right after them);
  2. during LatentRoleClient.generate_final_text (the Answerer: prompt prefill + every decode step) the sdpa
     attention of the selected layers gets an additive -inf on those columns
     (VLMAS_LATMASK=all -> all 36 layers, l0 -> layer 0 only). VLMAS_LATMASK_ROLES (default P,N) picks the roles
     (P = evidence_planner, N = navigator, R = reasoner). K/V, cache, positions and every other row are untouched.
  09-19 boost: VLMAS_LATBOOST=<alpha> (> 1) adds +ln(alpha) to the Answerer's attention logits on the latent columns of
     VLMAS_LATBOOST_ROLES (default R) in VLMAS_LATBOOST_LAYERS (all | l0 | a-b, e.g. 7-16). Before the softmax, so the
     Reasoner latents get alpha times their native weight relative to everything else. Can be combined with the mask.
  09-19 move: VLMAS_LATMOVE=<f in (0,1]> takes the fraction f of the Answerer's post-softmax attention mass on the latent
     columns of VLMAS_LATMOVE_FROM (default P) and gives it to the latent columns of VLMAS_LATMOVE_TO (default R),
     split in proportion to the attention those target columns already had (uniform if they had none), in
     VLMAS_LATMOVE_LAYERS (default 0; "a-b" or "all"). Every other column keeps exactly its weight, rows still sum to 1.
     Computed explicitly (softmax -> transfer -> A V) only in the selected layers; all other layers stay native sdpa.
  09-19 VLMAS_LATMASK_HEADS=h1,h2,..  restrict the mask to those query heads (default: all heads).
  09-19 VLMAS_LATVSCALE=match: in the MASK layers, instead of masking, the value vectors of the selected latent columns
     are rescaled per KV head to the median value norm of the non-latent columns (the latent is still read, but its
     4x-too-large layer-0 value no longer dominates). Mutually exclusive with masking in the same layer.
Log: [LATMASK] patched ... / [LATMASK] case cols=<n> layers=<spec> calls=<n masked attention calls>
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_f = os.path.join(os.path.dirname(_HERE), "latplan_hooks", "sitecustomize.py")
exec(compile(open(_f).read(), _f, "exec"), {"__file__": _f, "__name__": "sitecustomize_latplan"})

_MODE = os.environ.get("VLMAS_LATMASK", "").strip().lower()   # all | l0 | lt<k> (layers 0..k-1) | r<a>-<b> (layers a..b)
_BOOST = float(os.environ.get("VLMAS_LATBOOST", "1") or 1)
_MOVE = float(os.environ.get("VLMAS_LATMOVE", "0") or 0)
if _MODE in ("all", "l0") or _MODE.startswith("lt") or _MODE.startswith("r") or _BOOST > 1.0 or _MOVE > 0.0:
    import importlib.abc
    import importlib.machinery

    _ROLES = {"P": "evidence_planner", "N": "navigator", "R": "reasoner"}
    _WANT = {_ROLES[x.strip().upper()] for x in os.environ.get("VLMAS_LATMASK_ROLES", "P,N").split(",") if x.strip()}
    _HEADS = [int(x) for x in os.environ.get("VLMAS_LATMASK_HEADS", "").split(",") if x.strip()]
    _VSCALE = os.environ.get("VLMAS_LATVSCALE", "").strip().lower() == "match"
    _BWANT = {_ROLES[x.strip().upper()] for x in os.environ.get("VLMAS_LATBOOST_ROLES", "R").split(",") if x.strip()}
    _BL = os.environ.get("VLMAS_LATBOOST_LAYERS", "all").strip().lower()
    _MFROM = {_ROLES[x.strip().upper()] for x in os.environ.get("VLMAS_LATMOVE_FROM", "P").split(",") if x.strip()}
    _MTO = {_ROLES[x.strip().upper()] for x in os.environ.get("VLMAS_LATMOVE_TO", "R").split(",") if x.strip()}
    _ML = os.environ.get("VLMAS_LATMOVE_LAYERS", "0").strip().lower()
    _ST = {"spans": [], "last_ids": None, "active": False, "cols": None, "bcols": None, "calls": 0,
           "mfrom": None, "mto": None, "moved": []}

    def _move_layer(li):
        if _MOVE <= 0.0:
            return False
        if _ML == "all":
            return True
        if "-" in _ML:
            a, b = (int(v) for v in _ML.split("-"))
            return a <= li <= b
        return li == int(_ML)

    def _mask_layer(li):
        if _MODE.startswith("lt"):
            return li < int(_MODE[2:])
        if _MODE.startswith("r"):
            a, b = (int(v) for v in _MODE[1:].split("-"))
            return a <= li <= b
        return _MODE == "all" or (_MODE == "l0" and li == 0)

    def _boost_layer(li):
        if _BOOST <= 1.0:
            return False
        if _BL == "all":
            return True
        if _BL == "l0":
            return li == 0
        a, b = (int(v) for v in _BL.split("-"))
        return a <= li <= b

    def _install_attention():
        import torch
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        orig = ALL_ATTENTION_FUNCTIONS["sdpa"]

        def sdpa_masked(module, query, key, value, attention_mask, *a, **k):
            cols, bcols = _ST["cols"], _ST["bcols"]
            li = getattr(module, "layer_idx", None)
            if not _ST["active"] or cols is None or li is None or not (_mask_layer(li) or _boost_layer(li) or _move_layer(li)):
                return orig(module, query, key, value, attention_mask, *a, **k)
            q_len, kv = int(query.shape[2]), int(key.shape[2])
            neg = torch.finfo(query.dtype).min
            if attention_mask is None:
                bias = torch.zeros(1, 1, q_len, kv, dtype=query.dtype, device=query.device)
                if q_len > 1:
                    i = torch.arange(q_len, device=query.device)[:, None]
                    j = torch.arange(kv, device=query.device)[None, :]
                    bias = bias.masked_fill((j > (kv - q_len) + i)[None, None], neg)
            elif attention_mask.dtype == torch.bool:
                bias = torch.zeros(attention_mask.shape, dtype=query.dtype, device=query.device).masked_fill(~attention_mask, neg)
            else:
                bias = attention_mask.to(query.dtype).clone()
            bias = bias.expand(bias.shape[0], bias.shape[1], q_len, kv).clone()
            if _boost_layer(li):
                bc = bcols[bcols < kv].to(query.device)
                if bc.numel():
                    import math
                    bias[..., bc] = bias[..., bc] + math.log(_BOOST)
            if _mask_layer(li):
                c = cols[cols < kv].to(query.device)
                if c.numel() and _VSCALE:
                    vn = value.float().norm(dim=-1)                                   # [B, Hkv, kv]
                    keep = torch.ones(kv, dtype=torch.bool, device=value.device); keep[c] = False
                    med = vn[..., keep].median(dim=-1).values                        # [B, Hkv]
                    fac = (med[..., None] / vn[..., c].clamp_min(1e-9)).clamp(max=1.0)  # only shrink
                    value = value.clone()
                    value[:, :, c, :] = (value[:, :, c, :].float() * fac[..., None]).to(value.dtype)
                elif c.numel():
                    if _HEADS:
                        H = int(query.shape[1])
                        if bias.shape[1] != H:
                            bias = bias.expand(bias.shape[0], H, q_len, kv).clone()
                        hs = torch.tensor([h for h in _HEADS if h < H], dtype=torch.long, device=bias.device)
                        sub = bias[:, hs]
                        sub[..., c] = neg
                        bias[:, hs] = sub
                    else:
                        bias[..., c] = neg
            _ST["calls"] += 1
            k.pop("is_causal", None)
            if _move_layer(li):
                import math
                H, Hkv = int(query.shape[1]), int(key.shape[1])
                kk = key.repeat_interleave(H // Hkv, dim=1) if Hkv != H else key
                vv = value.repeat_interleave(H // Hkv, dim=1) if Hkv != H else value
                sc = k.get("scaling") or (1.0 / math.sqrt(int(query.shape[-1])))
                A = torch.softmax((query.float() @ kk.float().transpose(-1, -2)) * sc + bias.float(), dim=-1)  # [B,H,q,kv]
                fr = _ST["mfrom"][_ST["mfrom"] < kv].to(A.device)
                to = _ST["mto"][_ST["mto"] < kv].to(A.device)
                if fr.numel() and to.numel():
                    take = A[..., fr] * _MOVE
                    tot = take.sum(-1, keepdim=True)                                  # [B,H,q,1]
                    At = A[..., to]
                    w = At / At.sum(-1, keepdim=True).clamp_min(1e-12)
                    w = torch.where(At.sum(-1, keepdim=True) > 1e-12, w, torch.full_like(w, 1.0 / to.numel()))
                    A[..., fr] = A[..., fr] - take
                    A[..., to] = At + tot * w
                    _ST["moved"].append(float(tot[..., -1, 0].mean()))               # last row, head mean
                out = (A.to(vv.dtype) @ vv).transpose(1, 2).contiguous()
                return out, None
            return orig(module, query, key, value, bias, *a, is_causal=False, **k)

        ALL_ATTENTION_FUNCTIONS["sdpa"] = sdpa_masked
        print(f"[LATMASK] patched sdpa attention (mode={_MODE or 'none'} roles={sorted(_WANT)} boost={_BOOST:g} "
              f"boost_roles={sorted(_BWANT)} boost_layers={_BL} move={_MOVE:g} {sorted(_MFROM)}->{sorted(_MTO)} "
              f"move_layers={_ML} heads={_HEADS or 'all'} vscale={_VSCALE})", flush=True)

    def _wrap_client(cls):
        orig_app = cls.append_only
        orig_fin = cls.generate_final_text

        def append_only(self, *, role, images, image_labels, system_prompt, user_prompt):
            eng = self._backend._engine
            bb = eng._backbone
            if not getattr(bb.processor, "_latmask_wrapped", False):
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
                bb.processor._latmask_wrapped = True
            if role == "evidence_planner":
                _ST["spans"] = []
            before = int(self._state.cache_length) if self._state is not None and self._state.cache_length else 0
            _ST["last_ids"] = None
            rec = orig_app(self, role=role, images=images, image_labels=image_labels,
                           system_prompt=system_prompt, user_prompt=user_prompt)
            ids = _ST["last_ids"] or []
            m = int(self._latent_steps)
            lat0 = before + len(ids)
            _ST["spans"].append({"role": role, "lat": list(range(lat0, lat0 + m))})
            return rec

        def generate_final_text(self, *a, **k):
            import torch
            cols = sorted(c for sp in _ST["spans"] if sp["role"] in _WANT for c in sp["lat"])
            _ST["cols"] = torch.tensor(cols, dtype=torch.long)
            _ST["bcols"] = torch.tensor(sorted(c for sp in _ST["spans"] if sp["role"] in _BWANT for c in sp["lat"]),
                                        dtype=torch.long)
            _ST["mfrom"] = torch.tensor(sorted(c for sp in _ST["spans"] if sp["role"] in _MFROM for c in sp["lat"]), dtype=torch.long)
            _ST["mto"] = torch.tensor(sorted(c for sp in _ST["spans"] if sp["role"] in _MTO for c in sp["lat"]), dtype=torch.long)
            _ST["active"], _ST["calls"], _ST["moved"] = True, 0, []
            try:
                return orig_fin(self, *a, **k)
            finally:
                _ST["active"] = False
                print(f"[LATMASK] case cols={len(cols)} roles={sorted({sp['role'] for sp in _ST['spans'] if sp['role'] in _WANT})} "
                      f"layers={_MODE or 'none'} boost={_BOOST:g}x{len(_ST['bcols'])}@{_BL} calls={_ST['calls']} "
                      f"move={_MOVE:g}@{_ML} moved_first_row={(_ST['moved'][0] if _ST['moved'] else 0):.4f}", flush=True)

        cls.append_only = append_only
        cls.generate_final_text = generate_final_text

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != "vision_text_mas.latent_onepass":
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or spec.loader is None:
                return spec
            orig = spec.loader.exec_module

            def exec_module(module):
                orig(module)
                _wrap_client(module.LatentRoleClient)
                _install_attention()

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
