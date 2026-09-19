"""0919 Order 93 — Native Role-Boundary Latent Pre-Read (NRLP). Original files untouched.

Chains tmp_hj/latplan_hooks (latplan base: no Planner text, no target lines in the nav4 prompt).

VLMAS_NRLP=1 turns the method on. At a role boundary the receiver spends ONE step reading only the
immediate predecessor's native latent bank (m=10 columns), then everything else runs natively:

  Planner -> Navigator, Navigator -> Reasoner   (budget preserving)
      the receiver's FIRST latent step (step 0) keeps its own query/position but its attention mask
      admits only the sender's latent columns, so
          m_L = sum_k softmax_k(q_C K_{S,k}^T / sqrt(d)) V_{S,k}
      is the whole attention read of that step. Steps 1..m-1 are untouched, so the latent budget,
      the cache length and the number of forwards are exactly the native ones.

  Reasoner -> Answerer   (+1 step; the Answerer has no latent slot to reuse)
      the terminal decode is split: the Answerer prompt minus its last token is prefilled natively,
      ONE pre-read step runs on the Reasoner latent bank, and the last prompt token + generation then
      run natively on top of it (the last token is split out so the pre-read can reach the FIRST
      answer token). The terminal decode already runs on a deepcopy of the cache, so the extra
      column never enters the persistent state.

Env:
  VLMAS_NRLP=1                 enable
  VLMAS_NRLP_BOUNDARIES        receivers, comma separated (default "navigator,reasoner,answerer")
  VLMAS_NRLP_SELF=1            also let the pre-read step attend to its own column (default 0 = the
                               canonical softmax over the sender bank only)
  VLMAS_NRLP_SINK=1            also admit cache column 0 (attention-sink escape valve; default 0)
  VLMAS_NRLP_LOG=<file>        append one json line per pre-read (column counts, hidden norms)
  VLMAS_NRLP_MODE=<mode>       the method (default "read") or one of the page's §10 ablations, applied to the
                               pre-read step only and undone right after it:
                                 read      native QK read of the sender bank (the method)
                                 uniform   every sender key replaced by the bank mean -> equal scores, so the
                                           step reads the mean of the sender values (is QK selection needed?)
                                 shuffle_v sender values permuted against their keys (is the k-th value's
                                           association with the k-th key needed?)
                                 donor     the previous case's sender bank stands in for this one (is the
                                           communication paired with THIS case?); the first case of a run only
                                           captures a donor and must be dropped from that comparison
                                 full      the step keeps the whole cache (control: is it just the extra step?)
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_f = os.path.join(os.path.dirname(_HERE), "latplan_hooks", "sitecustomize.py")
exec(compile(open(_f).read(), _f, "exec"), {"__file__": _f, "__name__": "sitecustomize_latplan"})

if os.environ.get("VLMAS_NRLP", "").strip() == "1":
    import importlib.abc
    import json

    _BOUNDARIES = tuple(
        s.strip() for s in
        os.environ.get("VLMAS_NRLP_BOUNDARIES", "navigator,reasoner,answerer").split(",")
        if s.strip()
    )
    _SELF = os.environ.get("VLMAS_NRLP_SELF", "") == "1"
    _SINK = os.environ.get("VLMAS_NRLP_SINK", "") == "1"
    _LOG = os.environ.get("VLMAS_NRLP_LOG", "").strip()
    # VLMAS_NRLP_EXTRA=1: the pre-read is an EXTRA step instead of the receiver's first latent step,
    # so every native latent step survives (the role runs m+1 steps). The sender bank stays the role's
    # m ordinary latent columns; the pre-read column itself is not handed on.
    _EXTRA = os.environ.get("VLMAS_NRLP_EXTRA", "") == "1"
    _MODE = os.environ.get("VLMAS_NRLP_MODE", "read").strip() or "read"
    assert _MODE in ("read", "uniform", "shuffle_v", "donor", "full"), _MODE
    _DONOR = {}          # role -> the previous case's sender bank (list of (K, V) per layer)

    # role name -> the predecessor whose latent bank it reads
    _PRED = {"navigator": "evidence_planner", "reasoner": "navigator", "answerer": "reasoner"}

    # per-case state: latent columns of every role appended so far, and the armed pre-read
    _S = {"lat": {}, "armed": None, "ids": None, "answerer": False, "case": None, "hooked": False}

    def _log(rec):
        print("[NRLP] " + json.dumps(rec, ensure_ascii=False), flush=True)
        if _LOG:
            with open(_LOG, "a") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _allowed_mask(cols, kv_len, self_col, device, dtype):
        """[1, kv_len] 0/1 mask: 1 on the sender bank (+ self / sink when enabled)."""
        import torch
        mask = torch.zeros((1, kv_len), dtype=dtype, device=device)
        mask[0, torch.tensor(cols, dtype=torch.long, device=device)] = 1
        if _SINK:
            mask[0, 0] = 1
        if _SELF and self_col is not None:
            mask[0, self_col] = 1
        return mask

    def _mode_apply(cache, cols, role):
        """§10 ablations: perturb the sender bank for this one step. Returns (saved, note)."""
        import torch
        if _MODE in ("read", "full"):
            return None, ""
        idx = torch.tensor(cols, dtype=torch.long, device=cache.layers[0].keys.device)
        saved = [(l.keys.index_select(2, idx).clone(), l.values.index_select(2, idx).clone())
                 for l in cache.layers]
        note = _MODE
        if _MODE == "uniform":
            for l in cache.layers:
                k = l.keys.index_select(2, idx)
                l.keys.index_copy_(2, idx, k.mean(dim=2, keepdim=True).expand_as(k).contiguous())
        elif _MODE == "shuffle_v":
            g = torch.Generator().manual_seed(42)
            perm = torch.randperm(len(cols), generator=g).to(idx.device)
            for l, (_, v) in zip(cache.layers, saved):
                l.values.index_copy_(2, idx, v.index_select(2, perm))
        elif _MODE == "donor":
            donor = _DONOR.get(role)
            if donor is None:
                note = "donor_capture(no swap)"
            else:
                for l, (dk, dv) in zip(cache.layers, donor):
                    n = min(int(dk.shape[2]), len(cols))
                    sub = idx[:n]
                    l.keys.index_copy_(2, sub, dk[:, :, :n].to(l.keys.dtype))
                    l.values.index_copy_(2, sub, dv[:, :, :n].to(l.values.dtype))
        return (idx, saved, role), note

    def _mode_restore(cache, state):
        if state is None:
            return
        idx, saved, role = state
        for l, (k, v) in zip(cache.layers, saved):
            l.keys.index_copy_(2, idx, k)
            l.values.index_copy_(2, idx, v)
        if _MODE == "donor":
            _DONOR[role] = saved          # this case's native bank is the next case's donor

    # ---------------------------------------------------------------- latent-step pre-read
    def _install_lm_hook(bb):
        """One pre-hook on the text model: the first 1-token forward after arming gets the mask."""
        if _S["hooked"]:
            return

        def pre_hook(module, args, kwargs):
            armed = _S["armed"]
            if armed is None:
                return None
            embeds = kwargs.get("inputs_embeds")
            if embeds is None or int(embeds.shape[1]) != 1:
                return None          # prefill chunk, not the latent step
            cache = kwargs.get("past_key_values")
            if cache is None:
                return None
            role, cols = armed
            _S["armed"] = None       # step 0 only
            kv_len = int(bb._kv_len(cache)) + 1
            cols = [c for c in cols if c < kv_len - 1]
            if not cols:
                _log({"event": "skip", "role": role, "why": "no sender columns", "kv_len": kv_len})
                return None
            state, note = _mode_apply(cache, cols, role)
            if _MODE != "full":
                kwargs["attention_mask"] = _allowed_mask(cols, kv_len, kv_len - 1,
                                                         embeds.device, embeds.dtype)
            _S["pending"] = {"role": role, "n_sender": len(cols), "kv_len": kv_len,
                             "in_norm": float(embeds.float().norm())}
            _S["restore"] = (cache, state, note)
            return args, kwargs

        def post_hook(module, args, kwargs, output):
            pend = _S.pop("pending", None)
            if pend is None:
                return None
            cache, state, note = _S.pop("restore", (None, None, ""))
            if cache is not None:
                _mode_restore(cache, state)
            h = getattr(output, "last_hidden_state", None)
            if h is None and isinstance(output, tuple):
                h = output[0]
            pend.update(event="preread", boundary="latent_step0", mode=_MODE, note=note,
                        out_norm=float(h[0, -1].float().norm()) if h is not None else None,
                        case=_S["case"], self_col=_SELF, sink=_SINK)
            _log(pend)
            return None

        bb.lm.register_forward_pre_hook(pre_hook, with_kwargs=True)
        bb.lm.register_forward_hook(post_hook, with_kwargs=True)
        _S["hooked"] = True

    # ---------------------------------------------------------------- Answerer (+1 step)
    def _wrap_generate(bb):
        if getattr(bb.model, "_nrlp_wrapped", False):
            return
        orig_generate = bb.model.generate

        def generate(**kw):
            import torch
            cols = _S["lat"].get("reasoner") or []
            ids = kw.get("input_ids")
            cache = kw.get("past_key_values")
            pos = kw.get("position_ids")
            if (not _S["answerer"] or not cols or ids is None or cache is None or pos is None
                    or int(ids.shape[1]) < 2 or "prefix_allowed_tokens_fn" in kw):
                return orig_generate(**kw)
            _S["answerer"] = False          # the first terminal decode of this call only
            prompt_len = int(ids.shape[1])
            past_len = int(bb._kv_len(cache))
            start = int(pos[0, 0, 0])
            head = ids[:, :-1]
            with torch.no_grad():
                embeds = bb.model.get_input_embeddings()(head)
                out = bb.lm(
                    inputs_embeds=embeds,
                    attention_mask=torch.ones((1, past_len + prompt_len - 1),
                                              dtype=torch.long, device=bb.device),
                    past_key_values=cache,
                    position_ids=pos[..., :-1],
                    use_cache=True,
                )
                cache = out.past_key_values
                h = out.last_hidden_state[:, -1:, :]
                le = bb._apply_realign(h)
                kv_len = past_len + prompt_len            # head columns + this step
                keep = [c for c in cols if c < kv_len - 1]
                state, note = _mode_apply(cache, keep, "answerer")
                mask = (None if _MODE == "full"
                        else _allowed_mask(keep, kv_len, kv_len - 1, le.device, le.dtype))
                out2 = bb.lm(
                    inputs_embeds=le,
                    attention_mask=mask,
                    past_key_values=cache,
                    position_ids=bb._text_positions(1, start + prompt_len - 1),
                    use_cache=True,
                )
                cache = out2.past_key_values
                _mode_restore(cache, state)
            _log({"event": "preread", "boundary": "answerer_plus1", "role": "answerer",
                  "mode": _MODE, "note": note,
                  "case": _S["case"], "n_sender": len(keep), "kv_len": kv_len,
                  "prompt_len": prompt_len, "in_norm": float(le.float().norm()),
                  "out_norm": float(out2.last_hidden_state[0, -1].float().norm()),
                  "self_col": _SELF, "sink": _SINK})
            kw = dict(kw)
            kw["input_ids"] = ids[:, -1:]
            kw["past_key_values"] = cache
            kw["attention_mask"] = torch.ones((1, past_len + prompt_len + 1),
                                              dtype=torch.long, device=bb.device)
            kw["position_ids"] = bb._text_positions(1, start + prompt_len)
            for crit in kw.get("stopping_criteria", []) or []:
                if hasattr(crit, "prompt_len"):
                    crit.prompt_len = 1
            generated = orig_generate(**kw)
            # the caller slices generated[:, prompt_len:]; pad the head back so that still holds
            return torch.cat([head, generated], dim=1)

        bb.model.generate = generate
        bb.model._nrlp_wrapped = True

    # ---------------------------------------------------------------- role bookkeeping
    def _wrap_client(cls):
        orig_append = cls.append_only
        orig_final = cls.generate_final_text

        def _wrap_processor(bb):
            if getattr(bb.processor, "_nrlp_wrapped", False):
                return
            p_orig = bb.processor.__class__.__call__

            def p_call(pself, *a, **k):
                out = p_orig(pself, *a, **k)
                if k.get("images") is not None:
                    try:
                        _S["ids"] = int(out["input_ids"].shape[1])
                    except Exception:
                        pass
                return out
            bb.processor.__class__.__call__ = p_call
            bb.processor._nrlp_wrapped = True

        def append_only(self, *, role, images, image_labels, system_prompt, user_prompt):
            bb = self._backend._engine._backbone
            _wrap_processor(bb)
            _install_lm_hook(bb)
            _wrap_generate(bb)
            m = int(self._latent_steps)
            before = int(self._state.cache_length) if self._state is not None and self._state.cache_length else 0
            if before == 0:                      # first role of a case
                _S["lat"] = {}
                _S["case"] = int(getattr(self._state, "dataset_index", -1)) if self._state is not None else None
            sender = _S["lat"].get(_PRED.get(role, ""))
            armed = bool(role in _BOUNDARIES and sender)
            if armed:
                _S["armed"] = (role, sender)
            _S["ids"] = None
            if armed and _EXTRA:
                self._latent_steps = m + 1       # the pre-read is an extra step, not step 0's job
            try:
                rec = orig_append(self, role=role, images=images, image_labels=image_labels,
                                  system_prompt=system_prompt, user_prompt=user_prompt)
            finally:
                self._latent_steps = m
            _S["armed"] = None                   # never leak into the next role
            n_ids = _S["ids"]
            if n_ids:
                lat0 = before + n_ids
                if armed and _EXTRA:
                    lat0 += 1                    # hand on the m ordinary steps, not the pre-read column
                _S["lat"][role] = list(range(lat0, lat0 + m))
            _S["case"] = int(self._state.dataset_index)
            return rec

        def generate_final_text(self, **kw):
            _S["answerer"] = "answerer" in _BOUNDARIES and bool(_S["lat"].get("reasoner"))
            try:
                return orig_final(self, **kw)
            finally:
                _S["answerer"] = False

        cls.append_only = append_only
        cls.generate_final_text = generate_final_text

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != "vision_text_mas.latent_onepass":
                return None
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
                _wrap_client(module.LatentRoleClient)
                print(f"[NRLP] patched {fullname} boundaries={_BOUNDARIES} self={_SELF} sink={_SINK}",
                      flush=True)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
