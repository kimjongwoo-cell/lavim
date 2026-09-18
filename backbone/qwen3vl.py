"""Qwen3-VL-4B-Thinking backbone for latent reasoning.

Model structure:
  model.model.visual          → Qwen3VLVisionModel (ViT)
  model.model.language_model  → Qwen3VLTextModel
    layers[i].self_attn       → Qwen3VLTextAttention (ALL 36 layers, full attention)

All layers are full attention → clean KV cache for all layers.

Vision merging (mirrors Qwen3VLModel.forward):
  1. processor(images) → pixel_values, image_grid_thw
  2. model.model.get_image_features(pixel_values, image_grid_thw) → image_embeds
  3. embed_tokens(text_ids) → text_embeds
  4. masked_scatter image_embeds into text_embeds at <|image_pad|> positions
  5. compute_3d_position_ids() → position_ids (MRoPE)
  6. language_model(inputs_embeds, position_ids, ...) → forward
"""

from __future__ import annotations

import copy
import contextlib
import math
import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb, repeat_kv

from backbone.efficiency import EfficiencyMixin


def _transport_embeddings(
    prefill_input_embeddings: torch.Tensor,
    latent_trajectory: list[torch.Tensor],
) -> torch.Tensor:
    """Carry only recurrent latent inputs into the terminal readout."""
    if not latent_trajectory:
        return prefill_input_embeddings[:0]
    return torch.stack(latent_trajectory, dim=0)


class Qwen3VLBackbone(EfficiencyMixin):
    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        realign_method: str = "wa",   # "wa" | "softmax" | "identity_norm"
        realign_tau: float = 1.0,     # temperature for softmax method
        instruct: bool = False,
        mrope_pos: bool = True,       # use true MRoPE 3D positions (vs sequential)
        vision_inject: str = "grounded",  # "grounded" (template-internal, default) | "bolton" (legacy)
        build_realign: bool = True,   # False for text_mas (no latent steps → realign unused)
    ) -> None:
        self.device = device
        self.dtype  = dtype
        self._realign_method = realign_method
        self._realign_tau    = realign_tau
        self.instruct = instruct
        self.mrope_pos = mrope_pos
        self.vision_inject = vision_inject
        self._last_grid_thw = None
        # token-usage log for text_mas (filled by vlm_generate, read by pipeline)
        self.usage: list[dict] = []
        attention_backend = os.environ.get("VLMAS_ATTN_IMPLEMENTATION", "eager")
        if attention_backend not in {"eager", "sdpa"}:
            raise ValueError(
                "VLMAS_ATTN_IMPLEMENTATION must be 'eager' or 'sdpa'"
            )

        print(f"[Backbone] Loading {model_path} ...")
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=dtype, device_map=device,
            attn_implementation=attention_backend,
        )
        self.model.eval()
        print(f"[Backbone] Attention implementation: {attention_backend}", flush=True)

        self.vlm  = self.model.model          # Qwen3VLModel
        self.lm   = self.vlm.language_model   # Qwen3VLTextModel
        self.visual = self.vlm.visual         # Qwen3VLVisionModel

        # additive efficiency instrumentation (FLOPs profiler + TTFT); no-op if
        # it fails to init, and it never alters generated text.
        self._init_efficiency()

        # All layers are full attention for Qwen3-VL
        self.full_attn_indices: list[int] = list(range(len(self.lm.layers)))
        print(f"[Backbone] Layers: {len(self.lm.layers)} (all full attention)")

        # cache embeddings for realignment (both methods need them)
        self._input_emb: torch.Tensor | None = None   # E [V, D]
        self._output_emb: torch.Tensor | None = None  # U [V, D]
        # W_a specific
        self._realign_matrix: torch.Tensor | None = None
        self._realign_target_norm: torch.Tensor | None = None
        # text_mas decodes via plain model.generate (no latent steps), so the
        # realign map is never applied — skip building it (and its logs) there.
        if build_realign:
            self._build_realign_matrix()
            print(f"[Backbone] Realign method: {realign_method}"
                  + (f", tau={realign_tau}" if realign_method == "softmax" else ""))

    # ── latent realignment (LatentMAS style) ──────────────────────────────────
    def _build_realign_matrix(self) -> None:
        """Cache embeddings and build W_a (used only when realign_method='wa')."""
        dev = torch.device(self.device)

        input_emb  = self.model.get_input_embeddings().weight.detach().float()
        target_norm = input_emb.norm(dim=1).mean()
        self._realign_target_norm = target_norm.to(
            device=dev,
            dtype=torch.float32,
        )
        if self._realign_method == "identity_norm":
            print(f"[Backbone] Identity norm target={target_norm:.4f}")
            return
        output_emb = self.model.get_output_embeddings().weight.detach().float()

        # cache for softmax method
        self._input_emb  = input_emb.to(dev)   # E [V, D]
        self._output_emb = output_emb.to(dev)  # U [V, D]

        # W_a (least squares)
        gram = output_emb.T @ output_emb
        reg  = 1e-5 * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        rhs  = output_emb.T @ input_emb
        matrix = torch.linalg.solve(gram + reg, rhs)
        self._realign_matrix      = matrix.to(device=dev, dtype=torch.float32)
        print(f"[Backbone] W_a built: {tuple(matrix.shape)}, "
              f"target_norm={target_norm:.4f}")

    def _apply_realign(self, hidden: torch.Tensor) -> torch.Tensor:
        """Dispatch to W_a or softmax realignment.  hidden: [B, 1, D]"""
        if self._realign_method == "identity_norm":
            out = self._apply_realign_identity_norm(hidden)
        elif self._realign_matrix is None:
            return hidden
        elif self._realign_method == "softmax":
            out = self._apply_realign_softmax(hidden)
        else:
            out = self._apply_realign_wa(hidden)
        return self._latent_noise(out)

    def _latent_noise(self, z: torch.Tensor) -> torch.Tensor:
        """Stochastic latent (VLMAS_LATENT_NOISE=sigma, off by default): add
        seeded isotropic noise with relative scale sigma to the realigned
        latent and re-normalise to the target norm. Diagnostic lever against
        the latent-recurrence fixed point (super: VL-4B collapses at step 4,
        sigma=0.3 keeps PR ~5 like a text LLM). Seed = 42 + case counter so
        runs are reproducible; the noise is applied only to latent-step
        feedback (this method is only called there)."""
        import os as _os
        sigma = float(_os.environ.get("VLMAS_LATENT_NOISE", "0") or 0)
        if sigma <= 0:
            return z
        gen = getattr(self, "_latent_noise_gen", None)
        if gen is None:
            gen = torch.Generator(device=z.device).manual_seed(42)
            self._latent_noise_gen = gen
        zf = z.float()
        noise = torch.randn(zf.shape, generator=gen, device=zf.device) / (zf.shape[-1] ** 0.5)
        zf = zf + sigma * zf.norm(dim=-1, keepdim=True) * noise
        if self._realign_target_norm is not None:
            zf = zf * (self._realign_target_norm / zf.norm(dim=-1, keepdim=True).clamp_min(1e-6))
        return zf.to(z.dtype)

    def _apply_realign_identity_norm(self, hidden: torch.Tensor) -> torch.Tensor:
        """Match upstream's default identity projection plus norm scaling."""
        if self._realign_target_norm is None:
            return hidden
        original_dtype = hidden.dtype
        normalized = hidden.float()
        norm = normalized.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        normalized = normalized * (self._realign_target_norm / norm)
        return normalized.to(original_dtype)

    def _apply_realign_wa(self, hidden: torch.Tensor) -> torch.Tensor:
        """W_a least-squares map.  hidden: [B, 1, D] → [B, 1, D]"""
        orig_dtype = hidden.dtype
        h = hidden.float().squeeze(1)                          # [B, D]
        aligned = h @ self._realign_matrix                     # [B, D]
        norm = aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        aligned = aligned * (self._realign_target_norm / norm)
        return aligned.to(orig_dtype).unsqueeze(1)

    def _apply_realign_softmax(self, hidden: torch.Tensor) -> torch.Tensor:
        """Softmax realignment:  z̃_t = E^T · softmax(U z_t / τ)

        Result is a convex combination of input embeddings (lies in their
        convex hull by construction). Training-free, uses U and E only.

        hidden: [B, 1, D] → [B, 1, D]
        """
        orig_dtype = hidden.dtype
        h = hidden.float().squeeze(1)          # [B, D]
        U = self._output_emb                   # [V, D]
        E = self._input_emb                    # [V, D]

        logits  = h @ U.T / self._realign_tau  # [B, V]
        p       = torch.softmax(logits, dim=-1) # [B, V]  (on embedding simplex)
        z_tilde = p @ E                         # [B, D]

        return z_tilde.to(orig_dtype).unsqueeze(1)  # [B, 1, D]

    # ── vision encoding ────────────────────────────────────────────────────────
    @torch.no_grad()
    def encode_patches(self, images: list[Image.Image]) -> tuple[torch.Tensor, int]:
        """Encode patch images → visual token embeddings.

        Uses Qwen3VLModel.get_image_features() which runs:
          pixel_values → ViT → pooler → merged [n_vis_total, hidden]

        Returns:
            visual_embeds : [n_patches * n_vis_per_patch, hidden]
            n_vis_per_patch: visual tokens per patch image
        """
        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": "."})
        msgs = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False,
            enable_thinking=False,
        )
        inputs = self.processor(
            text=[text], images=images, return_tensors="pt"
        )
        pv  = inputs["pixel_values"].to(self.device, dtype=self.dtype)
        thw = inputs["image_grid_thw"].to(self.device)

        img_out = self.vlm.get_image_features(pv, thw, return_dict=True)
        # pooler_output: list of per-image tensors [n_vis_i, hidden]
        vis_feat = torch.cat(img_out.pooler_output, dim=0)  # [total_vis, hidden]

        n_vis_per_patch = vis_feat.shape[0] // len(images)
        # remember the per-patch grids so position ids can use true MRoPE coords
        self._last_grid_thw = thw  # [n_patches, 3] (t,h,w) raw grid
        return vis_feat, n_vis_per_patch, thw

    # ── MRoPE 3D position ids ────────────────────────────────────────────────────
    def vision_coords(
        self,
        grids: torch.Tensor | None,
        start: int = 0,
    ) -> tuple[torch.Tensor, int]:
        """Per-visual-token MRoPE coords for a [vis_1..vis_k] layout.

        Replicates get_rope_index's per-image logic: each image's tokens get
        (t,h,w) grid coordinates offset by the running cursor, and the cursor
        advances by the *compressed* span max(h,w)//merge (not the token count).

        Returns (coords [3, n_vis], next_cursor). Token i's coordinate is
        coords[:, i] — pruning simply keeps a subset of columns, so positions
        for a kept subset are positions_from_coords(coords[:, kept_idx], ...).
        """
        sms = self.model.config.vision_config.spatial_merge_size
        cur = start
        pieces = []
        if grids is not None:
            for g in grids:  # g: (t,h,w)
                vp = self.vlm.get_vision_position_ids(cur, g, 1, sms, device=self.device)
                pieces.append(vp)  # [3, n_vis_i]
                cur += max(int(g[1]), int(g[2])) // sms
        if pieces:
            coords = torch.cat(pieces, dim=1)  # [3, n_vis]
        else:
            coords = torch.zeros(3, 0, dtype=torch.long, device=self.device)
        return coords, cur

    def positions_from_coords(
        self,
        vis_coords: torch.Tensor,   # [3, n_vis] (possibly a kept subset)
        n_text: int,
        text_start: int,
    ) -> tuple[torch.Tensor, int]:
        """Assemble [3,1,seq] position ids from per-token visual coords plus
        n_text trailing text tokens (all three axes equal, from text_start)."""
        pieces = [vis_coords]
        cur = text_start
        if n_text > 0:
            t = torch.arange(n_text, device=self.device).view(1, -1).expand(3, -1) + cur
            pieces.append(t)
            cur += n_text
        pos = torch.cat(pieces, dim=1).unsqueeze(1)  # [3,1,seq]
        return pos, cur

    def build_mrope_positions(
        self,
        grids: torch.Tensor | None,
        n_text: int,
        start: int = 0,
    ) -> tuple[torch.Tensor, int]:
        """Convenience wrapper: full-patch [vis|text] positions.

        Equivalent to keeping *all* visual tokens. Pruning paths should instead
        use vision_coords() + positions_from_coords() with a kept subset.

        Returns (position_ids [3,1,seq], next_cursor).
        """
        vis_coords, vcur = self.vision_coords(grids, start=start)
        return self.positions_from_coords(vis_coords, n_text, vcur)

    def _text_positions(self, n: int, start: int) -> torch.Tensor:
        """[3,1,n] MRoPE positions for n text-like tokens, all axes equal."""
        t = torch.arange(start, start + n, device=self.device).view(1, -1).expand(3, -1)
        return t.unsqueeze(1)  # [3,1,n]

    @contextlib.contextmanager
    def _capture_selected_latent_attention(
        self,
        *,
        layer_indices: list[int],
        vision_columns: torch.Tensor,
        visual_only_scores: bool = False,
    ):
        """Capture selected-layer visual saliency without materializing full attention."""
        captured: dict[int, torch.Tensor] = {}
        handles: list[torch.utils.hooks.RemovableHandle] = []
        columns = vision_columns.detach().to(device=self.device, dtype=torch.long)

        def make_hook(layer_index: int):
            def hook(module, args, kwargs, _output):
                hidden = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
                position_embeddings = (
                    kwargs["position_embeddings"]
                    if "position_embeddings" in kwargs else args[1]
                )
                attention_mask = (
                    kwargs["attention_mask"] if "attention_mask" in kwargs else args[2]
                )
                cache = (
                    kwargs["past_key_values"]
                    if "past_key_values" in kwargs
                    else (args[3] if len(args) > 3 else None)
                )
                if cache is None:
                    return
                key_states = cache.layers[layer_index].keys
                valid_columns = columns[columns < key_states.shape[2]]
                if valid_columns.numel() == 0:
                    return
                query_shape = (*hidden.shape[:-1], -1, module.head_dim)
                query = module.q_norm(module.q_proj(hidden).view(query_shape)).transpose(1, 2)
                cos, sin = position_embeddings
                query, _ = apply_rotary_pos_emb(query, torch.zeros_like(query), cos, sin)
                if visual_only_scores:
                    key_states = key_states.index_select(2, valid_columns)
                key_states = repeat_kv(key_states, module.num_key_value_groups)
                scores = torch.matmul(query[:, :, -1:, :], key_states.transpose(-2, -1))
                scores = scores * module.scaling
                if visual_only_scores:
                    captured[layer_index] = scores[0, :, 0, :].float()
                else:
                    if attention_mask is not None:
                        scores = scores + attention_mask[:, :, -1:, : key_states.shape[2]]
                    captured[layer_index] = torch.softmax(scores, dim=-1)[
                        0, :, 0, valid_columns
                    ].float()

            return hook

        for index in layer_indices:
            handles.append(
                self.lm.layers[index].self_attn.register_forward_hook(
                    make_hook(index),
                    with_kwargs=True,
                )
            )
        try:
            yield captured
        finally:
            for handle in handles:
                handle.remove()

    @staticmethod
    def _continuation_user_text(system_prompt: str | None, user_text: str) -> str:
        """Encode a later agent role inside a valid user turn.

        Qwen chat templates permit one leading system message.  In a cumulative
        latent rollout, later roles must therefore be represented as new user
        turns after the preceding assistant turn has been closed, not as another
        system message injected into the middle of the transcript.
        """
        if not system_prompt:
            return user_text
        return f"Next-stage role instructions:\n{system_prompt}\n\n{user_text}"

    @torch.no_grad()
    def close_assistant_turn(
        self,
        past_kv,
        past_pos_cursor: int,
        *,
        close_thinking: bool = False,
    ):
        """Append Qwen's assistant terminator before the next cumulative role."""
        boundary_ids = self.processor.tokenizer(
            ("</think>\n" if close_thinking else "") + "<|im_end|>\n",
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"].to(self.device)
        boundary_embeds = self.lm.embed_tokens(boundary_ids)
        result = self.lm(
            inputs_embeds=boundary_embeds,
            position_ids=self._text_positions(boundary_ids.shape[1], past_pos_cursor),
            past_key_values=past_kv,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        next_cache = result.past_key_values
        return (
            next_cache,
            self._kv_len(next_cache),
            past_pos_cursor + int(boundary_ids.shape[1]),
        )

    # ── text embedding ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def embed_text(
        self,
        text: str,
        system_prompt: str | None = None,
        enable_thinking: bool = False,
    ) -> torch.Tensor:
        """Tokenise and embed text. Returns [n_q, hidden].

        Args:
            text          : user message content
            system_prompt : optional system role description (agent role)
            enable_thinking: whether to open <think> block (for Reasoner/Refiner)
        """
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": text})

        if self.instruct:
            prompt = self.processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        else:
            prompt = self.processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )
            # Thinking model always appends <think>\n; close it if thinking is disabled
            _tt = os.environ.get("VLMAS_THINK_TAGS", "").strip()
            if _tt == "strip" and prompt.endswith("<think>\n"):
                prompt = prompt[: -len("<think>\n")]
            elif _tt == "open":
                pass
            elif not enable_thinking:
                prompt = prompt + "\n</think>\n\n"

        ids = self.processor.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False,
        )["input_ids"].to(self.device)
        return self.lm.embed_tokens(ids).squeeze(0)  # [n_q, hidden]

    # ── plain VLM text generation over images (TextMAS path) ───────────────────
    @torch.no_grad()
    def vlm_generate(
        self,
        images: list[Image.Image] | None,
        system_prompt: str,
        user_text: str,
        max_new_tokens: int = 512,
        tag: str = "",
    ) -> str:
        """Standard processor + model.generate pass over zero or more images.

        Unlike generate_on_kv (which decodes on an existing latent/text KV), the
        agent SEES the crop images and DECODES natural-language text. This is the
        token-space channel for the text_mas method: the Reasoner describes
        patch-level observations, and (optionally) the Diagnosis answers with the
        crops re-fed. images=[]/None → text-only generation.
        """
        imgs = list(images or [])
        content: list[dict] = [{"type": "image", "image": im} for im in imgs]
        content.append({"type": "text", "text": user_text})
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        # Thinking model appends "<think>\n"; close it so the agent answers directly.
        if not self.instruct and text.endswith("<think>\n"):
            text = text + "</think>\n\n"

        proc_kwargs: dict = dict(text=[text], return_tensors="pt")
        if imgs:
            proc_kwargs["images"] = imgs
        inputs = self.processor(**proc_kwargs).to(self.model.device)

        _timer, _t0 = self._ttft_timer()
        _gk = dict(max_new_tokens=max_new_tokens, do_sample=False)
        if _timer is not None:
            _gk["logits_processor"] = [_timer]
        ids = self.model.generate(**inputs, **_gk)
        self._record_ttft(_timer, _t0)
        n_prompt = int(inputs["input_ids"].shape[1])
        new_ids = ids[:, n_prompt:]
        n_gen = int(new_ids.shape[1])
        # token-usage bookkeeping (per-agent generated + prefill cost). The
        # pipeline resets self.usage per item and aggregates after the loop.
        self.usage.append({"tag": tag, "prompt_tokens": n_prompt, "gen_tokens": n_gen})
        raw = self.processor.batch_decode(new_ids, skip_special_tokens=False)[0]
        if "</think>" in raw:
            raw = raw.split("</think>", 1)[-1]
        raw = re.sub(r"<[^>]+>", "", raw).strip()
        return raw

    # ── position ids for latent tokens ────────────────────────────────────────
    def _make_position_ids(
        self, seq_len: int, past_len: int = 0
    ) -> torch.Tensor:
        """Simple 1D position ids for text-only or latent tokens.
        Shape: [3, 1, seq_len] (MRoPE format: temporal, height, width all same).
        """
        pos = torch.arange(past_len, past_len + seq_len,
                           device=self.device).unsqueeze(0)  # [1, seq_len]
        return pos.unsqueeze(0).expand(3, 1, seq_len)         # [3, 1, seq_len]

    # ── latent step loop ───────────────────────────────────────────────────────
    @torch.no_grad()
    def forward_with_latent_steps(
        self,
        visual_embeds: torch.Tensor,   # [n_vis, hidden]
        text_embeds:   torch.Tensor,   # [n_q,  hidden]
        m: int,
        l_mid_layers: list[int],
        past_kv=None,                  # carried-over KV from a prior iteration
        past_len: int = 0,             # current seq length already in past_kv
        grids: torch.Tensor | None = None,   # per-patch (t,h,w) for MRoPE
        past_pos_cursor: int = 0,      # max MRoPE position+1 already in past_kv
    ) -> dict:
        """Prefill [vis | text], then run m latent steps.

        If past_kv is provided, the new [vis | text] tokens are appended after
        the existing KV, so reasoning continues from prior iterations' context.

        Position ids: with self.mrope_pos, visual tokens get true 3D MRoPE
        (h,w) grid coordinates and the rope cursor advances by the *compressed*
        span (not token count); otherwise falls back to sequential ids. The
        rope cursor is tracked separately from the cache token count (past_len).

        Returns the usual keys plus:
          past_len   : total seq length (token count) after this call
          pos_cursor : max MRoPE position+1 after this call (for continuation)
        """
        n_vis = visual_embeds.shape[0]
        n_q   = text_embeds.shape[0]
        seq   = n_vis + n_q

        # column offset of this iteration's visual tokens within the full
        # attention matrix (0 when no past, else right after carried-over KV)
        vis_col0 = past_len

        input_embeds = torch.cat(
            [visual_embeds.unsqueeze(0), text_embeds.unsqueeze(0)], dim=1
        )  # [1, seq, hidden]

        if grids is None:
            grids = self._last_grid_thw

        # position_ids for prefill
        if self.mrope_pos and grids is not None:
            # keep per-token visual coords so future pruning can re-anchor a
            # kept subset without recomputing from grids
            vis_coords, vcur = self.vision_coords(grids, start=past_pos_cursor)
            pos_ids, cursor = self.positions_from_coords(vis_coords, n_q, vcur)
        else:
            vis_coords = None
            pos_ids = self._make_position_ids(seq, past_len=past_len)  # sequential
            cursor = past_len + seq

        # ── prefill ────────────────────────────────────────────────────────────
        prune_args = getattr(self, "_prune_args", None)
        dynamic_prune = bool(
            prune_args is not None
            and getattr(prune_args, "prune_dynamic_ratio", False)
        )
        out = self.lm(
            inputs_embeds=input_embeds,
            position_ids=pos_ids,
            past_key_values=past_kv,
            use_cache=True,
            # Dynamic pruning derives saliency from latent-step attention.  The
            # prefill attention tensor contains all 36 layers and is otherwise
            # a large transient allocation with no consumer on this path.
            output_attentions=not dynamic_prune,
            output_hidden_states=True,
        )
        past_kv     = out.past_key_values
        last_hidden = out.hidden_states[-1][:, -1:, :]  # [1, 1, hidden]

        # query-to-vision attention from prefill
        # attn rows are this iteration's tokens; cols span [0, past_len+seq).
        # this iteration's question rows: [vis_col0+n_vis : vis_col0+seq]
        # this iteration's visual cols  : [vis_col0 : vis_col0+n_vis]
        query_to_vision_attn: dict[int, torch.Tensor] = {}
        # sink/row-sum diagnostics (P1): col 0 = attention-sink token; row-sum (~1.0)
        # lets a downstream probe quantify how much of each row's softmax mass is
        # eaten by the sink before renormalizing over visual tokens.
        query_to_vision_sink:   dict[int, torch.Tensor] = {}
        query_to_vision_rowsum: dict[int, torch.Tensor] = {}
        for li in l_mid_layers:
            a = out.attentions[li]
            if a is not None:
                q2v = a[0, :, n_vis:, vis_col0:vis_col0 + n_vis]  # [n_heads, n_q, n_vis]
                query_to_vision_attn[li] = q2v.cpu()
                query_to_vision_sink[li]   = a[0, :, n_vis:, 0].cpu()         # [n_heads, n_q]
                query_to_vision_rowsum[li] = a[0, :, n_vis:, :].sum(-1).cpu() # [n_heads, n_q]

        # ── latent steps ───────────────────────────────────────────────────────
        latent_trajectory: list[torch.Tensor] = []
        l2v_lists: dict[int, list[torch.Tensor]] = {li: [] for li in l_mid_layers}
        l2v_sink_lists:   dict[int, list[torch.Tensor]] = {li: [] for li in l_mid_layers}
        l2v_rowsum_lists: dict[int, list[torch.Tensor]] = {li: [] for li in l_mid_layers}

        for step in range(m):
            latent_embed = self._apply_realign(last_hidden)  # [1, 1, hidden]
            if self.mrope_pos and grids is not None:
                pos_ids_step = self._text_positions(1, start=cursor + step)
            else:
                pos_ids_step = self._make_position_ids(1, past_len=past_len + seq + step)

            out = self.lm(
                inputs_embeds=latent_embed,
                position_ids=pos_ids_step,
                past_key_values=past_kv,
                use_cache=True,
                output_attentions=True,
                output_hidden_states=True,
            )
            past_kv     = out.past_key_values
            last_hidden = out.hidden_states[-1][:, -1:, :]
            latent_trajectory.append(latent_embed.squeeze().cpu())

            # latent-to-vision attention
            # attn: [1, n_heads, 1, past_len+seq+step+1]
            for li in l_mid_layers:
                a = out.attentions[li]
                if a is not None:
                    l2v = a[0, :, 0, vis_col0:vis_col0 + n_vis]   # [n_heads, n_vis]
                    l2v_lists[li].append(l2v.cpu())
                    l2v_sink_lists[li].append(a[0, :, 0, 0].cpu())          # [n_heads]
                    l2v_rowsum_lists[li].append(a[0, :, 0, :].sum(-1).cpu()) # [n_heads]

        # stack: [m, n_heads, n_vis]
        latent_to_vision_attn: dict[int, torch.Tensor] = {
            li: torch.stack(lst, dim=0)
            for li, lst in l2v_lists.items() if lst
        }
        # stack: [m, n_heads]
        latent_to_vision_sink: dict[int, torch.Tensor] = {
            li: torch.stack(lst, dim=0) for li, lst in l2v_sink_lists.items() if lst
        }
        latent_to_vision_rowsum: dict[int, torch.Tensor] = {
            li: torch.stack(lst, dim=0) for li, lst in l2v_rowsum_lists.items() if lst
        }

        # ── extract KV for agent transfer ──────────────────────────────────────
        full_attn_kv = self._extract_full_attn_kv(past_kv, l_mid_layers)

        return {
            "past_key_values":        past_kv,
            "latent_trajectory":      latent_trajectory,
            "latent_to_vision_attn":  latent_to_vision_attn,
            "query_to_vision_attn":   query_to_vision_attn,
            # sink/row-sum diagnostics (P1) for the saliency probe
            "latent_to_vision_sink":   latent_to_vision_sink,
            "latent_to_vision_rowsum": latent_to_vision_rowsum,
            "query_to_vision_sink":    query_to_vision_sink,
            "query_to_vision_rowsum":  query_to_vision_rowsum,
            "full_attn_kv":           full_attn_kv,
            "past_len":               past_len + seq + m,
            "pos_cursor":             cursor + m + _visual_relays,
            "vis_coords":             (vis_coords.cpu() if vis_coords is not None else None),
            # span types of the columns this call appended: visual | prompt | latent
            "spans":                  [s for s in (("V", n_vis), ("P", n_q), ("L", m)) if s[1] > 0],
        }

    def _locate_role_spans(self, ids, tail_start: int, tail_end: int,
                           role_targets: list) -> list:
        """Find token ranges of each target string inside ids[tail_start:tail_end].

        Returns (start, end) ranges in TAIL-RELATIVE coords (0 == tail_start), i.e.
        directly indexing the q_rows / n_q saliency axis. The post-image tail is
        pure text (no image-token expansion), so a contiguous id-subsequence match
        is exact; we try a few leading-whitespace variants and silently skip any
        target not found (Rel then just uses the spans that matched)."""
        tok = self.processor.tokenizer
        id_list = ids[tail_start:tail_end].tolist()
        n = len(id_list)
        spans: list = []
        for tgt in (role_targets or []):
            tgt = (str(tgt) if tgt is not None else "").strip()
            if not tgt:
                continue
            for variant in (" " + tgt, "\n" + tgt, tgt):
                try:
                    seq = tok.encode(variant, add_special_tokens=False)
                except Exception:
                    seq = []
                L = len(seq)
                if L == 0 or L > n:
                    continue
                hit = next((i for i in range(0, n - L + 1)
                            if id_list[i:i + L] == seq), None)
                if hit is not None:
                    spans.append((hit, hit + L))
                    break
        # Isolated tokenization above can miss a phrase when the first target
        # token was merged with the preceding label/punctuation by BPE. Recover
        # spans against the exact decoded tail and only accept the offset map
        # when decode→encode is lossless for these cache rows.
        if len(spans) < len(role_targets or []):
            decoded = tok.decode(
                id_list,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            encoded = tok(
                decoded,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            encoded_ids = encoded["input_ids"]
            offsets = encoded["offset_mapping"]
            if encoded_ids == id_list:
                for tgt in (role_targets or []):
                    if not tgt:
                        continue
                    char_start = decoded.find(str(tgt).strip())
                    if char_start < 0:
                        continue
                    char_end = char_start + len(str(tgt).strip())
                    token_indices = [
                        index
                        for index, (start, end) in enumerate(offsets)
                        if start < char_end and end > char_start
                    ]
                    if not token_indices:
                        continue
                    span = (token_indices[0], token_indices[-1] + 1)
                    if span not in spans:
                        spans.append(span)
        return spans

    # ── grounded prefill (template-internal vision injection) + latent steps ────
    @torch.no_grad()
    def grounded_prefill_and_latent(
        self,
        images: list,                  # list[PIL.Image]
        system_prompt: str | None,
        user_text: str,
        m: int,
        l_mid_layers: list[int],
        enable_thinking: bool = False,
        want_attn: bool = True,
        want_vis_hidden: bool = False,
        role_targets: list | None = None,   # P1-2: strings to tag as Q_q (question/choices)
        bg_targets: list | None = None,     # --saliency_bg_correct: template rows B strings
    ) -> dict:
        """Inject the image INSIDE the chat template (with <|vision_start|>/<|image_pad|>
        markers) so the model actually grounds in it, then run m latent steps on the
        grounded KV. Mirrors the validated isolated test (diag_latent_grounding).

        The full model.forward builds the grounded prefill (handles deepstack/markers/
        MRoPE); latent steps continue through the language model directly. Returns the
        same dict shape as forward_with_latent_steps (+ vis_cols = image column range).
        """
        img_id = getattr(self.model.config, "image_token_id", 151655)

        content = [{"type": "image", "image": im} for im in images]
        content.append({"type": "text", "text": user_text})
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": content})

        text = self.processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        _tt = os.environ.get("VLMAS_THINK_TAGS", "").strip()
        if _tt == "strip" and text.endswith("<think>\n"):
            text = text[: -len("<think>\n")]
        elif _tt == "open":
            pass
        elif not self.instruct and not enable_thinking:
            text = text + "\n</think>\n\n"   # match embed_text convention

        inputs = self.processor(text=[text], images=images, return_tensors="pt").to(self.device)
        ids = inputs["input_ids"][0]
        is_vis = (ids == img_id)
        vis_idx = is_vis.nonzero(as_tuple=True)[0]              # exact image-token cols
        last = int(vis_idx[-1]); nV = int(vis_idx.numel())
        seq_prefill = len(ids)

        # RLE column-type map (multiple crops -> several image runs separated by
        # vision_end/start markers, so V is NOT one contiguous block).
        spans: list[tuple[str, int]] = []
        run_t, run_n = None, 0
        for k in range(seq_prefill):
            t = "V" if bool(is_vis[k]) else "P"
            if t == run_t:
                run_n += 1
            else:
                if run_t is not None:
                    spans.append((run_t, run_n))
                run_t, run_n = t, 1
        spans.append((run_t, run_n))
        prefill_spans = list(spans)            # P/V runs only (pre-latent)
        if m > 0:
            spans.append(("L", m))

        prune_args = getattr(self, "_prune_args", None)
        dynamic_prune = bool(
            prune_args is not None
            and getattr(prune_args, "prune_dynamic_ratio", False)
        )
        sparsevlm_active = bool(
            getattr(prune_args, "prune_query_adaptive", False)
            and getattr(self, "_prune_morphology_enabled", False)
        )
        atp_sap_active = bool(
            getattr(prune_args, "prune_atp_sap", False)
            and getattr(self, "_prune_morphology_enabled", False)
        )
        atp_all_layers = bool(
            atp_sap_active and getattr(prune_args, "prune_atp_all_layers", False)
        )
        visual_probe_layers = (
            list(range(len(self.lm.layers))) if atp_all_layers else l_mid_layers
        )
        visual_boundary_pruning = sparsevlm_active or atp_sap_active
        capture_prefill_attn = (
            want_attn and not dynamic_prune and not visual_boundary_pruning
        )
        q_rows = torch.arange(last + 1, seq_prefill, device=self.device)
        _sc_context = (
            self._capture_prefill_q2v(visual_probe_layers, q_rows, vis_idx)
            if visual_boundary_pruning
            else contextlib.nullcontext(None)
        )
        _self_context = (
            self._capture_prefill_visual_self_score(
                visual_probe_layers, vis_idx, vis_idx
            )
            if atp_sap_active
            else contextlib.nullcontext(None)
        )
        # AAVM (env VLMAS_AAVM=1, off = no hooks at all): collect the pre-RoPE
        # visual keys and their rotation phases during this one prefill, so the
        # acquisition re-addressing at the consolidation boundary needs no
        # second forward over the crops. See memory/aavm_runtime.py.
        from memory import aavm_runtime as _aavm
        with (
            _sc_context as _sc_q2v,
            _self_context as _self_scores,
            _aavm.capture_visual_raw_keys(self, vis_idx) as _aavm_capture,
        ):
            out = self.model(
                **inputs,
                use_cache=True,
                output_attentions=capture_prefill_attn,
                output_hidden_states=True,
            )
        prefill_input_embeddings = out.hidden_states[0][0].detach().cpu()
        past_kv = out.past_key_values
        last_hidden = out.hidden_states[-1][:, -1:, :]
        # last-layer hidden over the vision tokens (ROI/crop order) for z_i; the
        # full hidden states already exist here, so this is just a slice+copy.
        vis_hidden = (out.hidden_states[-1][0, vis_idx, :].float().cpu()
                      if want_vis_hidden else None)

        # query->vision from the post-image text rows (question/choices region)
        query_to_vision_attn: dict[int, torch.Tensor] = {}
        query_to_vision_sink:   dict[int, torch.Tensor] = {}   # col-0 sink share
        query_to_vision_rowsum: dict[int, torch.Tensor] = {}   # ~1.0 sanity
        if capture_prefill_attn:
            for li in l_mid_layers:
                a = out.attentions[li]
                if a is not None:
                    query_to_vision_attn[li] = a[0][:, q_rows][:, :, vis_idx].cpu()
                    query_to_vision_sink[li]   = a[0][:, q_rows, 0].cpu()
                    query_to_vision_rowsum[li] = a[0][:, q_rows, :].sum(-1).cpu()
        elif _sc_q2v:
            query_to_vision_attn = _sc_q2v
        visual_self_score = _self_scores or None

        # P1-2: tag which of the q_rows are question/choice tokens (Q_q), in
        # tail-relative coords (== the n_q saliency axis). None → caller averages
        # over ALL post-image text rows (old behavior, incl. chat-template tail).
        qq_role_spans = None
        if visual_boundary_pruning and role_targets:
            qq_role_spans = self._locate_role_spans(
                ids[q_rows], 0, int(q_rows.numel()), role_targets
            )
        elif capture_prefill_attn and role_targets:
            qq_role_spans = self._locate_role_spans(ids, last + 1, seq_prefill, role_targets)
        # --saliency_bg_correct: locate the fixed template rows B (Eq.reason-branch prior)
        bg_role_spans = None
        if want_attn and not dynamic_prune and bg_targets:
            bg_role_spans = self._locate_role_spans(ids, last + 1, seq_prefill, bg_targets)

        cursor = int(self.vlm.compute_3d_position_ids(
            input_ids=inputs["input_ids"], inputs_embeds=None,
            image_grid_thw=inputs.get("image_grid_thw"),
            mm_token_type_ids=inputs.get("mm_token_type_ids")).max().item()) + 1

        # ── optional PREFILL-stage SAFE prune (sink + background), pre-latent ─────
        # Cleans obvious junk (glass background + attention-sink/register tokens)
        # BEFORE the latent steps, so the reasoning + streaming operate on de-junked
        # vision. Distinct criterion (safe) from the streaming/pass-end mode
        # (--kv_prune), enabling a safe@prefill → topk@stream cascade. q2v is sliced
        # to survivors for the streaming controller, then nulled at return (grids no
        # longer match the reduced set) so saliency/one-shot prune no-op — same
        # contract as streaming eviction.
        prefill_pruned = False
        _pp = getattr(getattr(self, "_prune_args", None), "prune_prefill", "none")
        if (_pp == "safe" and want_attn and nV > 0 and query_to_vision_attn
                and inputs.get("image_grid_thw") is not None):
            import numpy as _np
            from memory.prune import (safe_prune_mask, universal_attention,
                                      token_tissue_fraction, apply_kv_prune)
            from memory.saliency import build_roi_spans
            _sms = int(getattr(self.model.config.vision_config, "spatial_merge_size", 2))
            _rspans = build_roi_spans(inputs.get("image_grid_thw"), _sms)
            _sink = universal_attention(query_to_vision_attn, l_mid_layers)   # [nV] high=sink
            _tissue = token_tissue_fraction(images, _rspans)                  # [nV]
            _rel = []                                                         # contrast Rel proxy
            for _li in l_mid_layers:
                _a = query_to_vision_attn.get(_li)
                if _a is None:
                    continue
                _a = _a.float().mean(0)                                       # [n_q, nV]
                _a = _a / _a.sum(-1, keepdim=True).clamp_min(1e-9)
                if _a.shape[0] >= 2:
                    _a = (_a - _a.mean(0, keepdim=True)).clamp_min(0.0)
                _rel.append(_a.mean(0))
            _sal = (torch.stack(_rel).mean(0).numpy() if _rel else _np.ones(nV))
            _keep = safe_prune_mask(_tissue, _sal, _sink,
                                    tissue_th=getattr(self._prune_args, "kv_prune_tissue_th", 0.1),
                                    sink_frac=getattr(self._prune_args, "kv_prune_sink_frac", 0.05))
            if _keep is not None and not _keep.all():
                past_kv, seq_prefill, prefill_spans, _newvis = apply_kv_prune(
                    past_kv, seq_prefill, vis_idx.tolist(), _keep, prefill_spans, self.device)
                vis_idx = torch.tensor(_newvis, device=self.device, dtype=torch.long)
                nV = int(vis_idx.numel())
                if vis_hidden is not None:
                    vis_hidden = vis_hidden[torch.as_tensor(_keep, dtype=torch.bool)]
                _kidx = torch.as_tensor(_np.where(_keep)[0], dtype=torch.long)
                query_to_vision_attn = {li: a[:, :, _kidx]
                                        for li, a in query_to_vision_attn.items()}
                prefill_pruned = True
                print(f"[PrunePrefill] safe: dropped {int((~_keep).sum())}/{len(_keep)} "
                      f"vis (sink+bg) → prefill KV {seq_prefill} tokens")

        # ── Pathology-Structured Visual Memory Consolidation (§3.3, env-gated) ──
        self._pcsi_capture_query(ids, last, prefill_input_embeddings)
        past_kv, seq_prefill, prefill_spans, _consol_keep, _consol_vis = (
            self._wsi_consolidate_boundary(
                past_kv, seq_prefill, vis_idx,
                prefill_input_embeddings[vis_idx.cpu()],
                query_to_vision_attn, inputs.get("image_grid_thw"),
                prefill_spans, qq_role_spans,
                visual_self_score=visual_self_score))
        if _consol_keep is not None and not bool(_consol_keep.all()):
            vis_idx = _consol_vis
            nV = int(vis_idx.numel())
            _consol_cpu = _consol_keep.cpu()
            if vis_hidden is not None:
                vis_hidden = vis_hidden[_consol_cpu]
            _consol_idx = _consol_cpu.nonzero(as_tuple=True)[0]
            query_to_vision_attn = {
                li: a[:, :, _consol_idx]
                for li, a in query_to_vision_attn.items()
            }
            prefill_pruned = True
        self._record_visual_meta(inputs.get("image_grid_thw"), _consol_keep, vis_idx)

        # ── AAVM: re-address the retained crops with their acquisition event ──
        # Runs AFTER C1 and BEFORE the latent steps, so the Reasoner is what
        # integrates the re-addressed evidence (the Answerer still never sees
        # an image). Off = untouched keys.
        if _aavm.enabled() and getattr(self, "_prune_morphology_enabled", False):
            _aavm_counts = list(
                getattr(self, "_reassembly_meta", (None,))[0]
                or self._aavm_token_counts(inputs.get("image_grid_thw")))
            _aavm.apply_aavm(
                self, past_kv,
                capture=_aavm_capture,
                keep_mask=_consol_keep,
                kept_counts=_aavm_counts,
                crop_queries=list(getattr(self, "_aavm_event_queries", ()) or []),
                cache_columns=vis_idx,
            )

        latent_trajectory: list[torch.Tensor] = []
        l2v_lists: dict[int, list[torch.Tensor]] = {li: [] for li in l_mid_layers}
        l2v_sink_lists:   dict[int, list[torch.Tensor]] = {li: [] for li in l_mid_layers}
        l2v_rowsum_lists: dict[int, list[torch.Tensor]] = {li: [] for li in l_mid_layers}

        # ── optional intra-pass streaming prune (H2O-style) / saliency dump ──────
        # Policy lives in the model-agnostic PruneController; this loop only feeds
        # it per-step latent→vision attention and applies the keep-mask. None ⇒
        # original one-shot path (Reasoner._maybe_prune_kv) with zero overhead.
        from memory.prune import build_prune_controller, apply_kv_prune
        sms = int(getattr(self.model.config.vision_config, "spatial_merge_size", 2))
        # Roles with attention disabled have no saliency to observe, so they do
        # not construct a controller. Navigator is protected by the engine.
        ctrl = (
            build_prune_controller(getattr(self, "_prune_args", None))
            if want_attn else None
        )
        one_shot_pruning = bool(
            ctrl is not None
            and getattr(getattr(self, "_prune_args", None), "prune_one_shot", False)
        )
        event_probes_only = bool(
            ctrl is not None
            and getattr(getattr(self, "_prune_args", None), "prune_event_probes_only", False)
        )
        lightweight_attention = bool(
            ctrl is not None
            and (
                getattr(getattr(self, "_prune_args", None), "prune_dynamic_ratio", False)
                or event_probes_only
                # ATP-SAP obtains its visual self/cross scores from the
                # prefill hooks above.  During latent steps it only needs the
                # selected Q→V rows, which _capture_selected_latent_attention
                # computes without requesting full attention tensors.  Keep
                # this path enabled so the surrounding model can stay on
                # SDPA instead of materialising a 12k×12k attention matrix.
                or getattr(getattr(self, "_prune_args", None), "prune_atp_sap", False)
            )
        )
        stable_streaming = bool(
            ctrl is not None
            and getattr(getattr(self, "_prune_args", None), "prune_stable_streaming", False)
        )
        stable_visual_only_scores = bool(
            ctrl is not None
            and getattr(
                getattr(self, "_prune_args", None),
                "prune_stable_visual_only_scores",
                False,
            )
        )
        if ctrl is not None:
            ctrl.configure_for_latent_steps(m)
            ctrl.start(grids=inputs.get("image_grid_thw"), vis_idx=vis_idx, sms=sms,
                       images=images, q2v=query_to_vision_attn, l_mid_layers=l_mid_layers,
                       role_spans=qq_role_spans, bg_role_spans=bg_role_spans,
                       image_magnifications=getattr(self, "_prune_image_magnifications", ()),
                       image_parent_indices=getattr(self, "_prune_image_parent_indices", ()),
                       morphology_enabled=getattr(self, "_prune_morphology_enabled", False))
        cur_vis   = vis_idx            # current vision column indices into the cache
        cur_len   = seq_prefill        # current cache length
        cur_spans = list(prefill_spans)
        evicted   = False
        # Per-step natural latent->vision mass. Reallocation records it in the
        # eager hook without returning full attention tensors; pruning falls back
        # to the controller's materialized attention because it needs saliency.
        probe_mass_steps: list[float] = []

        for step in range(m):
            le = self._apply_realign(last_hidden)
            from vision_text_mas.latent_hybrid_variants import latent_reallocation
            capture_context = (
                self._capture_selected_latent_attention(
                    layer_indices=l_mid_layers,
                    vision_columns=cur_vis,
                    visual_only_scores=stable_visual_only_scores,
                )
                if lightweight_attention and (
                    (stable_streaming and ctrl.should_observe(step))
                    or
                    (one_shot_pruning and step == 0)
                    or (not one_shot_pruning and (
                        not event_probes_only or ctrl.is_eviction_step(step)
                    ))
                )
                else contextlib.nullcontext({})
            )
            with capture_context as selected_attention:
                carried_reallocation_vis = getattr(
                    self, "_reallocation_carried_vision_columns", None
                )
                reallocation_vis = cur_vis
                if isinstance(carried_reallocation_vis, torch.Tensor) and carried_reallocation_vis.numel():
                    reallocation_vis = torch.unique(torch.cat((
                        carried_reallocation_vis.to(device=self.device, dtype=torch.long),
                        cur_vis,
                    )), sorted=True)
                reallocation_context = (
                    latent_reallocation(self, reallocation_vis)
                    if ctrl is None or ctrl.allow_reallocation(step)
                    else contextlib.nullcontext(None)
                )
                with reallocation_context as realloc_probe:
                    visual_bind_context = (
                        self._visual_bind_step(past_kv, cur_vis, cur_len)
                        if step == m - 1 else contextlib.nullcontext(False)
                    )
                    with visual_bind_context, self._visual_cut_ctx(cur_vis):
                        o = self.lm(
                            inputs_embeds=le,
                            position_ids=self._text_positions(1, cursor + step),
                            past_key_values=past_kv,
                            use_cache=True,
                            output_attentions=want_attn and not lightweight_attention,
                            output_hidden_states=True,
                        )
            realloc_mass = getattr(realloc_probe, "mean_mass", None)
            if realloc_mass is not None:
                probe_mass_steps.append(float(realloc_mass))
                reallocation_config = getattr(self, "_latent_reallocation", None)
                if (
                    reallocation_config is not None
                    and reallocation_config.adaptive_within_role
                ):
                    self._latent_reallocation_probe = float(realloc_mass)
            past_kv = o.past_key_values
            last_hidden = o.hidden_states[-1][:, -1:, :]
            if os.environ.get("VLMAS_RETR", ""):
                # retrieval kill test: per-layer residual input at the latest
                # latent step (overwritten each step -> the last step remains)
                self._retr_hidden = [h[0, -1].detach().float().clone() for h in o.hidden_states[:-1]]
            latent_trajectory.append(le.squeeze().cpu())
            cur_len += 1
            if cur_spans and cur_spans[-1][0] == "L":
                cur_spans[-1] = ("L", cur_spans[-1][1] + 1)
            else:
                cur_spans.append(("L", 1))

            # original aggregate collection (skipped once streaming eviction starts,
            # since per-step column counts would differ and can't be stacked)
            if want_attn and not lightweight_attention and (ctrl is None or not ctrl.evict):
                for li in l_mid_layers:
                    a = o.attentions[li]
                    if a is not None:
                        l2v_lists[li].append(a[0][:, 0, vis_idx].cpu())
                        l2v_sink_lists[li].append(a[0][:, 0, 0].cpu())
                        l2v_rowsum_lists[li].append(a[0][:, 0, :].sum(-1).cpu())

            # ── prune controller hook (observe + maybe evict) ───────────────────
            if ctrl is not None and (
                not event_probes_only or ctrl.is_eviction_step(step)
            ):
                # keep the head axis [H, nV] so the controller reduces it per
                # --stream_saliency_reduce (mean/max/contrast). Layer-mean here is
                # order-independent vs head-mean → default (mean) stays byte-identical.
                mats = (
                    [selected_attention[li] for li in l_mid_layers if li in selected_attention]
                    if lightweight_attention else [
                        o.attentions[li][0][:, 0, cur_vis].float()
                        for li in l_mid_layers if o.attentions[li] is not None
                    ]
                )
                if mats:
                    step_l2v = torch.stack(mats, 0).mean(0)          # [H, nV] (layer-mean)
                    if stable_streaming:
                        ctrl.observe_gpu(step, step_l2v)
                    elif getattr(ctrl, "reduce_head_first", False):
                        # head-reduce per layer, then layer-mean → mean_l(reduce_h)
                        # (matches roi_saliency Eq.(reason-attn) order). observe() gets
                        # a 1-D vector and passes it through unchanged.
                        evec = _np.stack(
                            [ctrl._reduce_heads(m.cpu().numpy()) for m in mats], 0
                        ).mean(0)
                        ctrl.observe(step, evec)
                    else:
                        ctrl.observe(step, step_l2v.cpu().numpy())   # byte-identical path
                    # vision mass this step = sum over vision, mean over heads (∈[0,1]);
                    # ALWAYS from step_l2v so the reallocation probe is flag-invariant.
                    if realloc_mass is None and not stable_streaming:
                        probe_mass_steps.append(float(step_l2v.sum(-1).mean()))
                if ctrl.evict:
                    keep = (
                        ctrl.maybe_keep_mask_gpu(step)
                        if stable_streaming
                        else ctrl.maybe_keep_mask(step)
                    )
                    if keep is not None and not keep.all():
                        keep_for_cache = (
                            keep.detach().cpu().numpy()
                            if stable_streaming
                            else keep
                        )
                        past_kv, cur_len, cur_spans, new_vis = apply_kv_prune(
                            past_kv, cur_len, cur_vis.tolist(), keep_for_cache, cur_spans, self.device,
                            in_place=True)
                        cur_vis = torch.tensor(new_vis, device=self.device, dtype=torch.long)
                        if stable_streaming:
                            ctrl.commit_gpu(keep)
                        else:
                            ctrl.commit(keep)
                        evicted = True
                        print(f"[PruneCtrl] step {step}: dropped {int((~keep).sum())}/"
                              f"{len(keep)} vis → KV {cur_len} tokens")

        latent_to_vision_attn = {li: torch.stack(v, 0) for li, v in l2v_lists.items() if v}
        latent_to_vision_sink = {li: torch.stack(v, 0) for li, v in l2v_sink_lists.items() if v}
        latent_to_vision_rowsum = {li: torch.stack(v, 0) for li, v in l2v_rowsum_lists.items() if v}
        full_attn_kv = self._extract_full_attn_kv(past_kv, l_mid_layers)
        transport_embeddings = _transport_embeddings(
            prefill_input_embeddings,
            latent_trajectory,
        )

        # streaming eviction already shrank the KV; the per-token q2v/l2v are now
        # stale (per-step columns don't align), so null them. --stream_recompute_
        # saliency rebuilds a SURVIVOR-space saliency from the controller's EMA score
        # so the Reasoner registry / ground_cos survive (else they no-op on empty q2v).
        stream_saliency_spans, stream_saliency_grids = [], {}
        if evicted or prefill_pruned:
            query_to_vision_attn = {}
            latent_to_vision_attn = {}
            if evicted and ctrl is not None and getattr(
                    getattr(self, "_prune_args", None), "stream_recompute_saliency", False):
                stream_saliency_spans, stream_saliency_grids = ctrl.final_saliency()

        past_kv, cur_len, _visual_relays = self._append_visual_relays(
            past_kv, cur_len, cursor + m, last_hidden, cur_vis,
            inputs.get("image_grid_thw"), spans=cur_spans)

        return {
            "past_key_values":        past_kv,
            "latent_trajectory":      latent_trajectory,
            "transport_embeddings":   transport_embeddings,
            "latent_to_vision_attn":  latent_to_vision_attn,
            "query_to_vision_attn":   query_to_vision_attn,
            "latent_to_vision_sink":   latent_to_vision_sink,
            "latent_to_vision_rowsum": latent_to_vision_rowsum,
            "query_to_vision_sink":    query_to_vision_sink,
            "query_to_vision_rowsum":  query_to_vision_rowsum,
            "qq_role_spans":          qq_role_spans,   # P1-2: Q_q token ranges (n_q coords)
            "tail_ids":               ids[last + 1:seq_prefill].detach().cpu(),  # post-image text rows (n_q axis) for B-subset probes
            "full_attn_kv":           full_attn_kv,
            "past_len":               cur_len,
            "pos_cursor":             cursor + m,
            "vis_coords":             None,
            "vis_cols":               cur_vis.cpu(),   # exact image-token columns
            "grids":                  inputs.get("image_grid_thw").cpu()
                                        if inputs.get("image_grid_thw") is not None else None,
            "n_vis":                  int(cur_vis.numel()),
            "n_vis_per_patch":        nV // max(1, len(images)),
            "spans":                  cur_spans,
            "vis_hidden":             vis_hidden,   # [n_vis, hidden] or None (build_memory)
            "pruned_streaming":       evicted,
            "saliency_steps":         ctrl.history if ctrl is not None else None,
            "stream_saliency_spans":  stream_saliency_spans,   # --stream_recompute_saliency
            "stream_saliency_grids":  stream_saliency_grids,
            "probe_rv":               (sum(probe_mass_steps) / len(probe_mass_steps)
                                       if probe_mass_steps else None),
            # per-step vision mass (streaming-path only). Additive: nothing reads it
            # yet, so all existing behavior is byte-identical. Used by phase0j to check
            # whether the r_v ramp survives eviction before wiring a per-step schedule.
            "probe_mass_steps":       (list(probe_mass_steps) if probe_mass_steps else None),
        }

    @torch.no_grad()
    def continue_with_latent_steps(
        self,
        text_embeds: torch.Tensor,   # [n_t, hidden]  new agent prompt
        past_kv,                     # past_key_values from previous agent
        past_len: int,               # current sequence length (token count) in past_kv
        m: int,
        l_mid_layers: list[int],
        past_pos_cursor: int | None = None,  # max MRoPE position+1 in past_kv
        capture_hidden_layers: list[int] | None = None,  # per-layer hidden capture
        vis_cols=None,               # absolute vision cols → capture q2v/l2v (rescore)
        realloc_alpha_schedule=None, # optional [m] per-step α (S8 verifier schedule)
    ) -> dict:
        """Append a new prompt to existing KV and run m more latent steps.

        Used by Refiner to continue reasoning from Reasoner's KV. The new text
        and latent tokens are positioned from the *rope cursor* (max position+1),
        not the token count, so they stay adjacent to the (compressed) visual
        positions instead of being pushed thousands of steps away.

        Cross-pass rescore (piggyback): when `vis_cols` is given AND l_mid_layers is
        non-empty, also capture query_to_vision_attn (the new prompt rows → those
        vision cols, in ONE shared context = all crops) and latent_to_vision_attn
        (latent steps → those cols). Lets the caller re-run roi_saliency over all
        crops on a single softmax denominator → comparable raw_score. No extra
        forward: attention is read off the passes that already run here.

        Returns the usual keys plus pos_cursor.
        """
        n_t = text_embeds.shape[0]
        use_mrope = self.mrope_pos and past_pos_cursor is not None
        cursor = past_pos_cursor if use_mrope else past_len

        _cap = vis_cols is not None and l_mid_layers and (
            (int(vis_cols.numel()) if hasattr(vis_cols, "numel") else len(vis_cols)) > 0)
        _vc = None
        if _cap:
            _vc = torch.as_tensor([int(c) for c in (vis_cols.tolist()
                    if hasattr(vis_cols, "tolist") else vis_cols)],
                    dtype=torch.long, device=self.device)
        q2v: dict[int, torch.Tensor] = {}
        l2v_lists: dict[int, list] = {li: [] for li in (l_mid_layers or [])} if _cap else {}

        # ── prefill new prompt on top of past_kv ─────────────────────
        if use_mrope:
            pos_ids = self._text_positions(n_t, start=cursor)
        else:
            pos_ids = self._make_position_ids(n_t, past_len=past_len)

        out = self.lm(
            inputs_embeds=text_embeds.unsqueeze(0),
            position_ids=pos_ids,
            past_key_values=past_kv,
            use_cache=True,
            output_attentions=_cap,
            output_hidden_states=True,
        )
        prefill_input_embeddings = out.hidden_states[0][0].detach().cpu()
        past_kv     = out.past_key_values
        last_hidden = out.hidden_states[-1][:, -1:, :]  # [1, 1, hidden]
        # q2v: prompt(question) rows → vision cols  [H, n_t, n_vis]  (one context)
        if _cap and out.attentions is not None:
            for li in l_mid_layers:
                a = out.attentions[li]
                if a is not None:
                    q2v[li] = a[0][:, :, _vc].cpu()
        past_len   += n_t
        cursor     += n_t

        # ── latent steps ────────────────────────────────────
        latent_trajectory: list[torch.Tensor] = []
        # per-step latent→vision mass (needs _cap; else stays empty → None). Same
        # reduce as grounded_prefill_and_latent's probe: layer-mean → sum vision →
        # mean heads. Used by phase0k to check the Verifier refinement r_v ramp.
        probe_mass_steps: list[float] = []

        # S8 per-step α schedule: mutate the live reallocation strength before each
        # latent forward, then restore α_base after the loop so any downstream gate/
        # verdict reads inside the SAME reallocating() block use the base α (not the
        # last step's value). None → no mutation → byte-identical to the constant path.
        _sched = realloc_alpha_schedule
        _alpha_base = None
        if _sched is not None:
            from memory.reallocate import get_active_alpha
            _alpha_base = get_active_alpha()

        for step in range(m):
            if _sched is not None and step < len(_sched):
                from memory.reallocate import set_active_alpha
                set_active_alpha(_sched[step])
            latent_embed = self._apply_realign(last_hidden)  # [1, 1, hidden]
            if use_mrope:
                pos_ids_step = self._text_positions(1, start=cursor + step)
            else:
                pos_ids_step = self._make_position_ids(1, past_len=past_len + step)

            out = self.lm(
                inputs_embeds=latent_embed,
                position_ids=pos_ids_step,
                past_key_values=past_kv,
                use_cache=True,
                output_attentions=_cap,
                output_hidden_states=True,
            )
            past_kv     = out.past_key_values
            last_hidden = out.hidden_states[-1][:, -1:, :]
            latent_trajectory.append(latent_embed.squeeze().cpu())
            if _cap and out.attentions is not None:  # l2v: latent → vision  [H, n_vis]/step
                _step_mats = []
                for li in l_mid_layers:
                    a = out.attentions[li]
                    if a is not None:
                        _col = a[0][:, 0, _vc].cpu()   # [H, n_vis]
                        l2v_lists[li].append(_col)
                        _step_mats.append(_col.float())
                if _step_mats:
                    _step_l2v = torch.stack(_step_mats, 0).mean(0)   # [H, nV] layer-mean
                    probe_mass_steps.append(float(_step_l2v.sum(-1).mean()))

        # restore α_base so gate/verdict reads in the enclosing reallocating() block
        # are unaffected by the schedule (only the latent steps were modulated).
        if _sched is not None and _alpha_base is not None:
            from memory.reallocate import set_active_alpha
            set_active_alpha(_alpha_base)

        latent_to_vision_attn = {li: torch.stack(v, 0) for li, v in l2v_lists.items() if v}
        full_attn_kv = self._extract_full_attn_kv(past_kv, l_mid_layers)
        transport_embeddings = _transport_embeddings(
            prefill_input_embeddings,
            latent_trajectory,
        )

        # ── optional per-layer hidden capture (saliency-layer / decode-debug) ──
        layer_hidden: dict[int, torch.Tensor] = {}
        if capture_hidden_layers:
            for li in capture_hidden_layers:
                if 0 <= li < len(out.hidden_states):
                    layer_hidden[li] = out.hidden_states[li][:, -1:, :].squeeze().cpu()

        return {
            "past_key_values":    past_kv,
            "latent_trajectory":  latent_trajectory,
            "transport_embeddings": transport_embeddings,
            "full_attn_kv":       full_attn_kv,
            "past_len":           past_len + m,
            "pos_cursor":         cursor + m,
            # span types appended by this continuation: prompt | latent
            "spans":              [s for s in (("P", n_t), ("L", m)) if s[1] > 0],
            "layer_hidden":       layer_hidden,
            # cross-pass rescore (piggyback): empty unless vis_cols was given
            "query_to_vision_attn":  q2v,
            "latent_to_vision_attn": latent_to_vision_attn,
            # per-step vision mass (only when vis_cols captured; additive/byte-identical). phase0k.
            "probe_mass_steps":      (list(probe_mass_steps) if probe_mass_steps else None),
        }

    def _extract_full_attn_kv(
        self,
        past_kv,
        l_mid_layers: list[int],
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        """Extract (K, V) tensors for full_attention layers (Refiner injection).

        Qwen3-VL uses DynamicCache with .layers[i].keys / .layers[i].values API.
        """
        result: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for li in l_mid_layers:
            try:
                if hasattr(past_kv, "layers"):
                    # Qwen3-VL DynamicCache
                    layer = past_kv.layers[li]
                    K = layer.keys.cpu()
                    V = layer.values.cpu()
                elif hasattr(past_kv, "key_cache"):
                    K = past_kv.key_cache[li].cpu()
                    V = past_kv.value_cache[li].cpu()
                else:
                    K = past_kv[li][0].cpu()
                    V = past_kv[li][1].cpu()
                result[li] = (K, V)
            except (IndexError, TypeError, AttributeError):
                continue
        return result

    @torch.no_grad()
    @contextlib.contextmanager
    def _capture_prefill_q2v(self, layer_indices, query_rows, vision_columns_abs):
        """Memory-safe question->vision attention at PREFILL (s_c for §3.3).

        Same hook mechanics as _capture_selected_latent_attention, but the
        query slice is the question rows (chunk-relative) instead of the last
        token, and the softmax is taken OVER THE VISUAL COLUMNS ONLY — exactly
        the normalized relevance the task-weighted coverage needs, without
        materializing any full attention matrix (per layer: [H, nq, nV]).
        Yields {layer: tensor [H, nq, nV] (cpu)}.
        """
        captured: dict[int, torch.Tensor] = {}
        handles: list[torch.utils.hooks.RemovableHandle] = []
        columns = vision_columns_abs.detach().to(device=self.device, dtype=torch.long)
        rows = query_rows.detach().to(device=self.device, dtype=torch.long)

        def make_hook(layer_index: int):
            def hook(module, args, kwargs, _output):
                hidden = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
                position_embeddings = (
                    kwargs["position_embeddings"]
                    if "position_embeddings" in kwargs else args[1]
                )
                cache = (
                    kwargs["past_key_values"]
                    if "past_key_values" in kwargs
                    else (args[3] if len(args) > 3 else None)
                )
                if cache is None or int(hidden.shape[1]) <= int(rows.max()):
                    return
                key_states = cache.layers[layer_index].keys
                valid = columns[columns < key_states.shape[2]]
                if valid.numel() == 0:
                    return
                query_shape = (*hidden.shape[:-1], -1, module.head_dim)
                query = module.q_norm(
                    module.q_proj(hidden).view(query_shape)).transpose(1, 2)
                cos, sin = position_embeddings
                query, _ = apply_rotary_pos_emb(
                    query, torch.zeros_like(query), cos, sin)
                query = query[:, :, rows, :]                    # [1,H,nq,D]
                keys = key_states.index_select(2, valid)        # vis cols only
                keys = repeat_kv(keys, module.num_key_value_groups)
                scores = torch.matmul(query, keys.transpose(-2, -1)) * module.scaling
                captured[layer_index] = torch.softmax(
                    scores, dim=-1)[0].float().cpu()            # [H,nq,nV]

            return hook

        for index in layer_indices:
            handles.append(
                self.lm.layers[index].self_attn.register_forward_hook(
                    make_hook(index), with_kwargs=True))
        try:
            yield captured
        finally:
            for handle in handles:
                handle.remove()

    @torch.no_grad()
    @contextlib.contextmanager
    def _capture_prefill_visual_self_score(
        self, layer_indices, visual_query_rows, vision_columns_abs, past_len=0
    ):
        """Stream ATP's visual-query→visual-key attention-logit importance."""
        captured: dict[int, torch.Tensor] = {}
        handles: list[torch.utils.hooks.RemovableHandle] = []
        columns = vision_columns_abs.detach().to(device=self.device, dtype=torch.long)
        rows = visual_query_rows.detach().to(device=self.device, dtype=torch.long)

        def make_hook(layer_index: int):
            def hook(module, args, kwargs, _output):
                hidden = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
                position_embeddings = (
                    kwargs["position_embeddings"]
                    if "position_embeddings" in kwargs else args[1]
                )
                cache = (
                    kwargs["past_key_values"]
                    if "past_key_values" in kwargs
                    else (args[3] if len(args) > 3 else None)
                )
                if cache is None or rows.numel() == 0:
                    return
                key_states = cache.layers[layer_index].keys
                valid = columns[columns < key_states.shape[2]]
                if valid.numel() == 0:
                    return
                query_shape = (*hidden.shape[:-1], -1, module.head_dim)
                query = module.q_norm(
                    module.q_proj(hidden).view(query_shape)
                ).transpose(1, 2)
                cos, sin = position_embeddings
                query, _ = apply_rotary_pos_emb(
                    query, torch.zeros_like(query), cos, sin
                )
                keys = repeat_kv(
                    key_states.index_select(2, valid),
                    module.num_key_value_groups,
                )
                total = torch.zeros(valid.numel(), device=self.device, dtype=torch.float32)
                counts = torch.zeros_like(total)
                absolute_rows = rows + int(past_len)
                for start in range(0, rows.numel(), 128):
                    chunk_rows = rows[start:start + 128]
                    chunk_abs = absolute_rows[start:start + 128]
                    queries = query[:, :, chunk_rows, :]
                    logits = torch.matmul(
                        queries, keys.transpose(-2, -1)
                    ) * module.scaling
                    causal = valid.unsqueeze(0) <= chunk_abs.unsqueeze(1)
                    logits = logits.masked_fill(~causal[None, None, :, :], 0.0)
                    total += logits.float().sum(dim=(0, 1, 2))
                    counts += causal.sum(dim=0).float() * logits.shape[1]
                captured[layer_index] = (total / counts.clamp_min(1.0)).cpu()

            return hook

        for index in layer_indices:
            handles.append(
                self.lm.layers[index].self_attn.register_forward_hook(
                    make_hook(index), with_kwargs=True
                )
            )
        try:
            yield captured
        finally:
            for handle in handles:
                handle.remove()

    def _aavm_token_counts(self, grids) -> list[int]:
        """Per-crop visual token counts when C1 is off (no reassembly meta)."""
        if grids is None:
            return []
        merge = int(getattr(
            self.model.config.vision_config, "spatial_merge_size", 2))
        return [
            (int(h) // merge) * (int(w) // merge) for _t, h, w in grids.tolist()
        ]

    def _pcsi_capture_query(self, ids, last, prefill_input_embeddings):
        """PCSI (env VLMAS_WSI_CONSOL_MODE=pcsi, Reasoner stage only; else no-op).

        Question set Q = the "Question stem:" tokens of this chunk's post-image
        text, read from the SAME embedding-layer output as the visual states
        handed to _wsi_consolidate_boundary (shared LLM-input space). Stores
        (embeddings [nq, d] cpu, decoded question text) in self._pcsi_query.
        """
        self._pcsi_query = None
        if (os.environ.get("VLMAS_WSI_CONSOL", "") != "1"
                or os.environ.get("VLMAS_WSI_CONSOL_MODE", "") not in ("pcsi", "qpris")
                or not getattr(self, "_prune_morphology_enabled", False)):
            return
        from memory.pcsi_select import locate_question_rows
        tok = self.processor.tokenizer
        tail = ids[last + 1:].tolist()
        rows = locate_question_rows([tok.decode([t]) for t in tail])
        n_emb = int(prefill_input_embeddings.shape[0])
        rows = [r for r in rows if last + 1 + r < n_emb]
        if rows:
            index = torch.tensor([last + 1 + r for r in rows], dtype=torch.long)
            self._pcsi_query = (
                prefill_input_embeddings[index].detach().float().cpu(),
                tok.decode([tail[r] for r in rows]))

    def _record_visual_meta(self, grids, keep, abs_cols):
        """Provenance of the retained visual-KV columns, in cache column order (bookkeeping only).

        Written at the C1 boundary of the Reasoner stage whether or not C1 is on, so a
        receiver-side consumer (memory/psar.py) can map every retained column j to its
        physical support g(j) without re-deriving it. keep = the C1 keep mask over the
        pre-C1 visual order (None when C1 did not run), so selected indices are
        keep.nonzero(); apply_kv_prune keeps the surviving columns in ascending order, so the
        k-th retained column is the k-th True of keep. Sets self._visual_meta (dict) or None
        with self._visual_meta_note giving the reason. Never changes any tensor the model uses.
        """
        self._visual_meta = None
        self._visual_meta_note = ""
        if not getattr(self, "_prune_morphology_enabled", False):
            self._visual_meta_note = "not reasoner stage"
            return
        if grids is None or not isinstance(abs_cols, torch.Tensor):
            self._visual_meta_note = "no grids / columns"
            return
        try:
            sms = int(getattr(self.model.config.vision_config, "spatial_merge_size", 2))
            grid_hws = tuple((int(h) // sms, int(w) // sms) for _t, h, w in grids.tolist())
            counts = [gh * gw for gh, gw in grid_hws]
            n_obs = len(counts)
            n_pre = int(keep.numel()) if isinstance(keep, torch.Tensor) else int(abs_cols.numel())
            if sum(counts) != n_pre:
                self._visual_meta_note = f"token/grid mismatch ({n_pre} vs {sum(counts)})"
                return
            obs = torch.repeat_interleave(torch.arange(n_obs), torch.tensor(counts))
            tok = torch.cat([torch.arange(c) for c in counts]) if counts else torch.empty(0, dtype=torch.long)
            sel = (keep.detach().cpu().nonzero(as_tuple=True)[0] if isinstance(keep, torch.Tensor)
                   else torch.arange(n_pre))
            if int(sel.numel()) != int(abs_cols.numel()):
                self._visual_meta_note = f"kept {int(sel.numel())} != columns {int(abs_cols.numel())}"
                return
            mags = tuple(int(v) for v in getattr(self, "_prune_image_magnifications", ()) or ())
            mags = mags if len(mags) == n_obs else (0,) * n_obs
            parents = tuple(int(v) for v in getattr(self, "_prune_image_parent_indices", ()) or ())
            parents = parents if len(parents) == n_obs else (-1,) * n_obs
            boxes = tuple(getattr(self, "_prune_image_boxes", ()) or ())
            boxes = boxes if len(boxes) == n_obs else (None,) * n_obs
            pids = tuple(getattr(self, "_prune_image_patch_ids", ()) or ())
            pids = pids if len(pids) == n_obs else ("",) * n_obs
            xy = None
            if n_obs and all(b is not None for b in boxes):
                from memory.csis_select import token_coordinates
                xy = token_coordinates(counts, grid_hws, boxes)[sel].float()
            o_sel = obs[sel]
            self._visual_meta = {
                "abs_cols": abs_cols.detach().cpu().clone(),   # j (retained visual-KV column)
                "orig_index": sel,                              # selected_indices (pre-C1 visual order)
                "obs_id": o_sel,                                # observation / crop index
                "tok_in_crop": tok[sel],                        # merged-grid cell inside the crop
                "scale": torch.tensor(mags, dtype=torch.long)[o_sel] if n_obs else o_sel,
                "xy": xy,                                       # slide-space token centre or None
                "patch_id": pids, "parents": parents, "boxes": boxes,
                "magnifications": mags, "token_counts": tuple(counts), "grid_hws": grid_hws,
                "kept_counts": tuple(int((o_sel == o).sum()) for o in range(n_obs)),
                "c1": isinstance(keep, torch.Tensor),
            }
        except Exception as exc:  # bookkeeping must never break a run
            self._visual_meta = None
            self._visual_meta_note = f"error: {exc!r}"

    def _wsi_consolidate_boundary(self, past_kv, seq_len, vis_abs,
                                  vis_embeddings, q2v, grids, spans,
                                  question_spans=None, visual_self_score=None,
                                  spatial_token_indices: tuple[torch.Tensor, ...] | None = None):
        """Pathology-Structured Visual Memory Consolidation (Method §3.3;
        env-gated VLMAS_WSI_CONSOL=1, Reasoner boundary only; off = no-op).

        Task-weighted facility-location selection over the WSI-valid graph
        (memory/wsi_memory.py) followed by ONE physical apply_kv_prune of the
        non-selected ORIGINAL visual columns. z_c = the chunk's embedding-layer
        output at the visual positions (frozen merger output); s_c = mean
        question->vision attention from the SAME prefill (uniform fallback).
        Budget B = ceil(nV * VLMAS_WSI_CONSOL_KEEP, default 0.25).

        Returns (past_kv, seq_len, spans, keep_bool | None, new_vis_abs | None);
        None means untouched.
        """
        untouched = (past_kv, seq_len, spans, None, None)
        prune_args = getattr(self, "_prune_args", None)
        sparsevlm_active = bool(
            getattr(prune_args, "prune_query_adaptive", False)
        )
        atp_sap_active = bool(getattr(prune_args, "prune_atp_sap", False))
        if (
            os.environ.get("VLMAS_WSI_CONSOL", "") != "1"
            and not sparsevlm_active
            and not atp_sap_active
        ):
            return untouched
        if not getattr(self, "_prune_morphology_enabled", False):
            return untouched  # reasoner stage only
        nV = int(vis_abs.numel()) if isinstance(vis_abs, torch.Tensor) else 0
        if nV == 0 or grids is None:
            return untouched
        import math as _math

        from memory.prune import apply_kv_prune as _consol_prune
        from memory.wsi_memory import consolidation_keep_mask

        if atp_sap_active:
            if visual_self_score is None or not q2v:
                print("[ATP-SAP] missing visual self/cross scores; skipped", flush=True)
                return untouched
            if not question_spans:
                print("[ATP-SAP] question span unmatched; using text raters", flush=True)
            sms = int(getattr(self.model.config.vision_config, "spatial_merge_size", 2))
            grid_shapes = tuple(
                shape
                for temporal, height, width in grids.tolist()
                for shape in [(int(height) // sms, int(width) // sms)] * int(temporal)
            )
            from memory.atp_sap import atp_sap_keep_mask

            keep, atp_info = atp_sap_keep_mask(
                visual_self_score,
                q2v,
                question_spans=question_spans,
                grid_shapes=grid_shapes,
                spatial_token_indices=spatial_token_indices,
            )
            keep = keep.to(self.device)
            token_counts = tuple(
                int(indices.numel())
                for indices in spatial_token_indices
            ) if spatial_token_indices is not None else tuple(
                height * width for height, width in grid_shapes
            )
            morphology_swaps = 0
            cross_scale_swaps = 0
            if (
                getattr(prune_args, "prune_morphology_coverage_rescue", False)
                or getattr(prune_args, "prune_cross_scale_consolidation", False)
            ):
                from memory.consolidate import (
                    cross_scale_consolidate,
                    morphology_coverage_rescue,
                )

                features = vis_embeddings.to(device=keep.device)
                if getattr(prune_args, "prune_morphology_coverage_rescue", False):
                    keep, morphology_swaps, _ = morphology_coverage_rescue(
                        features, keep, token_counts
                    )
                if getattr(prune_args, "prune_cross_scale_consolidation", False):
                    parent_indices = tuple(
                        int(value)
                        for value in getattr(self, "_prune_image_parent_indices", ())
                        or ()
                    )
                    if len(parent_indices) == len(token_counts):
                        keep, cross_scale_swaps = cross_scale_consolidate(
                            features,
                            keep,
                            token_counts,
                            parent_indices,
                            margin=float(
                                getattr(prune_args, "prune_cross_scale_margin", 0.05)
                            ),
                        )
            print(
                f"[ATP-SAP] kept={atp_info.retained_total}/{nV} "
                f"redundancy={atp_info.retained_by_redundancy} "
                f"spatial={atp_info.retained_by_spatial_scaffold} "
                f"theta={atp_info.redundant_threshold:.4f} "
                f"layer_K={list(atp_info.retained_by_layer)} "
                f"morph_swaps={morphology_swaps} cross_scale_swaps={cross_scale_swaps}",
                flush=True,
            )
            if bool(keep.all()):
                return past_kv, seq_len, spans, keep, vis_abs
            past_kv, seq_len, spans, new_vis = _consol_prune(
                past_kv, seq_len, vis_abs.tolist(),
                keep.detach().cpu().numpy(), spans, self.device,
            )
            return (
                past_kv,
                seq_len,
                spans,
                keep,
                torch.tensor(new_vis, device=self.device, dtype=torch.long),
            )

        if sparsevlm_active:
            if not q2v:
                print("[SparseVLM] no question-to-vision attention; skipped", flush=True)
                return untouched
            if not question_spans:
                print(
                    "[SparseVLM] question span unmatched; using visual-relevant "
                    "available text raters",
                    flush=True,
                )
            from memory.sparsevlm import (
                SparseVLMPathologyLayout,
                sparsevlm_keep_mask,
                sparsevlm_pathology_keep_mask,
            )

            keep, sparse_info = sparsevlm_keep_mask(
                q2v,
                question_spans=question_spans,
                lambda_scale=float(
                    getattr(prune_args, "prune_query_adaptive_lambda", 0.5)
                ),
            )
            pathology_info = None
            if getattr(prune_args, "prune_query_pathology_context", False):
                sms = int(getattr(self.model.config.vision_config, "spatial_merge_size", 2))
                grid_shapes = tuple(
                    shape
                    for temporal, height, width in grids.tolist()
                    for shape in [(int(height) // sms, int(width) // sms)] * int(temporal)
                )
                boxes = tuple(getattr(self, "_prune_image_boxes", ()) or ())
                magnifications = tuple(
                    int(value)
                    for value in getattr(self, "_prune_image_magnifications", ()) or ()
                )
                parent_indices = tuple(
                    int(value)
                    for value in getattr(self, "_prune_image_parent_indices", ()) or ()
                )
                if len(grid_shapes) == len(boxes) == len(magnifications) == len(parent_indices):
                    keep, pathology_info = sparsevlm_pathology_keep_mask(
                        keep,
                        sparse_info.visual_scores,
                        SparseVLMPathologyLayout(
                            grid_shapes=grid_shapes,
                            boxes=boxes,
                            magnifications=magnifications,
                            parent_indices=parent_indices,
                        ),
                    )
                else:
                    print(
                        "[SparseVLM-Pathology] crop metadata/grid mismatch; "
                        "using SparseVLM mask only",
                        flush=True,
                    )
            keep = keep.to(self.device)
            if pathology_info is None:
                print(
                    f"[SparseVLM] kept {int(keep.sum())}/{nV} "
                    f"raters={sparse_info.rater_count} rank={sparse_info.rank} "
                    f"dropped={sparse_info.drop_count}",
                    flush=True,
                )
            else:
                print(
                    f"[SparseVLM-Pathology] kept={int(keep.sum())}/{nV} "
                    f"base={pathology_info.base_keep_count} rank={sparse_info.rank} "
                    f"spatial={pathology_info.spatial_anchor_count} "
                    f"neighbors={pathology_info.neighbor_count} "
                    f"parent_scale={pathology_info.parent_anchor_count} "
                    f"context_added={pathology_info.context_added_count} "
                    f"base_replaced={pathology_info.base_tokens_replaced_count}",
                    flush=True,
                )
            if bool(keep.all()):
                return past_kv, seq_len, spans, keep, vis_abs
            past_kv, seq_len, spans, new_vis = _consol_prune(
                past_kv, seq_len, vis_abs.tolist(),
                keep.detach().cpu().numpy(), spans, self.device,
            )
            return (
                past_kv,
                seq_len,
                spans,
                keep,
                torch.tensor(new_vis, device=self.device, dtype=torch.long),
            )

        sms = int(getattr(self.model.config.vision_config, "spatial_merge_size", 2))
        grid_hws = tuple(
            (int(h) // sms, int(w) // sms) for _t, h, w in grids.tolist()
        )
        token_counts = tuple(gh * gw for gh, gw in grid_hws)
        if sum(token_counts) != nV:
            print(f"[WSIConsol] grid/token mismatch ({sum(token_counts)} vs {nV}); skipped",
                  flush=True)
            return untouched
        relevance = None
        if q2v:
            relevance = torch.stack(
                [a.float().mean(dim=(0, 1)) for a in q2v.values()]
            ).mean(0).to(self.device)
            if int(relevance.numel()) != nV:
                relevance = None
        magnifications = tuple(
            int(v) for v in getattr(self, "_prune_image_magnifications", ())
        )
        boxes = tuple(getattr(self, "_prune_image_boxes", ()) or ())
        if len(magnifications) != len(token_counts):
            magnifications = (0,) * len(token_counts)
        if len(boxes) != len(token_counts):
            boxes = (None,) * len(token_counts)
        control = os.environ.get("VLMAS_WSI_CONSOL_CONTROL", "")
        if control == "same_scale":
            # causal ablation: drop all cross-scale (ancestor) edges — a
            # box-less image joins no ancestor relation in token_footprints.
            boxes = (None,) * len(token_counts)
        elif control == "shuffled_parent":
            from memory.wsi_memory import apply_parent_shuffle
            boxes = apply_parent_shuffle(boxes, magnifications)
        ratio = float(os.environ.get("VLMAS_WSI_CONSOL_KEEP", "0.25"))
        budget = max(1, _math.ceil(nV * ratio))
        if os.environ.get("VLMAS_WSI_CONSOL_BUDGET", "").strip():
            # WSI-aware spec (0910 v2): case-level absolute budget B (e.g. 64/96/128)
            budget = max(1, int(os.environ["VLMAS_WSI_CONSOL_BUDGET"]))
            ratio = round(budget / max(nV, 1), 4)
        kappa_shift = os.environ.get("VLMAS_WSI_CONSOL_KAPPA", "shifted") != "cos"
        if os.environ.get("VLMAS_WSI_GEOM_AUDIT", "") == "1":
            # Cen-Prune-style geometry assumption check (observe-only).
            from memory.anchor_residual import geometry_audit
            aud = geometry_audit(
                vis_embeddings.to(self.device).float(),
                token_counts, magnifications, boxes)
            print("[GeomAudit] " + " ".join(
                f"{k}={v:.4f}" for k, v in sorted(aud.items())), flush=True)
        mode = os.environ.get("VLMAS_WSI_CONSOL_MODE", "coverage")
        if mode in ("topo", "topo_shuf"):
            # C1 upgrade (0908): Navigation-Topology Preservation — mandatory
            # {crop representative ∪ zoom-edge anchor tokens} then global
            # max-gap filling. topo_shuf = deranged-edge falsification arm.
            from memory.stratified import topology_keep_mask
            keep = topology_keep_mask(
                token_counts, grid_hws, boxes,
                tuple(getattr(self, "_prune_image_parent_indices", ()) or ()),
                budget, shuffle_edges=(mode == "topo_shuf")).to(self.device)
        elif mode in ("scale", "scale_shuf"):
            # Scale-Structured Visual KV Compression (0910 spec v3): B_C =
            # floor(alpha*B) coarse (bridge tokens + childless one-centers,
            # then GLOBAL farthest-point coverage in the common WSI frame) and
            # B_F = B - B_C fine (b_c >= 1 ∝ N_c; one-center + FPS per child).
            # scale_shuf = deranged parents (relative center transplanted).
            from memory.stratified import scale_structured_keep_mask
            keep, sc_info = scale_structured_keep_mask(
                token_counts, grid_hws, boxes,
                tuple(getattr(self, "_prune_image_parent_indices", ()) or ()),
                magnifications, budget,
                alpha=float(os.environ.get("VLMAS_WSI_CONSOL_ALPHA", "0.5")),
                shuffle_parents=(mode == "scale_shuf"), return_info=True)
            keep = keep.to(self.device)
            print(
                f"[WSIConsol] scale B={budget} B_C={sc_info['B_C']} "
                f"B_F={sc_info['B_F']} coarse={sc_info['coarse']} "
                f"fine={sc_info['fine']} bridges={sc_info['bridges']} "
                f"mandatory_C={sc_info['mandatory_C']} "
                f"infeasible={sc_info['infeasible']} "
                f"wsi_coords={sc_info['wsi_coords']} parents={sc_info['parents']} "
                f"per_crop={sc_info['per_crop']}", flush=True)
        elif mode in ("wsi", "wsi_flat", "wsi_shuf", "scale_v2", "scale_v2_flat", "scale_v2_shuf"):
            # WSI-Aware Visual KV Compression (0910 spec v2): seeds = center
            # token per observation U bridge token (parent token nearest the
            # child's footprint center) per real zoom edge; remaining slots by
            # greedy covering-radius (largest R_p observation, farthest token).
            # Requires B >= B0 (seed set never cut). wsi_flat = no bridge
            # constraint; wsi_shuf = deranged parents (relative pos transplanted).
            from memory.stratified import wsi_aware_keep_mask
            keep, wsi_info = wsi_aware_keep_mask(
                token_counts, grid_hws, boxes,
                tuple(getattr(self, "_prune_image_parent_indices", ()) or ()),
                budget,
                constraint={"wsi": "hier", "wsi_flat": "flat", "wsi_shuf": "shuffled",
                            "scale_v2": "hier", "scale_v2_flat": "flat",
                            "scale_v2_shuf": "shuffled"}[mode],
                return_info=True,
                # scale_v2 (0911 spec): identical rule to wsi but the coverage
                # radius uses raw ||u_j - u_k|| (no per-grid diam scaling)
                normalize_diam=not mode.startswith("scale_v2"))
            keep = keep.to(self.device)
            print(
                f"[WSIConsol] wsi constraint={wsi_info['constraint']} "
                f"B={budget} B0={wsi_info['B0']} edges={wsi_info['edges']} "
                f"infeasible={wsi_info['infeasible']} "
                f"parents={wsi_info['parents']} per_crop={wsi_info['per_crop']}",
                flush=True)
        elif mode in ("scale_v3", "scale_v3_flat", "scale_v3_shuf"):
            # Scale-Structured Visual KV Compression v3 (0911 spec):
            # precedence-constrained context coverage. One submodular
            # per-observation coverage utility F(S) over the horizontal
            # adjacency A (edge-sharing token footprints, VLMAS_WSI_CONSOL_ADJ=4|8),
            # vertical 5x->20x provenance as a precedence constraint
            # (child kept => its bridge parent token kept), greedy by marginal
            # gain per closure cost. _flat = no precedence; _shuf = deranged
            # parents (child center transplanted). Saturation (F maxed before B)
            # is filled farthest-point and logged.
            from memory.stratified import precedence_coverage_keep_mask
            keep, pc_info = precedence_coverage_keep_mask(
                token_counts, grid_hws, boxes,
                tuple(getattr(self, "_prune_image_parent_indices", ()) or ()),
                budget,
                constraint={"scale_v3": "hier", "scale_v3_flat": "flat",
                            "scale_v3_shuf": "shuffled"}[mode],
                adjacency=int(os.environ.get("VLMAS_WSI_CONSOL_ADJ", "4")),
                return_info=True)
            keep = keep.to(self.device)
            print(
                f"[WSIConsol] scale_v3 constraint={pc_info['constraint']} "
                f"adj={pc_info['adjacency']} B={budget} edges={pc_info['edges']} "
                f"closure_adds={pc_info['closure_adds']} "
                f"saturated_at={pc_info['saturated_at']} F={pc_info['F']:.3f} "
                f"parents={pc_info['parents']} per_crop={pc_info['per_crop']}",
                flush=True)
        elif mode in ("hier", "hier_shuf"):
            # Hierarchy-Constrained Spatial Pruning (0910 spec): mandatory
            # {>=m tokens per crop  U  parent token pi(c) per real 5x->20x
            # edge}, remainder by the strat rule (b_p ∝ |V_p|, equal-area
            # thinning). No scores. hier_shuf = Shuffled-Hierarchy control
            # (children re-attached to a different anchor; crops, tokens and
            # budget unchanged).
            from memory.stratified import hierarchy_keep_mask
            keep, hier_info = hierarchy_keep_mask(
                token_counts, grid_hws, boxes,
                tuple(getattr(self, "_prune_image_parent_indices", ()) or ()),
                budget,
                min_per_crop=int(os.environ.get("VLMAS_WSI_CONSOL_HIER_MIN", "1")),
                shuffle_parents=(mode == "hier_shuf"), return_info=True)
            keep = keep.to(self.device)
            print(
                f"[WSIConsol] hier mandatory={hier_info['mandatory']} "
                f"edges={hier_info['edges']} parents={hier_info['parents']} "
                f"per_crop={hier_info['per_crop']}", flush=True)
        elif mode == "dcs":
            # Directed Cross-Scale Submodular Visual KV Selection (C1 candidate,
            # Notion 0912): F(S) = sum_i s_i max_{j in S} a_ij, |S| <= B, greedy.
            # s = cross-observation debiased q2v salience (needs
            # VLMAS_WSI_CONSOL_SC=1; VLMAS_DCS_SAL=raw|uniform), a_ij = k_sem
            # (same scale) or k_sem * |Om_i & Om_j| / |Om_i| (cross scale;
            # VLMAS_DCS_AFF=symmetric -> IoU; VLMAS_DCS_Z=none -> spatial-only).
            # No hard 5x->20x precedence. Wrong-parent falsifier =
            # VLMAS_WSI_CONSOL_CONTROL=shuffled_parent (boxes remapped above).
            from memory.dcs_select import dcs_keep_mask
            keep, dcs_info = dcs_keep_mask(
                vis_embeddings.to(self.device).float(), relevance,
                token_counts, grid_hws, magnifications, boxes, budget,
                aff=os.environ.get("VLMAS_DCS_AFF", "directed"),
                z_mode=os.environ.get("VLMAS_DCS_Z", "merger"),
                sal=os.environ.get("VLMAS_DCS_SAL", "debiased"),
                return_info=True)
            _dcs_dump = os.environ.get("VLMAS_DCS_DUMP", "").strip()
            if _dcs_dump:
                os.makedirs(_dcs_dump, exist_ok=True)
                _dn = getattr(self, "_dcs_dump_n", 0)
                self._dcs_dump_n = _dn + 1
                torch.save({
                    "z": vis_embeddings.detach().to(torch.float16).cpu(),
                    "u": (relevance.detach().float().cpu()
                          if relevance is not None else None),
                    "token_counts": token_counts, "grid_hws": grid_hws,
                    "magnifications": magnifications, "boxes": boxes,
                    "parents": tuple(getattr(self, "_prune_image_parent_indices", ()) or ()),
                    "q2v_layers": sorted(q2v.keys()) if q2v else [],
                }, os.path.join(_dcs_dump, f"dcs_{_dn:04d}.pt"))
            keep = keep.to(self.device)
            print("[WSIConsol] dcs " + " ".join(
                f"{k}={v}" for k, v in dcs_info.items())
                + f" control={control or 'none'}"
                + f" q2v_layers={sorted(q2v.keys()) if q2v else []}", flush=True)
        elif mode == "pcsi":
            # Provenance-Conditioned Submodular Information Selection (C1 candidate,
            # Notion PCSI section 10): F(S) = sum_o I_f(S & V_o; Q | P_o), facility-
            # location CMI, P_o = V_parent(o) from the acquisition path
            # (_prune_image_parent_indices), s = [cos]_+ in the LLM-input space,
            # Q = "Question stem:" token embeddings (_pcsi_capture_query), |S| <= B,
            # greedy argmax marginal gain. Arms: VLMAS_PCSI_COND=true|none|wrong
            # (none = generic SMI, wrong = deranged parents), VLMAS_PCSI_FILL=lex|none.
            # VLMAS_WSI_CONSOL_CONTROL=shuffled_parent remaps boxes only (unused here).
            from memory.pcsi_select import pcsi_keep_mask
            _pq = getattr(self, "_pcsi_query", None)
            _pparents = tuple(getattr(self, "_prune_image_parent_indices", ()) or ())
            keep, pcsi_info = pcsi_keep_mask(
                vis_embeddings.to(self.device).float(),
                (_pq[0].to(self.device).float() if _pq is not None else None),
                token_counts, _pparents, budget, magnifications=magnifications,
                cond=os.environ.get("VLMAS_PCSI_COND", "true"),
                fill=os.environ.get("VLMAS_PCSI_FILL", "lex"),
                return_info=True)
            _pcsi_dump = os.environ.get("VLMAS_PCSI_DUMP", "").strip()
            if _pcsi_dump:
                os.makedirs(_pcsi_dump, exist_ok=True)
                _pn = getattr(self, "_pcsi_dump_n", 0)
                self._pcsi_dump_n = _pn + 1
                torch.save({
                    "z": vis_embeddings.detach().to(torch.float16).cpu(),
                    "q": (_pq[0].detach().to(torch.float16).cpu()
                          if _pq is not None else None),
                    "q_text": (_pq[1] if _pq is not None else ""),
                    "u": (relevance.detach().float().cpu()
                          if relevance is not None else None),
                    "token_counts": token_counts, "grid_hws": grid_hws,
                    "magnifications": magnifications, "boxes": boxes,
                    "parents": _pparents, "keep": keep.detach().cpu(),
                    "info": dict(pcsi_info),
                }, os.path.join(_pcsi_dump, f"pcsi_{_pn:04d}.pt"))
            keep = keep.to(self.device)
            print("[WSIConsol] pcsi " + " ".join(
                f"{k}={v}" for k, v in pcsi_info.items() if k != "order")
                + f" q_text={(_pq[1] if _pq is not None else '')!r}"
                + f" control={control or 'none'}", flush=True)
        elif mode == "pcris":
            # Provenance-Conditioned Residual Information Selection (C1 candidate,
            # Notion PCRIS section 8): r_i = (I - U_o U_o^T) h_hat_i with U_o an
            # orthonormal basis (SVD) of the parent observation's normalized states
            # (_prune_image_parent_indices), r_i = h_hat_i for roots; F(S) =
            # log det(I + sum_{i in S} r_i r_i^T), |S| <= B, greedy marginal gain.
            # No question signal. Arms: VLMAS_PCRIS_COND=true|none|wrong,
            # VLMAS_PCRIS_CONTEXT=parent|ancestors, VLMAS_PCRIS_DUMP=<dir>.
            from memory.pcris_select import pcris_keep_mask
            _rparents = tuple(getattr(self, "_prune_image_parent_indices", ()) or ())
            keep, pcris_info = pcris_keep_mask(
                vis_embeddings.to(self.device).float(), token_counts, _rparents,
                budget, magnifications=magnifications,
                cond=os.environ.get("VLMAS_PCRIS_COND", "true"),
                context=os.environ.get("VLMAS_PCRIS_CONTEXT", "parent"),
                return_info=True)
            _pcris_dump = os.environ.get("VLMAS_PCRIS_DUMP", "").strip()
            if _pcris_dump:
                os.makedirs(_pcris_dump, exist_ok=True)
                _rn = getattr(self, "_pcris_dump_n", 0)
                self._pcris_dump_n = _rn + 1
                torch.save({
                    "z": vis_embeddings.detach().cpu(),   # native dtype, exact
                    "z_dtype": str(vis_embeddings.dtype),
                    "u": (relevance.detach().float().cpu()
                          if relevance is not None else None),
                    "token_counts": token_counts, "grid_hws": grid_hws,
                    "magnifications": magnifications, "boxes": boxes,
                    "parents": _rparents, "keep": keep.detach().cpu(),
                    "info": dict(pcris_info),
                    "attn_impl": os.environ.get("VLMAS_ATTN_IMPLEMENTATION", "eager"),
                }, os.path.join(_pcris_dump, f"pcris_{_rn:04d}.pt"))
            keep = keep.to(self.device)
            print("[WSIConsol] pcris " + " ".join(
                f"{k}={v}" for k, v in pcris_info.items() if k != "order")
                + f" attn={os.environ.get('VLMAS_ATTN_IMPLEMENTATION', 'eager')}"
                + f" control={control or 'none'}", flush=True)
        elif mode == "qpris":
            # Question-Conditioned Provenance Residual Information Selection (C1
            # candidate, Notion Q-PRIS sections 2-5): r~_i = (I - P_pi(o)) h_hat_i / nu_i
            # with nu_i the within-case alternative-parent null (PCRIS v2 calibration),
            # q_i = ||U_Q^T r~_i||^2 against the orthonormal question-token subspace at
            # the C1 boundary (_pcsi_capture_query), and
            #   F_Q(S) = log det(I + sum_{i in S} q_i r~_i r~_i^T),  |S| <= B, greedy.
            # No q2v salience, no pooled Q-V cosine, no scale quota, no learned selector.
            # Arms: VLMAS_QPRIS_Q=true|shuffled|none  (shuffled = the PREVIOUS case's
            # question, so the true question never leaks into that arm; the first case of
            # a run has no predecessor and falls back to q=none, reported as q_src),
            # VLMAS_QPRIS_COND=true|none|wrong, VLMAS_QPRIS_CONTEXT=parent|ancestors,
            # VLMAS_QPRIS_NULLCAL=0|1, VLMAS_QPRIS_DUMP=<dir>.
            from memory.qpris_select import qpris_keep_mask
            _qparents = tuple(getattr(self, "_prune_image_parent_indices", ()) or ())
            _qq = getattr(self, "_pcsi_query", None)
            _qarm = (os.environ.get("VLMAS_QPRIS_Q", "true").strip() or "true")
            _qcpu = (_qq[0] if _qq is not None else None)
            _qtext = (_qq[1] if _qq is not None else "")
            if _qarm == "shuffled":
                _bank = getattr(self, "_qpris_prev_query", None)
                self._qpris_prev_query = (_qcpu, _qtext)
                _qsrc = "prev_case" if _bank is not None else "none_first_case"
                _qcpu, _qtext = (_bank if _bank is not None else (None, ""))
            elif _qarm == "none":
                _qcpu, _qtext, _qsrc = None, "", "none"
            else:
                _qsrc = "true" if _qcpu is not None else "none_missing"
            _qdev = (_qcpu.to(self.device).float() if _qcpu is not None else None)
            keep, qpris_info = qpris_keep_mask(
                vis_embeddings.to(self.device).float(), _qdev,
                token_counts, _qparents, budget, magnifications=magnifications,
                cond=os.environ.get("VLMAS_QPRIS_COND", "true"),
                context=os.environ.get("VLMAS_QPRIS_CONTEXT", "parent"),
                q_mode=("true" if _qdev is not None else "none"),
                nullcal=(os.environ.get("VLMAS_QPRIS_NULLCAL", "1").strip() != "0"),
                return_info=True)
            _qpris_dump = os.environ.get("VLMAS_QPRIS_DUMP", "").strip()
            if _qpris_dump:
                os.makedirs(_qpris_dump, exist_ok=True)
                _qn = getattr(self, "_qpris_dump_n", 0)
                self._qpris_dump_n = _qn + 1
                torch.save({
                    "z": vis_embeddings.detach().cpu(),   # native dtype, exact
                    "z_dtype": str(vis_embeddings.dtype),
                    "q": (_qcpu.detach().float().cpu() if _qcpu is not None else None),
                    "q_text": _qtext, "q_arm": _qarm, "q_src": _qsrc,
                    "u": (relevance.detach().float().cpu()
                          if relevance is not None else None),
                    "token_counts": token_counts, "grid_hws": grid_hws,
                    "magnifications": magnifications, "boxes": boxes,
                    "parents": _qparents, "keep": keep.detach().cpu(),
                    "info": {k: v for k, v in qpris_info.items() if k != "nu"},
                    "attn_impl": os.environ.get("VLMAS_ATTN_IMPLEMENTATION", "eager"),
                }, os.path.join(_qpris_dump, f"qpris_{_qn:04d}.pt"))
            keep = keep.to(self.device)
            print("[WSIConsol] qpris " + " ".join(
                f"{k}={v}" for k, v in qpris_info.items() if k not in ("order", "parents"))
                + f" q_arm={_qarm} q_src={_qsrc} q_text={_qtext!r}"
                + f" attn={os.environ.get('VLMAS_ATTN_IMPLEMENTATION', 'eager')}"
                + f" control={control or 'none'}", flush=True)
        elif mode == "csis":
            # Cross-Scale Innovation-Support Selection (C1 candidate, Notion
            # "Cross-Scale Innovation-Support Selection"): the PCRIS v2 provenance
            # residual r~_i supplies the innovation axis, and redundancy is localized in
            # physical WSI space -- L = (r~ r~^T) (.) kappa_x with an RBF spatial kernel
            # on slide coordinates, so similar morphology repeated FAR apart survives
            # (extent / distribution evidence) while nearby repeats are discounted.
            # F(S) = log det(I + L_S), |S| <= B, greedy. Same three provenance controls
            # as PCRIS. Arms: VLMAS_CSIS_COND=true|none|wrong,
            # VLMAS_CSIS_CONTEXT=parent|ancestors, VLMAS_CSIS_NULLCAL=0|1,
            # VLMAS_CSIS_SIGMA=<slide px> (default: the smallest crop side in the case,
            # i.e. one x20 field), VLMAS_CSIS_SIGMA_SCALE=<factor>, VLMAS_CSIS_DUMP=<dir>.
            # VLMAS_CSIS_SUPPORT=branch switches to the canonical question-agnostic form
            # (Notion "Question-Agnostic Cross-Scale Innovation-Support Selection"):
            # L = D_a^{1/2} (C (.) B) D_a^{1/2}, B = same top-level 5x acquisition branch,
            # no sigma. Default rbf keeps the archived RBF-distance arm unchanged.
            from memory.csis_select import csis_keep_mask, default_sigma
            _cparents = tuple(getattr(self, "_prune_image_parent_indices", ()) or ())
            _sig = os.environ.get("VLMAS_CSIS_SIGMA", "").strip()
            _sigma = float(_sig) if _sig else default_sigma(boxes)
            _sscale = os.environ.get("VLMAS_CSIS_SIGMA_SCALE", "").strip()
            if _sscale:
                _sigma *= float(_sscale)
            keep, csis_info = csis_keep_mask(
                vis_embeddings.to(self.device).float(), token_counts, _cparents, budget,
                grid_hws=grid_hws, boxes=boxes, magnifications=magnifications,
                sigma=_sigma,
                cond=os.environ.get("VLMAS_CSIS_COND", "true"),
                context=os.environ.get("VLMAS_CSIS_CONTEXT", "parent"),
                nullcal=(os.environ.get("VLMAS_CSIS_NULLCAL", "1").strip() != "0"),
                self_parent=os.environ.get("VLMAS_CSIS_SELF_PARENT", "root"),
                support=os.environ.get("VLMAS_CSIS_SUPPORT", "rbf").strip() or "rbf",
                return_info=True)
            _csis_dump = os.environ.get("VLMAS_CSIS_DUMP", "").strip()
            if _csis_dump:
                os.makedirs(_csis_dump, exist_ok=True)
                _cn = getattr(self, "_csis_dump_n", 0)
                self._csis_dump_n = _cn + 1
                torch.save({
                    "z": vis_embeddings.detach().cpu(),   # native dtype, exact
                    "z_dtype": str(vis_embeddings.dtype),
                    "token_counts": token_counts, "grid_hws": grid_hws,
                    "magnifications": magnifications, "boxes": boxes,
                    "parents": _cparents, "keep": keep.detach().cpu(),
                    "info": dict(csis_info),
                    "attn_impl": os.environ.get("VLMAS_ATTN_IMPLEMENTATION", "eager"),
                }, os.path.join(_csis_dump, f"csis_{_cn:04d}.pt"))
            keep = keep.to(self.device)
            print("[WSIConsol] csis " + " ".join(
                f"{k}={v}" for k, v in csis_info.items() if k not in ("order", "parents"))
                + f" attn={os.environ.get('VLMAS_ATTN_IMPLEMENTATION', 'eager')}"
                + f" control={control or 'none'}", flush=True)
        elif mode == "topk":
            # Score pruning reference arm (matched-budget accuracy comparison): keep the B
            # visual tokens with the highest question->vision salience (VLMAS_WSI_CONSOL_SC=1),
            # no coverage, no provenance, no representation similarity.
            # VLMAS_TOPK_PER_CROP=1 keeps a proportional share inside every crop instead.
            from memory.topk_select import salience_topk_keep_mask
            _per_crop = os.environ.get("VLMAS_TOPK_PER_CROP", "").strip() == "1"
            keep = salience_topk_keep_mask(
                relevance, token_counts, budget, per_crop=_per_crop).to(self.device)
            print(f"[WSIConsol] topk rel={'q2v' if relevance is not None else 'uniform'} "
                  f"per_crop={int(_per_crop)} kept={int(keep.sum())}/{nV} "
                  f"attn={os.environ.get('VLMAS_ATTN_IMPLEMENTATION', 'eager')}", flush=True)
        elif mode == "score":
            # SCoRe (Xu et al., CVPR 2026) Algorithm 1 — the literature selection arm for
            # the matched-budget accuracy table: greedy weighted k-center with cosine
            # distance on the visual states, salience^alpha as the weight
            # (VLMAS_SCORE_ALPHA, paper best 0.8). VLMAS_WSI_CONSOL_SC=1 supplies the
            # q2v salience; without it the weights are uniform (plain k-center).
            from memory.score_select import score_keep_mask
            _alpha = float(os.environ.get("VLMAS_SCORE_ALPHA", "0.8") or 0.8)
            keep, score_info = score_keep_mask(
                vis_embeddings, relevance, token_counts, budget,
                alpha=_alpha, return_info=True)
            keep = keep.to(self.device)
            print("[WSIConsol] score " + " ".join(
                f"{k}={v}" for k, v in score_info.items() if k != "per_crop")
                + f" per_crop={score_info.get('per_crop')}"
                + f" attn={os.environ.get('VLMAS_ATTN_IMPLEMENTATION', 'eager')}"
                + f" control={control or 'none'}", flush=True)
        elif mode == "strat":
            # C1 = Observation-Stratified compression (method draft 0907):
            # spatial thinning per crop, budget b_p ∝ |V_p|, no scores.
            from memory.stratified import stratified_keep_mask
            keep = stratified_keep_mask(
                token_counts, grid_hws, budget).to(self.device)
        elif mode == "flat":
            # Stage-1 control C: flat every-k-th sampling, no crop awareness.
            from memory.stratified import flat_keep_mask
            keep = flat_keep_mask(nV, budget).to(self.device)
        elif mode == "random_budget":
            # Stage-1 control B: uniform random keep at matched budget.
            from memory.stratified import random_keep_mask
            keep = random_keep_mask(nV, budget, seed=42).to(self.device)
        elif mode == "hier_preserve":
            # C1 = Hierarchy-PRESERVING compression (spec Sec. 3.1 final rev.):
            # within-crop reconstruction only, >=1 token per crop, global
            # greedy distortion reduction. Only knob: budget.
            from memory.subspace_select import hierarchy_preserving_keep_mask
            img_ids = torch.repeat_interleave(
                torch.arange(len(token_counts)),
                torch.tensor(token_counts)).to(self.device)
            keep = hierarchy_preserving_keep_mask(
                vis_embeddings.to(self.device).float(), img_ids, budget)
        elif mode == "subspace":
            # C1 = Hierarchy-Conditioned compression (spec Sec. 3.1 evening
            # rev.): greedy min of valid-subspace reconstruction distortion.
            from memory.subspace_select import subspace_keep_mask
            from memory.wsi_memory import (
                token_footprints, valid_representative_mask)
            rects, centers = token_footprints(boxes, grid_hws)
            img_ids = torch.repeat_interleave(
                torch.arange(len(token_counts)),
                torch.tensor(token_counts)).to(self.device)
            mags_t = torch.repeat_interleave(
                torch.tensor([float(m) for m in magnifications]),
                torch.tensor(token_counts)).to(self.device)
            valid = valid_representative_mask(
                img_ids, mags_t, rects.to(self.device), centers.to(self.device))
            keep = subspace_keep_mask(
                vis_embeddings.to(self.device).float(), valid, budget)
        elif mode == "anchor_residual":
            # C1 = WSI Anchor-Residual compression (family medoid anchor +
            # standardized-centered residual greedy; only knob = budget).
            from memory.anchor_residual import anchor_residual_keep_mask
            keep = anchor_residual_keep_mask(
                vis_embeddings.to(self.device).float(),
                token_counts,
                magnifications,
                boxes,
                budget,
            )
        else:
            keep = consolidation_keep_mask(
                vis_embeddings.to(self.device).float(),
                relevance,
                token_counts,
                magnifications,
                boxes,
                grid_hws,
                budget,
                kappa_shift=kappa_shift,
            )
        kept_per_image = []
        offset = 0
        for count in token_counts:
            kept_per_image.append(int(keep[offset:offset + count].sum()))
            offset += count
        # C2 reassembly metadata: kept-block sizes + acquisition structure,
        # consumed by restage_visual_kv when VLMAS_KV_RESTAGE_ORDER is set.
        self._reassembly_meta = (
            list(kept_per_image),
            list(getattr(self, "_prune_image_parent_indices", ()) or
                 [-1] * len(token_counts)),
            list(magnifications),
        )
        print(
            f"[WSIConsol] kept {int(keep.sum())}/{nV} (ratio {ratio}) "
            f"per_image={kept_per_image} "
            f"rel={'q2v' if relevance is not None else 'uniform'} "
            f"kappa={'shifted' if kappa_shift else 'cos'} "
            f"mode={mode} control={control or 'none'} "
            f"boxes={'yes' if any(b is not None for b in boxes) else 'no'}",
            flush=True,
        )
        if bool(keep.all()):
            return past_kv, seq_len, spans, keep, vis_abs
        keep_np = keep.detach().cpu().numpy()
        past_kv, seq_len, spans, new_vis = _consol_prune(
            past_kv, seq_len, vis_abs.tolist(), keep_np, spans, self.device)
        new_vis_abs = torch.tensor(new_vis, device=self.device, dtype=torch.long)
        return past_kv, seq_len, spans, keep, new_vis_abs

    @staticmethod
    def _reanchor_positions(old_pos, position_cursor, device):
        """MRoPE-preserving re-anchor (env VLMAS_KV_RESTAGE_MROPE).

        method.md "Latent Communication and Re-anchoring": a transmitted subset
        of visual KV cannot inherit its sender positions, so each retained
        token is re-anchored from its provenance -- and for multimodal RoPE the
        rotary coordinate is THREE-dimensional, its height/width axes coming
        from the patch grid. The default path flattens that to a 1-D run
        (arange broadcast over all three axes), which discards the within-crop
        spatial layout the prefill encoded.

        Here the stored (t,h,w) block is translated as a rigid body so its
        minimum lands on position_cursor: relative h/w structure is preserved
        exactly, only the anchor moves. The cursor then advances by the
        compressed span (max coordinate + 1), matching the convention the
        prefill uses for images, instead of by the token count.

        mode "shuffle" is the falsification arm: still 3-D, still the same
        coordinate multiset, but the h/w axes are permuted so the grid layout
        is destroyed while every other property is held fixed.

        Returns (new_pos [3, n], next_cursor).
        """
        import os as _os

        mode = _os.environ.get("VLMAS_KV_RESTAGE_MROPE", "").strip()
        n = int(old_pos.shape[-1])
        if mode not in ("1", "shuffle"):
            seq = torch.arange(position_cursor, position_cursor + n, device=device)
            return seq.unsqueeze(0).expand(3, -1), position_cursor + n
        base = old_pos.min(dim=1, keepdim=True).values
        new_pos = old_pos - base + position_cursor
        if mode == "shuffle":
            generator = torch.Generator(device="cpu")
            generator.manual_seed(42)
            perm = torch.randperm(n, generator=generator).to(device)
            new_pos = torch.stack([
                new_pos[0],
                new_pos[1].index_select(0, perm),
                new_pos[2].index_select(0, perm),
            ])
        return new_pos, int(new_pos.max().item()) + 1

    @torch.no_grad()
    def restage_visual_kv(self, cache, vis_cols, old_positions, position_cursor):
        """KV re-staging (env VLMAS_KV_RESTAGE=1; the refeed-minus-pixels test).

        Physically moves the reasoner's visual K/V columns to the cache TAIL
        right before the terminal prompt, re-rotating K from their original
        MRoPE positions to sequential text positions starting at
        position_cursor (V is position-free). Content and computation are the
        ORIGINAL cached states — only the position changes, which is exactly
        the confound refeed couldn't separate.

        VLMAS_KV_RESTAGE_IDENTITY=1: move the columns but KEEP their original
        positions (delta = 0). Column order is invisible to full-visibility
        terminal attention, so this mode must reproduce the un-restaged
        output — the plumbing correctness check.

        Returns (new_position_cursor, n_moved). Not compatible with
        reallocation arms (vision_mask columns go stale after the move).
        """
        from memory.restage import delta_cos_sin, rerotate_keys
        if not hasattr(cache, "layers"):
            return position_cursor, 0
        # Parking restore: the columns were REMOVED at the reasoner boundary and
        # live in the bank — append them at recency, nothing to delete.
        park_bank = getattr(self, "_park_bank", None)
        if (
            os.environ.get("VLMAS_KV_PARK", "") == "1"
            or os.environ.get("VLMAS_KV_PULSE", "") == "1"
        ) and park_bank is not None:
            # C2 = Cross-Scale Evidence Reassembly (method draft 0907): before
            # the recency restore, optionally permute the SAME B tokens into
            # navigation-derived coarse-to-fine bundles (or a control order).
            # Content and budget unchanged; only ordering differs.
            order_mode = os.environ.get("VLMAS_KV_RESTAGE_ORDER", "original")
            meta = getattr(self, "_reassembly_meta", None)
            if order_mode != "original" and meta is not None:
                from memory.stratified import reassembly_order
                kept_counts, parent_ids, mags = meta
                if sum(kept_counts) == int(park_bank["pos"].shape[-1]):
                    perm = torch.tensor(
                        reassembly_order(
                            kept_counts, parent_ids, mags, order_mode),
                        device=park_bank["pos"].device, dtype=torch.long)
                    park_bank = {
                        "k": [k[:, :, perm, :] for k in park_bank["k"]],
                        "v": [v[:, :, perm, :] for v in park_bank["v"]],
                        "pos": park_bank["pos"][..., perm],
                    }
                    print(f"[KVReassembly] order={order_mode} "
                          f"blocks={kept_counts}", flush=True)
                else:
                    print(f"[KVReassembly] SKIP: meta {sum(kept_counts)} != "
                          f"bank {int(park_bank['pos'].shape[-1])}", flush=True)
            n = int(park_bank["pos"].shape[-1])
            old_pos = park_bank["pos"].to(device=self.device, dtype=torch.long)
            new_pos, park_next_cursor = self._reanchor_positions(
                old_pos, position_cursor, self.device)
            sample = cache.layers[0].keys
            cos_old, sin_old = self.lm.rotary_emb(sample, old_pos.unsqueeze(1))
            cos_new, sin_new = self.lm.rotary_emb(sample, new_pos.unsqueeze(1))
            cos_delta, sin_delta = delta_cos_sin(
                cos_old.float(), sin_old.float(),
                cos_new.float(), sin_new.float())
            for layer_index, layer in enumerate(cache.layers):
                restored_k = rerotate_keys(
                    park_bank["k"][layer_index].float(), cos_delta, sin_delta
                ).to(layer.keys.dtype)
                layer.keys = torch.cat([layer.keys, restored_k], dim=2)
                layer.values = torch.cat(
                    [layer.values,
                     park_bank["v"][layer_index].to(layer.values.dtype)], dim=2)
            self._park_bank = None
            print(f"[KVPark] restored {n} visual cols at recency "
                  f"(pos {position_cursor}..{park_next_cursor - 1}, "
                  f"mrope={os.environ.get('VLMAS_KV_RESTAGE_MROPE', '') or 'off'})",
                  flush=True)
            return park_next_cursor, n
        cols = vis_cols.to(device=self.device, dtype=torch.long)
        n = int(cols.numel())
        total = int(cache.layers[0].keys.shape[2])
        if n == 0 or int(old_positions.shape[-1]) != n or int(cols.max()) >= total:
            print(f"[KVRestage] skipped (n={n}, pos={tuple(old_positions.shape)}, "
                  f"cache={total})", flush=True)
            return position_cursor, 0
        identity = os.environ.get("VLMAS_KV_RESTAGE_IDENTITY", "") == "1"
        wrong_slide = os.environ.get("VLMAS_KV_RESTAGE_MODE", "matched") == "wrong_slide"
        old_pos = old_positions.to(device=self.device, dtype=torch.long)  # [3, n]
        source_note = ""
        # RSVMH (memory/rsvmh.py, env VLMAS_KV_RESTAGE_ORDER in rsvmh.ORDERS): place the SAME
        # retained supports in ascending order of the Reasoner's closed-form support-deletion
        # sensitivity (or a control score) before the recency move. Off = unchanged.
        _rsvmh_counts = None
        _rsvmh_order = os.environ.get("VLMAS_KV_RESTAGE_ORDER", "").strip()
        if _rsvmh_order and not identity and not wrong_slide:
            from memory import rsvmh as _rsvmh
            if _rsvmh.enabled_order(_rsvmh_order):
                _seed = int(getattr(getattr(self, "_rss_ctx", None), "pre_len", 0) or 0)
                _perm, _counts, _note = _rsvmh.plan(self, cols, _rsvmh_order, case_seed=_seed)
                if _perm is not None:
                    cols = cols.index_select(0, _perm)
                    old_pos = old_pos.index_select(1, _perm)
                    _rsvmh_counts = _counts
                print(f"[RSVMH] {_note}" if _perm is not None else f"[RSVMH] SKIP {_note}", flush=True)
        if wrong_slide:
            # Content-specificity control: place a FROZEN donor slide's visual
            # K/V at recency instead of this case's own (first restage case
            # captures the donor and stays matched — exclude it from analysis).
            donor = getattr(self, "_restage_donor", None)
            if donor is None:
                self._restage_donor = {
                    "k": [layer.keys[:, :, cols, :].detach().clone()
                          for layer in cache.layers],
                    "v": [layer.values[:, :, cols, :].detach().clone()
                          for layer in cache.layers],
                    "pos": old_pos.detach().clone(),
                }
                source_note = " (donor case: own KV, matched)"
            else:
                n = min(n, int(donor["pos"].shape[-1]))
                old_pos = donor["pos"][:, :n]
                source_note = f" (WRONG-SLIDE donor, n={n})"
        if identity:
            new_pos, restage_next_cursor = old_pos, position_cursor + n
        elif _rsvmh_counts is not None:
            from memory.paging import page_anchor_positions
            new_pos, restage_next_cursor = page_anchor_positions(
                old_pos, _rsvmh_counts, position_cursor)
        else:
            new_pos, restage_next_cursor = self._reanchor_positions(
                old_pos, position_cursor, self.device)
        sample = cache.layers[0].keys
        cos_old, sin_old = self.lm.rotary_emb(sample, old_pos.unsqueeze(1))
        cos_new, sin_new = self.lm.rotary_emb(sample, new_pos.unsqueeze(1))
        cos_delta, sin_delta = delta_cos_sin(
            cos_old.float(), sin_old.float(), cos_new.float(), sin_new.float())
        keep = torch.ones(total, dtype=torch.bool, device=self.device)
        keep[cols] = False
        donor_bank = getattr(self, "_restage_donor", None) if wrong_slide else None
        use_donor = wrong_slide and not source_note.startswith(" (donor case")
        # VLMAS_KV_RESTAGE_VSCALE (beta < 1): attenuate the restaged VALUES so
        # the readout blends prior reasoning with visual evidence instead of
        # the all-or-nothing gate — the fix knob for prior-dominant backbones
        # (8B reversal). K untouched: addressing stays exact.
        vscale = float(os.environ.get("VLMAS_KV_RESTAGE_VSCALE", "1.0"))
        for layer_index, layer in enumerate(cache.layers):
            if use_donor:
                source_k = donor_bank["k"][layer_index][:, :, :n, :]
                source_v = donor_bank["v"][layer_index][:, :, :n, :]
            else:
                source_k = layer.keys[:, :, cols, :]
                source_v = layer.values[:, :, cols, :]
            moved_k = rerotate_keys(
                source_k.float(), cos_delta, sin_delta).to(layer.keys.dtype)
            moved_v = source_v.to(layer.values.dtype)
            if vscale != 1.0:
                moved_v = moved_v * vscale
            layer.keys = torch.cat([layer.keys[:, :, keep, :], moved_k], dim=2)
            layer.values = torch.cat(
                [layer.values[:, :, keep, :], moved_v], dim=2)
        new_cursor = position_cursor if identity else restage_next_cursor
        print(
            f"[KVRestage] moved {n}/{total} visual cols to tail; "
            f"{'IDENTITY positions' if identity else f'pos {position_cursor}..{restage_next_cursor - 1}'}"
            f" mrope={os.environ.get('VLMAS_KV_RESTAGE_MROPE', '') or 'off'}"
            f"{source_note}",
            flush=True,
        )
        return new_cursor, n

    def _visual_cut_ctx(self, cur_vis):
        """Visual-contribution cut at Reasoner latent steps (env VLMAS_VCUT,
        VLMAS_VCUT_STAGE contains R; off = nullcontext)."""
        import contextlib as _cl
        from memory import visual_cut as _vc
        if not _vc.enabled("R") or not getattr(self, "_prune_morphology_enabled", False):
            return _cl.nullcontext(None)
        if not isinstance(cur_vis, torch.Tensor) or cur_vis.numel() == 0:
            return _cl.nullcontext(None)
        return _vc.install_visual_cut(self, cur_vis.to(self.device), stage="R")

    @contextlib.contextmanager
    def _visual_bind_step(self, past_kv, cur_vis, cur_len):
        """Visual Binding Step (env-gated experiment; off = no-op).

        Wrapped around the FINAL Reasoner latent-step forward only. At the
        layers in VLMAS_VISUAL_BIND_LAYERS a forward_pre_hook replaces the
        attention mask with one that admits ONLY the visual KV columns plus
        the token itself, so h_m^{l+1} = h_m^l + Attn(h_m^l, K_V, V_V): the
        residual stream keeps the Reasoner's question/task context while the
        binding layer injects WSI evidence. All other layers, the realign
        W_a, positions, and the KV append are untouched; step count stays m.

        VLMAS_VISUAL_BIND_MODE controls the causal arms:
          matched      bind to this slide's visual KV (the method)
          wrong_slide  bind to a FROZEN donor slide's visual K/V (first case
                       of the run captures the donor; that case is a no-swap
                       donor and must be excluded from the matched-vs-wrong
                       comparison) — swapped only at binding layers, only
                       for the duration of this forward, then restored
          shuffled_v   keep K but PERMUTE V across the visual columns at the
                       binding layers (destroys position-content binding
                       while keeping the content marginals), restored after
          random_v     keep K (routing) but replace V at binding layers with
                       deterministic mean/std-matched Gaussian (content
                       control), restored after the forward

        Yields True when the bind is active, False when it is a no-op.
        """
        if os.environ.get("VLMAS_VISUAL_BIND", "") != "1":
            yield False
            return
        if not getattr(self, "_prune_morphology_enabled", False):
            yield False
            return  # reasoner stage only
        if not isinstance(cur_vis, torch.Tensor) or cur_vis.numel() == 0:
            yield False
            return
        from vision_text_mas.visual_bind import (
            build_bind_mask, matched_noise_like, parse_bind_layers,
            shuffle_permutation,
        )
        num_layers = len(self.lm.layers)
        layer_ids = parse_bind_layers(
            os.environ.get("VLMAS_VISUAL_BIND_LAYERS", ""), num_layers)
        mode = os.environ.get("VLMAS_VISUAL_BIND_MODE", "matched")
        vis = cur_vis.to(device=self.device, dtype=torch.long)
        cache_layers = past_kv.layers
        kv_len = cur_len + 1  # this token's K/V are appended before attention
        # VLMAS_VISUAL_BIND_NO_SELF=1: spec-faithful Sec. 3.4.2 (softmax over
        # the visual columns only); default keeps the token itself readable.
        self_col = None if os.environ.get(
            "VLMAS_VISUAL_BIND_NO_SELF", "") == "1" else cur_len
        mask = build_bind_mask(
            kv_len, vis, self_col, cache_layers[0].keys.dtype, self.device)

        saved: list[tuple[int, torch.Tensor, torch.Tensor | None, torch.Tensor]] = []
        donor_note = ""
        if mode == "wrong_slide":
            donor = getattr(self, "_bind_donor", None)
            if donor is None:
                self._bind_donor = {
                    li: (
                        cache_layers[li].keys[:, :, vis, :].detach().clone(),
                        cache_layers[li].values[:, :, vis, :].detach().clone(),
                    )
                    for li in layer_ids
                }
                donor_note = " (donor case: captured own KV, NO swap)"
            else:
                for li in layer_ids:
                    if li not in donor:
                        continue
                    donor_k, donor_v = donor[li]
                    n = min(int(donor_k.shape[2]), int(vis.numel()))
                    cols = vis[:n]
                    layer = cache_layers[li]
                    saved.append((
                        li, cols,
                        layer.keys[:, :, cols, :].detach().clone(),
                        layer.values[:, :, cols, :].detach().clone(),
                    ))
                    layer.keys[:, :, cols, :] = donor_k[:, :, :n, :].to(layer.keys.dtype)
                    layer.values[:, :, cols, :] = donor_v[:, :, :n, :].to(layer.values.dtype)
        elif mode == "shuffled_v":
            permutation = shuffle_permutation(int(vis.numel()), 42)
            for li in layer_ids:
                layer = cache_layers[li]
                original = layer.values[:, :, vis, :].detach().clone()
                saved.append((li, vis, None, original))
                layer.values[:, :, vis, :] = original[:, :, permutation, :]
        elif mode == "random_v":
            for li in layer_ids:
                layer = cache_layers[li]
                original = layer.values[:, :, vis, :].detach().clone()
                saved.append((li, vis, None, original))
                layer.values[:, :, vis, :] = matched_noise_like(original, 42 + li)

        def bind_pre_hook(module, args, kwargs):
            if "attention_mask" in kwargs or len(args) < 3:
                kwargs["attention_mask"] = mask
                return args, kwargs
            new_args = list(args)
            new_args[2] = mask
            return tuple(new_args), kwargs

        handles = [
            self.lm.layers[li].self_attn.register_forward_pre_hook(
                bind_pre_hook, with_kwargs=True)
            for li in layer_ids
        ]
        print(
            f"[VisualBind] layers={layer_ids} mode={mode} vis={int(vis.numel())} "
            f"kv_len={kv_len}{donor_note}",
            flush=True,
        )
        try:
            yield True
        finally:
            for handle in handles:
                handle.remove()
            for li, cols, original_k, original_v in saved:
                layer = cache_layers[li]
                if original_k is not None:
                    layer.keys[:, :, cols, :] = original_k
                layer.values[:, :, cols, :] = original_v

    def _append_visual_relays(self, past_kv, cur_len, next_pos, last_hidden,
                              vis_cols, grids, spans=None):
        """ROI-conditioned visual latent relay (env-gated experiment; off = no-op).

        VLMAS_VISUAL_RELAY=global|roi: after the Reasoner's latent rollout, run
        one latent-style step per group whose attention is RESTRICTED (via
        attention_mask) to that group's visual KV plus the new token itself:
        r_g = F(q_reason; K_g, V_g) with input e = W_a(q_reason). Each relay's
        KV is appended at recency, so the Answerer reads visual-conditioned
        latent messages without touching raw visual KV attention.
        VLMAS_VISUAL_RELAY_CROSS=1: mismatched-content control — relays read a
        FROZEN donor slide's V (first case of the run captures the donor; that
        case is a no-swap donor and must be excluded from C-vs-D analysis).
        """
        mode = os.environ.get("VLMAS_VISUAL_RELAY", "")
        if mode not in ("global", "roi"):
            return past_kv, cur_len, 0
        if not getattr(self, "_prune_morphology_enabled", False):
            return past_kv, cur_len, 0  # reasoner stage only
        if not isinstance(vis_cols, torch.Tensor) or vis_cols.numel() == 0:
            return past_kv, cur_len, 0
        vis = vis_cols.to(device=self.device, dtype=torch.long)
        if mode == "roi" and grids is not None:
            sms = int(getattr(self.model.config.vision_config, "spatial_merge_size", 2))
            counts = [int(t) * (int(h) // sms) * (int(w) // sms)
                      for t, h, w in grids.tolist()]
            if sum(counts) == int(vis.numel()):
                groups, offset = [], 0
                for count in counts:
                    groups.append(vis[offset:offset + count])
                    offset += count
            else:
                print(
                    f"[VisualRelay] ROI split mismatch ({sum(counts)} vs "
                    f"{int(vis.numel())}); falling back to global", flush=True,
                )
                groups = [vis]
        else:
            groups = [vis]
        cross = os.environ.get("VLMAS_VISUAL_RELAY_CROSS", "") == "1"
        layers = past_kv.layers if hasattr(past_kv, "layers") else None
        swapped = None
        donor_note = ""
        if cross and layers is not None:
            donor = getattr(self, "_relay_v_donor", None)
            if donor is None:
                self._relay_v_donor = [
                    layer.values[:, :, vis, :].detach().clone() for layer in layers
                ]
                donor_note = " (donor case: captured own V, NO swap)"
            else:
                swapped = []
                for layer_index, layer in enumerate(layers):
                    bank = donor[layer_index]
                    n = min(int(bank.shape[2]), int(vis.numel()))
                    cols = vis[:n]
                    swapped.append(
                        (layer_index, cols, layer.values[:, :, cols, :].detach().clone())
                    )
                    layer.values[:, :, cols, :] = bank[:, :, :n, :].to(layer.values.dtype)
        # Role-Boundary Visual Read (C2 candidate, 0907 spec): additionally
        # admit the prompt-text ("P") spans, so the single read step computes
        # r_vis = Attn(q_read, [K_Q; K_R*]) — a question-CONDITIONED read of
        # the compressed bank instead of the 0903 relay's visual-only read.
        q_read_cols = None
        if os.environ.get("VLMAS_VISUAL_RELAY_WITH_Q", "") == "1" and spans:
            cols, off = [], 0
            for span_type, span_len in spans:
                if span_type == "P":
                    cols.extend(range(off, off + span_len))
                off += span_len
            cols = [c for c in cols if c < cur_len]
            if cols:
                q_read_cols = torch.tensor(
                    cols, device=self.device, dtype=torch.long)
        added = 0
        try:
            for group in groups:
                le = self._apply_realign(last_hidden)
                allowed = torch.zeros(cur_len + 1, dtype=torch.long, device=self.device)
                allowed[group] = 1
                if q_read_cols is not None:
                    allowed[q_read_cols] = 1
                allowed[cur_len] = 1
                o = self.lm(
                    inputs_embeds=le,
                    position_ids=self._text_positions(1, next_pos + added),
                    attention_mask=allowed.unsqueeze(0),
                    past_key_values=past_kv,
                    use_cache=True,
                    output_hidden_states=True,
                )
                past_kv = o.past_key_values
                cur_len += 1
                added += 1
        finally:
            if swapped is not None:
                relay_layers = past_kv.layers
                for layer_index, cols, original in swapped:
                    relay_layers[layer_index].values[:, :, cols, :] = original
        print(
            f"[VisualRelay] mode={mode} relays={added} "
            f"with_q={0 if q_read_cols is None else int(q_read_cols.numel())} "
            f"group_sizes={[int(g.numel()) for g in groups]} "
            f"cross_swap={int(swapped is not None)}{donor_note}",
            flush=True,
        )
        return past_kv, cur_len, added

    @torch.no_grad()
    def _hierarchy_pruned_vision_features(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        *,
        parent_indices: tuple[int, ...],
        keep_ratio: float | None,
        min_tokens_per_image: int,
        significance_sigma: float | None = None,
        observe_blocks: int = 4,
        spectral_energy_ratio: float | None = None,
        spectral_min_keep_ratio: float = 0.25,
        spectral_max_keep_ratio: float = 0.50,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Observe every WSI patch, then prune merge-groups for deep encoding.

        The first vision blocks see every image token.  Group-level morphology
        novelty then drives one hierarchy-constrained selection, and only the
        retained groups traverse the remaining vision and language blocks.
        Keeping complete spatial-merge groups preserves the pretrained merger.
        """
        from memory.prune import hierarchy_prefill_keep_mask, spectral_energy_keep_ratio

        visual = self.visual
        hidden = visual.patch_embed(pixel_values.type(visual.dtype))
        hidden = hidden + visual.fast_pos_embed_interpolate(grid_thw)

        rotary = visual.rot_pos_emb(grid_thw)
        seq_len = int(hidden.shape[0])
        rotary = rotary.reshape(seq_len, -1)
        embedding = torch.cat((rotary, rotary), dim=-1)
        position_embeddings = (embedding.cos(), embedding.sin())

        raw_counts = tuple(int(t * h * w) for t, h, w in grid_thw.tolist())
        cu_seqlens = torch.tensor(
            (0, *torch.tensor(raw_counts).cumsum(0).tolist()),
            dtype=torch.int32,
            device=hidden.device,
        )
        split_at = min(max(1, observe_blocks), len(visual.blocks) - 1)
        for block in visual.blocks[:split_at]:
            hidden = block(
                hidden,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )

        merge_unit = int(visual.spatial_merge_unit)
        token_counts = tuple(count // merge_unit for count in raw_counts)
        grouped = hidden.reshape(-1, merge_unit, hidden.shape[-1]).float()
        pooled = grouped.mean(dim=1)
        scores = torch.empty(pooled.shape[0], dtype=torch.float32, device=pooled.device)
        offset = 0
        for count in token_counts:
            patch_features = pooled[offset : offset + count]
            center = patch_features.mean(dim=0, keepdim=True)
            novelty = 1.0 - F.cosine_similarity(patch_features, center, dim=-1)
            texture = grouped[offset : offset + count].var(dim=1, unbiased=False).mean(dim=-1)
            novelty = (novelty - novelty.min()) / (novelty.max() - novelty.min() + 1e-6)
            texture = (texture - texture.min()) / (texture.max() - texture.min() + 1e-6)
            scores[offset : offset + count] = novelty + 0.25 * texture
            offset += count

        adaptive_keep_ratio = keep_ratio
        if spectral_energy_ratio is not None:
            spectral_budget = 0
            spectral_offset = 0
            spectral_image_budgets = []
            for count in token_counts:
                patch_ratio = spectral_energy_keep_ratio(
                    pooled[spectral_offset : spectral_offset + count],
                    energy_ratio=spectral_energy_ratio,
                    min_keep_ratio=spectral_min_keep_ratio,
                    max_keep_ratio=spectral_max_keep_ratio,
                )
                image_budget = math.ceil(count * patch_ratio)
                spectral_budget += image_budget
                spectral_image_budgets.append((image_budget, count))
                spectral_offset += count
            adaptive_keep_ratio = spectral_budget / pooled.shape[0]
            print(
                "[PruneVision/adaptive] spectral-energy budget: "
                f"{adaptive_keep_ratio:.2%} (tau={spectral_energy_ratio:.2%}) "
                f"per_image_K={spectral_image_budgets}",
                flush=True,
            )
        # C2 #14 EOVC (memory/eovc.py, env VLMAS_EOVC=1): the same early-observation group
        # states, but the keep mask minimises the support variation left unexplained by the
        # retained span instead of ranking novelty + 0.25*texture under a per-image quota.
        # Off = unchanged.
        from memory import eovc as _eovc
        # C1 #19 SEC-LD (memory/secld_select.py, env VLMAS_SECLD=1): pixel-space spatial evidence
        # demand e_i = 1 - b_i times the same early-observation morphology direction, one support-wise
        # log-det set objective, no per-crop quota. Off = unchanged.
        from memory import secld_select as _secld
        if _secld.enabled():
            _secld_cfg = _secld.config()
            _secld_ratio = float(adaptive_keep_ratio) if _secld_cfg["keep"] is None else float(_secld_cfg["keep"])
            _secld_budget = int(round(_secld_ratio * int(pooled.shape[0])))
            _secld_pv = getattr(self, "_secld_pixel_values_fp32", None)
            if _secld_pv is None or tuple(_secld_pv.shape) != tuple(pixel_values.shape):
                _secld_pv = pixel_values
            _secld_info = _secld.select(_secld_pv.to(pooled.device), grid_thw, pooled, token_counts,
                                        _secld_budget, _secld_cfg)
            keep_groups = _secld_info["keep"].to(device=pooled.device)
            print(f"[SECLD] {_secld.summary(_secld_info)} keep_ratio={_secld_ratio:.3f}", flush=True)
        elif _eovc.enabled():
            _eovc_budget = int(round(float(adaptive_keep_ratio) * int(pooled.shape[0])))
            _eovc_info = _eovc.select(grouped, token_counts, _eovc_budget)
            keep_groups = _eovc_info["keep"].to(device=pooled.device)
            print(f"[EOVC] {_eovc.summary(_eovc_info)} keep_ratio={float(adaptive_keep_ratio):.3f}",
                  flush=True)
        else:
            keep_groups = hierarchy_prefill_keep_mask(
                scores,
                token_counts=token_counts,
                parent_indices=parent_indices,
                keep_ratio=adaptive_keep_ratio,
                min_tokens_per_image=min_tokens_per_image,
                significance_sigma=significance_sigma,
            )
        # Pathology-structured consolidation (env-gated post-correction; both
        # components swap ORIGINAL groups within one image so budget/quota are
        # unchanged — see memory/consolidate.py). Off = byte-identical.
        prune_args = getattr(self, "_prune_args", None)
        rescue_on = bool(
            getattr(prune_args, "prune_morphology_coverage_rescue", False)
            or os.environ.get("VLMAS_MORPH_RESCUE", "") == "1"
        )
        cross_on = bool(
            getattr(prune_args, "prune_cross_scale_consolidation", False)
            or os.environ.get("VLMAS_CROSS_SCALE", "") == "1"
        )
        if rescue_on or cross_on:
            from memory.consolidate import (
                cross_scale_consolidate, morphology_coverage_rescue,
            )
            budget_before = int(keep_groups.sum().item())
            rescue_swaps, rescue_gain, cross_swaps = 0, 0.0, 0
            if rescue_on:
                keep_groups, rescue_swaps, rescue_gain = morphology_coverage_rescue(
                    pooled, keep_groups, token_counts,
                    max_swaps_per_image=int(
                        os.environ.get("VLMAS_RESCUE_MAX_SWAPS", "0")) or None,
                )
            if cross_on:
                keep_groups, cross_swaps = cross_scale_consolidate(
                    pooled, keep_groups, token_counts, parent_indices,
                    margin=float(os.environ.get("VLMAS_CROSS_SCALE_MARGIN", "0.05")),
                    max_swaps_per_image=int(
                        os.environ.get("VLMAS_CROSS_SCALE_MAX_SWAPS", "0")) or None,
                )
            budget_after = int(keep_groups.sum().item())
            assert budget_after == budget_before, (
                f"consolidation changed the KV budget "
                f"({budget_before} -> {budget_after})"
            )
            print(
                f"[Consolidate] rescue_swaps={rescue_swaps} "
                f"rescue_gain={rescue_gain:.3f} cross_swaps={cross_swaps} "
                f"budget={budget_after}/{int(keep_groups.numel())} "
                f"parents={list(parent_indices)}",
                flush=True,
            )
        # Sender-relay experiment: preserve the FINAL pruning-decision score of
        # every surviving merge-group (= one LLM visual token), in survivor
        # order, plus its original group index and source image. Read-only
        # bookkeeping — the selection itself is untouched.
        image_of_group = torch.repeat_interleave(
            torch.arange(len(token_counts)),
            torch.tensor(token_counts, dtype=torch.long),
        )
        kept_cpu = keep_groups.detach().cpu()
        self._prefill_prune_survivor_scores = scores.detach().float().cpu()[kept_cpu]
        self._prefill_prune_survivor_group_ids = (
            torch.nonzero(kept_cpu, as_tuple=True)[0]
        )
        self._prefill_prune_survivor_image_ids = image_of_group[kept_cpu]
        raw_keep = keep_groups.repeat_interleave(merge_unit)
        hidden = hidden[raw_keep]
        position_embeddings = (
            position_embeddings[0][raw_keep],
            position_embeddings[1][raw_keep],
        )

        kept_per_image: list[int] = []
        offset = 0
        for count in token_counts:
            kept_per_image.append(int(keep_groups[offset : offset + count].sum().item()))
            offset += count
        retained_raw_counts = tuple(count * merge_unit for count in kept_per_image)
        cu_seqlens = torch.tensor(
            (0, *torch.tensor(retained_raw_counts).cumsum(0).tolist()),
            dtype=torch.int32,
            device=hidden.device,
        )
        for block in visual.blocks[split_at:]:
            hidden = block(
                hidden,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )
        return visual.merger(hidden), keep_groups

    # ── unified text decode: model.generate on top of a latent KV ───────────────
    @staticmethod
    def _kv_len(kv) -> int:
        try:
            if hasattr(kv, "layers"):
                return kv.layers[0].keys.shape[-2]
            if hasattr(kv, "key_cache"):
                return kv.key_cache[0].shape[-2]
            return kv[0][0].shape[-2]
        except Exception:
            return 0

    @torch.no_grad()
    def retain_last_kv(self, past_kv, keep_tokens: int):
        """Discard every physical KV position except the final recurrent window."""
        if keep_tokens < 1:
            raise ValueError("keep_tokens must be positive")
        if self._kv_len(past_kv) <= keep_tokens:
            return past_kv
        if hasattr(past_kv, "layers"):
            for layer in past_kv.layers:
                layer.keys = layer.keys[..., -keep_tokens:, :].contiguous()
                layer.values = layer.values[..., -keep_tokens:, :].contiguous()
            return past_kv
        if hasattr(past_kv, "key_cache"):
            past_kv.key_cache = [
                key[..., -keep_tokens:, :].contiguous()
                for key in past_kv.key_cache
            ]
            past_kv.value_cache = [
                value[..., -keep_tokens:, :].contiguous()
                for value in past_kv.value_cache
            ]
            return past_kv
        if isinstance(past_kv, tuple):
            return tuple(
                tuple(tensor[..., -keep_tokens:, :].contiguous() for tensor in layer)
                for layer in past_kv
            )
        raise TypeError(f"unsupported KV cache type: {type(past_kv)!r}")

    @torch.no_grad()
    def _cjk_suppress_ids(self) -> list[int]:
        """Token ids whose surface form contains any CJK / kana / fullwidth
        char. Built once and cached; passed to generate(suppress_tokens=…) so
        greedy decoding cannot emit Chinese subwords (e.g. ``aden癌``, ``染色``)
        that leak from Qwen's bilingual vocab when decoding OOD latent-KV.
        Special tokens are left untouched so EOS/pad still work.
        """
        cached = getattr(self, "_cjk_ids_cache", None)
        if cached is not None:
            return cached
        import re as _re
        _cjk = _re.compile(
            r"[　-〿぀-ヿ㐀-䶿一-鿿"
            r"豈-﫿＀-￯]")
        tok = self.processor.tokenizer
        special = set(tok.all_special_ids or [])
        vocab = tok.get_vocab()  # {piece: id}
        ids: list[int] = []
        for tid in vocab.values():
            if tid in special:
                continue
            piece = tok.decode([tid])
            if _cjk.search(piece):
                ids.append(tid)
        self._cjk_ids_cache = ids
        print(f"[qwen3vl] english_only: suppressing {len(ids)} CJK token ids")
        return ids

    def generate_on_kv(
        self,
        system_prompt: str,
        user_text: str,
        past_kv,
        max_new_tokens: int,
        clone_cache: bool = False,
        pos_cursor: int | None = None,
        do_sample: bool = False,
        rep_penalty: float = 1.0,
        no_repeat: int = 0,
        temperature: float = 0.7,
        top_p: float = 0.9,
        enable_thinking: bool = False,
        english_only: bool = False,
        continuation_user_turn: bool = False,
    ) -> str:
        """Single decode path for ALL agents (Diagnosis, Verifier, …).

        Re-prefills the (already system-folded) prompt as real text tokens via
        DynamicCache + model.generate, so decoding starts at a clean assistant
        boundary (no manual inputs_embeds / latent-hidden loop). This is the
        KVComm-style path that avoids the off-distribution echo seen when
        decoding straight off a latent hidden state.

        The anti-repetition knobs default OFF (LatentMAS-faithful / verified
        best); pass rep_penalty!=1.0 or no_repeat>0 only deliberately.
        """
        tok = self.processor.tokenizer

        if hasattr(self.processor, "apply_chat_template"):
            msgs = []
            if continuation_user_turn:
                msgs.append({
                    "role": "user",
                    "content": self._continuation_user_text(system_prompt, user_text),
                })
            else:
                if system_prompt:
                    msgs.append({"role": "system", "content": system_prompt})
                msgs.append({"role": "user", "content": user_text})
            try:
                prompt = self.processor.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
            except TypeError:
                prompt = self.processor.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                )
            if prompt.endswith("<think>\n") and not enable_thinking:
                prompt = prompt + "</think>\n\n"
        else:
            from llava.conversation import conv_templates

            conv = conv_templates["llava_v1"].copy()
            if system_prompt:
                conv.system = system_prompt
            conv.append_message(conv.roles[0], user_text)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

        input_ids = tok(
            prompt, return_tensors="pt", add_special_tokens=False,
        )["input_ids"].to(self.device)
        attention_mask = torch.ones_like(input_ids, device=self.device)

        cache_source = copy.deepcopy(past_kv) if clone_cache else past_kv
        dynamic_cache = (
            self._kv_to_dynamic_cache(cache_source)
            if hasattr(self, "_kv_to_dynamic_cache")
            else cache_source
        )
        if dynamic_cache is not None:
            past_len = self._kv_len(dynamic_cache)
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)
                # MRoPE: land the first generated token at pos_cursor (adjacent to
                # the compressed visual positions) instead of ~token_count away.
                if getattr(self, "mrope_pos", False) and pos_cursor is not None:
                    self.vlm.rope_deltas = torch.tensor(
                        [[pos_cursor - past_len]], dtype=torch.long, device=self.device)
                else:
                    self.vlm.rope_deltas = None

        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=dynamic_cache,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
            use_cache=True,
        )
        if rep_penalty and rep_penalty != 1.0:
            gen_kwargs["repetition_penalty"] = rep_penalty
        if no_repeat and no_repeat > 0:
            gen_kwargs["no_repeat_ngram_size"] = no_repeat
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        if getattr(self, "mrope_pos", False):
            position_start = pos_cursor if pos_cursor is not None else past_len
            gen_kwargs["position_ids"] = self._text_positions(
                input_ids.shape[1],
                start=position_start,
            )
        if english_only:
            _sup = self._cjk_suppress_ids()
            if _sup:
                gen_kwargs["suppress_tokens"] = _sup

        _timer, _t0 = self._ttft_timer()
        _lp = []
        if _timer is not None:
            _lp.append(_timer)
        # ── PROBE_FINALIZER_EOS (off by default = byte-identical): capture the
        #    first-generated-token logits so we can see WHY greedy emits <|im_end|>
        #    (empty answer) on the carried latent KV. Dumps top-k + EOS rank/prob.
        import os as _os
        _probe_path = _os.environ.get("PROBE_FINALIZER_EOS")
        _cap = {}
        if _probe_path:
            class _CapLP:
                def __call__(self, ids, scores):
                    if "first" not in _cap:
                        _cap["first"] = scores[0].detach().float().cpu()
                    return scores
            _lp.append(_CapLP())
        if _lp:
            gen_kwargs["logits_processor"] = _lp
        output_ids = self.model.generate(**gen_kwargs)
        self._record_ttft(_timer, _t0)
        new_ids = output_ids[:, input_ids.shape[1]:]
        _text = tok.decode(new_ids[0], skip_special_tokens=True).strip()
        # Preserve the full Thinking Judger readout. The Answerer parser extracts
        # the final boxed payload, while the artifact retains the preceding
        # reasoning for parity with the original LatentMAS output.
        if _probe_path and "first" in _cap:
            import json as _json
            _s = _cap["first"]
            _eos = tok.eos_token_id
            _probs = torch.softmax(_s, dim=-1)
            _tk = torch.topk(_s, 10)
            _rec = {
                "prompt_snip": (user_text or "")[:70],
                "gen_text": _text[:60],
                "gen_empty": (_text == ""),
                "argmax_tok": tok.decode([int(_s.argmax())]),
                "eos_id": int(_eos) if _eos is not None else None,
                "eos_logit": float(_s[_eos]) if _eos is not None else None,
                "eos_prob": float(_probs[_eos]) if _eos is not None else None,
                "eos_rank": int((_s > _s[_eos]).sum()) if _eos is not None else None,
                "top10": [[tok.decode([int(i)]), round(float(_probs[i]), 4),
                           round(float(_s[i]), 2)] for i in _tk.indices.tolist()],
            }
            try:
                with open(_probe_path, "a") as _f:
                    _f.write(_json.dumps(_rec, ensure_ascii=False) + "\n")
            except Exception:
                pass
        return _text

    @torch.no_grad()
    def generate_on_kv_manual(
        self,
        system_prompt: str,
        user_text: str,
        past_kv,
        max_new_tokens: int,
        *,
        pos_cursor: int | None = None,
        enable_thinking: bool = True,
    ) -> str:
        """0717-style manual terminal readout without retaining hidden-state stacks."""
        embeds = self.embed_text(
            user_text,
            system_prompt=system_prompt,
            enable_thinking=enable_thinking,
        )
        cache = (
            self._kv_to_dynamic_cache(past_kv)
            if hasattr(self, "_kv_to_dynamic_cache")
            else past_kv
        )
        cache_len = self._kv_len(cache)
        start = pos_cursor if self.mrope_pos and pos_cursor is not None else cache_len
        positions = self._text_positions(embeds.shape[0], start=start)
        output = self.lm(
            inputs_embeds=embeds.unsqueeze(0),
            position_ids=positions,
            past_key_values=cache,
            use_cache=True,
            output_hidden_states=False,
        )
        cache = output.past_key_values
        hidden = output.last_hidden_state[:, -1:, :]
        head = self.model.lm_head
        dtype = head.weight.dtype
        token_ids: list[int] = []
        eos_id = self.processor.tokenizer.eos_token_id
        next_start = start + embeds.shape[0]
        for offset in range(max_new_tokens):
            token_id = int(head(hidden.to(dtype))[0, -1].argmax().item())
            if token_id == eos_id:
                break
            token_ids.append(token_id)
            token = torch.tensor([[token_id]], device=self.device)
            output = self.lm(
                inputs_embeds=self.lm.embed_tokens(token),
                position_ids=self._text_positions(1, start=next_start + offset),
                past_key_values=cache,
                use_cache=True,
                output_hidden_states=False,
            )
            cache = output.past_key_values
            hidden = output.last_hidden_state[:, -1:, :]
        return self.processor.tokenizer.decode(token_ids, skip_special_tokens=True).strip()

    @torch.no_grad()
    def generate_on_kv_realloc(self, system_prompt: str, user_text: str, past_kv,
                               vis_cols, alpha: float, layers,
                               max_new_tokens: int, pos_cursor: int | None = None,
                               clone_cache: bool = True, vis_weight=None) -> str:
        """generate_on_kv with S8 attention reallocation active during decode.

        Uses a MANUAL eager decode loop (each step passes output_attentions=True,
        which forces the eager attention path so memory/reallocate's monkeypatch
        fires). Reallocation is applied at DECODE only (prefill is untouched),
        mirroring the isolation-validated diag_reallocation. Same finalizer prompt
        and MRoPE position handling as generate_on_kv, so the alpha=0 path matches
        the greedy baseline up to the eager/sdpa numerical difference.
        """
        from memory.reallocate import reallocating
        tok = self.processor.tokenizer

        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": user_text})
        try:
            prompt = self.processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            prompt = self.processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
        if prompt.endswith("<think>\n"):
            prompt = prompt + "</think>\n\n"

        input_ids = tok(prompt, return_tensors="pt",
                        add_special_tokens=False)["input_ids"].to(self.device)
        L = int(input_ids.shape[1])

        kv = copy.deepcopy(past_kv) if clone_cache else past_kv
        kv = self._kv_to_dynamic_cache(kv) if hasattr(self, "_kv_to_dynamic_cache") else kv
        past_len = self._kv_len(kv)
        if getattr(self, "mrope_pos", False) and pos_cursor is not None:
            pos = self._text_positions(L, start=pos_cursor)
            cursor = pos_cursor + L
        else:
            pos = self._make_position_ids(L, past_len=past_len)
            cursor = past_len + L

        embeds = self.lm.embed_tokens(input_ids)                 # [1, L, H]
        # prefill (no reallocation — decode-only, matches isolation)
        out = self.lm(inputs_embeds=embeds, position_ids=pos, past_key_values=kv,
                      use_cache=True, output_hidden_states=True)
        kv = out.past_key_values
        last = out.hidden_states[-1][:, -1:, :]

        lm_head = self.model.lm_head
        wd = lm_head.weight.dtype
        eos_id = tok.eos_token_id
        ids: list[int] = []
        vt = torch.as_tensor(vis_cols, dtype=torch.long, device=self.device)
        with reallocating(vt, alpha, layers, device=self.device, vis_weight=vis_weight):
            for t in range(max_new_tokens):
                nid = lm_head(last.to(wd))[0, -1].argmax()
                tid = int(nid.item())
                if tid == eos_id:
                    break
                ids.append(tid)
                o = self.lm(inputs_embeds=self.lm.embed_tokens(nid.view(1, 1)),
                            position_ids=self._text_positions(1, start=cursor + t),
                            past_key_values=kv, use_cache=True,
                            output_attentions=True, output_hidden_states=True)
                kv = o.past_key_values
                last = o.hidden_states[-1][:, -1:, :]
        return tok.decode(ids, skip_special_tokens=True).strip()

    # ── flexible per-entry vision prune (fires at EVERY vision entry) ───────────
    @torch.no_grad()
    def _prune_vision_entry(self, past_kv, cache_len: int, vis_cols, spans,
                            l2v_attn: dict, grids, images=None):
        """One-shot intra-pass prune of the JUST-ADDED vision block, by saliency.

        Runs at every vision entry when --kv_prune != none (the user's "always prune
        when vision enters" policy). The keep-mask decision goes through the shared
        `keep_mask_for_block` seam, so a cross-iteration RETENTION strategy (which
        ROIs survive long-term) can later wrap/override it without touching callers.

        Returns (past_kv, cache_len, spans, vis_cols, ndrop). Leaves l2v_attn/grids
        to the caller untouched (callers reuse the FULL saliency for ROI/zoom meta).
        """
        args = getattr(self, "_prune_args", None)
        mode = getattr(args, "kv_prune", "none") if args is not None else "none"
        scope = getattr(args, "prune_scope", "all") if args is not None else "all"
        n = (int(vis_cols.numel()) if hasattr(vis_cols, "numel") else len(vis_cols)) \
            if vis_cols is not None else 0
        # scope='initial' → only the Reasoner prefill prunes; appended crops kept whole
        if mode == "none" or scope != "all" or not l2v_attn or n == 0:
            return past_kv, cache_len, spans, vis_cols, 0
        import numpy as np
        from memory.prune import (keep_mask_for_block, apply_kv_prune,
                                  token_tissue_fraction)
        sms = int(getattr(self.model.config.vision_config, "spatial_merge_size", 2))

        # aggregate l2v over scoring layers (+steps,+heads) → per-token saliency
        mats = []
        for t in l2v_attn.values():
            a = t.float()
            while a.dim() > 1:
                a = a.mean(0)
            mats.append(a)
        if not mats:
            return past_kv, cache_len, spans, vis_cols, 0
        sal = torch.stack(mats, 0).mean(0).cpu().numpy()
        if sal.shape[0] != n:
            return past_kv, cache_len, spans, vis_cols, 0

        # roi_of_token + (safe) tissue from the patch grid
        roi, gh_gw = [], []
        if grids is not None:
            for i, (t, h, w) in enumerate(grids.tolist()):
                c = int(t * h * w // (sms * sms))
                roi += [i] * c
                gh_gw.append((int(h // sms), int(w // sms)))
        if len(roi) != n:
            roi, gh_gw = [0] * n, []
        roi = np.asarray(roi, dtype=np.int64)
        tissue = None
        if mode == "safe" and images is not None and gh_gw:
            from memory.saliency import RoiSpan
            rs, start = [], 0
            for i, (gh, gw) in enumerate(gh_gw):
                rs.append(RoiSpan(id=i, start=start, length=gh * gw, h=gh, w=gw))
                start += gh * gw
            imgs = images if isinstance(images, list) else [images]
            try:
                tissue = token_tissue_fraction(imgs, rs)
            except Exception as e:
                print(f"[VisionPrune] tissue fraction failed ({e}); safe→keep all")

        keep = keep_mask_for_block(
            mode, sal, roi, tissue=tissue, sink=None,
            b_span=int(getattr(args, "kv_prune_b_span", 32)),
            radius=int(getattr(args, "kv_prune_radius", 1)),
            tissue_th=float(getattr(args, "kv_prune_tissue_th", 0.1)),
            sink_frac=float(getattr(args, "kv_prune_sink_frac", 0.05)),
            random_seed=int(getattr(args, "prune_random_seed", 0)))
        ndrop = int((~keep).sum())
        if ndrop == 0:
            return past_kv, cache_len, spans, vis_cols, 0
        vc = vis_cols.tolist() if hasattr(vis_cols, "tolist") else list(vis_cols)
        pk, new_len, new_spans, new_vis = apply_kv_prune(
            past_kv, cache_len, vc, keep, spans, self.device)
        print(f"[VisionPrune] entry {mode}: dropped {ndrop}/{n} new-crop vis "
              f"→ KV {new_len} tokens")
        return pk, new_len, new_spans, new_vis, ndrop

    @torch.no_grad()
    def saliency_scan(
        self,
        image: Image.Image,
        prompt_text: str,
        m: int,
        l_mid_layers: list[int],
    ) -> dict:
        """Fresh (no prior KV) content-driven saliency over ONE image, for the
        Scanner's `--scanner_method saliency` path. Mirrors
        append_grounded_vision_and_latent but from an EMPTY KV, and also returns
        vis_hidden — so the Scanner's saliency→bbox mappers pick regions WITHOUT any
        coordinate decode. OPT-IN: only reached when --scanner_method saliency (default
        thumbnail_vl), so the default qwen path is byte-unchanged. Qwen3-VL/MRoPE;
        grids=image_grid_thw, Scanner reads vision_config.spatial_merge_size.

        Returns {grids, n_vis, vis_hidden, latent_trajectory, latent_to_vision_attn}
        (same keys the Scanner consumes; cf. backbone/huatuogpt.py::saliency_scan)."""
        img_id = getattr(self.model.config, "image_token_id", 151655)
        chunk_text = f"{prompt_text}<|vision_start|><|image_pad|><|vision_end|>"
        inputs = self.processor(text=[chunk_text], images=[image],
                                return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        ids = input_ids[0]
        L = int(ids.numel())
        pv  = inputs["pixel_values"].to(self.device, dtype=self.dtype)
        thw = inputs["image_grid_thw"].to(self.device)

        embeds = self.lm.embed_tokens(input_ids)
        img_feat = torch.cat(
            self.vlm.get_image_features(pv, thw, return_dict=True).pooler_output, dim=0)
        vis_mask = (ids == img_id)
        embeds[0, vis_mask] = img_feat.to(embeds.dtype)

        vis_idx_local = vis_mask.nonzero(as_tuple=True)[0]
        first_v = int(vis_idx_local[0]); n_v = int(vis_idx_local.numel())
        P = first_v
        Q = L - (first_v + n_v)
        pre = self._text_positions(P, start=0)
        vis_coords, vcur = self.vision_coords(thw, start=P)
        vis = vis_coords.unsqueeze(1)
        post = self._text_positions(Q, start=vcur)
        pos = torch.cat([pre, vis, post], dim=2)
        cursor = vcur + Q

        out = self.lm(inputs_embeds=embeds, position_ids=pos, use_cache=True,
                      output_attentions=True, output_hidden_states=True)
        past_kv = out.past_key_values
        last_hidden = out.hidden_states[-1][:, -1:, :]
        vis_hidden = out.hidden_states[-1][0, vis_idx_local, :].float().cpu()

        latent_trajectory: list[torch.Tensor] = []
        l2v_lists: dict[int, list[torch.Tensor]] = {li: [] for li in l_mid_layers}
        for step in range(m):
            le = self._apply_realign(last_hidden)
            o = self.lm(inputs_embeds=le,
                        position_ids=self._text_positions(1, cursor + step),
                        past_key_values=past_kv, use_cache=True,
                        output_attentions=True, output_hidden_states=True)
            past_kv = o.past_key_values
            last_hidden = o.hidden_states[-1][:, -1:, :]
            if os.environ.get("VLMAS_RETR", ""):
                # retrieval kill test: per-layer residual input at the latest
                # latent step (overwritten each step -> the last step remains)
                self._retr_hidden = [h[0, -1].detach().float().clone() for h in o.hidden_states[:-1]]
            latent_trajectory.append(le.squeeze().cpu())
            for li in l_mid_layers:
                a = o.attentions[li]
                if a is not None:
                    l2v_lists[li].append(a[0][:, 0, 0:L][:, vis_idx_local].cpu())
        latent_to_vision_attn = {li: torch.stack(v, 0) for li, v in l2v_lists.items() if v}

        return {
            "grids":                 thw.cpu(),
            "n_vis":                 n_v,
            "vis_hidden":            vis_hidden,
            "latent_trajectory":     latent_trajectory,
            "latent_to_vision_attn": latent_to_vision_attn,
        }

    @torch.no_grad()
    def grounded_prefill_and_latent_on_kv(
        self,
        past_kv,
        past_pos_cursor: int,
        images: list,
        system_prompt: str | None,
        user_text: str,
        m: int,
        l_mid_layers: list[int],
        enable_thinking: bool = False,
        want_attn: bool = True,
        want_vis_hidden: bool = False,
        role_targets: list | None = None,
        bg_targets: list | None = None,     # --saliency_bg_correct: template rows B strings
        continuation_user_turn: bool = False,
    ) -> dict:
        """KV-CONTINUATION counterpart of grounded_prefill_and_latent (--thread_initial_scan).

        Prefills the SAME [system, crops, user] chat template, but ON TOP OF an
        existing `past_kv` (the initial Scanner's thumbnail-latent, vision-dropped),
        with MRoPE positions continued from `past_pos_cursor`. Returns the same dict
        shape so the Reasoner is unchanged downstream.

        Built by combining grounded_prefill_and_latent's input/saliency logic with
        append_grounded_vision_and_latent's proven continuation (self.lm + past_kv +
        offset positions). The ORIGINAL grounded_prefill_and_latent is untouched, so
        the default (--thread_initial_scan off) path is byte-identical.

        Intra-pass streaming prune (--kv_prune + --prune_every) IS replicated here
        (Plan B): the PruneController runs over ABSOLUTE vision cols (past_len+vis_idx)
        so apply_kv_prune drops only vision and keeps the base prefix. The prefill
        safe-prune is still NOT ported (asserted off). full_attn_kv stays None.

        ⚠️ UNVERIFIED (GPU): past_len attention-column offsets, MRoPE continuation, AND
        the streaming eviction on this path all need a GPU smoke (byte-identical when
        prune off + functional when on). Model-free units can't exercise absolute-col
        eviction. GAP2: cur_spans omits the base prefix → apply_kv_prune RLE falls back
        to stale spans (harmless unless combined with --carry_kv_mode).
        """
        _pa = getattr(self, "_prune_args", None)
        # intra-pass streaming (--kv_prune + --prune_every) IS ported below (Plan B);
        # only the prefill-time safe-prune is still unsupported on this threaded path.
        assert getattr(_pa, "prune_prefill", "none") == "none", \
            "grounded_prefill_and_latent_on_kv does not support prune_prefill yet"

        img_id = getattr(self.model.config, "image_token_id", 151655)
        past_len = self._kv_len(past_kv)

        # ── same chat-template prompt as grounded_prefill_and_latent ──────────────
        content = [{"type": "image", "image": im} for im in images]
        content.append({
            "type": "text",
            "text": (
                self._continuation_user_text(system_prompt, user_text)
                if continuation_user_turn else user_text
            ),
        })
        msgs = (
            [] if continuation_user_turn else
            ([{"role": "system", "content": system_prompt}] if system_prompt else [])
        )
        msgs.append({"role": "user", "content": content})
        text = self.processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        _tt = os.environ.get("VLMAS_THINK_TAGS", "").strip()
        if _tt == "strip" and text.endswith("<think>\n"):
            text = text[: -len("<think>\n")]
        elif _tt == "open":
            pass
        elif not self.instruct and not enable_thinking:
            text = text + "\n</think>\n\n"
        inputs = self.processor(text=[text], images=images, return_tensors="pt").to(self.device)
        ids = inputs["input_ids"][0]
        is_vis = (ids == img_id)
        vis_idx = is_vis.nonzero(as_tuple=True)[0]          # image-token cols in THIS chunk
        last = int(vis_idx[-1]); nV = int(vis_idx.numel())
        L = int(ids.numel())

        # ── grounded embeds via manual scatter (self.lm path, like append) ────────
        embeds = self.lm.embed_tokens(inputs["input_ids"])
        pv  = inputs["pixel_values"].to(self.device, dtype=self.dtype)
        thw = inputs["image_grid_thw"].to(self.device)
        # SEC-LD / NOVA read the un-cast fp32 pixels (bf16 pv moves a gray level by <= 0.25); None when off.
        self._secld_pixel_values_fp32 = (
            inputs["pixel_values"].detach().to(torch.float32)
            if (os.environ.get("VLMAS_SECLD", "").strip() == "1"
                or os.environ.get("VLMAS_C1_SELECT", "").strip() == "nova") else None
        )

        # ── MRoPE positions for the chunk, CONTINUED from past_pos_cursor ─────────
        # compute_3d_position_ids builds the multi-image layout from 0; offsetting all
        # 3 dims by past_pos_cursor slots the chunk right after the base KV.
        pos = self.vlm.compute_3d_position_ids(
            input_ids=inputs["input_ids"], inputs_embeds=None,
            image_grid_thw=thw, mm_token_type_ids=inputs.get("mm_token_type_ids"))
        pos = pos + int(past_pos_cursor)
        cursor = int(pos.max().item()) + 1

        # V4 observes every requested image in the early vision blocks, then
        # performs one hierarchy-constrained merge-group selection.  Retained
        # groups alone traverse the deep vision blocks and decoder prefill.
        hierarchy_prefill = bool(
            _pa is not None
            and getattr(_pa, "prune_prefill_hierarchy_aware", False)
            and getattr(self, "_prune_morphology_enabled", False)
        )
        # Sender-relay experiment: never let a previous role's survivor scores
        # leak into this role's bookkeeping.
        self._prefill_prune_survivor_scores = None
        self._prefill_prune_survivor_group_ids = None
        self._prefill_prune_survivor_image_ids = None
        # C1 #14 Question-Agnostic Salience-Coverage Log-Det Selection (memory/qasc_select.py,
        # env VLMAS_C1_QASC=1, Reasoner stage): full vision tower, joint x5/x20 log-det greedy on
        # merged tokens; retained tokens alone enter the decoder prefill. Off = unchanged.
        _qasc_prefill = bool(
            os.environ.get("VLMAS_C1_QASC", "") == "1"
            and getattr(self, "_prune_morphology_enabled", False)
        )
        prefill_spatial_indices = None
        if (hierarchy_prefill or _qasc_prefill) and nV > 0:
            if _qasc_prefill:
                from memory import qasc_select as _qasc
            img_feat, keep_visual = _qasc.select_vision_features(self, pv, thw) if _qasc_prefill else self._hierarchy_pruned_vision_features(
                pv,
                thw,
                parent_indices=tuple(
                    int(value)
                    for value in getattr(self, "_prune_image_parent_indices", ())
                ),
                keep_ratio=getattr(_pa, "prune_keep_ratio", None),
                min_tokens_per_image=int(getattr(_pa, "kv_prune_b_span", 16)),
                significance_sigma=getattr(
                    _pa, "prune_prefill_significance_sigma", None
                ),
                observe_blocks=int(getattr(_pa, "prune_prefill_observe_blocks", 4)),
                spectral_energy_ratio=getattr(
                    _pa, "prune_spectral_energy_ratio", None
                ),
                spectral_min_keep_ratio=float(
                    getattr(_pa, "prune_spectral_min_keep_ratio", 0.25)
                ),
                spectral_max_keep_ratio=float(
                    getattr(_pa, "prune_spectral_max_keep_ratio", 0.50)
                ),
            )
            spatial_segments = []
            spatial_offset = 0
            _sms = int(getattr(self.model.config.vision_config, "spatial_merge_size", 2))
            for temporal, height, width in thw.tolist():
                frame_tokens = (int(height) // _sms) * (int(width) // _sms)
                for _ in range(int(temporal)):
                    frame_keep = keep_visual[
                        spatial_offset:spatial_offset + frame_tokens
                    ]
                    spatial_segments.append(
                        frame_keep.nonzero(as_tuple=True)[0].detach()
                    )
                    spatial_offset += frame_tokens
            if spatial_offset != int(keep_visual.numel()):
                raise ValueError("prefill visual mask does not match image grids")
            prefill_spatial_indices = tuple(spatial_segments)
            sequence_keep = ~is_vis
            sequence_keep[vis_idx[keep_visual]] = True
            original_visual_count = nV
            embeds = embeds[:, sequence_keep]
            pos = pos[:, :, sequence_keep]
            ids = ids[sequence_keep]
            is_vis = is_vis[sequence_keep]
            vis_idx = is_vis.nonzero(as_tuple=True)[0]
            nV = int(vis_idx.numel())
            L = int(ids.numel())
            last = int(vis_idx[-1])
            embeds[0, is_vis] = img_feat.to(embeds.dtype)
            print(
                "[PruneVision/v4] hierarchy-constrained visual tokens: "
                f"{nV}/{original_visual_count} across {len(images)} images; "
                f"all tokens observed for {getattr(_pa, 'prune_prefill_observe_blocks', 4)}/24 vision blocks"
            )
        else:
            img_feat = torch.cat(
                self.vlm.get_image_features(pv, thw, return_dict=True).pooler_output,
                dim=0,
            )
            embeds[0, is_vis] = img_feat.to(embeds.dtype)
        # Position-consistency test (env VLMAS_POS_GAP=<G>; unset/0 = untouched): leave G empty MRoPE
        # positions between the Reasoner's last visual token and everything after it, so the prompt
        # tail, the latent steps and the Answerer are all COMPUTED with the visual block (and everything
        # before it) G positions farther away. Consistent counterpart of the post-hoc canon_diag arm
        # gap<G>, which re-rotates the cached keys of columns 0..last-visual by -G instead.
        _pos_gap = int(os.environ.get("VLMAS_POS_GAP", "0") or 0)
        if _pos_gap and getattr(self, "_prune_morphology_enabled", False) and nV > 0:
            pos = pos.clone()
            pos[:, :, last + 1:] += _pos_gap
            cursor = int(pos.max().item()) + 1
            print(f"[PosGap] reasoner: +{_pos_gap} MRoPE positions after visual col {past_len + last} "
                  f"({L - last - 1} tail tokens), cursor {cursor}", flush=True)
        # VISTOK_VIZ (opt-in): vision embeddings exactly as scattered into the
        # Reasoner prefill (post ViT+merger), with per-image grids.
        _vistok_dir = getattr(self, "_vistok_dump_dir", None)
        if _vistok_dir and getattr(self, "_prune_morphology_enabled", False):
            torch.save({
                "img_feat": img_feat.detach().float().cpu(),
                "grid_thw": thw.detach().cpu(),
                "n_images": len(images),
                "n_vis": int(nV),
            }, os.path.join(_vistok_dir, "reasoner_vis.pt"))
            self._vistok_dump_dir = None
        # retrieval kill tests: keep the encoder-side embeddings of the visual
        # tokens as scattered (same order as vis_idx) for feature-similarity
        # controls (memory/vsim_dump.py). Bookkeeping only.
        if os.environ.get("VLMAS_RETR", "").strip():
            self._retr_img_feat = img_feat.detach().float().cpu()

        dynamic_prune = bool(
            _pa is not None and getattr(_pa, "prune_dynamic_ratio", False)
        )
        event_probes_only = bool(
            _pa is not None and getattr(_pa, "prune_event_probes_only", False)
        )
        sparsevlm_active = bool(
            getattr(getattr(self, "_prune_args", None), "prune_query_adaptive", False)
        )
        atp_sap_active = bool(
            getattr(getattr(self, "_prune_args", None), "prune_atp_sap", False)
            and getattr(self, "_prune_morphology_enabled", False)
        )
        atp_all_layers = bool(
            atp_sap_active
            and getattr(getattr(self, "_prune_args", None), "prune_atp_all_layers", False)
        )
        visual_probe_layers = (
            list(range(len(self.lm.layers))) if atp_all_layers else l_mid_layers
        )
        visual_boundary_pruning = sparsevlm_active or atp_sap_active
        capture_prefill_attn = (
            want_attn and not dynamic_prune and not event_probes_only
            and not visual_boundary_pruning
        )
        # §3.3 task-weighting (VLMAS_WSI_CONSOL_SC=1): hook-based question->vision
        # capture at prefill — memory-safe s_c for variants that never materialize
        # full attention (base / hierarchy-aware pruning).
        _consol_sc_on = (
            (
                os.environ.get("VLMAS_WSI_CONSOL", "") == "1"
                and os.environ.get("VLMAS_WSI_CONSOL_SC", "") == "1"
            )
            or visual_boundary_pruning
        ) and (
            getattr(self, "_prune_morphology_enabled", False)
            and nV > 0
            and not capture_prefill_attn
        )
        q2v_rows = (
            (~is_vis).nonzero(as_tuple=True)[0]
            if visual_boundary_pruning
            else torch.arange(last + 1, L, device=self.device)
        )
        _sc_context = (
            self._capture_prefill_q2v(
                visual_probe_layers, q2v_rows, past_len + vis_idx)
            if _consol_sc_on else contextlib.nullcontext(None)
        )
        _self_context = (
            self._capture_prefill_visual_self_score(
                visual_probe_layers, vis_idx, past_len + vis_idx, past_len=past_len
            )
            if atp_sap_active else contextlib.nullcontext(None)
        )
        from memory import aavm_runtime as _aavm
        from memory import aiva as _aiva
        # AIVA (env VLMAS_AIVA=1, off = no hooks): count-corrected group access
        # for every attention call of this role -- prefill rows AND latent
        # steps. Columns are re-read per call, so the consolidation boundary's
        # renumbering below is picked up automatically.
        self._aiva_visual_cols = (past_len + vis_idx).detach()
        _aiva_handles = _aiva.install_aiva(self)
        # Access decomposition (env VLMAS_ACCESS_DIAG=<dir>): read the logits
        # the latent steps already produce and split M_V into compatibility and
        # cardinality. No extra forward. Off = no hooks.
        from memory import access_diag as _accdiag
        _acc_handles = (
            _accdiag.capture_access(self) if _accdiag.enabled() else None)
        if _acc_handles is not None:
            _acc_steps = _acc_handles.__enter__()
        with (
            _sc_context as _sc_q2v,
            _self_context as _self_scores,
            _aavm.capture_visual_raw_keys(self, past_len + vis_idx) as _aavm_capture,
        ):
            out = self.lm(inputs_embeds=embeds, position_ids=pos, past_key_values=past_kv,
                          use_cache=True, output_attentions=capture_prefill_attn,
                          output_hidden_states=True)
        prefill_input_embeddings = out.hidden_states[0][0].detach().cpu()
        past_kv = out.past_key_values
        last_hidden = out.hidden_states[-1][:, -1:, :]
        vis_hidden = (out.hidden_states[-1][0, vis_idx, :].float().cpu()
                      if want_vis_hidden else None)

        seq_prefill = past_len + L
        spans: list[tuple[str, int]] = []                    # P/V runs (this chunk only)
        run_t, run_n = None, 0
        for k in range(L):
            t = "V" if bool(is_vis[k]) else "P"
            if t == run_t:
                run_n += 1
            else:
                if run_t is not None:
                    spans.append((run_t, run_n))
                run_t, run_n = t, 1
        spans.append((run_t, run_n))

        # ── q2v: post-image text rows (query dim 0..L-1) → vision (key dim + past_len)
        query_to_vision_attn: dict[int, torch.Tensor] = {}
        query_to_vision_sink:   dict[int, torch.Tensor] = {}
        query_to_vision_rowsum: dict[int, torch.Tensor] = {}
        if capture_prefill_attn:
            q_rows = torch.arange(last + 1, L, device=self.device)
            vkeys  = (past_len + vis_idx)
            for li in l_mid_layers:
                a = out.attentions[li]
                if a is not None:
                    query_to_vision_attn[li]   = a[0][:, q_rows][:, :, vkeys].cpu()
                    query_to_vision_sink[li]   = a[0][:, q_rows, 0].cpu()
                    query_to_vision_rowsum[li] = a[0][:, q_rows, :].sum(-1).cpu()
        elif _sc_q2v:
            query_to_vision_attn = _sc_q2v
        visual_self_score = _self_scores or None

        qq_role_spans = None
        if visual_boundary_pruning and role_targets:
            qq_role_spans = self._locate_role_spans(
                ids[q2v_rows], 0, int(q2v_rows.numel()), role_targets)
        elif capture_prefill_attn and role_targets:
            # role spans are tail-relative to (last+1 .. L); _locate_role_spans works
            # on the chunk ids exactly as in the fresh path.
            qq_role_spans = self._locate_role_spans(ids, last + 1, L, role_targets)
        bg_role_spans = None
        if capture_prefill_attn and bg_targets:
            bg_role_spans = self._locate_role_spans(ids, last + 1, L, bg_targets)

        # ── Pathology-Structured Visual Memory Consolidation (§3.3, env-gated) ──
        # Absolute cols = past_len + chunk-relative vis_idx; the base prefix is
        # never touched, so surviving absolute cols map back by subtracting
        # past_len.
        self._pcsi_capture_query(ids, last, prefill_input_embeddings)
        past_kv, seq_prefill, spans, _consol_keep, _consol_vis = (
            self._wsi_consolidate_boundary(
                past_kv, seq_prefill, past_len + vis_idx,
                prefill_input_embeddings[vis_idx.cpu()],
                (_sc_q2v if _sc_q2v else query_to_vision_attn), thw, spans,
                qq_role_spans, visual_self_score=visual_self_score,
                spatial_token_indices=prefill_spatial_indices))
        if _consol_keep is not None and not bool(_consol_keep.all()):
            vis_idx = _consol_vis - past_len
            nV = int(vis_idx.numel())
            _consol_cpu = _consol_keep.cpu()
            if vis_hidden is not None:
                vis_hidden = vis_hidden[_consol_cpu]
            _consol_idx = _consol_cpu.nonzero(as_tuple=True)[0]
            query_to_vision_attn = {
                li: a[:, :, _consol_idx]
                for li, a in query_to_vision_attn.items()
            }
        self._record_visual_meta(thw, _consol_keep, past_len + vis_idx)

        self._aiva_visual_cols = (past_len + vis_idx).detach()

        # Single-Pass Dual-Address Visual Routing (env VLMAS_KV_ROUTE=1):
        # remember the retained visual columns, their native 3-D MRoPE
        # coordinates and the per-observation page sizes at the Reasoner
        # boundary; the terminal Answerer resolves each page's address from
        # its own boundary query (memory/route_address.py). Base layout only.
        # (Restored 0910 22:40 -- the block had been dropped by a concurrent
        # edit; without it ROUTE_MODE=canonical logs "SKIP: no visual
        # bookkeeping" and silently runs the base layout.)
        # Also recorded (bookkeeping only, no routing) when VLMAS_RETR is set so
        # the retrieval kill tests can run on the base layout.
        if ((os.environ.get("VLMAS_KV_ROUTE", "") == "1" or os.environ.get("VLMAS_RETR", "").strip())
                and getattr(self, "_prune_morphology_enabled", False)):
            self._route_vis_cols = (past_len + vis_idx).detach().clone()
            self._route_vis_pos = pos[:, 0, vis_idx].detach().clone()
            self._route_pages = list(
                getattr(self, "_reassembly_meta", (None,))[0]
                or self._aavm_token_counts(inputs.get("image_grid_thw")))

        # Level 3/4 probe: substitute a frozen donor slide's visual K/V so the
        # latent steps reason over another slide while everything else is held
        # fixed. Reasoner stage only; off = untouched.
        from memory import rpath_probe as _rpath
        if _rpath.wrong_visual_enabled() and getattr(
                self, "_prune_morphology_enabled", False):
            _rpath_note = _rpath.swap_visual_for_donor(
                self, past_kv, past_len + vis_idx)
            print(f"[RPath] wrong_visual{_rpath_note}", flush=True)

        # ── AAVM: re-address retained crops with their acquisition event ──
        # After C1, before the latent steps: the Reasoner is what integrates the
        # re-addressed evidence. Reasoner stage only (crops, not control images).
        if _aavm.enabled() and getattr(self, "_prune_morphology_enabled", False):
            _aavm.apply_aavm(
                self, past_kv,
                capture=_aavm_capture,
                keep_mask=_consol_keep,
                kept_counts=list(
                    getattr(self, "_reassembly_meta", (None,))[0]
                    or self._aavm_token_counts(inputs.get("image_grid_thw"))),
                crop_queries=list(getattr(self, "_aavm_event_queries", ()) or []),
                cache_columns=past_len + vis_idx,
            )

        # KV re-staging support: remember the visual tokens' original MRoPE
        # positions (aligned with the absolute vision columns) for the
        # terminal-time re-rotation. Env-gated; tiny tensor.
        # LADR (C2 12.8, memory/ladr.py) reads the same provenance to build its role-local frame.
        if os.environ.get("VLMAS_KV_RESTAGE", "") == "1" or os.environ.get("VLMAS_LADR", "").strip():
            self._restage_vis_positions = pos[:, 0, vis_idx].detach().clone()

        # ── restage-on-boundary / parking (pruning·binding ON TOP of restage) ──
        # VLMAS_KV_RESTAGE_AT=boundary: move visual to recency BEFORE the latent
        #   steps, so z_t are generated with visual at recency (binding substrate);
        #   the terminal move is then skipped.
        # VLMAS_KV_PARK=1: bank the visual K/V and REMOVE it from the cache for
        #   the whole latent phase (shorter cache every step), then the terminal
        #   restage restores the bank at recency — temporal pruning, zero
        #   information loss.
        _restage_at = os.environ.get("VLMAS_KV_RESTAGE_AT", "terminal")
        _park = os.environ.get("VLMAS_KV_PARK", "") == "1"
        # VLMAS_KV_PULSE=1: per-handoff evidence rebinding — boundary restage
        # here (consumer's latent steps see visual at recency), the engine
        # RE-PARKS the block after the agent finishes, and the terminal restore
        # rebinds it again. Every consumer reads evidence at its recency window.
        _pulse = os.environ.get("VLMAS_KV_PULSE", "") == "1"
        if (
            os.environ.get("VLMAS_KV_RESTAGE", "") == "1"
            and getattr(self, "_prune_morphology_enabled", False)
            and nV > 0
            and (_park or _pulse or _restage_at == "boundary")
        ):
            import numpy as _np_park

            from memory.prune import apply_kv_prune as _park_prune
            _abs_vis = (past_len + vis_idx).to(self.device)
            _old_pos = pos[:, 0, vis_idx].detach().clone()
            if _park and not _pulse:
                self._park_bank = {
                    "k": [layer.keys[:, :, _abs_vis, :].detach().clone()
                          for layer in past_kv.layers],
                    "v": [layer.values[:, :, _abs_vis, :].detach().clone()
                          for layer in past_kv.layers],
                    "pos": _old_pos,
                }
                keep_none = _np_park.zeros(int(_abs_vis.numel()), dtype=bool)
                past_kv, seq_prefill, spans, _ = _park_prune(
                    past_kv, seq_prefill, _abs_vis.tolist(), keep_none, spans,
                    self.device)
                vis_idx = torch.empty(0, dtype=torch.long, device=self.device)
                nV = 0
                # C2-alone support: reassembly ordering needs block metadata
                # even when C1 consolidation is OFF — rebuild it from the raw
                # grids so full-KV runs can still reorder at the restore.
                sms_park = int(getattr(
                    self.model.config.vision_config, "spatial_merge_size", 2))
                counts_park = [
                    int(t) * (int(h) // sms_park) * (int(w) // sms_park)
                    for t, h, w in thw.tolist()]
                if sum(counts_park) == int(_old_pos.shape[-1]):
                    self._reassembly_meta = (
                        counts_park,
                        list(getattr(self, "_prune_image_parent_indices", ())
                             or [-1] * len(counts_park)),
                        list(getattr(self, "_prune_image_magnifications", ())
                             or [0] * len(counts_park)),
                    )
                print(f"[KVPark] parked {int(_old_pos.shape[-1])} visual cols; "
                      f"latent-phase cache {seq_prefill}", flush=True)
            else:
                _new_cursor, _moved = self.restage_visual_kv(
                    past_kv, _abs_vis, _old_pos, cursor)
                if _moved:
                    cursor = _new_cursor
                    vis_idx = (torch.arange(
                        seq_prefill - _moved, seq_prefill, device=self.device)
                        - past_len)
                    self._restage_vis_positions = (
                        torch.arange(cursor - _moved, cursor, device=self.device)
                        .unsqueeze(0).expand(3, -1).clone())
                    # absolute cols of the restaged block (stable: nothing
                    # before them is ever removed) — for the terminal-drop
                    # carrier probe.
                    self._early_vis_cols = torch.arange(
                        seq_prefill - _moved, seq_prefill, device=self.device)
                    print("[KVRestage@boundary] visual at tail during latent "
                          "steps", flush=True)

        # ── latent steps (continue from cursor), l2v over the NEW vision cols ──────
        latent_trajectory: list[torch.Tensor] = []
        l2v_lists:        dict[int, list[torch.Tensor]] = {li: [] for li in l_mid_layers}
        l2v_sink_lists:   dict[int, list[torch.Tensor]] = {li: [] for li in l_mid_layers}
        l2v_rowsum_lists: dict[int, list[torch.Tensor]] = {li: [] for li in l_mid_layers}

        # ── intra-pass streaming prune (Plan B: threaded twin of the fresh path) ──
        # Same PruneController policy, but the vision cols here are ABSOLUTE
        # (past_len + vis_idx) into a cache that already holds the base (Scanner
        # thumbnail-latent) prefix. apply_kv_prune drops only the vision cols so the
        # base + latent tokens survive; pos_cursor is untouched (rotary phase baked).
        from memory.prune import build_prune_controller, apply_kv_prune
        sms = int(getattr(self.model.config.vision_config, "spatial_merge_size", 2))
        # Continuation counterpart of the fresh-role guard above: a role with
        # attention disabled must not enter the controller's saliency path.
        ctrl = build_prune_controller(_pa) if want_attn else None
        one_shot_pruning = bool(
            ctrl is not None and getattr(_pa, "prune_one_shot", False)
        )
        event_probes_only = bool(
            ctrl is not None and getattr(_pa, "prune_event_probes_only", False)
        )
        lightweight_attention = bool(
            ctrl is not None
            and (
                getattr(_pa, "prune_dynamic_ratio", False)
                or event_probes_only
                # ATP-SAP's prefill score hooks already provide the adaptive
                # pruning signal; latent saliency can use the lightweight
                # selected-row capture while the model remains on SDPA.
                or getattr(_pa, "prune_atp_sap", False)
            )
        )
        stable_streaming = bool(
            ctrl is not None and getattr(_pa, "prune_stable_streaming", False)
        )
        stable_visual_only_scores = bool(
            ctrl is not None
            and getattr(_pa, "prune_stable_visual_only_scores", False)
        )
        if ctrl is not None:
            ctrl.configure_for_latent_steps(m)
            ctrl.start(grids=thw, vis_idx=vis_idx, sms=sms, images=images,
                       q2v=query_to_vision_attn, l_mid_layers=l_mid_layers,
                       role_spans=qq_role_spans, bg_role_spans=bg_role_spans,
                       image_magnifications=getattr(self, "_prune_image_magnifications", ()),
                       image_parent_indices=getattr(self, "_prune_image_parent_indices", ()),
                       morphology_enabled=getattr(self, "_prune_morphology_enabled", False))
        cur_vis   = (past_len + vis_idx)   # ABSOLUTE vision cols into the cache
        cur_len   = seq_prefill            # FULL cache length (base + this chunk)
        cur_spans = list(spans)            # chunk-only spans (base prefix omitted; GAP2)
        evicted   = False
        # Hook probe for reallocation; pruning controller provides the fallback.
        probe_mass_steps: list[float] = []

        # Efficient Role-Support Sensitivity (memory/rss_diag.py, env VLMAS_RSS): closed-form
        # support-deletion proxy at the last latent step + state for the Answerer-side LOO.
        # Off = no-op.
        _rss = None
        if (os.environ.get("VLMAS_RSS", "").strip()
                or os.environ.get("VLMAS_KV_RESTAGE_ORDER", "").strip()
                in ("sensitivity", "sensitivity_rev", "mass", "size", "page_shuffle")) \
                and getattr(self, "_prune_morphology_enabled", False):
            from memory import rss_diag as _rssd
            _rss = _rssd.begin(self, cur_vis=cur_vis, cursor=cursor, cur_len=cur_len,
                               last_hidden=last_hidden, m=m, thw=thw)

        for step in range(m):
            if _rss is not None:
                _rss.before_step(step)
            le = self._apply_realign(last_hidden)
            from vision_text_mas.latent_hybrid_variants import latent_reallocation
            capture_context = (
                self._capture_selected_latent_attention(
                    layer_indices=l_mid_layers,
                    vision_columns=cur_vis,
                    visual_only_scores=stable_visual_only_scores,
                )
                if lightweight_attention and (
                    (stable_streaming and ctrl.should_observe(step))
                    or
                    (one_shot_pruning and step == 0)
                    or (not one_shot_pruning and (
                        not event_probes_only or ctrl.is_eviction_step(step)
                    ))
                )
                else contextlib.nullcontext({})
            )
            with capture_context as selected_attention:
                carried_reallocation_vis = getattr(
                    self, "_reallocation_carried_vision_columns", None
                )
                reallocation_vis = cur_vis
                if isinstance(carried_reallocation_vis, torch.Tensor) and carried_reallocation_vis.numel():
                    reallocation_vis = torch.unique(torch.cat((
                        carried_reallocation_vis.to(device=self.device, dtype=torch.long),
                        cur_vis,
                    )), sorted=True)
                reallocation_context = (
                    latent_reallocation(self, reallocation_vis)
                    if ctrl is None or ctrl.allow_reallocation(step)
                    else contextlib.nullcontext(None)
                )
                with reallocation_context as realloc_probe:
                    visual_bind_context = (
                        self._visual_bind_step(past_kv, cur_vis, cur_len)
                        if step == m - 1 else contextlib.nullcontext(False)
                    )
                    with visual_bind_context, self._visual_cut_ctx(cur_vis):
                        o = self.lm(
                            inputs_embeds=le,
                            position_ids=self._text_positions(1, cursor + step),
                            past_key_values=past_kv,
                            use_cache=True,
                            output_attentions=(
                                want_attn
                                and not lightweight_attention
                                and (not event_probes_only or ctrl.is_eviction_step(step))
                            ),
                            output_hidden_states=True,
                        )
            realloc_mass = getattr(realloc_probe, "mean_mass", None)
            if realloc_mass is not None:
                probe_mass_steps.append(float(realloc_mass))
                reallocation_config = getattr(self, "_latent_reallocation", None)
                if (
                    reallocation_config is not None
                    and reallocation_config.adaptive_within_role
                ):
                    self._latent_reallocation_probe = float(realloc_mass)
            past_kv = o.past_key_values
            last_hidden = o.hidden_states[-1][:, -1:, :]
            if _rss is not None:
                _rss.after_step(step, o, cur_vis)
            if os.environ.get("VLMAS_RETR", ""):
                # retrieval kill test: per-layer residual input at the latest
                # latent step (overwritten each step -> the last step remains)
                self._retr_hidden = [h[0, -1].detach().float().clone() for h in o.hidden_states[:-1]]
            latent_trajectory.append(le.squeeze().cpu())
            cur_len += 1                                 # GAP1: track full cache length
            if cur_spans and cur_spans[-1][0] == "L":    # GAP1: extend the latent run
                cur_spans[-1] = ("L", cur_spans[-1][1] + 1)
            else:
                cur_spans.append(("L", 1))
            # collect l2v only before eviction starts (post-evict cols misalign to stack)
            if (
                want_attn
                and not lightweight_attention
                and (not event_probes_only or ctrl.is_eviction_step(step))
                and (ctrl is None or not ctrl.evict)
            ):
                for li in l_mid_layers:
                    a = o.attentions[li]
                    if a is not None:
                        l2v_lists[li].append(a[0][:, 0, cur_vis].cpu())
                        l2v_sink_lists[li].append(a[0][:, 0, 0].cpu())
                        l2v_rowsum_lists[li].append(a[0][:, 0, :].sum(-1).cpu())
            # ── prune controller hook (observe head-resolved + maybe evict) ─────────
            if ctrl is not None and (
                not event_probes_only or ctrl.is_eviction_step(step)
            ):
                # keep the head axis [H, nV] so the controller reduces it per
                # --stream_saliency_reduce (mean/max/contrast); cur_vis is ABSOLUTE.
                mats = (
                    [selected_attention[li] for li in l_mid_layers if li in selected_attention]
                    if lightweight_attention else [
                        o.attentions[li][0][:, 0, cur_vis].float()
                        for li in l_mid_layers if o.attentions[li] is not None
                    ]
                )
                if mats:
                    step_l2v = torch.stack(mats, 0).mean(0)
                    if stable_streaming:
                        ctrl.observe_gpu(step, step_l2v)
                    elif getattr(ctrl, "reduce_head_first", False):
                        import numpy as _np
                        # head-reduce per layer, then layer-mean → mean_l(reduce_h)
                        # (matches roi_saliency Eq.(reason-attn) order); observe passes
                        # the 1-D vector through unchanged.
                        evec = _np.stack(
                            [ctrl._reduce_heads(m.cpu().numpy()) for m in mats], 0
                        ).mean(0)
                        ctrl.observe(step, evec)
                    else:
                        ctrl.observe(step, step_l2v.cpu().numpy())   # byte-identical path
                    # ALWAYS from step_l2v → reallocation probe is flag-invariant.
                    if realloc_mass is None and not stable_streaming:
                        probe_mass_steps.append(float(step_l2v.sum(-1).mean()))
                if ctrl.evict:
                    keep = (
                        ctrl.maybe_keep_mask_gpu(step)
                        if stable_streaming
                        else ctrl.maybe_keep_mask(step)
                    )
                    if keep is not None and not keep.all():
                        keep_for_cache = (
                            keep.detach().cpu().numpy()
                            if stable_streaming
                            else keep
                        )
                        past_kv, cur_len, cur_spans, new_vis = apply_kv_prune(
                            past_kv, cur_len, cur_vis.tolist(), keep_for_cache, cur_spans, self.device,
                            in_place=True)
                        cur_vis = torch.tensor(new_vis, device=self.device, dtype=torch.long)
                        if stable_streaming:
                            ctrl.commit_gpu(keep)
                        else:
                            ctrl.commit(keep)
                        evicted = True
                        print(f"[PruneCtrl/thread] step {step}: dropped "
                              f"{int((~keep).sum())}/{len(keep)} vis → KV {cur_len} tokens")

        # cur_spans already carries the ("L", m) run (extended per-step above).
        latent_to_vision_attn    = {li: torch.stack(v, 0) for li, v in l2v_lists.items() if v}
        latent_to_vision_sink    = {li: torch.stack(v, 0) for li, v in l2v_sink_lists.items() if v}
        latent_to_vision_rowsum  = {li: torch.stack(v, 0) for li, v in l2v_rowsum_lists.items() if v}
        transport_embeddings = _transport_embeddings(
            prefill_input_embeddings,
            latent_trajectory,
        )

        # streaming eviction shrank the KV → per-token q2v/l2v are stale, null them;
        # --stream_recompute_saliency rebuilds a survivor-space saliency (+ regidx via
        # pipeline) so registry / ground_cos / joint survive (mirror of the fresh path).
        stream_saliency_spans, stream_saliency_grids = [], {}
        if evicted:
            query_to_vision_attn = {}
            latent_to_vision_attn = {}
            if ctrl is not None and getattr(_pa, "stream_recompute_saliency", False):
                stream_saliency_spans, stream_saliency_grids = ctrl.final_saliency()

        _aiva.remove_aiva(_aiva_handles)
        if getattr(self, "_prune_morphology_enabled", False):
            _rpath.dump_latent_states(self, latent_trajectory)
        if _acc_handles is not None:
            _acc_handles.__exit__(None, None, None)

        # Probe W: dump the final latent hidden (z_m) per case, sequential index.
        if os.environ.get("VLMAS_DUMP_ZM", ""):
            _dump_dir = os.environ["VLMAS_DUMP_ZM"]
            os.makedirs(_dump_dir, exist_ok=True)
            _zm_index = getattr(self, "_zm_dump_index", 0)
            torch.save(
                last_hidden.detach().float().cpu(),
                os.path.join(_dump_dir, f"zm_{_zm_index:04d}.pt"))
            self._zm_dump_index = _zm_index + 1

        past_kv, cur_len, _visual_relays = self._append_visual_relays(
            past_kv, cur_len, cursor + m, last_hidden, cur_vis, thw,
            spans=cur_spans)

        return {
            "past_key_values":        past_kv,
            "latent_trajectory":      latent_trajectory,
            "transport_embeddings":   transport_embeddings,
            "latent_to_vision_attn":  latent_to_vision_attn,
            "query_to_vision_attn":   query_to_vision_attn,
            "latent_to_vision_sink":   latent_to_vision_sink,
            "latent_to_vision_rowsum": latent_to_vision_rowsum,
            "query_to_vision_sink":    query_to_vision_sink,
            "query_to_vision_rowsum":  query_to_vision_rowsum,
            "qq_role_spans":          qq_role_spans,
            "tail_ids":               ids[last + 1:L].detach().cpu(),  # post-image text rows (n_q axis) for --saliency_bg_correct
            "full_attn_kv":           None,          # separate feature; unused here
            "past_len":               cur_len,       # shrunk by streaming eviction
            "pos_cursor":             cursor + m + _visual_relays,    # eviction keeps baked rotary phase
            "vis_coords":             None,
            "vis_cols":               cur_vis.cpu(),  # SURVIVING vision cols (ABSOLUTE)
            "grids":                  thw.cpu(),
            "n_vis":                  int(cur_vis.numel()),
            "n_vis_per_patch":        nV // max(1, len(images)),
            "spans":                  cur_spans,
            "vis_hidden":             vis_hidden,
            "pruned_streaming":       evicted,
            "saliency_steps":         ctrl.history if ctrl is not None else None,
            "stream_saliency_spans":  stream_saliency_spans,   # --stream_recompute_saliency
            "stream_saliency_grids":  stream_saliency_grids,
            "probe_rv":               (sum(probe_mass_steps) / len(probe_mass_steps)
                                       if probe_mass_steps else None),
            # per-step vision mass (streaming-path only; additive, byte-identical). See phase0j.
            "probe_mass_steps":       (list(probe_mass_steps) if probe_mass_steps else None),
        }

    @torch.no_grad()
    def append_grounded_vision_and_latent(
        self,
        past_kv,
        past_pos_cursor: int,
        image,                          # one PIL.Image (the new crop)
        framing_text: str,
        m: int,
        l_mid_layers: list[int],
        want_attn: bool = True,
    ) -> dict:
        """Append ONE marker-wrapped image (+ optional latent steps) to an EXISTING
        KV — the KV-carry counterpart of grounded_prefill_and_latent.

        Layout appended: [framing text, <|vision_start|>, image tokens, <|vision_end|>].
        The image tokens keep their <|vision_start|>/<|image_pad|>/<|vision_end|>
        markers (so grounding survives, unlike the marker-less 'bolton' append) and
        get true MRoPE vision coords continued from past_pos_cursor via vision_coords.
        Returns the new KV, pos_cursor, the appended vision columns, and the
        latent→vision attention needed for saliency on the new crop.
        """
        img_id = getattr(self.model.config, "image_token_id", 151655)
        past_len = self._kv_len(past_kv)

        chunk_text = f"{framing_text}<|vision_start|><|image_pad|><|vision_end|>"
        inputs = self.processor(text=[chunk_text], images=[image],
                                return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]                      # [1, L]
        ids = input_ids[0]
        L = int(ids.numel())
        pv  = inputs["pixel_values"].to(self.device, dtype=self.dtype)
        thw = inputs["image_grid_thw"].to(self.device)

        # ── grounded embeds: scatter image features into <|image_pad|> slots ────
        embeds = self.lm.embed_tokens(input_ids)             # [1, L, H]
        img_feat = torch.cat(
            self.vlm.get_image_features(pv, thw, return_dict=True).pooler_output, dim=0)
        vis_mask = (ids == img_id)
        embeds[0, vis_mask] = img_feat.to(embeds.dtype)

        # ── positions: [pre-text | vision MRoPE | post-text], continued from cursor ─
        vis_idx_local = vis_mask.nonzero(as_tuple=True)[0]
        first_v = int(vis_idx_local[0]); n_v = int(vis_idx_local.numel())
        P = first_v                       # framing + <|vision_start|>
        Q = L - (first_v + n_v)           # <|vision_end|> (+ trailing)
        pre = self._text_positions(P, start=past_pos_cursor)
        vis_coords, vcur = self.vision_coords(thw, start=past_pos_cursor + P)
        vis = vis_coords.unsqueeze(1)                         # [3,1,n_v]
        post = self._text_positions(Q, start=vcur)
        pos = torch.cat([pre, vis, post], dim=2)             # [3,1,L]
        cursor = vcur + Q

        out = self.lm(inputs_embeds=embeds, position_ids=pos, past_key_values=past_kv,
                      use_cache=True, output_attentions=want_attn,
                      output_hidden_states=True)
        past_kv = out.past_key_values
        last_hidden = out.hidden_states[-1][:, -1:, :]

        # visual token hidden states at last layer (for --missing_score hidden)
        vis_hidden = out.hidden_states[-1][0, vis_idx_local, :].float().cpu()  # (n_vis, d)

        # absolute columns of the appended vision inside the full attention matrix
        vis_cols = (past_len + vis_idx_local).cpu()
        spans: list[tuple[str, int]] = [("P", P), ("V", n_v), ("P", Q)]

        # ── latent steps + latent→vision attention (for saliency on the new crop) ─
        latent_trajectory: list[torch.Tensor] = []
        l2v_lists: dict[int, list[torch.Tensor]] = {li: [] for li in l_mid_layers}
        for step in range(m):
            le = self._apply_realign(last_hidden)
            o = self.lm(inputs_embeds=le,
                        position_ids=self._text_positions(1, cursor + step),
                        past_key_values=past_kv, use_cache=True,
                        output_attentions=want_attn, output_hidden_states=True)
            past_kv = o.past_key_values
            last_hidden = o.hidden_states[-1][:, -1:, :]
            if os.environ.get("VLMAS_RETR", ""):
                # retrieval kill test: per-layer residual input at the latest
                # latent step (overwritten each step -> the last step remains)
                self._retr_hidden = [h[0, -1].detach().float().clone() for h in o.hidden_states[:-1]]
            latent_trajectory.append(le.squeeze().cpu())
            if want_attn:
                for li in l_mid_layers:
                    a = o.attentions[li]
                    if a is not None:
                        l2v_lists[li].append(a[0][:, 0, past_len:past_len + L][:, vis_idx_local].cpu())
        if m > 0:
            spans.append(("L", m))
        latent_to_vision_attn = {li: torch.stack(v, 0) for li, v in l2v_lists.items() if v}

        # ── always prune the just-added vision block (flexible seam; a
        #    cross-iteration RETENTION strategy layers on top of this later).
        #    l2v_attn/grids are kept FULL so the carry loop's ROI/zoom saliency
        #    metadata is unaffected; only the KV vision columns are dropped. ──────
        cache_len = past_len + L + m
        past_kv, cache_len, spans, vis_cols, _ndrop = self._prune_vision_entry(
            past_kv, cache_len, vis_cols, spans, latent_to_vision_attn, thw,
            images=[image])
        n_vis_kept = int(vis_cols.numel()) if hasattr(vis_cols, "numel") else len(vis_cols)

        return {
            "past_key_values":       past_kv,
            "latent_trajectory":     latent_trajectory,
            "latent_to_vision_attn": latent_to_vision_attn,   # FULL (ROI/zoom meta)
            "vis_hidden":            vis_hidden,               # (n_vis, d) for hidden-sim
            "past_len":              cache_len,
            "pos_cursor":            cursor + m,
            "vis_cols":              vis_cols,           # appended image columns (post-prune)
            "grids":                 thw.cpu(),          # FULL grid (zoom metadata)
            "n_vis":                 n_vis_kept,
            "spans":                 spans,
        }

    @torch.no_grad()
    def drop_vision_columns(self, past_kv, past_len: int, vis_cols, spans):
        """Drop the ENTIRE given vision block from a KV, keeping everything else.

        Unlike _prune_vision_entry (saliency-partial, gated by --kv_prune), this is
        an unconditional full drop of the columns in `vis_cols`. Used by the KV-carry
        Scanner's `latent_only` thread mode: after appending a thumbnail/ROI + m
        latent steps, the raw vision is discarded so ONLY the Scanner's latent
        "thought" (and framing text) survive into the threaded KV — no gigapixel
        thumbnail bloats/pollutes the downstream reasoning KV.

        Returns (past_kv, new_len, new_spans). No-op when vis_cols is empty.
        """
        import numpy as np
        n = (int(vis_cols.numel()) if hasattr(vis_cols, "numel") else len(vis_cols)) \
            if vis_cols is not None else 0
        if n == 0:
            return past_kv, past_len, spans
        keep = np.zeros(n, dtype=bool)   # drop all vision columns
        from memory.prune import apply_kv_prune
        pk, new_len, new_spans, _new_vis = apply_kv_prune(
            past_kv, past_len, vis_cols, keep, spans, self.device)
        print(f"[VisionDrop] latent_only: dropped all {n} vision cols "
              f"→ KV {past_len} → {new_len} tokens")
        return pk, new_len, new_spans

    @torch.no_grad()
    def last_token_logits(
        self,
        system_prompt: str,
        user_text: str,
        past_kv,
        pos_cursor: int | None = None,
        clone_cache: bool = True,
    ) -> torch.Tensor:
        """Prefill a short prompt on top of past_kv and return the NEXT-token
        logit vector [vocab] — without generating any token.

        Used for no-decode decisions (e.g. the Verifier's sufficiency gate): read
        the distribution at the clean assistant boundary and compare candidate
        word logits, instead of decoding text. clone_cache keeps past_kv pristine
        so the same KV can be read multiple times.
        """
        embeds = self.embed_text(
            user_text, system_prompt=system_prompt, enable_thinking=False
        ).unsqueeze(0)
        L = embeds.shape[1]
        past = copy.deepcopy(past_kv) if clone_cache else past_kv
        past_len = self._kv_len(past)
        if getattr(self, "mrope_pos", False) and pos_cursor is not None:
            pos = self._text_positions(L, start=pos_cursor)
        else:
            pos = self._make_position_ids(L, past_len=past_len)
        out = self.lm(
            inputs_embeds=embeds, position_ids=pos, past_key_values=past,
            use_cache=True, output_hidden_states=True,
        )
        h = out.hidden_states[-1][:, -1, :]
        wd = self.model.lm_head.weight.dtype
        return self.model.lm_head(h.to(wd))[0].float()

    # ── span-aware KV pruning (LatentMAS-style retention, WSI-adapted) ──────────
    @torch.no_grad()
    def prune_kv(self, past_kv, span_map, keep_types):
        """Keep only KV columns whose span type is in keep_types, dropping the
        rest — a span-aware gather, NOT a tail-slice (visual evidence lives at
        the front and must survive while prompt text in the middle is dropped).

        span_map: ordered list of (type, n_cols) describing how the cache was
        built, e.g. [("V",961),("P",30),("L",10),("P",24),("L",10)].

        Returns (pruned_cache, new_token_count, filtered_span_map). The cache is
        deep-copied so the caller's chain is untouched. pos_cursor is unchanged
        by the caller: kept columns keep their baked-in RoPE phase, and new
        tokens continue from the same rope cursor (holes in position are fine).
        """
        keep_idx: list[int] = []
        new_map: list[tuple[str, int]] = []
        col = 0
        for t, n in span_map:
            if t in keep_types:
                keep_idx.extend(range(col, col + n))
                new_map.append((t, n))
            col += n

        if len(keep_idx) == col:          # nothing dropped → no-op (e.g. full mode)
            return past_kv, col, list(span_map)
        if not keep_idx:                  # degenerate: keep at least nothing-dropped
            return past_kv, col, list(span_map)

        pruned = copy.deepcopy(past_kv)
        idx = torch.tensor(keep_idx, device=self.device, dtype=torch.long)
        layers = pruned.layers if hasattr(pruned, "layers") else None
        if layers is not None:
            for layer in layers:
                layer.keys = layer.keys.index_select(2, idx).contiguous()
                layer.values = layer.values.index_select(2, idx).contiguous()
        return pruned, len(keep_idx), new_map
