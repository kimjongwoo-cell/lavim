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

        print(f"[Backbone] Loading {model_path} ...")
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=dtype, device_map=device,
            attn_implementation="eager",  # required for output_attentions=True
        )
        self.model.eval()

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
            return self._apply_realign_identity_norm(hidden)
        if self._realign_matrix is None:
            return hidden
        if self._realign_method == "softmax":
            return self._apply_realign_softmax(hidden)
        return self._apply_realign_wa(hidden)

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
            if not enable_thinking:
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
            "pos_cursor":             cursor + m,
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
        if not self.instruct and not enable_thinking:
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
        out = self.model(
            **inputs,
            use_cache=True,
            output_attentions=want_attn and not dynamic_prune,
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
        if want_attn and not dynamic_prune:
            q_rows = torch.arange(last + 1, seq_prefill, device=self.device)
            for li in l_mid_layers:
                a = out.attentions[li]
                if a is not None:
                    query_to_vision_attn[li] = a[0][:, q_rows][:, :, vis_idx].cpu()
                    query_to_vision_sink[li]   = a[0][:, q_rows, 0].cpu()
                    query_to_vision_rowsum[li] = a[0][:, q_rows, :].sum(-1).cpu()

        # P1-2: tag which of the q_rows are question/choice tokens (Q_q), in
        # tail-relative coords (== the n_q saliency axis). None → caller averages
        # over ALL post-image text rows (old behavior, incl. chat-template tail).
        qq_role_spans = None
        if want_attn and not dynamic_prune and role_targets:
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Observe every WSI patch, then prune merge-groups for deep encoding.

        The first vision blocks see every image token.  Group-level morphology
        novelty then drives one hierarchy-constrained selection, and only the
        retained groups traverse the remaining vision and language blocks.
        Keeping complete spatial-merge groups preserves the pretrained merger.
        """
        from memory.prune import hierarchy_prefill_keep_mask

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

        keep_groups = hierarchy_prefill_keep_mask(
            scores,
            token_counts=token_counts,
            parent_indices=parent_indices,
            keep_ratio=keep_ratio,
            min_tokens_per_image=min_tokens_per_image,
            significance_sigma=significance_sigma,
        )
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
        if not self.instruct and not enable_thinking:
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
        if hierarchy_prefill and nV > 0:
            img_feat, keep_visual = self._hierarchy_pruned_vision_features(
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
            )
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

        dynamic_prune = bool(
            _pa is not None and getattr(_pa, "prune_dynamic_ratio", False)
        )
        event_probes_only = bool(
            _pa is not None and getattr(_pa, "prune_event_probes_only", False)
        )
        capture_prefill_attn = want_attn and not dynamic_prune and not event_probes_only
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

        qq_role_spans = None
        if capture_prefill_attn and role_targets:
            # role spans are tail-relative to (last+1 .. L); _locate_role_spans works
            # on the chunk ids exactly as in the fresh path.
            qq_role_spans = self._locate_role_spans(ids, last + 1, L, role_targets)
        bg_role_spans = None
        if capture_prefill_attn and bg_targets:
            bg_role_spans = self._locate_role_spans(ids, last + 1, L, bg_targets)

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
            and (getattr(_pa, "prune_dynamic_ratio", False) or event_probes_only)
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
            "pos_cursor":             cursor + m,    # eviction keeps baked rotary phase
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
