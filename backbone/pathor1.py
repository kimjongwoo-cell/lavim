"""Patho-R1-7B backbone for latent reasoning (Qwen2.5-VL family).

Ported 1:1 from qwen3vl.py — same latent/grounded/carry/prune logic. Patho-R1 is
a Qwen2_5_VLForConditionalGeneration checkpoint, and in transformers>=5.8 the
Qwen2.5-VL model exposes the SAME MRoPE helpers as Qwen3-VL
(get_image_features / get_vision_position_ids / compute_3d_position_ids /
get_rope_index), so the port is nearly byte-identical.

Deltas vs qwen3vl.py (the ONLY places this file diverges):
  1. class name only; the loader (AutoModelForImageTextToText) resolves the
     Qwen2.5-VL class automatically from config, so __init__ is unchanged.
  2. thinking hooks are GUARDED by `prompt.endswith("<think>\\n")`: Patho-R1's
     chat template never auto-opens a <think> block (it uses a system-prompt
     <think>…</think><answer>…</answer> format), so the qwen3 unconditional
     `</think>` suffix must NOT fire here. Guarded → no-op for Patho, identical
     behaviour for a real thinking model.
  3. compute_3d_position_ids() is called with video_grid_thw/attention_mask/
     past_key_values explicitly (Qwen2.5-VL lists them as required positional
     args; the keyword call is valid on both families).

Model structure:
  model.model.visual          → Qwen2_5_VisionTransformerPretrainedModel (ViT)
  model.model.language_model  → Qwen2_5_VLTextModel
    layers[i].self_attn       → Qwen2_5_VLAttention (ALL 28 layers, full attention)
"""

from __future__ import annotations

import copy
import re

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

from backbone.efficiency import EfficiencyMixin


class PathoR1Backbone(EfficiencyMixin):
    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        realign_method: str = "wa",   # "wa" | "softmax"
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

        self.vlm  = self.model.model          # Qwen2_5_VLModel
        self.lm   = self.vlm.language_model   # Qwen2_5_VLTextModel
        self.visual = self.vlm.visual         # Qwen2_5_VisionTransformer

        # additive efficiency instrumentation (FLOPs profiler + TTFT); no-op on failure.
        self._init_efficiency()

        # All 28 layers are full attention for Patho-R1 (no sliding window used)
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
        output_emb = self.model.get_output_embeddings().weight.detach().float()

        # cache for softmax method
        self._input_emb  = input_emb.to(dev)   # E [V, D]
        self._output_emb = output_emb.to(dev)  # U [V, D]

        # W_a (least squares)
        gram = output_emb.T @ output_emb
        reg  = 1e-5 * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        rhs  = output_emb.T @ input_emb
        matrix = torch.linalg.solve(gram + reg, rhs)
        target_norm = input_emb.norm(dim=1).mean()

        self._realign_matrix      = matrix.to(device=dev, dtype=torch.float32)
        self._realign_target_norm = target_norm.to(device=dev, dtype=torch.float32)
        print(f"[Backbone] W_a built: {tuple(matrix.shape)}, "
              f"target_norm={target_norm:.4f}")

    def _apply_realign(self, hidden: torch.Tensor) -> torch.Tensor:
        """Dispatch to W_a or softmax realignment.  hidden: [B, 1, D]"""
        if self._realign_method == "softmax":
            return self._apply_realign_softmax(hidden)
        return self._apply_realign_wa(hidden)

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

    @torch.no_grad()
    def close_assistant_turn(
        self,
        past_kv,
        past_pos_cursor: int,
        *,
        close_thinking: bool = False,
    ):
        """Append the chat terminator before the next cumulative latent role."""
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
            # Close an auto-opened <think> block only if the template actually
            # opened one. Patho-R1's template never does (system-prompt thinking
            # format), so this is a no-op here; kept for thinking-model parity.
            if not enable_thinking and prompt.endswith("<think>\n"):
                prompt = prompt + "</think>\n\n"

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
        # Anti-repetition: patho_r1 is loop-prone under pure greedy decode — it
        # collapses into degenerate phrase/token repetition ("The left panel ... The
        # right panel ..." ×N, "des des des ..."). A moderate repetition_penalty +
        # a 3-gram block breaks the loop without sampling noise. text_mas vlm_generate
        # only (Reasoner/Verifier/Diagnosis); the latent path is untouched.
        _gk = dict(max_new_tokens=max_new_tokens, do_sample=False,
                   repetition_penalty=float(getattr(self, "vlm_rep_penalty", 1.3)),
                   no_repeat_ngram_size=int(getattr(self, "vlm_no_repeat", 3)))
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
        out = self.lm(
            inputs_embeds=input_embeds,
            position_ids=pos_ids,
            past_key_values=past_kv,
            use_cache=True,
            output_attentions=True,
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
            latent_trajectory.append(last_hidden.squeeze().cpu())

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
        if not self.instruct and not enable_thinking and text.endswith("<think>\n"):
            text = text + "</think>\n\n"   # close think only if template opened it

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

        out = self.model(**inputs, use_cache=True,
                         output_attentions=want_attn, output_hidden_states=True)
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
        if want_attn:
            q_rows = torch.arange(last + 1, seq_prefill, device=self.device)
            for li in l_mid_layers:
                a = out.attentions[li]
                if a is not None:
                    query_to_vision_attn[li] = a[0][:, q_rows][:, :, vis_idx].cpu()
                    query_to_vision_sink[li]   = a[0][:, q_rows, 0].cpu()
                    query_to_vision_rowsum[li] = a[0][:, q_rows, :].sum(-1).cpu()

        # Qwen2.5-VL lists video_grid_thw/attention_mask/past_key_values as
        # required args (Qwen3-VL defaults them); pass explicitly so the same
        # keyword call is valid on both families.
        cursor = int(self.vlm.compute_3d_position_ids(
            input_ids=inputs["input_ids"], inputs_embeds=None,
            image_grid_thw=inputs.get("image_grid_thw"),
            video_grid_thw=None, attention_mask=None, past_key_values=None,
            mm_token_type_ids=inputs.get("mm_token_type_ids")).max().item()) + 1

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
        ctrl = build_prune_controller(getattr(self, "_prune_args", None))
        if ctrl is not None:
            ctrl.start(grids=inputs.get("image_grid_thw"), vis_idx=vis_idx, sms=sms,
                       images=images, q2v=query_to_vision_attn, l_mid_layers=l_mid_layers,
                       image_magnifications=getattr(self, "_prune_image_magnifications", ()),
                       morphology_enabled=getattr(self, "_prune_morphology_enabled", False))
        cur_vis   = vis_idx            # current vision column indices into the cache
        cur_len   = seq_prefill        # current cache length
        cur_spans = list(prefill_spans)
        evicted   = False

        for step in range(m):
            le = self._apply_realign(last_hidden)
            o = self.lm(inputs_embeds=le, position_ids=self._text_positions(1, cursor + step),
                        past_key_values=past_kv, use_cache=True,
                        output_attentions=want_attn, output_hidden_states=True)
            past_kv = o.past_key_values
            last_hidden = o.hidden_states[-1][:, -1:, :]
            latent_trajectory.append(last_hidden.squeeze().cpu())
            cur_len += 1
            if cur_spans and cur_spans[-1][0] == "L":
                cur_spans[-1] = ("L", cur_spans[-1][1] + 1)
            else:
                cur_spans.append(("L", 1))

            # original aggregate collection (skipped once streaming eviction starts,
            # since per-step column counts would differ and can't be stacked)
            if want_attn and (ctrl is None or not ctrl.evict):
                for li in l_mid_layers:
                    a = o.attentions[li]
                    if a is not None:
                        l2v_lists[li].append(a[0][:, 0, vis_idx].cpu())
                        l2v_sink_lists[li].append(a[0][:, 0, 0].cpu())
                        l2v_rowsum_lists[li].append(a[0][:, 0, :].sum(-1).cpu())

            # ── prune controller hook (observe + maybe evict) ───────────────────
            if ctrl is not None and o.attentions is not None:
                vecs = [o.attentions[li][0][:, 0, cur_vis].float().mean(0)
                        for li in l_mid_layers if o.attentions[li] is not None]
                if vecs:
                    ctrl.observe(step, torch.stack(vecs, 0).mean(0).cpu().numpy())
                if ctrl.evict:
                    keep = ctrl.maybe_keep_mask(step)
                    if keep is not None and not keep.all():
                        past_kv, cur_len, cur_spans, new_vis = apply_kv_prune(
                            past_kv, cur_len, cur_vis.tolist(), keep, cur_spans, self.device)
                        cur_vis = torch.tensor(new_vis, device=self.device, dtype=torch.long)
                        ctrl.commit(keep)
                        evicted = True
                        print(f"[PruneCtrl] step {step}: dropped {int((~keep).sum())}/"
                              f"{len(keep)} vis → KV {cur_len} tokens")

        latent_to_vision_attn = {li: torch.stack(v, 0) for li, v in l2v_lists.items() if v}
        latent_to_vision_sink = {li: torch.stack(v, 0) for li, v in l2v_sink_lists.items() if v}
        latent_to_vision_rowsum = {li: torch.stack(v, 0) for li, v in l2v_rowsum_lists.items() if v}
        full_attn_kv = self._extract_full_attn_kv(past_kv, l_mid_layers)

        # streaming eviction already shrank the KV; null the now-stale per-token
        # attention so the Reasoner's one-shot prune / saliency / memory paths
        # gracefully no-op (they guard on empty q2v) instead of double-pruning.
        if evicted:
            query_to_vision_attn = {}
            latent_to_vision_attn = {}

        return {
            "past_key_values":        past_kv,
            "latent_trajectory":      latent_trajectory,
            "latent_to_vision_attn":  latent_to_vision_attn,
            "query_to_vision_attn":   query_to_vision_attn,
            "latent_to_vision_sink":   latent_to_vision_sink,
            "latent_to_vision_rowsum": latent_to_vision_rowsum,
            "query_to_vision_sink":    query_to_vision_sink,
            "query_to_vision_rowsum":  query_to_vision_rowsum,
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
        continuation_user_turn: bool = False,
    ) -> dict:
        """Append a grounded Patho-R1 role to an existing latent cache.

        This is the cumulative counterpart of ``grounded_prefill_and_latent``.
        It intentionally keeps the continuation path free of pruning policy: the
        base transport must first preserve a valid multimodal cache, while pruning
        variants can act on the returned absolute vision columns.
        """
        img_id = getattr(self.model.config, "image_token_id", 151655)
        past_len = self._kv_len(past_kv)
        content = [{"type": "image", "image": image} for image in images]
        continuation_text = user_text
        if continuation_user_turn and system_prompt:
            continuation_text = f"Next-stage role instructions:\n{system_prompt}\n\n{user_text}"
        content.append({"type": "text", "text": continuation_text})
        messages = [] if continuation_user_turn else (
            [{"role": "system", "content": system_prompt}] if system_prompt else []
        )
        messages.append({"role": "user", "content": content})
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if not self.instruct and not enable_thinking and text.endswith("<think>\n"):
            text += "</think>\n\n"
        inputs = self.processor(
            text=[text], images=images, return_tensors="pt"
        ).to(self.device)
        input_ids = inputs["input_ids"]
        ids = input_ids[0]
        is_vis = ids == img_id
        vis_idx = is_vis.nonzero(as_tuple=True)[0]
        if int(vis_idx.numel()) == 0:
            raise ValueError("grounded continuation requires at least one image token")
        last_visual = int(vis_idx[-1])
        sequence_length = int(ids.numel())

        embeds = self.lm.embed_tokens(input_ids)
        pixel_values = inputs["pixel_values"].to(self.device, dtype=self.dtype)
        image_grid = inputs["image_grid_thw"].to(self.device)
        image_features = torch.cat(
            self.vlm.get_image_features(
                pixel_values, image_grid, return_dict=True
            ).pooler_output,
            dim=0,
        )
        embeds[0, is_vis] = image_features.to(embeds.dtype)
        positions = self.vlm.compute_3d_position_ids(
            input_ids=input_ids,
            inputs_embeds=None,
            image_grid_thw=image_grid,
            video_grid_thw=None,
            attention_mask=None,
            past_key_values=None,
            mm_token_type_ids=inputs.get("mm_token_type_ids"),
        ) + int(past_pos_cursor)
        cursor = int(positions.max().item()) + 1
        output = self.lm(
            inputs_embeds=embeds,
            position_ids=positions,
            past_key_values=past_kv,
            use_cache=True,
            output_attentions=want_attn,
            output_hidden_states=True,
        )
        past_kv = output.past_key_values
        last_hidden = output.hidden_states[-1][:, -1:, :]
        absolute_vis_cols = past_len + vis_idx
        vis_hidden = (
            output.hidden_states[-1][0, vis_idx, :].float().cpu()
            if want_vis_hidden
            else None
        )
        query_to_vision_attn: dict[int, torch.Tensor] = {}
        if want_attn:
            query_rows = torch.arange(
                last_visual + 1, sequence_length, device=self.device
            )
            for layer_index in l_mid_layers:
                attention = output.attentions[layer_index]
                if attention is not None:
                    query_to_vision_attn[layer_index] = attention[0][:, query_rows][
                        :, :, absolute_vis_cols
                    ].cpu()

        latent_trajectory: list[torch.Tensor] = []
        latent_to_vision: dict[int, list[torch.Tensor]] = {
            layer_index: [] for layer_index in l_mid_layers
        }
        for step in range(m):
            latent_embed = self._apply_realign(last_hidden)
            output = self.lm(
                inputs_embeds=latent_embed,
                position_ids=self._text_positions(1, cursor + step),
                past_key_values=past_kv,
                use_cache=True,
                output_attentions=want_attn,
                output_hidden_states=True,
            )
            past_kv = output.past_key_values
            last_hidden = output.hidden_states[-1][:, -1:, :]
            latent_trajectory.append(last_hidden.squeeze().cpu())
            if want_attn:
                for layer_index in l_mid_layers:
                    attention = output.attentions[layer_index]
                    if attention is not None:
                        latent_to_vision[layer_index].append(
                            attention[0][:, 0, absolute_vis_cols].cpu()
                        )
        latent_to_vision_attn = {
            layer_index: torch.stack(values, 0)
            for layer_index, values in latent_to_vision.items()
            if values
        }
        return {
            "past_key_values": past_kv,
            "latent_trajectory": latent_trajectory,
            "latent_to_vision_attn": latent_to_vision_attn,
            "query_to_vision_attn": query_to_vision_attn,
            "full_attn_kv": self._extract_full_attn_kv(past_kv, l_mid_layers),
            "past_len": past_len + sequence_length + m,
            "pos_cursor": cursor + m,
            "vis_cols": absolute_vis_cols.cpu(),
            "grids": image_grid.cpu(),
            "n_vis": int(vis_idx.numel()),
            "n_vis_per_patch": int(vis_idx.numel()) // max(1, len(images)),
            "spans": [("P", sequence_length), ("L", m)],
            "vis_hidden": vis_hidden,
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
    ) -> dict:
        """Append a new prompt to existing KV and run m more latent steps.

        Used by Refiner to continue reasoning from Reasoner's KV. The new text
        and latent tokens are positioned from the *rope cursor* (max position+1),
        not the token count, so they stay adjacent to the (compressed) visual
        positions instead of being pushed thousands of steps away.

        Returns the usual keys plus pos_cursor.
        """
        n_t = text_embeds.shape[0]
        use_mrope = self.mrope_pos and past_pos_cursor is not None
        cursor = past_pos_cursor if use_mrope else past_len

        # ── prefill new prompt on top of past_kv ──────────────────────────────
        if use_mrope:
            pos_ids = self._text_positions(n_t, start=cursor)
        else:
            pos_ids = self._make_position_ids(n_t, past_len=past_len)

        out = self.lm(
            inputs_embeds=text_embeds.unsqueeze(0),
            position_ids=pos_ids,
            past_key_values=past_kv,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=True,
        )
        past_kv     = out.past_key_values
        last_hidden = out.hidden_states[-1][:, -1:, :]  # [1, 1, hidden]
        past_len   += n_t
        cursor     += n_t

        # ── latent steps ───────────────────────────────────────────────────────
        latent_trajectory: list[torch.Tensor] = []

        for step in range(m):
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
                output_attentions=False,
                output_hidden_states=True,
            )
            past_kv     = out.past_key_values
            last_hidden = out.hidden_states[-1][:, -1:, :]
            latent_trajectory.append(last_hidden.squeeze().cpu())

        full_attn_kv = self._extract_full_attn_kv(past_kv, l_mid_layers)

        return {
            "past_key_values":    past_kv,
            "latent_trajectory":  latent_trajectory,
            "full_attn_kv":       full_attn_kv,
            "past_len":           past_len + m,
            "pos_cursor":         cursor + m,
            # span types appended by this continuation: prompt | latent
            "spans":              [s for s in (("P", n_t), ("L", m)) if s[1] > 0],
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
                msgs.append(
                    {
                        "role": "user",
                        "content": (
                            f"Next-stage role instructions:\n{system_prompt}\n\n{user_text}"
                            if system_prompt
                            else user_text
                        ),
                    }
                )
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
            if prompt.endswith("<think>\n"):
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

        _timer, _t0 = self._ttft_timer()
        if _timer is not None:
            gen_kwargs["logits_processor"] = [_timer]
        output_ids = self.model.generate(**gen_kwargs)
        self._record_ttft(_timer, _t0)
        new_ids = output_ids[:, input_ids.shape[1]:]
        return tok.decode(new_ids[0], skip_special_tokens=True).strip()

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
            sink_frac=float(getattr(args, "kv_prune_sink_frac", 0.05)))
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
        vis_hidden — so the Scanner's saliency→bbox mappers can pick regions WITHOUT
        any coordinate decode (patho_r1 is a diagnosis-reasoning model that cannot
        serialize bbox coords). Qwen2.5-VL / MRoPE; grids come from image_grid_thw
        and the Scanner reads vision_config.spatial_merge_size (=2) — no
        scan_spatial_merge_size attr needed.

        Returns dict {grids, n_vis, vis_hidden, latent_trajectory,
        latent_to_vision_attn} — same keys the Scanner consumes (cf.
        backbone/huatuogpt.py::saliency_scan).
        """
        img_id = getattr(self.model.config, "image_token_id", 151655)
        chunk_text = f"{prompt_text}<|vision_start|><|image_pad|><|vision_end|>"
        inputs = self.processor(text=[chunk_text], images=[image],
                                return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        ids = input_ids[0]
        L = int(ids.numel())
        pv  = inputs["pixel_values"].to(self.device, dtype=self.dtype)
        thw = inputs["image_grid_thw"].to(self.device)

        # grounded embeds: scatter image features into <|image_pad|> slots
        embeds = self.lm.embed_tokens(input_ids)
        img_feat = torch.cat(
            self.vlm.get_image_features(pv, thw, return_dict=True).pooler_output, dim=0)
        vis_mask = (ids == img_id)
        embeds[0, vis_mask] = img_feat.to(embeds.dtype)

        # positions: [pre-text | vision MRoPE | post-text], fresh from 0
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
            latent_trajectory.append(last_hidden.squeeze().cpu())
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
    def drop_vision_columns(self, past_kv, past_len: int, vis_cols, spans):
        """Drop the whole appended vision block from a KV (Scanner scan_on_kv
        latent_only thread), keeping the latent thought. Reuses the backbone-agnostic
        apply_kv_prune (patho DynamicCache). Returns (kv, new_len, spans).
        No-op when vis_cols is empty. Needed for KV-carry rescan (MAX_ITERATIONS>0)."""
        import numpy as np
        from memory.prune import apply_kv_prune
        n = (int(vis_cols.numel()) if hasattr(vis_cols, "numel") else len(vis_cols)) \
            if vis_cols is not None else 0
        if n == 0:
            return past_kv, past_len, spans
        keep = np.zeros(n, dtype=bool)                    # drop all vision columns
        pk, new_len, new_spans, _ = apply_kv_prune(
            past_kv, past_len, vis_cols, keep, spans, self.device)
        print(f"[VisionDrop/patho] dropped {n} vision cols → KV {past_len} → {new_len}")
        return pk, new_len, new_spans

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
        # visual-token hidden states (FULL, pre-prune) for hidden-cosine saliency in
        # the Scanner's kv_scan rescan (missing_score=hidden); avoids attn corner-sink.
        vis_hidden = out.hidden_states[-1][0, vis_idx_local, :].float().cpu()

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
            latent_trajectory.append(last_hidden.squeeze().cpu())
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
            "past_len":              cache_len,
            "pos_cursor":            cursor + m,
            "vis_cols":              vis_cols,           # appended image columns (post-prune)
            "grids":                 thw.cpu(),          # FULL grid (zoom metadata)
            "n_vis":                 n_vis_kept,
            "vis_hidden":            vis_hidden,   # (n_vis, d) for hidden-cosine rescan
            "spans":                 spans,
        }

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
