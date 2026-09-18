"""InternVL3-8B backbone (InternViT-300M + Qwen2-7B + pixel-shuffle projector).

Same LatentMAS structure as llava_onevision.py: the LLM is Qwen2-7B (hidden 3584,
28 layers, 4 KV heads) — IDENTICAL to LLaVA-OneVision — so the realign / DynamicCache /
generate-on-KV paths are copied verbatim. Only vision encoding, image-marker prompt,
and module loading differ.

Deltas vs llava_onevision.py:
  1. Model is the HF-NATIVE `InternVLForConditionalGeneration` (OpenGVLab/InternVL3-8B-hf,
     downloaded to ...-hf). The original OpenGVLab `InternVLChatModel` (trust_remote_code)
     will not load under tf5.10 → use the -hf conversion (same rationale as llava-ov).
  2. Vision: InternViT-300M @448/patch14 → 32×32=1024 patches; get_image_features drops
     CLS (strategy "default") and applies pixel-shuffle (downsample 0.5) → 16×16 = 256
     tokens/tile → multi_modal_projector. We force a SINGLE 448 tile per crop (bypass
     InternVL dynamic anyres tiling) so the vision block is a clean fixed 16×16 grid —
     the Scanner's saliency→bbox mappers need this.
  3. Prompt: Qwen2 chatml; image marker = "<IMG_CONTEXT>" (context_image_token,
     id 151667 — a REAL vocab token). The InternVL processor expands each <IMG_CONTEXT>
     to <img>…IMG_CONTEXT*256…</img> for the generate paths; the manual latent path
     locates the single 151667 token and splices 256 vision embeds (mirrors llava-ov's
     single-marker design — no manual <img>/</img> boundary tokens).
  4. 28 layers / hidden 3584 (Qwen2-7B), GQA (4 KV heads); LM = model.model.language_model.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from typing import Any

import torch
from PIL import Image

from transformers import (InternVLForConditionalGeneration, AutoProcessor,
                          AutoTokenizer)

from backbone.efficiency import EfficiencyMixin

_N_VIS_PER_IMAGE = 256          # InternViT @448/patch14 → 32×32 → pixel-shuffle 0.5 → 16×16
_VIS_GRID = 16
_IMAGE_TOKEN = "<IMG_CONTEXT>"  # context_image_token, id 151667


class _TokenizerWrapper:
    """Expose bk.processor.tokenizer.* uniformly (agents call bk.processor.tokenizer)."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def decode(self, *a, **k):
        return self.tokenizer.decode(*a, **k)

    def apply_chat_template(self, *a, **k):
        return self.tokenizer.apply_chat_template(*a, **k)


class InternVLBackbone(EfficiencyMixin):
    """InternVL3-8B (HF-native InternVL) adapter with LatentMAS realignment."""

    # InternVL3's Qwen2 text model uses ordinary one-dimensional RoPE; the
    # terminal decoder must therefore not construct Qwen3-VL MRoPE positions.
    mrope_pos = False

    def __init__(
        self,
        model_path: str = "/media/super/4TB/hj/InternVL3-8B-hf",
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        latent_space_realign: bool = True,
        realign_method: str = "wa",     # "wa" | "softmax"
        realign_tau: float = 1.0,
    ) -> None:
        self.device = device
        self.dtype  = dtype
        self._latent_space_realign = latent_space_realign
        self._realign_method = realign_method
        self._realign_tau    = realign_tau

        print(f"[Backbone] Loading {model_path} ...")
        self.model = InternVLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device,
            attn_implementation="eager",
        ).eval()

        # Processor drives the (multi-image) generate paths; tokenizer drives the
        # manual latent-injection path. Fall back to a bare tokenizer if the saved
        # processor is unavailable.
        try:
            self.processor_full = AutoProcessor.from_pretrained(model_path)
            self.tokenizer = self.processor_full.tokenizer
        except Exception:
            self.processor_full = None
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        self.model.config.tokenizer_padding_side = "left"

        # chatml stop ids: pin <|im_end|> (+ eos) so decoding stops at the turn
        # boundary rather than rolling into a second assistant turn.
        _im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.eos_ids = sorted({i for i in (_im_end, self.tokenizer.eos_token_id)
                               if isinstance(i, int) and i >= 0})

        self.image_token_id = int(self.model.config.image_token_id)   # 151667
        self._vfl  = self.model.config.vision_feature_layer            # -1
        self._vfss = self.model.config.vision_feature_select_strategy  # "default"

        # Eager attention so output_attentions=True populates (tf5 resolves the impl
        # from config._attn_implementation at forward time → set the config, not just
        # per-module attrs).
        self.model.config._attn_implementation = "eager"
        self.model.config.text_config._attn_implementation = "eager"

        # module handles (wrapper.model = InternVLModel)
        self.lm      = self.model.model.language_model      # Qwen2Model
        self.vt      = self.model.model.vision_tower        # InternVisionModel
        self.mm_proj = self.model.model.multi_modal_projector
        if getattr(self.lm, "config", None) is not None:
            self.lm.config._attn_implementation = "eager"
        for layer in self.lm.layers:
            if hasattr(layer.self_attn, "_attn_implementation"):
                layer.self_attn._attn_implementation = "eager"

        # InternViT preprocessing constants (ImageNet mean/std, resize 448 bicubic).
        self._img_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self._img_std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        self._img_size = 448

        self.processor = _TokenizerWrapper(self.tokenizer)

        # No grounded_prefill_and_latent on LLaVA-family backbones → Reasoner takes the
        # non-grounded path (encode_patches + <IMG_CONTEXT>-injected forward_with_latent).
        self.vision_inject = "bolton"
        # Pixel-shuffled grid = 16×16, no further spatial merge (unlike Qwen merge=2).
        self.scan_spatial_merge_size = 1
        # Reasoner._compute_saliency / M4 read spatial_merge_size off
        # model.config.vision_config; pin 1 so build_roi_spans keeps the full 16×16=256 grid.
        try:
            self.model.config.vision_config.spatial_merge_size = 1
        except Exception:
            pass

        n_layers = len(self.lm.layers)
        self.full_attn_indices: list[int] = list(range(n_layers))
        print(f"[Backbone] Layers: {n_layers} (all full attention), vis {_N_VIS_PER_IMAGE}/img")

        self._init_efficiency()
        self._image_token_pos: int | None = None
        self._build_realign_matrix()

    # ── realignment (identical to huatuo/quilt/llavaov) ─────────────────────────
    def _build_realign_matrix(self) -> None:
        dev = torch.device(self.device)
        input_emb  = self.model.get_input_embeddings().weight.detach().float()
        output_emb = self.model.get_output_embeddings().weight.detach().float()

        self._input_emb  = input_emb.to(dev)
        self._output_emb = output_emb.to(dev)

        gram   = output_emb.T @ output_emb
        reg    = 1e-5 * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        rhs    = output_emb.T @ input_emb
        matrix = torch.linalg.solve(gram + reg, rhs)
        target_norm = input_emb.norm(dim=1).mean()

        if not self._latent_space_realign:
            matrix = torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)

        self._realign_matrix      = matrix.to(device=dev, dtype=torch.float32)
        self._realign_target_norm = target_norm.to(device=dev, dtype=torch.float32)
        label = "identity (ablation)" if not self._latent_space_realign else f"{tuple(matrix.shape)}"
        print(f"[Backbone] W_a built: {label}, target_norm={target_norm:.4f}")
        print(f"[Backbone] Realign method: {self._realign_method}"
              + (f", tau={self._realign_tau}" if self._realign_method == "softmax" else ""))

    def _apply_realign(self, hidden: torch.Tensor) -> torch.Tensor:
        if self._realign_method == "softmax":
            return self._apply_realign_softmax(hidden)
        return self._apply_realign_wa(hidden)

    def _apply_realign_wa(self, hidden: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden.dtype
        h = hidden.float().squeeze(1)
        aligned = h @ self._realign_matrix
        norm = aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        aligned = aligned * (self._realign_target_norm / norm)
        return aligned.to(orig_dtype).unsqueeze(1)

    def _apply_realign_softmax(self, hidden: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden.dtype
        h = hidden.float().squeeze(1)
        logits  = h @ self._output_emb.T / self._realign_tau
        p       = torch.softmax(logits, dim=-1)
        z_tilde = p @ self._input_emb
        return z_tilde.to(orig_dtype).unsqueeze(1)

    # ── image token locate (151667 is a real vocab token) ───────────────────────
    def _tokenizer_image_token(self, prompt: str) -> torch.Tensor:
        ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    # ── vision encoding ─────────────────────────────────────────────────────────
    @staticmethod
    def _to_pil(img) -> Image.Image:
        if isinstance(img, Image.Image):
            return img.convert("RGB")
        import numpy as np
        if isinstance(img, torch.Tensor):
            arr = img.detach().cpu()
            if arr.ndim == 3 and arr.shape[0] in (1, 3):
                arr = arr.permute(1, 2, 0)
            arr = arr.float().numpy()
            if arr.max() <= 1.0:
                arr = arr * 255.0
            return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert("RGB")
        if isinstance(img, np.ndarray):
            arr = img.astype(np.float32)
            if arr.max() <= 1.0:
                arr = arr * 255.0
            return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert("RGB")
        raise TypeError(f"encode_patches: unsupported image type {type(img)}")

    def _preprocess(self, im: Image.Image) -> torch.Tensor:
        """resize 448 (single tile, no anyres) → [0,1] → ImageNet normalize → [3,448,448]."""
        import numpy as np
        im = im.convert("RGB").resize((self._img_size, self._img_size), Image.BICUBIC)
        arr = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0)  # [H,W,3]
        arr = arr.permute(2, 0, 1)                                        # [3,H,W]
        arr = (arr - self._img_mean) / self._img_std
        return arr

    @torch.no_grad()
    def encode_patches(self, images: list) -> tuple[torch.Tensor, int, torch.Tensor]:
        """Single-tile InternViT → pixel-shuffle → projector → [n_images*256, hidden].
        Delegates to the model's own get_image_features (vision_tower → CLS-drop →
        pixel_shuffle(0.5) → multi_modal_projector), then flattens.

        Returns (vis, n_vis_per_patch, grids) — grids = [[1,16,16]]×N of (t,h,w) per
        crop, which the Reasoner needs to map bboxes → ROI spans. sms=1 for this
        backbone, so 16//1=16 → 256 tokens/ROI.
        """
        pil = [self._to_pil(im) for im in images]
        pv = torch.stack([self._preprocess(im) for im in pil]).to(
            self.device, dtype=self.dtype)                     # [N,3,448,448]
        vo = self.model.model.get_image_features(
            pixel_values=pv, vision_feature_layer=self._vfl,
            vision_feature_select_strategy=self._vfss)
        feat = vo.pooler_output                                # [N,256,hidden]
        vis = feat.reshape(-1, feat.shape[-1])                 # [N*256, hidden]
        grids = torch.tensor([[1, _VIS_GRID, _VIS_GRID]] * len(pil), dtype=torch.long)
        return vis, _N_VIS_PER_IMAGE, grids

    # ── text embedding (chatml) ─────────────────────────────────────────────────
    def _build_prompt(self, text: str, system_prompt: str | None, with_image: bool) -> str:
        body = text.replace(_IMAGE_TOKEN, "")
        if with_image:
            body = f"{_IMAGE_TOKEN}\n" + body
        sys_seg = (f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                   if system_prompt else "")
        return (f"{sys_seg}<|im_start|>user\n{body}<|im_end|>\n"
                f"<|im_start|>assistant\n")

    @torch.no_grad()
    def embed_text(
        self,
        text: str,
        system_prompt: str | None = None,
        enable_thinking: bool = False,   # unused (Qwen2 has no thinking token)
        with_image: bool = False,
    ) -> torch.Tensor:
        """Tokenise + embed. With image, inserts a zero placeholder at the
        <IMG_CONTEXT> slot (filled by forward_with_latent_steps); records
        self._image_token_pos."""
        prompt = self._build_prompt(text, system_prompt, with_image)

        if with_image:
            ids = self._tokenizer_image_token(prompt)
            img_positions = (ids == self.image_token_id).nonzero(as_tuple=True)[0]
            if len(img_positions) > 0:
                self._image_token_pos = int(img_positions[0].item())
                D = self.lm.embed_tokens.embedding_dim
                pre_ids  = ids[:self._image_token_pos]
                post_ids = ids[self._image_token_pos + 1:]
                pre_emb  = (self.lm.embed_tokens(pre_ids.unsqueeze(0)).squeeze(0)
                            if pre_ids.numel() > 0
                            else torch.empty(0, D, device=self.device, dtype=self.dtype))
                post_emb = self.lm.embed_tokens(post_ids.unsqueeze(0)).squeeze(0)
                dummy    = torch.zeros(1, D, device=self.device, dtype=self.dtype)
                return torch.cat([pre_emb, dummy, post_emb], dim=0)
            self._image_token_pos = None
        else:
            self._image_token_pos = None

        ids = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False,
        )["input_ids"].to(self.device)
        return self.lm.embed_tokens(ids).squeeze(0)

    # ── position ids (1D) ────────────────────────────────────────────────────────
    def _make_position_ids(self, seq_len: int, past_len: int = 0) -> torch.Tensor:
        return torch.arange(past_len, past_len + seq_len, device=self.device).unsqueeze(0)

    @contextmanager
    def _attn_impl(self, impl: str):
        """Temporarily switch the LM attention implementation. The backbone loads
        eager (so latent/saliency forwards can read output_attentions), but eager
        materializes the full [H, S, S] score matrix — a long carried KV → O(S²) OOM.
        Wrap generate() paths (which never read attentions) in "sdpa" to avoid it.
        """
        targets = [self.model.config,
                   getattr(self.model.config, "text_config", None),
                   getattr(self.lm, "config", None)]
        saved = [(t, getattr(t, "_attn_implementation", None)) for t in targets if t is not None]
        layer_saved = [(l.self_attn, l.self_attn._attn_implementation)
                       for l in self.lm.layers if hasattr(l.self_attn, "_attn_implementation")]
        for t, _ in saved:
            t._attn_implementation = impl
        for sa, _ in layer_saved:
            sa._attn_implementation = impl
        try:
            yield
        finally:
            for t, v in saved:
                t._attn_implementation = v
            for sa, v in layer_saved:
                sa._attn_implementation = v

    # ── latent step loop (identical structure to huatuo/llavaov) ────────────────
    @torch.no_grad()
    def forward_with_latent_steps(
        self,
        visual_embeds: torch.Tensor,   # [n_vis, hidden]
        text_embeds:   torch.Tensor,   # [n_q,  hidden]  (may hold <IMG_CONTEXT> placeholder)
        m: int,
        l_mid_layers: list[int],
    ) -> dict:
        n_vis = visual_embeds.shape[0]

        img_pos = getattr(self, "_image_token_pos", None)
        if img_pos is not None:
            pre_emb  = text_embeds[:img_pos]
            post_emb = text_embeds[img_pos + 1:]
            input_embeds = torch.cat([pre_emb, visual_embeds, post_emb], dim=0).unsqueeze(0)
            n_pre = pre_emb.shape[0]
            n_q   = pre_emb.shape[0] + post_emb.shape[0]
            vis_start = n_pre
        else:
            input_embeds = torch.cat(
                [visual_embeds.unsqueeze(0), text_embeds.unsqueeze(0)], dim=1)
            n_q       = text_embeds.shape[0]
            vis_start = 0

        seq = input_embeds.shape[1]
        pos_ids = self._make_position_ids(seq)

        out = self.lm(
            inputs_embeds=input_embeds, position_ids=pos_ids,
            use_cache=True, output_attentions=True, output_hidden_states=True,
        )
        past_kv     = out.past_key_values
        last_hidden = out.hidden_states[-1][:, -1:, :]
        vis_hidden = out.hidden_states[-1][0, vis_start:vis_start + n_vis, :].detach().cpu()

        query_to_vision_attn: dict[int, torch.Tensor] = {}
        attn_ok = out.attentions is not None and len(out.attentions) > 0
        for li in l_mid_layers:
            if not attn_ok or li >= len(out.attentions):
                continue
            a = out.attentions[li]
            if a is not None:
                text_start = vis_start + n_vis
                query_to_vision_attn[li] = a[0, :, text_start:, vis_start:vis_start + n_vis].cpu()

        latent_trajectory: list[torch.Tensor] = []
        l2v_lists: dict[int, list[torch.Tensor]] = {li: [] for li in l_mid_layers}

        for step in range(m):
            latent_embed = self._apply_realign(last_hidden)
            pos_ids_step = self._make_position_ids(1, past_len=seq + step)
            out = self.lm(
                inputs_embeds=latent_embed, position_ids=pos_ids_step,
                past_key_values=past_kv, use_cache=True,
                output_attentions=True, output_hidden_states=True,
            )
            past_kv     = out.past_key_values
            last_hidden = out.hidden_states[-1][:, -1:, :]
            latent_trajectory.append(last_hidden.squeeze().cpu())

            sa = out.attentions
            sa_ok = sa is not None and len(sa) > 0
            for li in l_mid_layers:
                if not sa_ok or li >= len(sa):
                    continue
                a = sa[li]
                if a is not None:
                    l2v_lists[li].append(a[0, :, 0, vis_start:vis_start + n_vis].cpu())

        latent_to_vision_attn = {
            li: torch.stack(lst, dim=0) for li, lst in l2v_lists.items() if lst
        }
        full_attn_kv = self._extract_full_attn_kv(past_kv, l_mid_layers)

        if img_pos is not None:
            n_pre, n_post = vis_start, n_q - vis_start
            spans = [s for s in (("P", n_pre), ("V", n_vis), ("P", n_post), ("L", m)) if s[1] > 0]
        else:
            spans = [s for s in (("V", n_vis), ("P", n_q), ("L", m)) if s[1] > 0]

        return {
            "past_key_values":       past_kv,
            "latent_trajectory":     latent_trajectory,
            "latent_to_vision_attn": latent_to_vision_attn,
            "query_to_vision_attn":  query_to_vision_attn,
            "full_attn_kv":          full_attn_kv,
            "past_len":              seq + m,
            "pos_cursor":            seq + m,
            "spans":                 spans,
            "vis_hidden":            vis_hidden,
            "vis_cols":              list(range(vis_start, vis_start + n_vis)),
        }

    @torch.no_grad()
    def grounded_prefill_and_latent(
        self,
        images: list[Image.Image],
        system_prompt: str,
        user_text: str,
        m: int,
        l_mid_layers: list[int],
        enable_thinking: bool,
        want_attn: bool,
        role_targets: list[str] | None = None,
        bg_targets: list[str] | None = None,
    ) -> dict:
        """Start a training-free latent handoff from grounded InternVL features."""
        _ = enable_thinking, role_targets, bg_targets
        return self._grounded_prefill(
            images=images,
            system_prompt=system_prompt,
            user_text=user_text,
            m=m,
            l_mid_layers=l_mid_layers,
            want_attn=want_attn,
            past_kv=None,
            past_len=0,
        )

    @torch.no_grad()
    def grounded_prefill_and_latent_on_kv(
        self,
        past_kv,
        past_pos_cursor: int,
        images: list[Image.Image],
        system_prompt: str,
        user_text: str,
        m: int,
        l_mid_layers: list[int],
        enable_thinking: bool,
        want_attn: bool,
        role_targets: list[str] | None = None,
        bg_targets: list[str] | None = None,
        continuation_user_turn: bool = False,
    ) -> dict:
        """Append a grounded InternVL role to an existing latent KV cache."""
        _ = past_pos_cursor, enable_thinking, role_targets, bg_targets, continuation_user_turn
        return self._grounded_prefill(
            images=images,
            system_prompt=system_prompt,
            user_text=user_text,
            m=m,
            l_mid_layers=l_mid_layers,
            want_attn=want_attn,
            past_kv=past_kv,
            past_len=self._extract_kv_len(past_kv),
        )

    @torch.no_grad()
    def _grounded_prefill(
        self,
        *,
        images: list[Image.Image],
        system_prompt: str,
        user_text: str,
        m: int,
        l_mid_layers: list[int],
        want_attn: bool,
        past_kv,
        past_len: int,
    ) -> dict:
        """Inject visual embeddings at the image marker and append latent tokens."""
        visual_embeds, _, _ = self.encode_patches(images)
        text_embeds = self.embed_text(
            user_text,
            system_prompt=system_prompt,
            with_image=True,
        )
        image_position = self._image_token_pos
        if image_position is None:
            raise RuntimeError("InternVL prompt did not contain an image marker")
        pre_embeds = text_embeds[:image_position]
        post_embeds = text_embeds[image_position + 1 :]
        inputs_embeds = torch.cat(
            [pre_embeds, visual_embeds, post_embeds], dim=0
        ).unsqueeze(0)
        sequence_length = inputs_embeds.shape[1]
        vision_start = image_position
        vision_end = vision_start + visual_embeds.shape[0]
        output = self.lm(
            inputs_embeds=inputs_embeds,
            position_ids=self._make_position_ids(sequence_length, past_len=past_len),
            past_key_values=past_kv,
            use_cache=True,
            # The prefill attention is never consumed below.  Requesting it with a
            # long carried KV materializes an eager O(S²) matrix and OOMs before
            # the one-token latent steps can collect their saliency.
            output_attentions=False,
            output_hidden_states=True,
        )
        cache = output.past_key_values
        hidden = output.hidden_states[-1][:, -1:, :]
        spans = [
            span
            for span in (
                ("P", int(pre_embeds.shape[0])),
                ("V", int(visual_embeds.shape[0])),
                ("P", int(post_embeds.shape[0])),
            )
            if span[1] > 0
        ]
        from memory.prune import apply_kv_prune, build_prune_controller

        controller = build_prune_controller(getattr(self, "_prune_args", None))
        vision_columns = torch.arange(
            past_len + vision_start,
            past_len + vision_end,
            device=self.device,
            dtype=torch.long,
        )
        cache_length = past_len + sequence_length
        if controller is not None:
            controller.start(
                grids=None,
                vis_idx=vision_columns,
                sms=1,
                images=images,
                q2v={},
                l_mid_layers=l_mid_layers,
                image_magnifications=getattr(self, "_prune_image_magnifications", ()),
                morphology_enabled=getattr(self, "_prune_morphology_enabled", False),
            )
        latent_to_vision: dict[int, list[torch.Tensor]] = {
            layer: [] for layer in l_mid_layers
        }
        evicted = False
        for step in range(m):
            output = self.lm(
                inputs_embeds=self._apply_realign(hidden),
                position_ids=self._make_position_ids(
                    1, past_len=past_len + sequence_length + step
                ),
                past_key_values=cache,
                use_cache=True,
                output_attentions=want_attn,
                output_hidden_states=True,
            )
            cache = output.past_key_values
            hidden = output.hidden_states[-1][:, -1:, :]
            cache_length += 1
            spans.append(("L", 1))
            if want_attn and output.attentions is not None:
                for layer in l_mid_layers:
                    attention = output.attentions[layer]
                    if attention is not None:
                        latent_to_vision[layer].append(
                            attention[0, :, 0, vision_columns]
                            .detach()
                            .cpu()
                        )
            if controller is not None and output.attentions is not None:
                matrices = [
                    output.attentions[layer][0, :, 0, vision_columns].float()
                    for layer in l_mid_layers
                    if output.attentions[layer] is not None
                ]
                if matrices:
                    controller.observe(step, torch.stack(matrices, dim=0).mean(0).cpu().numpy())
                keep = controller.maybe_keep_mask(step)
                if keep is not None and not keep.all():
                    cache, cache_length, spans, kept_columns = apply_kv_prune(
                        cache,
                        cache_length,
                        vision_columns.tolist(),
                        keep,
                        spans,
                        self.device,
                        in_place=True,
                    )
                    vision_columns = torch.tensor(kept_columns, device=self.device, dtype=torch.long)
                    controller.commit(keep)
                    evicted = True
                    print(
                        f"[PruneCtrl] step {step + 1}: dropped {int((~keep).sum())}/"
                        f"{len(keep)} vis → KV {cache_length} tokens",
                        flush=True,
                    )
        total_length = cache_length
        return {
            "past_key_values": cache,
            "past_len": total_length,
            "pos_cursor": total_length,
            "spans": spans,
            "vis_cols": vision_columns.tolist(),
            "latent_to_vision_attn": (
                {}
                if evicted
                else {
                    layer: torch.stack(values, dim=0)
                    for layer, values in latent_to_vision.items()
                    if values
                }
            ),
            "probe_rv": None,
        }

    @torch.no_grad()
    def close_assistant_turn(
        self,
        past_kv,
        past_pos_cursor: int,
        *,
        close_thinking: bool = False,
    ) -> tuple[object, int, int]:
        """Append InternVL's ChatML assistant-turn terminator to a carried KV cache."""
        _ = close_thinking
        boundary_ids = self.tokenizer(
            "<|im_end|>\n", return_tensors="pt", add_special_tokens=False
        )["input_ids"].to(self.device)
        boundary_embeds = self.lm.embed_tokens(boundary_ids)
        cache_length = self._extract_kv_len(past_kv)
        output = self.lm(
            inputs_embeds=boundary_embeds,
            position_ids=self._make_position_ids(
                boundary_embeds.shape[1], past_len=cache_length
            ),
            past_key_values=past_kv,
            use_cache=True,
            output_hidden_states=False,
        )
        updated_length = cache_length + boundary_embeds.shape[1]
        return output.past_key_values, updated_length, past_pos_cursor + boundary_embeds.shape[1]

    @torch.no_grad()
    def saliency_scan(
        self,
        image: Image.Image,
        prompt_text: str,
        m: int,
        l_mid_layers: list[int],
    ) -> dict:
        """Fresh (no prior KV) content-driven saliency over ONE image, for the
        Scanner's `--scanner_method saliency` path. See huatuogpt.saliency_scan."""
        vis, n_vis, _ = self.encode_patches([image])
        text_embeds = self.embed_text(prompt_text, system_prompt=None, with_image=True)

        img_pos = getattr(self, "_image_token_pos", None)
        if img_pos is not None:
            pre_emb  = text_embeds[:img_pos]
            post_emb = text_embeds[img_pos + 1:]
            input_embeds = torch.cat([pre_emb, vis, post_emb], dim=0).unsqueeze(0)
            vis_start = pre_emb.shape[0]
        else:
            input_embeds = torch.cat([vis.unsqueeze(0), text_embeds.unsqueeze(0)], dim=1)
            vis_start = 0

        seq = input_embeds.shape[1]
        out = self.lm(
            inputs_embeds=input_embeds, position_ids=self._make_position_ids(seq),
            use_cache=True, output_attentions=True, output_hidden_states=True,
        )
        past_kv     = out.past_key_values
        last_hidden = out.hidden_states[-1][:, -1:, :]
        vis_hidden  = out.hidden_states[-1][0, vis_start:vis_start + n_vis, :].float().cpu()

        latent_trajectory: list[torch.Tensor] = []
        l2v_lists: dict[int, list[torch.Tensor]] = {li: [] for li in l_mid_layers}
        for step in range(m):
            latent_embed = self._apply_realign(last_hidden)
            out = self.lm(
                inputs_embeds=latent_embed,
                position_ids=self._make_position_ids(1, past_len=seq + step),
                past_key_values=past_kv, use_cache=True,
                output_attentions=True, output_hidden_states=True,
            )
            past_kv     = out.past_key_values
            last_hidden = out.hidden_states[-1][:, -1:, :]
            latent_trajectory.append(last_hidden.squeeze().cpu())
            sa = out.attentions
            if sa is not None and len(sa) > 0:
                for li in l_mid_layers:
                    if li < len(sa) and sa[li] is not None:
                        l2v_lists[li].append(
                            sa[li][0, :, 0, vis_start:vis_start + n_vis].cpu())

        latent_to_vision_attn = {
            li: torch.stack(lst, dim=0) for li, lst in l2v_lists.items() if lst
        }
        gh = gw = _VIS_GRID                                    # 256 → 16×16
        grids = torch.tensor([[1, gh, gw]], dtype=torch.long)
        return {
            "grids":                 grids,
            "n_vis":                 n_vis,
            "vis_hidden":            vis_hidden,
            "latent_trajectory":     latent_trajectory,
            "latent_to_vision_attn": latent_to_vision_attn,
        }

    @torch.no_grad()
    def append_grounded_vision_and_latent(
        self, past_kv, past_pos_cursor, image: Image.Image, framing_text: str,
        m: int, l_mid_layers: list[int], want_attn: bool = True,
    ) -> dict:
        """Append [framing_text | vision(256)] onto a CARRIED KV + m latent steps —
        KV-carry counterpart of saliency_scan (Scanner scan_on_kv). See huatuo."""
        past_len = self._extract_kv_len(past_kv)

        fr_ids = self.tokenizer(framing_text, return_tensors="pt",
                                add_special_tokens=False)["input_ids"].to(self.device)
        fr_emb = self.lm.embed_tokens(fr_ids).squeeze(0)
        vis, n_vis, _ = self.encode_patches([image])
        n_pre = fr_emb.shape[0]
        inp = torch.cat([fr_emb, vis], dim=0).unsqueeze(0)
        L = inp.shape[1]
        vis_start = n_pre

        out = self.lm(inputs_embeds=inp,
                      position_ids=self._make_position_ids(L, past_len=past_len),
                      past_key_values=past_kv, use_cache=True,
                      output_attentions=want_attn, output_hidden_states=True)
        past_kv     = out.past_key_values
        last_hidden = out.hidden_states[-1][:, -1:, :]
        vis_hidden  = out.hidden_states[-1][0, vis_start:vis_start + n_vis, :].float().cpu()

        vis_cols = torch.arange(past_len + vis_start, past_len + vis_start + n_vis)
        spans = [("P", n_pre), ("V", n_vis)]
        seq_after = past_len + L

        latent_trajectory: list[torch.Tensor] = []
        l2v: dict[int, list[torch.Tensor]] = {li: [] for li in l_mid_layers}
        for step in range(m):
            le = self._apply_realign(last_hidden)
            out = self.lm(inputs_embeds=le,
                          position_ids=self._make_position_ids(1, past_len=seq_after + step),
                          past_key_values=past_kv, use_cache=True,
                          output_attentions=want_attn, output_hidden_states=True)
            past_kv     = out.past_key_values
            last_hidden = out.hidden_states[-1][:, -1:, :]
            latent_trajectory.append(last_hidden.squeeze().cpu())
            if want_attn and out.attentions:
                for li in l_mid_layers:
                    if li < len(out.attentions) and out.attentions[li] is not None:
                        l2v[li].append(out.attentions[li][0, :, 0, vis_cols].cpu())
        if m > 0:
            spans.append(("L", m))

        gh = gw = _VIS_GRID
        return {
            "past_key_values":       past_kv,
            "past_len":              seq_after + m,
            "pos_cursor":            seq_after + m,
            "vis_cols":              vis_cols.cpu(),
            "spans":                 spans,
            "grids":                 torch.tensor([[1, gh, gw]], dtype=torch.long),
            "n_vis":                 n_vis,
            "vis_hidden":            vis_hidden,
            "latent_trajectory":     latent_trajectory,
            "latent_to_vision_attn": {li: torch.stack(v, 0) for li, v in l2v.items() if v},
        }

    @torch.no_grad()
    def drop_vision_columns(self, past_kv, past_len: int, vis_cols, spans):
        """Drop the appended vision block from a KV (latent_only thread mode)."""
        import numpy as np
        from memory.prune import apply_kv_prune
        n = (int(vis_cols.numel()) if hasattr(vis_cols, "numel") else len(vis_cols)) \
            if vis_cols is not None else 0
        if n == 0:
            return past_kv, past_len, spans
        keep = np.zeros(n, dtype=bool)
        pk, new_len, new_spans, _ = apply_kv_prune(
            past_kv, past_len, vis_cols, keep, spans, self.device)
        print(f"[VisionDrop/internvl] dropped {n} vision cols → KV {past_len} → {new_len}")
        return pk, new_len, new_spans

    @torch.no_grad()
    def continue_with_latent_steps(
        self, text_embeds: torch.Tensor, past_kv, past_len: int, m: int,
        l_mid_layers: list[int], past_pos_cursor: int | None = None,  # accepted; 1D RoPE
        capture_hidden_layers: list[int] | None = None,   # per-layer hidden capture
        vis_cols=None,                # absolute vision cols → capture q2v/l2v (rescore)
        realloc_alpha_schedule=None,  # optional [m] per-step α (Verifier S8 schedule)
    ) -> dict:
        """Append a new prompt to existing KV and run m more latent steps (1D RoPE).

        Mirrors llavaov/qwen3vl.continue_with_latent_steps so the carry pipeline's
        Verifier refinement (registry_rescore joint / attn_realloc schedule) works.
        """
        n_t = text_embeds.shape[0]

        _cap = vis_cols is not None and l_mid_layers and (
            (int(vis_cols.numel()) if hasattr(vis_cols, "numel") else len(vis_cols)) > 0)
        _vc = None
        if _cap:
            _vc = torch.as_tensor([int(c) for c in (vis_cols.tolist()
                    if hasattr(vis_cols, "tolist") else vis_cols)],
                    dtype=torch.long, device=self.device)
        q2v: dict[int, torch.Tensor] = {}
        l2v_lists: dict[int, list] = {li: [] for li in (l_mid_layers or [])} if _cap else {}

        # ── prefill new prompt on top of past_kv ──────────────────────────────
        out = self.lm(
            inputs_embeds=text_embeds.unsqueeze(0),
            position_ids=self._make_position_ids(n_t, past_len=past_len),
            past_key_values=past_kv, use_cache=True,
            output_attentions=_cap, output_hidden_states=True,
        )
        past_kv     = out.past_key_values
        last_hidden = out.hidden_states[-1][:, -1:, :]
        if _cap and out.attentions is not None:
            for li in l_mid_layers:
                a = out.attentions[li]
                if a is not None:
                    q2v[li] = a[0][:, :, _vc].cpu()   # [H, n_t, n_vis]
        past_len   += n_t

        # S8 per-step α schedule (global reallocation strength); restore base after.
        _sched = realloc_alpha_schedule
        _alpha_base = None
        if _sched is not None:
            try:
                from memory.reallocate import get_active_alpha
                _alpha_base = get_active_alpha()
            except Exception:
                _sched = None

        latent_trajectory: list[torch.Tensor] = []
        for step in range(m):
            if _sched is not None and step < len(_sched):
                try:
                    from memory.reallocate import set_active_alpha
                    set_active_alpha(_sched[step])
                except Exception:
                    pass
            latent_embed = self._apply_realign(last_hidden)
            out = self.lm(
                inputs_embeds=latent_embed,
                position_ids=self._make_position_ids(1, past_len=past_len + step),
                past_key_values=past_kv, use_cache=True,
                output_attentions=_cap, output_hidden_states=True,
            )
            past_kv     = out.past_key_values
            last_hidden = out.hidden_states[-1][:, -1:, :]
            latent_trajectory.append(last_hidden.squeeze().cpu())
            if _cap and out.attentions is not None:
                for li in l_mid_layers:
                    a = out.attentions[li]
                    if a is not None:
                        l2v_lists[li].append(a[0][:, 0, _vc].cpu())   # [H, n_vis]

        if _sched is not None and _alpha_base is not None:
            try:
                from memory.reallocate import set_active_alpha
                set_active_alpha(_alpha_base)
            except Exception:
                pass

        latent_to_vision_attn = {li: torch.stack(v, 0) for li, v in l2v_lists.items() if v}
        layer_hidden: dict[int, torch.Tensor] = {}
        if capture_hidden_layers:
            for li in capture_hidden_layers:
                if 0 <= li < len(out.hidden_states):
                    layer_hidden[li] = out.hidden_states[li][:, -1:, :].squeeze().cpu()

        return {
            "past_key_values":   past_kv,
            "latent_trajectory": latent_trajectory,
            "full_attn_kv":      self._extract_full_attn_kv(past_kv, l_mid_layers),
            "past_len":          past_len + m,
            "pos_cursor":        past_len + m,
            "spans":             [s for s in (("P", n_t), ("L", m)) if s[1] > 0],
            "layer_hidden":          layer_hidden,
            "query_to_vision_attn":  q2v,
            "latent_to_vision_attn": latent_to_vision_attn,
        }

    # ── KV helpers ───────────────────────────────────────────────────────────────
    def _extract_full_attn_kv(self, past_kv, l_mid_layers: list[int]):
        result: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for li in l_mid_layers:
            try:
                if hasattr(past_kv, "layers"):
                    layer = past_kv.layers[li]
                    K, V = layer.keys.cpu(), layer.values.cpu()
                elif hasattr(past_kv, "key_cache"):
                    K, V = past_kv.key_cache[li].cpu(), past_kv.value_cache[li].cpu()
                else:
                    K, V = past_kv[li][0].cpu(), past_kv[li][1].cpu()
                result[li] = (K, V)
            except (IndexError, TypeError, AttributeError):
                continue
        return result

    @staticmethod
    def _extract_kv_len(kv) -> int:
        try:
            if hasattr(kv, "get_seq_length"):
                return int(kv.get_seq_length())
            if hasattr(kv, "layers"):
                return kv.layers[0].keys.shape[-2]
            if hasattr(kv, "key_cache"):
                return kv.key_cache[0].shape[-2]
            return kv[0][0].shape[-2]
        except Exception:
            return 0

    @staticmethod
    def _kv_len(kv) -> int:
        """Expose the cache-length protocol used by the shared terminal decoder."""
        return InternVLBackbone._extract_kv_len(kv)

    # ── Scanner interface: single-image VL generation ───────────────────────────
    @torch.no_grad()
    def vl_generate(
        self, image: Image.Image, prompt: str, system_prompt: str = "",
        max_new_tokens: int = 1024,
    ) -> str:
        text_prompt = self._build_prompt(prompt, system_prompt or None, with_image=True)
        return self._generate_with_images([image], text_prompt, max_new_tokens)

    # ── text_mas interface: multi-image VL generation ───────────────────────────
    @torch.no_grad()
    def vlm_generate(
        self, images, system_prompt: str, user_text: str,
        max_new_tokens: int = 512, tag: str = "",
    ) -> str:
        """text_mas token-space channel: SEE 0..N crops and DECODE text. N
        <IMG_CONTEXT> markers ↔ N images (processor expands each to its token count).
        images=[]/None → text-only. Only used by the text_mas channel."""
        imgs = [self._to_pil(im) for im in (images or [])]
        n = len(imgs)
        body = user_text.replace(_IMAGE_TOKEN, "")
        body = (f"{_IMAGE_TOKEN}\n" * n) + body
        sys_seg = (f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                   if system_prompt else "")
        text_prompt = (f"{sys_seg}<|im_start|>user\n{body}<|im_end|>\n"
                       f"<|im_start|>assistant\n")
        _rp = float(getattr(self, "textmas_rep_penalty", 1.05))
        _nr = int(getattr(self, "textmas_no_repeat_ngram", 0))
        return self._generate_with_images(
            imgs, text_prompt, max_new_tokens, rep_penalty=_rp, no_repeat=_nr, tag=tag)

    @torch.no_grad()
    def _generate_with_images(self, imgs, text_prompt, max_new_tokens,
                              rep_penalty: float = 1.0, no_repeat: int = 0,
                              tag: str = "") -> str:
        """Shared HF-processor generate: expands <IMG_CONTEXT> markers to image tokens."""
        _timer, _t0 = self._ttft_timer()
        _gk = dict(do_sample=False, max_new_tokens=max_new_tokens, use_cache=True,
                   eos_token_id=self.eos_ids,
                   pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        if rep_penalty and rep_penalty != 1.0:
            _gk["repetition_penalty"] = rep_penalty
        if no_repeat and no_repeat > 0:
            _gk["no_repeat_ngram_size"] = no_repeat
        if _timer is not None:
            _gk["logits_processor"] = [_timer]

        # sdpa during generate: eager would materialize O(S²) attention over the
        # (multi-crop) prompt → OOM. Generation never reads attentions.
        if imgs and self.processor_full is not None:
            inputs = self.processor_full(
                images=imgs, text=text_prompt, return_tensors="pt",
            ).to(self.device, self.dtype)
            in_len = inputs["input_ids"].shape[1]
            with self._attn_impl("sdpa"):
                out = self.model.generate(**inputs, **_gk)
        else:
            ids = self.tokenizer(text_prompt, return_tensors="pt",
                                 add_special_tokens=False)["input_ids"].to(self.device)
            in_len = ids.shape[1]
            with self._attn_impl("sdpa"):
                out = self.model.generate(input_ids=ids, **_gk)
        self._record_ttft(_timer, _t0)

        if getattr(self, "usage", None) is None:
            self.usage = []
        self.usage.append({"tag": tag, "prompt_tokens": int(in_len),
                           "gen_tokens": int(out[0].shape[0] - in_len)})
        new_ids = out[0, in_len:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    # ── unified text decode on top of a KV (Qwen2 native) ───────────────────────
    def _cjk_suppress_ids(self) -> list[int]:
        """Token ids whose decoded piece contains CJK — suppressed when english_only
        (mirrors qwen3vl). Cached after first build."""
        if getattr(self, "_cjk_ids_cache", None) is not None:
            return self._cjk_ids_cache
        import re as _re
        _cjk = _re.compile(r"[　-〿぀-ヿ㐀-䶿一-鿿豈-﫿＀-￯]")
        tok = self.tokenizer
        special = set(tok.all_special_ids or [])
        ids: list[int] = []
        for tid in tok.get_vocab().values():
            if tid in special:
                continue
            if _cjk.search(tok.decode([tid])):
                ids.append(tid)
        self._cjk_ids_cache = ids
        print(f"[internvl] english_only: suppressing {len(ids)} CJK token ids")
        return ids

    @torch.no_grad()
    def generate_on_kv(
        self, system_prompt: str, user_text: str, past_kv, max_new_tokens: int,
        clone_cache: bool = False, pos_cursor: int | None = None,
        do_sample: bool = False, rep_penalty: float = 1.0, no_repeat: int = 0,
        temperature: float = 0.7, top_p: float = 0.9, enable_thinking: bool = False,
        english_only: bool = False,
        continuation_user_turn: bool = False,
    ) -> str:
        """Re-prefill clean text at an assistant boundary and decode on top of the
        carried KV. Uses the LM's generate directly (text-only, no image branch)."""
        _ = continuation_user_turn
        prompt = self._build_prompt(user_text, system_prompt or None, with_image=False)
        input_ids = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False,
        )["input_ids"].to(self.device)
        attention_mask = torch.ones_like(input_ids)

        cache_source = copy.deepcopy(past_kv) if clone_cache else past_kv
        if cache_source is not None:
            past_len = self._extract_kv_len(cache_source)
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)

        gen_kwargs = dict(
            input_ids=input_ids, attention_mask=attention_mask,
            past_key_values=cache_source,
            do_sample=do_sample, max_new_tokens=max_new_tokens,
            eos_token_id=self.eos_ids,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            use_cache=True,
        )
        if rep_penalty and rep_penalty != 1.0:
            gen_kwargs["repetition_penalty"] = rep_penalty
        if no_repeat and no_repeat > 0:
            gen_kwargs["no_repeat_ngram_size"] = no_repeat
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        if english_only:
            _sup = self._cjk_suppress_ids()
            if _sup:
                gen_kwargs["suppress_tokens"] = _sup

        # Call the stock HF generate on the LM (Qwen2) so [past|new] mask +
        # past_key_values behave normally (no image branch on a carried KV).
        from transformers import GenerationMixin
        _timer, _t0 = self._ttft_timer()
        if _timer is not None:
            gen_kwargs["logits_processor"] = [_timer]
        with self._attn_impl("sdpa"):   # avoid O(S²) eager attn on the carried KV
            output_ids = GenerationMixin.generate(self.model, **gen_kwargs)
        self._record_ttft(_timer, _t0)
        new_ids = output_ids[:, input_ids.shape[1]:]
        return self.tokenizer.decode(new_ids[0], skip_special_tokens=True).strip()

    # ── no-decode logit read on top of a KV (Verifier sufficiency gate) ─────────
    @torch.no_grad()
    def last_token_logits(
        self, system_prompt: str, user_text: str, past_kv,
        pos_cursor: int | None = None, clone_cache: bool = True,
    ) -> torch.Tensor:
        embeds = self.embed_text(
            user_text, system_prompt=system_prompt, with_image=False,
        ).unsqueeze(0)
        L = embeds.shape[1]
        past = copy.deepcopy(past_kv) if clone_cache else past_kv
        past_len = self._extract_kv_len(past)
        out = self.lm(
            inputs_embeds=embeds,
            position_ids=self._make_position_ids(L, past_len=past_len),
            past_key_values=past, use_cache=True, output_hidden_states=True,
        )
        h = out.hidden_states[-1][:, -1, :]
        wd = self.model.lm_head.weight.dtype
        return self.model.lm_head(h.to(wd))[0].float()

    @torch.no_grad()
    def generate_on_kv_realloc(
        self, system_prompt: str, user_text: str, past_kv, vis_cols, alpha: float,
        layers, max_new_tokens: int, pos_cursor: int | None = None,
        clone_cache: bool = True, vis_weight=None,
    ) -> str:
        """S8 attention-reallocation decode. As with llava_onevision, the eager
        reallocating() monkeypatch is unvalidated for this HF-native attention stack,
        so we DELEGATE to plain generate_on_kv (decode reallocation is a no-op here).
        The run completes with faithful greedy output; only the experimental
        α-reallocation at the Diagnosis decode is inert. Prefer non-realloc variants.
        """
        print(f"[internvl] generate_on_kv_realloc → plain decode "
              f"(α={alpha} reallocation inert on this backbone)")
        return self.generate_on_kv(
            system_prompt=system_prompt, user_text=user_text, past_kv=past_kv,
            max_new_tokens=max_new_tokens, clone_cache=clone_cache,
            pos_cursor=pos_cursor,
            english_only=bool(getattr(self, "_diag_force_english", False)),
        )
