"""Quilt-LLaVA backbone for WSI-LatentMAS.

Model: LlavaLlamaForCausalLM (Llama-2 7B + CLIP ViT + mm_projector)
Checkpoint: /media/super/4TB/hj/wsi-llava-v1.5-7b-e1

Key differences from Qwen3-VL:
  - Vision: CLIP ViT → mm_projector → 576 fixed tokens per image
  - LM: Llama-2 (all layers are full attention, no sliding window)
  - Position IDs: standard 1D [1, seq] (not MRoPE)
  - KV cache: tuple-of-tuples  past_kv[layer][0/1]  (not DynamicCache)
  - Prompt: llava_v1 conversation template  (no thinking token management)
  - No image_grid_thw: pixel_values only
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image

# ── quilt-llava repo must be on sys.path ──────────────────────────────────────
_QUILT_REPO = Path("/home/super/hj/HJ/quilt-llava")
if str(_QUILT_REPO) not in sys.path:
    sys.path.insert(0, str(_QUILT_REPO))

# ── MPT compatibility shim ────────────────────────────────────────────────────
# quilt-llava's MPT code imports _expand_mask/_make_causal_mask from bloom,
# both removed in transformers ≥ 4.35. Stub the whole MPT subpackage so the
# rest of the llava codebase (LlavaLlamaForCausalLM etc.) loads cleanly.
import types as _types
for _mod_name in [
    "llava.model.language_model.mpt",
    "llava.model.language_model.mpt.modeling_mpt",
    "llava.model.language_model.mpt.hf_prefixlm_converter",
    "llava.model.language_model.mpt.configuration_mpt",
    "llava.model.language_model.mpt.adapt_tokenizer",
    "llava.model.language_model.mpt.meta_init_context",
    "llava.model.language_model.mpt.norm",
    "llava.model.language_model.mpt.param_init_fns",
]:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = _types.ModuleType(_mod_name)

# Provide stub classes so llava_mpt.py can subclass them without errors
_mpt_stub = sys.modules["llava.model.language_model.mpt.modeling_mpt"]
for _sym in ("MPTConfig", "MPTForCausalLM", "MPTModel"):
    if not hasattr(_mpt_stub, _sym):
        setattr(_mpt_stub, _sym, type(_sym, (), {}))
# ─────────────────────────────────────────────────────────────────────────────

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images
from llava.conversation import conv_templates
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX


_N_VIS_PER_IMAGE = 576   # fixed for LLaVA-style CLIP ViT with 336px input


class _TokenizerWrapper:
    """Minimal wrapper so agents can use bk.processor.tokenizer uniformly."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


from backbone.efficiency import EfficiencyMixin


class QuiltLLaVABackbone(EfficiencyMixin):
    """Quilt-LLaVA adapter with LatentMAS-style hidden-state realignment."""

    def __init__(
        self,
        model_path: str = "/home/super/hj/HJ/quilt-llava/qllava_ckpt",
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float16,
        latent_space_realign: bool = True,
        realign_method: str = "wa",   # "wa" | "softmax"
        realign_tau: float = 1.0,
    ) -> None:
        self.device = device
        self.dtype  = dtype
        self._latent_space_realign = latent_space_realign
        self._realign_method = realign_method
        self._realign_tau    = realign_tau

        print(f"[Backbone] Loading {model_path} ...")
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_path,
            model_base=None,
            model_name=model_name,
            device_map=device,
        )
        self.model.eval()
        # Force legacy tuple KV cache: quilt-llava Llama-2 indexes past_key_values
        # as tuples, but newer transformers defaults to DynamicCache.
        if hasattr(self.model, "generation_config"):
            self.model.generation_config.cache_implementation = None

        # Eager attention so output_attentions=True populates. tf5 resolves the impl
        # from config._attn_implementation at FORWARD time, so the per-module attribute
        # alone is a no-op (SDPA stays → attentions return None) — set the config.
        self.model.config._attn_implementation = "eager"
        if getattr(getattr(self.model, "model", None), "config", None) is not None:
            self.model.model.config._attn_implementation = "eager"
        for layer in self.model.model.layers:
            if hasattr(layer.self_attn, "_attn_implementation"):
                layer.self_attn._attn_implementation = "eager"

        # Patch llava_arch.py:104-105 to support DynamicCache in
        # prepare_inputs_labels_for_multimodal. The original code does
        # past_key_values[-1][-1].shape[-2] (tuple indexing) to get seq length,
        # which crashes when DynamicCache is passed (e.g., from model.generate).
        self._patch_llava_arch()

        # Expose a minimal processor-like namespace so Diagnosis/other agents can
        # call bk.processor.tokenizer.eos_token_id / .decode() uniformly.
        self.processor = _TokenizerWrapper(self.tokenizer)

        # Llama-2: all layers are full attention
        self.lm = self.model.model          # LlavaLlamaModel
        n_layers = len(self.lm.layers)
        self.full_attn_indices: list[int] = list(range(n_layers))
        print(f"[Backbone] Layers: {n_layers} (all full attention)")

        # additive efficiency instrumentation (FLOPs profiler); no-op on failure.
        self._init_efficiency()

        self._build_realign_matrix()

    # ── realignment ───────────────────────────────────────────────────────────
    def _build_realign_matrix(self) -> None:
        """Cache embeddings and build W_a."""
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
        """Dispatch to W_a or softmax realignment.  hidden: [B, 1, D]"""
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
        """z̃_t = E^T · softmax(U z_t / τ)  — convex combination of input embeddings."""
        orig_dtype = hidden.dtype
        h = hidden.float().squeeze(1)           # [B, D]
        logits  = h @ self._output_emb.T / self._realign_tau  # [B, V]
        p       = torch.softmax(logits, dim=-1)                # [B, V]
        z_tilde = p @ self._input_emb                          # [B, D]
        return z_tilde.to(orig_dtype).unsqueeze(1)

    # ── vision encoding ───────────────────────────────────────────────────────
    @staticmethod
    def _to_pil(img) -> Image.Image:
        """Convert PIL / numpy array / torch.Tensor → PIL RGB."""
        if isinstance(img, Image.Image):
            return img.convert("RGB")
        import numpy as np
        if isinstance(img, torch.Tensor):
            arr = img.detach().cpu()
            if arr.ndim == 3 and arr.shape[0] in (1, 3):  # CHW → HWC
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

    @torch.no_grad()
    def encode_patches(
        self,
        images: list,   # PIL Image | np.ndarray | torch.Tensor
    ) -> tuple[torch.Tensor, int]:
        """Encode patch images → visual token embeddings.

        Accepts PIL Images, numpy arrays (HWC uint8/float), or torch.Tensors (CHW).
        Uses encode_images(): pixel_values → CLIP ViT → mm_projector → [N, 576, D]

        Returns:
            visual_embeds  : [n_patches * 576, hidden]
            n_vis_per_patch: 576
        """
        pil_images = [self._to_pil(img) for img in images]

        pixel_values = process_images(
            pil_images, self.image_processor, self.model.config
        ).to(self.device, dtype=self.dtype)             # [N, C, H, W]

        vis_feat = self.model.encode_images(pixel_values)  # [N, 576, hidden]
        vis_feat = vis_feat.view(-1, vis_feat.shape[-1])   # [N*576, hidden]
        return vis_feat, _N_VIS_PER_IMAGE

    # ── text embedding ────────────────────────────────────────────────────────
    @torch.no_grad()
    def embed_text(
        self,
        text: str,
        system_prompt: str | None = None,
        enable_thinking: bool = False,   # unused, kept for API compatibility
        with_image: bool = False,
    ) -> torch.Tensor:
        """Tokenise and embed text using llava_v1 conversation template.

        When with_image=True, prepends DEFAULT_IMAGE_TOKEN to the user message
        so visual features can be inserted at the correct training-time position.
        Stores the image token index in self._image_token_pos.

        Returns [n_tokens, hidden]  (image position holds a zero-vector placeholder).
        """
        from llava.mm_utils import tokenizer_image_token

        conv = conv_templates["llava_v1"].copy()
        if system_prompt:
            conv.system = system_prompt

        user_msg = (DEFAULT_IMAGE_TOKEN + "\n" + text) if with_image else text
        conv.append_message(conv.roles[0], user_msg)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        if with_image:
            ids = tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt",
            ).to(self.device)                              # [seq] with -200 placeholder

            img_positions = (ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
            if len(img_positions) > 0:
                self._image_token_pos = img_positions[0].item()
                # embed pre and post separately; insert zero vector at image pos
                D = self.lm.embed_tokens.embedding_dim
                pre_ids  = ids[:self._image_token_pos]
                post_ids = ids[self._image_token_pos + 1:]
                pre_emb  = self.lm.embed_tokens(pre_ids.unsqueeze(0)).squeeze(0) \
                           if pre_ids.numel() > 0 else torch.empty(0, D, device=self.device, dtype=self.dtype)
                post_emb = self.lm.embed_tokens(post_ids.unsqueeze(0)).squeeze(0)
                dummy    = torch.zeros(1, D, device=self.device, dtype=self.dtype)
                return torch.cat([pre_emb, dummy, post_emb], dim=0)
            # fallback: no image token found
            self._image_token_pos = None
        else:
            self._image_token_pos = None

        ids = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False,
        )["input_ids"].to(self.device)
        return self.lm.embed_tokens(ids).squeeze(0)    # [n_tokens, hidden]

    # ── position ids ──────────────────────────────────────────────────────────
    def _make_position_ids(self, seq_len: int, past_len: int = 0) -> torch.Tensor:
        """Standard 1D position ids.  Shape: [1, seq_len]."""
        return torch.arange(
            past_len, past_len + seq_len, device=self.device
        ).unsqueeze(0)

    # ── latent step loop ──────────────────────────────────────────────────────
    @torch.no_grad()
    def forward_with_latent_steps(
        self,
        visual_embeds: torch.Tensor,   # [n_vis, hidden]
        text_embeds:   torch.Tensor,   # [n_q,  hidden]  (may contain image placeholder)
        m: int,
        l_mid_layers: list[int],
    ) -> dict:
        """Prefill [pre_text | vis | post_text] then run m latent steps.

        If embed_text() was called with with_image=True, self._image_token_pos
        holds the placeholder position and visual_embeds are inserted there,
        matching the LLaVA training-time sequence structure.

        Returns:
            past_key_values        : KV cache
            latent_trajectory      : list[Tensor] length m, each [hidden]
            latent_to_vision_attn  : {layer_idx: Tensor [m, n_heads, n_vis]}
            query_to_vision_attn   : {layer_idx: Tensor [n_heads, n_q,  n_vis]}
            full_attn_kv           : {layer_idx: (K, V)}
        """
        n_vis = visual_embeds.shape[0]

        # ── build input_embeds with vis at correct position ────────────────────
        img_pos = getattr(self, "_image_token_pos", None)
        if img_pos is not None:
            # text_embeds: [pre_len + 1(dummy) + post_len, D]
            pre_emb  = text_embeds[:img_pos]           # before <image>
            post_emb = text_embeds[img_pos + 1:]       # after  <image>
            input_embeds = torch.cat(
                [pre_emb, visual_embeds, post_emb], dim=0
            ).unsqueeze(0)  # [1, pre+n_vis+post, D]
            n_pre = pre_emb.shape[0]
            n_q   = pre_emb.shape[0] + post_emb.shape[0]
            vis_start = n_pre                           # vis begins at n_pre
        else:
            # fallback: original [vis | text] order (Refiner/Diagnosis have no image)
            input_embeds = torch.cat(
                [visual_embeds.unsqueeze(0), text_embeds.unsqueeze(0)], dim=1
            )
            n_q       = text_embeds.shape[0]
            vis_start = 0

        seq = input_embeds.shape[1]
        pos_ids = self._make_position_ids(seq)

        # ── prefill ────────────────────────────────────────────────────────────
        out = self.lm(
            inputs_embeds=input_embeds,
            position_ids=pos_ids,
            use_cache=True,
            output_attentions=True,
            output_hidden_states=True,
        )
        past_kv     = out.past_key_values
        last_hidden = out.hidden_states[-1][:, -1:, :]   # [1, 1, hidden]

        # ── query-to-vision attention (post-text rows × vis cols) ─────────────
        query_to_vision_attn: dict[int, torch.Tensor] = {}
        attn_available = out.attentions is not None and len(out.attentions) > 0
        for li in l_mid_layers:
            if not attn_available or li >= len(out.attentions):
                continue
            a = out.attentions[li]
            if a is not None:
                # text rows attend to vis cols (vis_start..vis_start+n_vis)
                text_start = vis_start + n_vis
                q2v = a[0, :, text_start:, vis_start:vis_start + n_vis]
                query_to_vision_attn[li] = q2v.cpu()

        # ── latent steps ───────────────────────────────────────────────────────
        latent_trajectory: list[torch.Tensor] = []
        l2v_lists: dict[int, list[torch.Tensor]] = {li: [] for li in l_mid_layers}

        for step in range(m):
            latent_embed = self._apply_realign(last_hidden)   # [1, 1, hidden]
            pos_ids_step = self._make_position_ids(1, past_len=seq + step)

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

            step_attn = out.attentions
            step_attn_ok = step_attn is not None and len(step_attn) > 0
            for li in l_mid_layers:
                if not step_attn_ok or li >= len(step_attn):
                    continue
                a = step_attn[li]
                if a is not None:
                    l2v = a[0, :, 0, vis_start:vis_start + n_vis]  # [n_heads, n_vis]
                    l2v_lists[li].append(l2v.cpu())

        latent_to_vision_attn: dict[int, torch.Tensor] = {
            li: torch.stack(lst, dim=0)
            for li, lst in l2v_lists.items() if lst
        }
        full_attn_kv = self._extract_full_attn_kv(past_kv, l_mid_layers)

        return {
            "past_key_values":       past_kv,
            "latent_trajectory":     latent_trajectory,
            "latent_to_vision_attn": latent_to_vision_attn,
            "query_to_vision_attn":  query_to_vision_attn,
            "full_attn_kv":          full_attn_kv,
        }

    @torch.no_grad()
    def continue_with_latent_steps(
        self,
        text_embeds: torch.Tensor,
        past_kv,
        past_len: int,
        m: int,
        l_mid_layers: list[int],
    ) -> dict:
        """Append new prompt to existing KV and run m more latent steps.

        Used by Refiner to continue from Reasoner's KV.
        """
        n_t = text_embeds.shape[0]
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
        last_hidden = out.hidden_states[-1][:, -1:, :]
        past_len   += n_t

        latent_trajectory: list[torch.Tensor] = []
        for step in range(m):
            latent_embed = self._apply_realign(last_hidden)
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
            "past_key_values":   past_kv,
            "latent_trajectory": latent_trajectory,
            "full_attn_kv":      full_attn_kv,
            "past_len":          past_len + m,
        }

    # ── llava_arch patch ──────────────────────────────────────────────────────
    @staticmethod
    def _patch_llava_arch() -> None:
        """Monkey-patch prepare_inputs_labels_for_multimodal to support DynamicCache.

        llava_arch.py:105 does past_key_values[-1][-1].shape[-2] (tuple indexing)
        to get the past sequence length. DynamicCache doesn't support [] indexing,
        causing a TypeError when model.generate() passes DynamicCache internally.
        This patch replaces that line with a format-agnostic seq-length lookup.
        """
        import llava.model.llava_arch as _arch

        if getattr(_arch, "_dynamic_cache_patched", False):
            return  # already patched

        _orig = _arch.LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal

        def _patched(self, input_ids, attention_mask, past_key_values, labels, images):
            vision_tower = self.get_vision_tower()
            if (vision_tower is None or images is None
                    or input_ids.shape[1] == 1):
                if (past_key_values is not None
                        and vision_tower is not None
                        and images is not None
                        and input_ids.shape[1] == 1):
                    # ── format-agnostic seq length ────────────────────────────
                    if hasattr(past_key_values, "get_seq_length"):
                        past_len = past_key_values.get_seq_length()
                    elif hasattr(past_key_values, "key_cache"):
                        past_len = past_key_values.key_cache[0].shape[-2]
                    else:
                        past_len = past_key_values[-1][-1].shape[-2]  # legacy
                    attention_mask = torch.ones(
                        (attention_mask.shape[0], past_len + 1),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                return input_ids, attention_mask, past_key_values, None, labels
            # delegate remainder to original implementation
            return _orig(self, input_ids, attention_mask, past_key_values, labels, images)

        _arch.LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal = _patched
        _arch._dynamic_cache_patched = True
        print("[Backbone] llava_arch DynamicCache patch applied.")

    # ── KV helpers ────────────────────────────────────────────────────────────
    def _kv_to_dynamic_cache(self, past_kv):
        """Convert tuple-of-tuples KV → DynamicCache (no-op if already DynamicCache)."""
        from transformers import DynamicCache
        if isinstance(past_kv, DynamicCache):
            return past_kv
        dc = DynamicCache()
        for li, (K, V) in enumerate(past_kv):
            dc.update(
                K.to(device=self.device, dtype=self.dtype),
                V.to(device=self.device, dtype=self.dtype),
                li,
            )
        return dc

    def _extract_full_attn_kv(
        self,
        past_kv,
        l_mid_layers: list[int],
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        """Extract (K, V) — supports both tuple-of-tuples and DynamicCache."""
        result: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for li in l_mid_layers:
            try:
                if hasattr(past_kv, "key_cache"):          # DynamicCache
                    K = past_kv.key_cache[li].cpu()
                    V = past_kv.value_cache[li].cpu()
                else:                                       # legacy tuple
                    K = past_kv[li][0].cpu()
                    V = past_kv[li][1].cpu()
                result[li] = (K, V)
            except (IndexError, TypeError, AttributeError):
                continue
        return result

    # ── scanner interface ─────────────────────────────────────────────────────
    @torch.no_grad()
    def vl_generate(
        self,
        image: Image.Image,
        prompt: str,
        system_prompt: str = "",
        max_new_tokens: int = 1024,
    ) -> str:
        """Single-image VL generation for Scanner thumbnail analysis.

        Uses prepare_inputs_labels_for_multimodal + manual decode to avoid
        model.generate() which triggers DynamicCache incompatibility with
        quilt-llava's Llama-2 code on newer transformers versions.
        """
        from llava.mm_utils import tokenizer_image_token

        conv = conv_templates["llava_v1"].copy()
        if system_prompt:
            conv.system = system_prompt
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + prompt)
        conv.append_message(conv.roles[1], None)
        text_prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            text_prompt, self.tokenizer,
            image_token_index=IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).to(self.device)

        pixel_values = process_images(
            [image], self.image_processor, self.model.config
        ).to(self.device, dtype=self.dtype)

        # merge visual tokens into input embeds — no model.generate() needed
        # returns (None, attn_mask, past_kv, new_input_embeds, new_labels)
        _, _, _, inputs_embeds, _ = self.model.prepare_inputs_labels_for_multimodal(
            input_ids, None, None, None, pixel_values,
        )

        seq_len = inputs_embeds.shape[1]
        out = self.lm(
            inputs_embeds=inputs_embeds,
            position_ids=self._make_position_ids(seq_len),
            use_cache=True,
            output_hidden_states=True,
        )
        past_kv  = out.past_key_values
        last_h   = out.hidden_states[-1][:, -1:, :]
        past_len = seq_len

        eos_id = self.tokenizer.eos_token_id
        generated_ids: list[int] = []

        for _ in range(max_new_tokens):
            logits  = self.model.lm_head(last_h)
            next_id = int(logits[0, 0].argmax(-1))
            if next_id == eos_id:
                break
            generated_ids.append(next_id)
            past_len += 1
            tok_emb = self.lm.embed_tokens(
                torch.tensor([[next_id]], device=self.device)
            )
            out_i = self.lm(
                inputs_embeds=tok_emb,
                position_ids=self._make_position_ids(1, past_len=past_len),
                past_key_values=past_kv,
                use_cache=True,
                output_hidden_states=True,
            )
            past_kv = out_i.past_key_values
            last_h  = out_i.hidden_states[-1][:, -1:, :]

        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # ── generate ──────────────────────────────────────────────────────────────
    @torch.no_grad()
    def generate(
        self,
        visual_embeds: torch.Tensor,   # [n_vis, hidden]
        text_embeds:   torch.Tensor,   # [n_q,  hidden]
        past_kv=None,
        past_len: int = 0,
        max_new_tokens: int = 256,
        **_: Any,
    ) -> str:
        """Generate text given visual+text embeds (and optional prior KV).

        If past_kv is provided, text_embeds are appended on top of it.
        Otherwise prefills from scratch with [vis | text].
        """
        if past_kv is not None:
            n_t = text_embeds.shape[0]
            pos_ids = self._make_position_ids(n_t, past_len=past_len)
            out = self.lm(
                inputs_embeds=text_embeds.unsqueeze(0),
                position_ids=pos_ids,
                past_key_values=past_kv,
                use_cache=True,
                output_hidden_states=True,
            )
            past_kv  = out.past_key_values
            past_len += n_t
        else:
            seq = visual_embeds.shape[0] + text_embeds.shape[0]
            input_embeds = torch.cat(
                [visual_embeds.unsqueeze(0), text_embeds.unsqueeze(0)], dim=1
            )
            pos_ids = self._make_position_ids(seq)
            out = self.lm(
                inputs_embeds=input_embeds,
                position_ids=pos_ids,
                use_cache=True,
                output_hidden_states=True,
            )
            past_kv  = out.past_key_values
            past_len = seq

        last_h = out.hidden_states[-1][:, -1:, :]   # [1, 1, hidden]
        eos_id = self.tokenizer.eos_token_id
        generated_ids: list[int] = []

        for _ in range(max_new_tokens):
            logits  = self.model.lm_head(last_h)     # [1, 1, vocab]
            next_id = int(logits[0, 0].argmax(-1))
            if next_id == eos_id:
                break
            generated_ids.append(next_id)
            past_len += 1

            tok_embed = self.lm.embed_tokens(
                torch.tensor([[next_id]], device=self.device)
            )
            pos_ids_i = self._make_position_ids(1, past_len=past_len)
            out_i = self.lm(
                inputs_embeds=tok_embed,
                position_ids=pos_ids_i,
                past_key_values=past_kv,
                use_cache=True,
                output_hidden_states=True,
            )
            past_kv = out_i.past_key_values
            last_h  = out_i.hidden_states[-1][:, -1:, :]

        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)
