"""LLaVA-Med v1.5 backbone (LlavaMistralForCausalLM = Mistral-7B + CLIP ViT + mm_projector).

Same LatentMAS structure as huatuogpt.py (LLaVA-family: <image> token injection,
CLIP encode_images, 1D RoPE, all-full-attention) with a MODERN Mistral LLM, so the
DynamicCache / model.generate paths work natively — a near drop-in analog of huatuogpt.py.

Deltas vs huatuogpt.py (the ONLY places this file diverges):
  1. Loader: LlavaMistralForCausalLM.from_pretrained + vision_tower.load_model()
     (from the LLaVA-Med repo, matches its builder.load_pretrained_model).
  2. Vision preprocess: process_images() honours model.config.image_aspect_ratio="pad"
     (expand2square with CLIP mean bg) + CLIP-L/14-336.
  3. Prompt format: LLaVA-Med conv_templates["mistral_instruct"] ([INST]…[/INST],
     LLAMA_2 sep_style). tokenizer_image_token from the repo's mm_utils.
  4. llava_arch DynamicCache patch: prepare_inputs_labels_for_multimodal does a tuple
     index `past_key_values[-1][-1].shape[-2]` on the images!=None decode step, which
     crashes under DynamicCache (Mistral on transformers>=5). vl_generate hits it; the
     patch makes the seq-length lookup format-agnostic (mirrors quilt_llava._patch_llava_arch).
  5. 32 layers / hidden 4096 (Mistral-7B), GQA (8 KV heads).

Model structure:
  model.model.vision_tower  → CLIPVisionTower
  model.model.mm_projector  → mlp2x_gelu  → 576 vis tokens per image
  model.model.layers[i]     → MistralDecoderLayer (all 32, full attention)
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image

# ── LLaVA-Med repo must be on sys.path (its own LLaVA fork; defines llava_mistral) ──
# NOTE: quilt-llava / huatuogpt-vision also ship a top-level `llava` package. Only ONE
# llava-family backbone may be imported per process (first import wins sys.modules).
_LLAVAMED_REPO = Path("/home/super/hj/HJ/LLaVA-Med")
if str(_LLAVAMED_REPO) not in sys.path:
    sys.path.insert(0, str(_LLAVAMED_REPO))

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN  # noqa: E402
from llava.model.language_model.llava_mistral import LlavaMistralForCausalLM  # noqa: E402
from llava.mm_utils import (  # noqa: E402
    process_images, tokenizer_image_token, get_model_name_from_path,
)
from llava.conversation import conv_templates  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from backbone.efficiency import EfficiencyMixin  # noqa: E402

_N_VIS_PER_IMAGE = 576
_CONV = "mistral_instruct"


class _TokenizerWrapper:
    """Expose bk.processor.tokenizer.* uniformly (agents call bk.processor.tokenizer)."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def decode(self, *a, **k):
        return self.tokenizer.decode(*a, **k)


class LLaVAMedBackbone(EfficiencyMixin):
    """LLaVA-Med v1.5 (LlavaMistral) adapter with LatentMAS hidden-state realignment."""

    def __init__(
        self,
        model_path: str = "/media/super/4TB/hj/llava-med-v1.5-mistral-7b",
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
        self.model = LlavaMistralForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=False, torch_dtype=dtype,
        )
        self.model = self.model.to(device).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.config.tokenizer_padding_side = "left"

        # Vision tower is not loaded by from_pretrained → load + move to device/dtype
        # (matches LLaVA-Med builder.load_pretrained_model).
        vision_tower = self.model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device=device, dtype=dtype)
        self.model.model.mm_projector.to(device=device, dtype=dtype)
        self.image_processor = vision_tower.image_processor

        # Eager attention so output_attentions=True populates. tf5 resolves the impl
        # from config._attn_implementation at FORWARD time, so the per-module attribute
        # alone is a no-op (SDPA stays → attentions return None) — set the config.
        self.model.config._attn_implementation = "eager"
        if getattr(getattr(self.model, "model", None), "config", None) is not None:
            self.model.model.config._attn_implementation = "eager"
        for layer in self.model.model.layers:
            if hasattr(layer.self_attn, "_attn_implementation"):
                layer.self_attn._attn_implementation = "eager"

        # DynamicCache-safe prepare_inputs_labels_for_multimodal (vl_generate decode).
        self._patch_llava_arch()

        self.processor = _TokenizerWrapper(self.tokenizer)

        # No grounded_prefill_and_latent on LLaVA backbones → Reasoner takes the
        # non-grounded path (encode_patches + <image>-injected forward_with_latent_steps).
        self.vision_inject = "bolton"

        self.lm = self.model.model               # LlavaMistralModel (MistralModel + LLaVA)
        n_layers = len(self.lm.layers)
        self.full_attn_indices: list[int] = list(range(n_layers))
        print(f"[Backbone] Layers: {n_layers} (all full attention)")

        # additive efficiency instrumentation (FLOPs profiler + TTFT); no-op on failure.
        self._init_efficiency()

        self._image_token_pos: int | None = None
        self._build_realign_matrix()

    # ── realignment (identical to huatuogpt) ────────────────────────────────────
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

    # ── vision encoding ──────────────────────────────────────────────────────────
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

    @torch.no_grad()
    def encode_patches(self, images: list) -> tuple[torch.Tensor, int]:
        """process_images (pad+CLIP per model.config) → encode_images → [n_patches*576, hidden]."""
        pil = [self._to_pil(im) for im in images]
        pv = process_images(pil, self.image_processor, self.model.config).to(
            self.device, dtype=self.dtype)               # [N, 3, 336, 336]
        vis = self.model.encode_images(pv)               # [N, 576, hidden]
        vis = vis.reshape(-1, vis.shape[-1])             # [N*576, hidden]
        return vis, _N_VIS_PER_IMAGE

    # ── prompt building (LLaVA-Med mistral_instruct conv) ───────────────────────
    def _build_prompt(self, text: str, system_prompt: str | None, with_image: bool) -> str:
        conv = conv_templates[_CONV].copy()
        if system_prompt:
            conv.system = system_prompt
        body = text.replace("<image>", "").replace("<s>", "").replace("</s>", "")
        if with_image:
            body = f"{DEFAULT_IMAGE_TOKEN}\n" + body
        conv.append_message(conv.roles[0], body)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    @torch.no_grad()
    def embed_text(
        self,
        text: str,
        system_prompt: str | None = None,
        enable_thinking: bool = False,   # unused (Mistral has no thinking token)
        with_image: bool = False,
    ) -> torch.Tensor:
        """Tokenise + embed. With image, inserts a zero placeholder at the <image> slot
        (filled by forward_with_latent_steps); records position in self._image_token_pos.
        """
        prompt = self._build_prompt(text, system_prompt, with_image)

        if with_image:
            ids = tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt",
            ).to(self.device)                                # [seq] with -200 marker
            img_positions = (ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
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

    # ── position ids (1D) ─────────────────────────────────────────────────────────
    def _make_position_ids(self, seq_len: int, past_len: int = 0) -> torch.Tensor:
        return torch.arange(past_len, past_len + seq_len, device=self.device).unsqueeze(0)

    # ── latent step loop (identical structure to huatuogpt) ─────────────────────
    @torch.no_grad()
    def forward_with_latent_steps(
        self,
        visual_embeds: torch.Tensor,   # [n_vis, hidden]
        text_embeds:   torch.Tensor,   # [n_q,  hidden]  (may hold <image> placeholder)
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
            "past_len":              seq + m,     # 1D RoPE: cache length == positions
            "pos_cursor":            seq + m,
            "spans":                 spans,
        }

    @torch.no_grad()
    def continue_with_latent_steps(
        self, text_embeds: torch.Tensor, past_kv, past_len: int, m: int,
        l_mid_layers: list[int], past_pos_cursor: int = 0,   # accepted; 1D RoPE
    ) -> dict:
        n_t = text_embeds.shape[0]
        out = self.lm(
            inputs_embeds=text_embeds.unsqueeze(0),
            position_ids=self._make_position_ids(n_t, past_len=past_len),
            past_key_values=past_kv, use_cache=True,
            output_attentions=False, output_hidden_states=True,
        )
        past_kv     = out.past_key_values
        last_hidden = out.hidden_states[-1][:, -1:, :]
        past_len   += n_t

        latent_trajectory: list[torch.Tensor] = []
        for step in range(m):
            latent_embed = self._apply_realign(last_hidden)
            out = self.lm(
                inputs_embeds=latent_embed,
                position_ids=self._make_position_ids(1, past_len=past_len + step),
                past_key_values=past_kv, use_cache=True,
                output_attentions=False, output_hidden_states=True,
            )
            past_kv     = out.past_key_values
            last_hidden = out.hidden_states[-1][:, -1:, :]
            latent_trajectory.append(last_hidden.squeeze().cpu())

        return {
            "past_key_values":   past_kv,
            "latent_trajectory": latent_trajectory,
            "full_attn_kv":      self._extract_full_attn_kv(past_kv, l_mid_layers),
            "past_len":          past_len + m,
            "pos_cursor":        past_len + m,
            "spans":             [s for s in (("P", n_t), ("L", m)) if s[1] > 0],
        }

    # ── llava_arch DynamicCache patch (vl_generate decode path) ─────────────────
    @staticmethod
    def _patch_llava_arch() -> None:
        """Make prepare_inputs_labels_for_multimodal DynamicCache-safe.

        On the images!=None decode step (input_ids.shape[1]==1), the original does
        `past_key_values[-1][-1].shape[-2]` (tuple index) which crashes under
        DynamicCache (Mistral on transformers>=5). vl_generate re-feeds images every
        decode step, so it hits this. Replace the seq-length lookup with a
        format-agnostic one; the full multimodal path delegates to the original.
        """
        import llava.model.llava_arch as _arch

        if getattr(_arch, "_dynamic_cache_patched", False):
            return
        _orig = _arch.LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal

        def _patched(self, input_ids, position_ids, attention_mask, past_key_values,
                     labels, images, image_sizes=None):
            vision_tower = self.get_vision_tower()
            if vision_tower is None or images is None or input_ids.shape[1] == 1:
                if (past_key_values is not None and vision_tower is not None
                        and images is not None and input_ids.shape[1] == 1):
                    if hasattr(past_key_values, "get_seq_length"):
                        past_len = int(past_key_values.get_seq_length())
                    elif hasattr(past_key_values, "layers"):
                        past_len = past_key_values.layers[0].keys.shape[-2]
                    elif hasattr(past_key_values, "key_cache"):
                        past_len = past_key_values.key_cache[0].shape[-2]
                    else:
                        past_len = past_key_values[-1][-1].shape[-2]
                    target_shape = past_len + 1
                    attention_mask = torch.cat((attention_mask, torch.ones(
                        (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                        dtype=attention_mask.dtype, device=attention_mask.device)), dim=1)
                    position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
                return input_ids, position_ids, attention_mask, past_key_values, None, labels
            return _orig(self, input_ids, position_ids, attention_mask,
                         past_key_values, labels, images, image_sizes)

        _arch.LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal = _patched
        _arch._dynamic_cache_patched = True
        print("[Backbone] llava_arch DynamicCache patch applied.")

    # ── KV helpers ────────────────────────────────────────────────────────────────
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

    # ── Scanner interface: single-image VL generation (LLaVA-Med native) ────────
    @torch.no_grad()
    def vl_generate(
        self, image: Image.Image, prompt: str, system_prompt: str = "",
        max_new_tokens: int = 1024,
    ) -> str:
        text_prompt = self._build_prompt(prompt, system_prompt or None, with_image=True)
        input_ids = tokenizer_image_token(
            text_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt",
        ).unsqueeze(0).to(self.device)
        pv = process_images(
            [self._to_pil(image)], self.image_processor, self.model.config,
        ).to(self.device, dtype=self.dtype)
        _timer, _t0 = self._ttft_timer()
        _gk = dict(do_sample=False, max_new_tokens=max_new_tokens, use_cache=True,
                   pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        if _timer is not None:
            _gk["logits_processor"] = [_timer]
        # native LlavaMistral.generate (needs the images path → keep the override)
        out = self.model.generate(input_ids, images=pv, **_gk)
        self._record_ttft(_timer, _t0)
        # LlavaMistral.generate feeds inputs_embeds → output has ONLY new tokens
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()

    # ── unified text decode on top of a KV (KVComm-style; Mistral native) ───────
    @torch.no_grad()
    def generate_on_kv(
        self, system_prompt: str, user_text: str, past_kv, max_new_tokens: int,
        clone_cache: bool = False, pos_cursor: int | None = None,
        do_sample: bool = False, rep_penalty: float = 1.0, no_repeat: int = 0,
        temperature: float = 0.7, top_p: float = 0.9, enable_thinking: bool = False,
    ) -> str:
        """Re-prefill clean text at an assistant boundary and decode with model.generate
        on top of the carried KV (avoids latent-hidden garbage). images=None, so the
        llava_arch tuple-index branch is skipped."""
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

        # Bypass LlavaMistral's generate override (it forces inputs_embeds + drops the
        # attention_mask, which breaks text decoding on a carried KV). Call the stock
        # HF generate so input_ids + [past|new] mask + past_key_values behave normally.
        from transformers import GenerationMixin
        _timer, _t0 = self._ttft_timer()
        if _timer is not None:
            gen_kwargs["logits_processor"] = [_timer]
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
        """Prefill a short prompt on top of past_kv and return the NEXT-token logit
        vector [vocab] without generating. clone_cache keeps past_kv pristine so the
        same KV can be read multiple times (1D RoPE, so pos_cursor is unused)."""
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
