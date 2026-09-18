"""Qwen3.5-4B VLM backbone for latent reasoning.

Model structure:
  model.model.visual          → vision encoder (ViT)
  model.model.language_model  → Qwen3_5TextModel (hybrid)
    layers[i].self_attn       → Qwen3_5Attention       (full attention, i=3,7,11,15,19,23,27,31)
    layers[i].linear_attn     → Qwen3_5GatedDeltaNet   (recurrent, others)

Latent step loop:
  prefill([vis | text]) → past_kv, last_hidden
  for m steps:
    lm(W_a(last_hidden), past_kv) → past_kv, last_hidden, attn_weights

KV transfer between agents:
  pass past_key_values whole (includes both full_attn KV + GatedDeltaNet state)
  for manual injection: use full_attn_indices only
"""

from __future__ import annotations

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

from backbone.efficiency import EfficiencyMixin


class Qwen35Backbone(EfficiencyMixin):
    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.device = device
        self.dtype  = dtype

        print(f"[Backbone] Loading {model_path} ...")
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=dtype, device_map=device,
        )
        self.model.eval()

        self.lm     = self.model.model.language_model
        self.visual = self.model.model.visual

        # additive efficiency instrumentation (FLOPs profiler); no-op on failure.
        self._init_efficiency()

        # full_attention layer indices (Qwen3_5Attention, not GatedDeltaNet)
        self.full_attn_indices: list[int] = [
            i for i, layer in enumerate(self.lm.layers)
            if hasattr(layer, "self_attn")
        ]
        print(f"[Backbone] Full-attention layers: {self.full_attn_indices}")

        # W_a alignment: identity init, can be trained later
        hidden = self.lm.config.hidden_size
        self.W_a = nn.Linear(hidden, hidden, bias=False).to(device=device, dtype=dtype)
        nn.init.eye_(self.W_a.weight)
        self.W_a.eval()

    # ── vision encoding ────────────────────────────────────────────────────────
    @torch.no_grad()
    def encode_patches(self, images: list[Image.Image]) -> tuple[torch.Tensor, int]:
        """Run images through Qwen3.5 vision encoder + projector.

        Returns:
            visual_embeds: [n_patches * n_vis_per_patch, hidden]
            n_vis_per_patch: number of visual tokens per patch
        """
        # build prompt with one image token per patch
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

        vis_feat = self.visual(pv, grid_thw=thw)  # [total_vis, hidden]
        n_vis_per_patch = vis_feat.shape[0] // len(images)
        return vis_feat, n_vis_per_patch

    # ── text embedding ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def embed_text(
        self, text: str, enable_thinking: bool = False
    ) -> torch.Tensor:
        """Tokenise and embed question text. Returns [n_q, hidden]."""
        msgs = [{"role": "user", "content": text}]
        prompt = self.processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        ids = self.processor.tokenizer(
            prompt, return_tensors="pt"
        )["input_ids"].to(self.device)
        return self.lm.embed_tokens(ids).squeeze(0)  # [n_q, hidden]

    # ── latent step loop ───────────────────────────────────────────────────────
    @torch.no_grad()
    def forward_with_latent_steps(
        self,
        visual_embeds: torch.Tensor,   # [n_vis, hidden]
        text_embeds:   torch.Tensor,   # [n_q,  hidden]
        m: int,
        l_mid_layers: list[int],       # full_attn layer indices to extract from
    ) -> dict:
        """Prefill [vis | text], then run m latent steps.

        Returns
        -------
        past_key_values   : full HF cache (full_attn KV + GatedDeltaNet state)
        latent_trajectory : list[Tensor] length m, each [hidden]
        latent_to_vision_attn : {layer_idx: Tensor [m, n_heads, n_vis]}
        query_to_vision_attn  : {layer_idx: Tensor [n_heads, n_q,  n_vis]}
        full_attn_kv          : {layer_idx: (K, V)} for agent-to-agent transfer
        """
        n_vis = visual_embeds.shape[0]
        n_q   = text_embeds.shape[0]

        # ── prefill ────────────────────────────────────────────────────────────
        input_embeds = torch.cat(
            [visual_embeds.unsqueeze(0), text_embeds.unsqueeze(0)], dim=1
        )  # [1, n_vis+n_q, hidden]

        out = self.lm(
            inputs_embeds=input_embeds,
            use_cache=True,
            output_attentions=True,
            output_hidden_states=True,
        )
        past_kv     = out.past_key_values
        last_hidden = out.hidden_states[-1][:, -1:, :]  # [1, 1, hidden]

        # query-to-vision attention from prefill
        # attn[layer]: [B, n_heads, n_vis+n_q, n_vis+n_q] for full_attn, else None
        query_to_vision_attn: dict[int, torch.Tensor] = {}
        for li in l_mid_layers:
            if out.attentions[li] is not None:
                attn = out.attentions[li]          # [1, nh, seq, seq]
                # question token rows → visual token columns
                q2v = attn[0, :, n_vis:, :n_vis]  # [nh, n_q, n_vis]
                query_to_vision_attn[li] = q2v.cpu()

        # ── latent steps ───────────────────────────────────────────────────────
        latent_trajectory: list[torch.Tensor] = []
        l2v_lists: dict[int, list[torch.Tensor]] = {li: [] for li in l_mid_layers}

        for _ in range(m):
            latent_embed = self.W_a(last_hidden)  # [1, 1, hidden]

            out = self.lm(
                inputs_embeds=latent_embed,
                past_key_values=past_kv,
                use_cache=True,
                output_attentions=True,
                output_hidden_states=True,
            )
            past_kv     = out.past_key_values
            last_hidden = out.hidden_states[-1][:, -1:, :]
            latent_trajectory.append(last_hidden.squeeze().cpu())

            # latent-to-vision attention
            # attn[layer]: [1, nh, 1, n_vis+n_q+step+1]
            for li in l_mid_layers:
                if out.attentions[li] is not None:
                    attn = out.attentions[li]
                    l2v  = attn[0, :, 0, :n_vis]  # [nh, n_vis]
                    l2v_lists[li].append(l2v.cpu())

        # stack per layer: [m, nh, n_vis]
        latent_to_vision_attn: dict[int, torch.Tensor] = {}
        for li, lst in l2v_lists.items():
            if lst:
                latent_to_vision_attn[li] = torch.stack(lst, dim=0)

        # ── extract full_attn KV for agent transfer ────────────────────────────
        full_attn_kv = self._extract_full_attn_kv(past_kv, l_mid_layers)

        return {
            "past_key_values":        past_kv,
            "latent_trajectory":      latent_trajectory,
            "latent_to_vision_attn":  latent_to_vision_attn,
            "query_to_vision_attn":   query_to_vision_attn,
            "full_attn_kv":           full_attn_kv,
        }

    def _extract_full_attn_kv(
        self,
        past_kv,
        l_mid_layers: list[int],
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        """Extract (K, V) from full_attention layers only (for Refiner injection)."""
        result: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for li in l_mid_layers:
            if li not in self.full_attn_indices:
                continue
            try:
                if hasattr(past_kv, "key_cache"):
                    # DynamicCache
                    K = past_kv.key_cache[li].cpu()
                    V = past_kv.value_cache[li].cpu()
                else:
                    K = past_kv[li][0].cpu()
                    V = past_kv[li][1].cpu()
                result[li] = (K, V)
            except (IndexError, TypeError, AttributeError):
                continue
        return result
