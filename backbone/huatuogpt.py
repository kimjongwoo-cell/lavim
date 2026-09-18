"""HuatuoGPT-Vision-7B backbone (LlavaQwen2 = Qwen2-7B + CLIP ViT + mm_projector).

Same LatentMAS structure as quilt_llava.py (LLaVA-family: <image> token injection,
CLIP encode_images, 1D RoPE, all-full-attention), but the LLM is a MODERN Qwen2, so
the DynamicCache / model.generate paths work natively (no old-Llama-2 workarounds).

Deltas vs quilt_llava.py:
  1. Loader: LlavaQwen2ForCausalLM.from_pretrained(init_vision_encoder_from_ckpt=False)
     then vision_tower.load_model()  (matches HuatuoGPT cli.py).
  2. Vision preprocess: expand2square(mean bg) + CLIP processor  (HuatuoGPT convention).
  3. Prompt format: HuatuoGPT "<|user|>\n<image>\n{text}\n<|assistant|>\n" (no system
     role → any system_prompt is folded into the user text). tokenizer_image_token
     replicated from cli.py.
  4. 28 layers / hidden 3584 (Qwen2-7B), GQA (4 KV heads).
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image

# ── HuatuoGPT-Vision repo must be on sys.path (its own LLaVA fork) ─────────────
_HUATUO_REPO = Path("/home/users/whddn12316/wsi-root/huatuogpt-vision")
if str(_HUATUO_REPO) not in sys.path:
    sys.path.insert(0, str(_HUATUO_REPO))

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN  # noqa: E402
from llava.model.language_model.llava_qwen2 import LlavaQwen2ForCausalLM  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from backbone.efficiency import EfficiencyMixin  # noqa: E402

_N_VIS_PER_IMAGE = 576


class _TokenizerWrapper:
    """Expose bk.processor.tokenizer.* uniformly (agents call bk.processor.tokenizer)."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def decode(self, *a, **k):
        return self.tokenizer.decode(*a, **k)


class HuatuoGPTBackbone(EfficiencyMixin):
    """HuatuoGPT-Vision (LlavaQwen2) adapter with LatentMAS hidden-state realignment."""

    def __init__(
        self,
        model_path: str = "/media/super/4TB/hj/HuatuoGPT-Vision-7B",
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
        # Vision tower init inside __init__ fails under a meta-device context on newer
        # transformers, so defer it (matches HuatuoGPT cli.py init_components).
        self.model, _info = LlavaQwen2ForCausalLM.from_pretrained(
            model_path,
            init_vision_encoder_from_ckpt=False,
            output_loading_info=True,
            torch_dtype=dtype,
        )
        self.model = self.model.to(device).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.config.tokenizer_padding_side = "left"

        vision_tower = self.model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(dtype=dtype, device=device)
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

        self.processor = _TokenizerWrapper(self.tokenizer)

        # No grounded_prefill_and_latent on LLaVA backbones → take the Reasoner's
        # non-grounded path (encode_patches + <image>-injected forward_with_latent_steps).
        self.vision_inject = "bolton"

        # CLIP ViT-L/14 @336 → 24×24 = 576 vision tokens, no spatial merge (unlike
        # Qwen's merge_size=2). Read by the Scanner's saliency→bbox mappers so the
        # grid math is correct for this backbone; qwen backbones lack this attr and
        # keep their config.vision_config.spatial_merge_size path unchanged.
        self.scan_spatial_merge_size = 1

        self.lm = self.model.model               # LlavaQwen2Model (Qwen2Model + LLaVA)
        n_layers = len(self.lm.layers)
        self.full_attn_indices: list[int] = list(range(n_layers))
        print(f"[Backbone] Layers: {n_layers} (all full attention)")

        # additive efficiency instrumentation (FLOPs profiler + TTFT); no-op on failure.
        self._init_efficiency()

        self._image_token_pos: int | None = None
        self._build_realign_matrix()

    # ── realignment (identical to quilt_llava) ─────────────────────────────────
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

    # ── HuatuoGPT tokenizer_image_token (replicated from cli.py) ────────────────
    def _tokenizer_image_token(self, prompt: str) -> torch.Tensor:
        chunks = [self.tokenizer(c, add_special_tokens=False).input_ids
                  for c in prompt.split("<image>")]

        def _sep(X, s):
            return [e for pair in zip(X, [s] * len(X)) for e in pair][:-1]

        input_ids: list[int] = []
        offset = 0
        if chunks and chunks[0] and chunks[0][0] == self.tokenizer.bos_token_id:
            offset = 1
            input_ids.append(chunks[0][0])
        for x in _sep(chunks, [IMAGE_TOKEN_INDEX] * (offset + 1)):
            input_ids.extend(x[offset:])
        return torch.tensor(input_ids, dtype=torch.long, device=self.device)

    # ── vision encoding ────────────────────────────────────────────────────────
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

    def _expand2square(self, im: Image.Image) -> Image.Image:
        bg = tuple(int(x * 255) for x in self.image_processor.image_mean)
        w, h = im.size
        if w == h:
            return im
        s = max(w, h)
        r = Image.new(im.mode, (s, s), bg)
        r.paste(im, ((s - w) // 2, (s - h) // 2))
        return r

    @torch.no_grad()
    def encode_patches(self, images: list) -> tuple[torch.Tensor, int]:
        """expand2square + CLIP preprocess → encode_images → [n_patches*576, hidden]."""
        pil = [self._expand2square(self._to_pil(im)) for im in images]
        pv = torch.stack([
            self.image_processor.preprocess(im, return_tensors="pt")["pixel_values"][0]
            for im in pil
        ]).to(self.device, dtype=self.dtype)         # [N, 3, 336, 336]
        vis = self.model.encode_images(pv)           # [N, 576, hidden]
        vis = vis.reshape(-1, vis.shape[-1])         # [N*576, hidden]
        return vis, _N_VIS_PER_IMAGE

    # ── text embedding (HuatuoGPT format) ───────────────────────────────────────
    def _build_prompt(self, text: str, system_prompt: str | None, with_image: bool) -> str:
        # HuatuoGPT has no system role → fold system into the user turn.
        body = text if not system_prompt else f"{system_prompt}\n\n{text}"
        body = body.replace("<image>", "").replace("<s>", "").replace("</s>", "")
        if with_image:
            body = f"{DEFAULT_IMAGE_TOKEN}\n" + body
        return f"<|user|>\n{body}\n<|assistant|>\n"

    @torch.no_grad()
    def embed_text(
        self,
        text: str,
        system_prompt: str | None = None,
        enable_thinking: bool = False,   # unused (Qwen2/HuatuoGPT has no thinking token)
        with_image: bool = False,
    ) -> torch.Tensor:
        """Tokenise + embed. With image, inserts a zero placeholder at the <image> slot
        (filled by forward_with_latent_steps); records position in self._image_token_pos.
        """
        prompt = self._build_prompt(text, system_prompt, with_image)

        if with_image:
            ids = self._tokenizer_image_token(prompt)           # [seq] with -200 marker
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

    # ── position ids (1D) ───────────────────────────────────────────────────────
    def _make_position_ids(self, seq_len: int, past_len: int = 0) -> torch.Tensor:
        return torch.arange(past_len, past_len + seq_len, device=self.device).unsqueeze(0)

    # ── latent step loop (identical structure to quilt_llava) ───────────────────
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
        # §10 ground_cos gate input (bolton path): vision-token last hidden captured
        # from the PREFILL out, BEFORE the latent loop below overwrites `out`. The
        # Reasoner mean-pools this into vis_pool when --build_memory is on. Cheap
        # slice moved to CPU; only huatuo's forward is touched (isolation).
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

        # 1D-RoPE column layout for downstream synthesis/verify/prune. With the
        # <image> path the vision block sits between pre/post text; otherwise it is
        # a leading [vis | text] block.
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
            "vis_hidden":            vis_hidden,   # [n_vis, hidden] — ground_cos vis_pool
            "vis_cols":              list(range(vis_start, vis_start + n_vis)),
        }

    @torch.no_grad()
    def saliency_scan(
        self,
        image: Image.Image,
        prompt_text: str,
        m: int,
        l_mid_layers: list[int],
    ) -> dict:
        """Fresh (no prior KV) content-driven saliency over ONE image, for the
        Scanner's `--scanner_method saliency` path.

        Prefills [<image> tokens | prompt] then runs m realigned latent "thought"
        steps, exactly like the Reasoner's forward_with_latent_steps, but returns
        the pieces the Scanner's saliency→bbox mappers need — WITHOUT asking the
        model to verbalize any coordinates (which HuatuoGPT/LLaVA-family cannot do
        reliably). The model only has to LOOK; we read where its routing latent
        attends (latent_to_vision_attn) and/or how similar the latent is to each
        visual token (vis_hidden + latent_trajectory) → top-k regions.

        Returns dict with:
          grids                 : [[1, gh, gw]] token grid (24×24 for CLIP@336)
          n_vis                 : number of vision tokens
          vis_hidden            : (n_vis, d) last-layer hidden at vision positions
          latent_trajectory     : list of (d,) per latent step (last = routing z)
          latent_to_vision_attn : {layer: [m, heads, n_vis]}
        """
        vis, n_vis = self.encode_patches([image])          # [n_vis, H]
        text_embeds = self.embed_text(prompt_text, system_prompt=None, with_image=True)

        # assemble [pre-text | vision | post-text] (mirrors forward_with_latent_steps)
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
        # visual-token hidden states (for hidden-cosine saliency; escapes attn corner-sink)
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
        gh = gw = int(round(n_vis ** 0.5))                 # 576 → 24×24
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
        """Append [framing_text | vision(576)] onto a CARRIED KV + m latent steps —
        the KV-carry counterpart of saliency_scan, for the Scanner's scan_on_kv
        (rescan / reveal) path. huatuo 1D-RoPE / DynamicCache / no markers.

        Returns the scanner-contract dict (past_key_values, past_len, pos_cursor,
        vis_cols, spans, grids, n_vis, vis_hidden, latent_trajectory,
        latent_to_vision_attn) so the existing saliency mappers can select regions
        without any coordinate decode. Only reached when huatuo runs rescan
        (max_iterations>0); inert otherwise.
        """
        past_len = self._extract_kv_len(past_kv)          # carried KV len = 1D cursor

        fr_ids = self.tokenizer(framing_text, return_tensors="pt",
                                add_special_tokens=False)["input_ids"].to(self.device)
        fr_emb = self.lm.embed_tokens(fr_ids).squeeze(0)  # [n_pre, H]
        vis, n_vis = self.encode_patches([image])         # [576, H]
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

        gh = gw = int(round(n_vis ** 0.5))                # 576 → 24×24
        return {
            "past_key_values":       past_kv,
            "past_len":              seq_after + m,
            "pos_cursor":            seq_after + m,       # 1D RoPE: len == cursor
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
        """Drop the whole appended vision block from a KV (latent_only thread mode),
        keeping the Scanner's latent thought. Reuses the backbone-agnostic
        apply_kv_prune (huatuo native DynamicCache). Returns (kv, new_len, spans)."""
        import numpy as np
        from memory.prune import apply_kv_prune
        n = (int(vis_cols.numel()) if hasattr(vis_cols, "numel") else len(vis_cols)) \
            if vis_cols is not None else 0
        if n == 0:
            return past_kv, past_len, spans
        keep = np.zeros(n, dtype=bool)                    # drop all vision columns
        pk, new_len, new_spans, _ = apply_kv_prune(
            past_kv, past_len, vis_cols, keep, spans, self.device)
        print(f"[VisionDrop/huatuo] dropped {n} vision cols → KV {past_len} → {new_len}")
        return pk, new_len, new_spans

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

    # ── KV helpers (3-way cache layout, like the fixed quilt) ───────────────────
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

    # ── Scanner interface: single-image VL generation (HuatuoGPT native) ────────
    @torch.no_grad()
    def vl_generate(
        self, image: Image.Image, prompt: str, system_prompt: str = "",
        max_new_tokens: int = 1024,
    ) -> str:
        text_prompt = self._build_prompt(prompt, system_prompt or None, with_image=True)
        input_ids = self._tokenizer_image_token(text_prompt).unsqueeze(0)
        pil = self._expand2square(self._to_pil(image))
        pv = self.image_processor.preprocess(pil, return_tensors="pt")["pixel_values"].to(
            self.device, dtype=self.dtype)
        _timer, _t0 = self._ttft_timer()
        _gk = dict(do_sample=False, max_new_tokens=max_new_tokens, use_cache=True,
                   pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        if _timer is not None:
            _gk["logits_processor"] = [_timer]
        out = self.model.generate(input_ids, images=pv, **_gk)
        self._record_ttft(_timer, _t0)
        # LlavaQwen2.generate calls super().generate(inputs_embeds=...), so HF returns
        # ONLY the newly generated tokens (prompt not included). Slicing off
        # input_ids.shape[1] here discarded the real output → empty string. Decode
        # the full continuation directly.
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()

    # ── text_mas interface: multi-image VL generation (Reasoner/Verifier/Diagnosis) ─
    @torch.no_grad()
    def vlm_generate(
        self, images, system_prompt: str, user_text: str,
        max_new_tokens: int = 512, tag: str = "",
    ) -> str:
        """text_mas token-space channel: SEE 0..N crops and DECODE natural-language
        text. Mirrors patho/qwen `vlm_generate` but via HuatuoGPT's LLaVA-family
        <image> path (N <image> markers ↔ N pixel_values). images=[]/None → text-only
        (Verifier audit). Only used by the text_mas channel; latent path untouched.
        """
        imgs = [self._to_pil(im) for im in (images or [])]
        n = len(imgs)
        # HuatuoGPT has no system role → fold system into the user turn, then prepend
        # one <image> marker per crop (tokenizer_image_token handles N markers).
        body = user_text if not system_prompt else f"{system_prompt}\n\n{user_text}"
        body = body.replace("<image>", "").replace("<s>", "").replace("</s>", "")
        body = (f"{DEFAULT_IMAGE_TOKEN}\n" * n) + body
        text_prompt = f"<|user|>\n{body}\n<|assistant|>\n"
        input_ids = self._tokenizer_image_token(text_prompt).unsqueeze(0)

        # HuatuoGPT (weak LLaVA-family) collapses into copying the accumulated
        # text context / repeating the prompt's format list under pure greedy on a
        # long text_mas context (Qwen3-VL does not; see garbage-diagnosis 0710).
        # Anti-repetition breaks the degenerate loop. Qwen keeps pure greedy, so
        # this is a HUATUO-ONLY mitigation, not a matched-decode change.
        _rp = float(getattr(self, "textmas_rep_penalty", 1.2))
        _nr = int(getattr(self, "textmas_no_repeat_ngram", 3))
        _gk = dict(do_sample=False, max_new_tokens=max_new_tokens, use_cache=True,
                   pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        if _rp and _rp != 1.0:
            _gk["repetition_penalty"] = _rp
        if _nr and _nr > 0:
            _gk["no_repeat_ngram_size"] = _nr
        _timer, _t0 = self._ttft_timer()
        if _timer is not None:
            _gk["logits_processor"] = [_timer]
        if n:
            pv = torch.cat([
                self.image_processor.preprocess(
                    self._expand2square(im), return_tensors="pt")["pixel_values"]
                for im in imgs]).to(self.device, dtype=self.dtype)
            out = self.model.generate(input_ids, images=pv, **_gk)
        else:
            out = self.model.generate(input_ids, **_gk)
        self._record_ttft(_timer, _t0)

        # token-usage bookkeeping (pipeline resets self.usage per item, aggregates
        # after the text_mas loop). LlavaQwen2.generate returns ONLY the new tokens.
        if getattr(self, "usage", None) is None:
            self.usage = []
        self.usage.append({"tag": tag,
                           "prompt_tokens": int(input_ids.shape[1]),
                           "gen_tokens": int(out[0].shape[0])})
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()

    # ── unified text decode on top of a KV (KVComm-style; Qwen2 native) ─────────
    @torch.no_grad()
    def generate_on_kv(
        self, system_prompt: str, user_text: str, past_kv, max_new_tokens: int,
        clone_cache: bool = False, pos_cursor: int | None = None,
        do_sample: bool = False, rep_penalty: float = 1.0, no_repeat: int = 0,
        temperature: float = 0.7, top_p: float = 0.9, enable_thinking: bool = False,
    ) -> str:
        """Re-prefill clean text at a <|assistant|> boundary and decode with
        model.generate on top of the carried KV (avoids latent-hidden garbage)."""
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

        # Bypass LlavaQwen2's generate override (it forces inputs_embeds + drops the
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
