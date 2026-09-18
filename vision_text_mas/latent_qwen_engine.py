"""Physical Qwen KV engine for Latent VL-MAS without enabled pruning."""

from __future__ import annotations

import importlib
import contextlib
import os
import sys
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Generic, Literal, Protocol, TypedDict, TypeVar

import torch
from PIL import Image
from typing_extensions import assert_never

from vision_text_mas.latent_backend import LatentAppendResult
from vision_text_mas.latent_batch_reallocation import (
    TerminalReallocationConfig,
    terminal_reallocation,
)
from vision_text_mas.latent_hybrid_variants import (
    NativeHybridPruneConfig,
    attach_native_hybrid_variants,
)
from vision_text_mas.latent_kv_relay import (
    RelationalVisualContext,
    append_relay,
    extract_selected_latent_kv,
    relay_length,
    select_pathology_context_steps,
    select_visual_contrast_steps,
    select_visual_grounded_steps,
    select_visual_grounded_steps_from_masses,
)
from vision_text_mas.latent_terminal import generate_terminal_json
from vision_text_mas.relay_destination import (
    STATIC_RELAY_SOURCES,
    build_destination_weights,
    cosine_similarity,
    rank_correlation,
)
from vision_text_mas.pathology_reallocation import (
    build_pathology_reallocation_weights,
    pathology_question_profile,
    pathology_visual_patch_ids,
)

CacheT = TypeVar("CacheT")
# "upstream": 0717/upstream-LatentMAS turn framing — re-emit the full [system,user]
# chat template per agent (system repeated), NO close_assistant_turn, FULL cumulative
# carry (no tail-slice). Matches methods/latent_mas.py's default generate_latent_batch
# (threads past_kv, no truncation, no turn close). "cumulative" differs by closing each
# assistant turn (</think><|im_end|>) + continuation_user_turn — a chat-clean divergence
# from upstream. "cumulative_nothink" is "cumulative" with thinking held CLOSED per agent
# (0717-faithful): keep_thinking_open stays False so each agent's latent block is generated
# after </think> (empty think), and close_assistant_turn adds only <|im_end|> (no second
# </think>). This stops the accumulated KV from carrying an open thinking-register that made
# the Answerer ramble instead of committing (see debug/smoke_terminal_kv_reduce.py). Note:
# continuation_user_turn is gated on the cumulative FAMILY (both cumulative variants) so the
# chat-clean framing applies to both; keep_thinking_open is gated on "cumulative" ONLY, so
# "cumulative_nothink" and "upstream" both get think-off.
TransportMode = Literal[
    "latent_only", "sequential_info_only", "cumulative", "cumulative_nothink", "upstream"
]
LatentBackboneName = Literal[
    "qwen3-vl", "patho-r1", "octomed", "internvl3", "huatuo"
]


def _reasoner_attention_targets(user_prompt: str) -> tuple[list[str], list[str]]:
    """Return the question and fixed instruction spans for pruning V2.

    ``announce_latent_upstream`` can prepend an evidence-plan heading before the
    canonical Reasoner prompt, so the first line is not reliably the question.
    Keep the exact prompt substring that the tokenizer must locate.
    """
    question_start = user_prompt.find("Question stem: ")
    if question_start < 0:
        return [], []
    question_end = user_prompt.find("\nEvidence plan:", question_start)
    if question_end < 0:
        question_end = user_prompt.find("\n", question_start)
    if question_end < 0:
        question_end = len(user_prompt)
    question = user_prompt[question_start:question_end]
    question_text = user_prompt[question_start + len("Question stem: "):question_end]
    background_candidates = (
        "Describe every patch using patch_id, quality, architecture, cellularity",
        "Assess every patch for question-relevant visible morphology: architecture",
        "Assess every labeled patch for question-relevant visible morphology, "
        "including architecture",
    )
    backgrounds = [
        candidate for candidate in background_candidates if candidate in user_prompt
    ]
    return [question, question_text], backgrounds


def _role_attention_targets(user_prompt: str, stage: str) -> list[str]:
    """Return only question/evidence-target text to rate a role's visual KV."""
    if stage == "reasoner":
        return _reasoner_attention_targets(user_prompt)[0]
    labels = (
        ("Question stem: ",)
        if stage == "evidence_planner"
        else (
            "Question focus: ",
            "x5 evidence target: ",
            "x20 evidence target: ",
            "Overview target at x5: ",
            "Detail target at x20: ",
        )
    )
    targets: list[str] = []
    for line in user_prompt.splitlines():
        for label in labels:
            if line.startswith(label):
                value = line[len(label):].strip()
                if value:
                    targets.append(value)
                break
    return targets


def _pathology_reallocation_weights(
    *,
    columns: torch.Tensor | None,
    spans: object,
    grids: object,
    magnifications: tuple[int, ...],
    parent_indices: tuple[int, ...],
    spatial_merge_size: int,
    parent_anchor_strength: float,
    fine_evidence_boost: float,
    coarse_evidence_boost: float,
) -> torch.Tensor:
    """Return post-pruning weights aligned to surviving visual KV columns."""
    if not isinstance(columns, torch.Tensor) or columns.numel() == 0:
        return torch.empty(0, dtype=torch.float32)
    if not isinstance(grids, torch.Tensor) or not isinstance(spans, list):
        return torch.empty(0, dtype=torch.float32)
    retained = [int(count) for token_type, count in spans if token_type == "V"]
    if len(retained) != len(magnifications) or len(grids) != len(magnifications):
        return torch.empty(0, dtype=torch.float32)
    merge = max(1, int(spatial_merge_size))
    original = [
        int(grid[0]) * (int(grid[1]) // merge) * (int(grid[2]) // merge)
        for grid in grids
    ]
    try:
        return build_pathology_reallocation_weights(
            columns.detach().cpu(),
            retained_per_patch=retained,
            original_per_patch=original,
            magnifications=magnifications,
            parent_indices=parent_indices,
            parent_anchor_strength=parent_anchor_strength,
            fine_evidence_boost=fine_evidence_boost,
            coarse_evidence_boost=coarse_evidence_boost,
        )
    except ValueError:
        # A malformed external cache must retain the baseline redistribution,
        # never block a case after pruning completed successfully.
        return torch.empty(0, dtype=torch.float32)


def _backbone_load_spec(
    backbone_name: LatentBackboneName,
) -> tuple[str, str, tuple[int, ...]]:
    """Return the adapter and valid attention probe layers for one VLM family."""
    match backbone_name:
        case "qwen3-vl":
            return "backbone.qwen3vl", "Qwen3VLBackbone", (20, 24, 28, 32)
        case "patho-r1" | "octomed":
            return "backbone.pathor1", "PathoR1Backbone", (20, 22, 24, 26)
        case "internvl3":
            return "backbone.internvl", "InternVLBackbone", (20, 22, 24, 26)
        case "huatuo":
            return "backbone.huatuogpt", "HuatuoGPTBackbone", (20, 22, 24, 26)
        case unreachable:
            assert_never(unreachable)
class BackboneResult(TypedDict, Generic[CacheT]):
    past_key_values: CacheT
    past_len: int
    pos_cursor: int


class LatentBackbone(Protocol[CacheT]):
    """Validated subset of the preserved latent Qwen backbone."""

    _prune_image_magnifications: tuple[int, ...]
    _prune_image_parent_indices: tuple[int, ...]
    _prune_morphology_enabled: bool
    _latent_reallocation: TerminalReallocationConfig | None
    _reallocation_carried_vision_columns: torch.Tensor
    _reallocation_vision_weights: torch.Tensor
    _latent_reallocation_probe: float | None
    _latent_reallocation_role: str
    device: str

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
    ) -> BackboneResult[CacheT]: ...

    def grounded_prefill_and_latent_on_kv(
        self,
        past_kv: CacheT,
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
    ) -> BackboneResult[CacheT]: ...

    def close_assistant_turn(
        self,
        past_kv: CacheT,
        past_pos_cursor: int,
        *,
        close_thinking: bool = False,
    ) -> tuple[CacheT, int, int]: ...

    def embed_text(
        self,
        text: str,
        system_prompt: str | None = None,
        enable_thinking: bool = False,
    ) -> torch.Tensor: ...

    def continue_with_latent_steps(
        self,
        text_embeds: torch.Tensor,
        past_kv: CacheT,
        past_len: int,
        m: int,
        l_mid_layers: list[int],
        past_pos_cursor: int | None = None,
    ) -> BackboneResult[CacheT]: ...

    def generate_on_kv(
        self,
        system_prompt: str,
        user_text: str,
        past_kv: CacheT,
        max_new_tokens: int,
        *,
        clone_cache: bool = False,
        pos_cursor: int | None = None,
        do_sample: bool = False,
        rep_penalty: float = 1.0,
        no_repeat: int = 0,
        temperature: float = 0.7,
        top_p: float = 0.9,
        enable_thinking: bool = False,
        continuation_user_turn: bool = False,
    ) -> str: ...

    def retain_last_kv(self, past_kv: CacheT, keep_tokens: int) -> CacheT: ...

@dataclass(frozen=True, slots=True)
class LatentImageRequiredError(RuntimeError):
    """A role attempted to initialize or advance Qwen without its visual input."""

    stage: str

    def __str__(self) -> str:
        return f"latent Qwen append requires role images: {self.stage}"


class LatentQwenEngine(Generic[CacheT]):
    """Run every role on HF with selectable latent-KV transport semantics."""

    def __init__(
        self,
        backbone: LatentBackbone[CacheT],
        *,
        temperature: float = 0.6,
        top_p: float = 0.95,
        seed: int = 42,
        terminal_do_sample: bool = True,
        answerer_thinking: bool = True,
        terminal_no_repeat_ngram_size: int = 0,
        terminal_repetition_penalty: float = 1.0,
        transport_mode: TransportMode = "upstream",
        thinking_roles: frozenset[str] = frozenset(),
        prune_config: NativeHybridPruneConfig | None = None,
        reallocation: TerminalReallocationConfig | None = None,
        latent_probe_layers: tuple[int, ...] = (20, 24, 28, 32),
    ) -> None:
        self._backbone = backbone
        self._temperature = temperature
        self._top_p = top_p
        self._seed = seed
        self._terminal_do_sample = terminal_do_sample
        self._answerer_thinking = answerer_thinking
        self._terminal_no_repeat_ngram_size = terminal_no_repeat_ngram_size
        self._terminal_repetition_penalty = terminal_repetition_penalty
        self._transport_mode = transport_mode
        self._thinking_roles = thinking_roles
        self._capture_attention = bool(
            prune_config is not None
            and not prune_config.prefill_hierarchy_aware
            and not prune_config.query_adaptive
        )
        self._reallocation_enabled = reallocation is not None
        self._latent_probe_layers = latent_probe_layers
        self._last_probe_mass: float | None = None
        self._answerer_vision_columns = torch.empty(0, dtype=torch.long)
        self._answerer_vision_weights = torch.empty(0, dtype=torch.float32)
        self._answerer_visual_patch_ids = torch.empty(0, dtype=torch.long)
        self._answerer_parent_patch_indices: tuple[int, ...] = ()
        self._answerer_reallocation_scale = 1.0
        self._answerer_latent_kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        # Sender-relay experiment: surviving-token pruning priority (aligned to
        # _answerer_vision_columns) plus per-case audit payloads.
        self._answerer_vision_priority = torch.empty(0, dtype=torch.float32)
        self._sender_audit: dict | None = None
        self.last_relay_audit: dict | None = None
        # Reallocation-only runs preserve cache column positions.  Carrying this
        # set lets the receiving Reasoner read Navigator visual evidence as well
        # as its newly appended high-magnification patches.
        self._reallocation_carry_enabled = (
            reallocation is not None
            and prune_config is None
            and not reallocation.pathology_aware
        )
        self._carried_vision_columns = torch.empty(0, dtype=torch.long)
        if prune_config is not None or reallocation is not None:
            attach_native_hybrid_variants(
                backbone,
                prune_config=prune_config,
                reallocation=reallocation,
            )
        if reallocation is not None and (
            reallocation.latent_kv_relay
            or reallocation.visual_grounded_latent_kv_relay
            or reallocation.visual_contrast_latent_kv_relay
            or reallocation.pathology_context_latent_kv_relay
        ):
            # The relay only appends already-computed latent K/V; it does not
            # need full attention matrices.  SDPA keeps the 12,288-token
            # Reasoner pass within the GPU budget.
            backbone.model.config._attn_implementation = "sdpa"
            backbone.lm.config._attn_implementation = "sdpa"

    @classmethod
    def from_pretrained(
        cls,
        model_path: Path,
        device: str,
        *,
        preserved_root: Path,
        temperature: float = 0.6,
        top_p: float = 0.95,
        seed: int = 42,
        terminal_do_sample: bool = True,
        answerer_thinking: bool = True,
        terminal_no_repeat_ngram_size: int = 0,
        terminal_repetition_penalty: float = 1.0,
        transport_mode: TransportMode = "upstream",
        thinking_roles: frozenset[str] = frozenset(),
        realign_method: str = "identity_norm",
        prune_config: NativeHybridPruneConfig | None = None,
        reallocation: TerminalReallocationConfig | None = None,
        backbone_name: LatentBackboneName = "qwen3-vl",
    ) -> LatentQwenEngine[CacheT]:
        """Load one compatible latent VLM backbone with the selected ablation."""
        root_text = str(preserved_root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        module_name, class_name, latent_probe_layers = _backbone_load_spec(backbone_name)
        module = importlib.import_module(module_name)
        backbone_class = getattr(module, class_name)
        match backbone_name:
            case "internvl3" | "huatuo":
                backbone = backbone_class(
                    model_path=str(model_path),
                    device=device,
                    realign_method=realign_method,
                )
            case "qwen3-vl" | "patho-r1" | "octomed":
                backbone = backbone_class(
                    model_path=str(model_path),
                    device=device,
                    instruct=False,
                    mrope_pos=True,
                    vision_inject="grounded",
                    realign_method=realign_method,
                    build_realign=True,
                )
            case unreachable:
                assert_never(unreachable)
        model_layers = getattr(getattr(backbone, "lm", None), "layers", None)
        if model_layers is not None:
            latent_probe_layers = tuple(
                layer for layer in latent_probe_layers if layer < len(model_layers)
            )
            if not latent_probe_layers:
                raise ValueError("loaded backbone has no valid latent attention probe layers")
        return cls(
            backbone,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            terminal_do_sample=terminal_do_sample,
            answerer_thinking=answerer_thinking,
            terminal_no_repeat_ngram_size=terminal_no_repeat_ngram_size,
            terminal_repetition_penalty=terminal_repetition_penalty,
            transport_mode=transport_mode,
            thinking_roles=thinking_roles,
            prune_config=prune_config,
            reallocation=reallocation,
            latent_probe_layers=latent_probe_layers,
        )

    def append(
        self,
        *,
        cache: CacheT | None,
        cache_length: int,
        position_cursor: int,
        images: tuple[Image.Image, ...],
        system_prompt: str,
        user_prompt: str,
        latent_steps: int,
        stage: str = "",
    ) -> LatentAppendResult[CacheT]:
        """Append a role and retain its configured latent-KV handoff state."""
        self._backbone._reallocation_carried_vision_columns = (
            self._carried_vision_columns
            if self._reallocation_carry_enabled
            else torch.empty(0, dtype=torch.long)
        )
        self._backbone._reallocation_vision_weights = torch.empty(0, dtype=torch.float32)
        patch_ids = tuple(str(image.info.get("latent_patch_id", "")) for image in images)
        patch_index = {patch_id: index for index, patch_id in enumerate(patch_ids)}
        self._backbone._prune_image_patch_ids = patch_ids   # visual provenance (bookkeeping only)
        self._backbone._prune_image_magnifications = tuple(
            int(image.info.get("latent_magnification", 0)) for image in images
        )
        self._backbone._prune_image_parent_indices = tuple(
            patch_index.get(str(image.info.get("latent_parent_patch_id", "")), -1)
            if int(image.info.get("latent_magnification", 0)) > 5 else -1
            for image in images
        )
        # AAVM: per-crop acquisition-event query text (empty when the search
        # Navigator did not run, which makes AAVM a no-op for that case).
        self._backbone._aavm_event_queries = tuple(
            str(image.info.get("latent_event_query", "")) for image in images
        )
        self._backbone._prune_image_boxes = tuple(
            tuple(int(v) for v in image.info["latent_box"])
            if image.info.get("latent_box") else None
            for image in images
        )
        prune_args = getattr(self._backbone, "_prune_args", None)
        reasoner_only_pruning = bool(
            prune_args is not None and getattr(prune_args, "prune_reasoner_only", False)
        )
        configured_prune_roles = (
            getattr(prune_args, "prune_roles", None)
            if prune_args is not None else None
        )
        prune_this_stage = (
            stage in configured_prune_roles
            if configured_prune_roles is not None
            else not reasoner_only_pruning or stage == "reasoner"
        )
        sparsevlm_all_roles = bool(
            prune_args is not None
            and getattr(prune_args, "prune_query_adaptive", False)
            and not reasoner_only_pruning
        )
        atp_sap_all_roles = bool(
            prune_args is not None
            and getattr(prune_args, "prune_atp_sap", False)
            and not reasoner_only_pruning
        )
        hierarchy_pruning_all_roles = bool(
            prune_args is not None
            and getattr(prune_args, "prune_hierarchy_all_roles", False)
            and not reasoner_only_pruning
        )
        self._backbone._prune_morphology_enabled = (
            prune_this_stage and (
            stage == "reasoner"
            or sparsevlm_all_roles
            or atp_sap_all_roles
            or hierarchy_pruning_all_roles
            )
        )
        reasoner_targets, reasoner_background = (
            _reasoner_attention_targets(user_prompt)
            if stage == "reasoner" else (None, None)
        )
        attention_targets = (
            _role_attention_targets(user_prompt, stage)
            if sparsevlm_all_roles or atp_sap_all_roles
            else (reasoner_targets or [])
        )
        layers = list(self._latent_probe_layers)
        pruning_probe_layer = (
            getattr(prune_args, "prune_probe_layer", None)
            if prune_this_stage and (stage != "navigator" or atp_sap_all_roles)
            else None
        )
        if pruning_probe_layer is not None:
            layers = [int(pruning_probe_layer)]
        previous_cache_length = cache_length
        keep_thinking_open = (
            stage in self._thinking_roles
            if self._thinking_roles
            else self._transport_mode == "cumulative"
        )
        question_profile = pathology_question_profile(user_prompt)
        config = getattr(self._backbone, "_latent_reallocation", None)
        boundary_relay: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        if self._reallocation_enabled and config is not None:
            self._backbone._latent_reallocation_probe = config.initial_probe_for_role(
                stage, self._last_probe_mass
            )
            self._backbone._latent_reallocation_role = (
                ""
                if config.pathology_aware
                and stage == "reasoner"
                and not question_profile.enabled
                else stage
            )
        # C2 Latent-Provenance Visual Handoff (memory/lpvh.py, VLMAS_LPVH): record, during the
        # Reasoner's latent steps, the native attention every step gives each physical support.
        # Off / other stage = nullcontext (entered lazily, only on the image-bearing branches).
        from memory import lpvh as _lpvh
        _lpvh_r = _lpvh.reasoner_probe(
            self._backbone, m=latent_steps, stage=stage,
            seed=42 + int(getattr(self, "_rpath_case_index", 0) or 0))
        # C2 Support-Resolved Visual Write Handoff (memory/srvw.py, VLMAS_SRVW): accumulate, during the
        # Reasoner's latent steps, the value message alpha*V each physical support writes (per layer).
        # Off / other stage = nullcontext.
        from memory import srvw as _srvw
        _srvw_r = _srvw.reasoner_probe(self._backbone, m=latent_steps, stage=stage)
        # Latent Unsilencing (memory/lu.py, VLMAS_LU): capture the Reasoner prefill's question->visual
        # attention + input embeddings and the latent-step bookkeeping. Off / other stage = nullcontext.
        from memory import lu as _lu
        _lu_r = _lu.reasoner_probe(self._backbone, m=latent_steps, stage=stage, targets=attention_targets)
        # VELH (memory/velh.py, VLMAS_VELH): anchor query / cos-sin rows / latent bookkeeping of the Reasoner.
        from memory import velh as _velh
        _velh_r = _velh.reasoner_probe(self._backbone, m=latent_steps, stage=stage, targets=attention_targets)
        # Text-cut (memory/textcut.py, VLMAS_TEXTCUT): token ids / positions of every role's prefill rows.
        from memory import textcut as _tc
        if _tc.enabled() and cache is None:
            _tc.reset(self)
        _tc_r = _tc.stage_probe(self._backbone, stage)
        # RN-LCR (memory/rnlcr.py, VLMAS_RNLCR): question->visual relevance + prefill embeddings for grounding.
        from memory import rnlcr as _rn
        _rn_r = _rn.reasoner_probe(self._backbone, m=latent_steps, stage=stage, targets=attention_targets)
        # C2 13 RELH consumer hook: the Reasoner reads the Navigator endpoint at its own boundary.
        from memory import relh as _relh
        _relh_r = contextlib.nullcontext()
        if _relh.enabled() and _relh.active_at(stage) and stage == "reasoner":
            _rcfg = _relh.config()
            _reps = _relh.endpoints_for(self._backbone, "reasoner")
            if _reps:
                _relh_r = _relh.install_relh(
                    self._backbone, _reps, int(position_cursor) - 1, rows=_rcfg["rows"],
                    identity=(_rcfg["mode"] == "identity"))
                print(f"[RELH:reasoner] mode={_rcfg['mode']} endpoints={_reps} "
                      f"handoff={int(position_cursor) - 1}", flush=True)
        if not images and cache is not None:
            result = self._backbone.continue_with_latent_steps(
                self._backbone.embed_text(
                    user_prompt,
                    system_prompt,
                    enable_thinking=keep_thinking_open,
                ),
                cache,
                cache_length,
                latent_steps,
                layers,
                past_pos_cursor=position_cursor,
            )
        elif cache is None:
            if not images:
                raise LatentImageRequiredError(stage=user_prompt[:80])
            with _lpvh_r, _srvw_r, _relh_r, _lu_r, _velh_r, _tc_r, _rn_r:
                result = self._backbone.grounded_prefill_and_latent(
                    images=list(images),
                    system_prompt=system_prompt,
                    user_text=user_prompt,
                    m=latent_steps,
                    l_mid_layers=layers,
                    enable_thinking=keep_thinking_open,
                    want_attn=self._capture_attention and prune_this_stage,
                    role_targets=attention_targets,
                    bg_targets=reasoner_background,
                )
        else:
            with _lpvh_r, _srvw_r, _relh_r, _lu_r, _velh_r, _tc_r, _rn_r:
                result = self._backbone.grounded_prefill_and_latent_on_kv(
                    past_kv=cache,
                    past_pos_cursor=position_cursor,
                    images=list(images),
                    system_prompt=system_prompt,
                    user_text=user_prompt,
                    m=latent_steps,
                    l_mid_layers=layers,
                    enable_thinking=keep_thinking_open,
                    want_attn=self._capture_attention and prune_this_stage,
                    role_targets=attention_targets,
                    bg_targets=reasoner_background,
                    continuation_user_turn=(
                    os.environ.get("VLMAS_NO_CONTINUATION", "").strip() != "1"
                    and self._transport_mode in ("cumulative", "cumulative_nothink")
                ),
                )
        if _lu.enabled() and stage == "reasoner":
            # Latent Unsilencing: Z = native latent input embeddings -> Stage I/II -> crop + replay once.
            _lu.apply(self, result, m=latent_steps,
                      case_index=int(getattr(self, "_rpath_case_index", -1) or 0))
        if _velh.enabled() and stage == "reasoner":
            _velh.record_reasoner(self, result, m=latent_steps, case_index=int(getattr(self, "_rpath_case_index", -1) or 0))
        if stage == "reasoner":
            # Relay4/FULR selection is case-local; never carry a prior case's
            # latent mask into the next Answerer call.
            self._backbone._relay4_visual_mask = {}
        if _rn.enabled() and stage == "reasoner":
            # RN-LCR: record the latent columns; ground/full also ground the trajectory and replay it once.
            _rn.apply_reasoner(self, result, m=latent_steps, case_index=int(getattr(self, "_rpath_case_index", -1) or 0))
        if (
            config is not None
            and (
                config.visual_grounded_latent_kv_relay
                or config.visual_contrast_latent_kv_relay
            )
            and stage != "reasoner"
            and stage in config.visual_grounded_latent_kv_relay_stages
        ):
            stage_vision_columns = result.get("vis_cols")
            selected_steps = (
                select_visual_contrast_steps(
                    result["past_key_values"],
                    latent_steps=latent_steps,
                    vision_columns=(
                        stage_vision_columns
                        if isinstance(stage_vision_columns, torch.Tensor)
                        else torch.empty(0, dtype=torch.long)
                    ),
                    text_start=previous_cache_length,
                )
                if config.visual_contrast_latent_kv_relay
                else select_visual_grounded_steps_from_masses(
                    result.get("probe_mass_steps") or (),
                    threshold=config.visual_grounding_threshold,
                    top_k=config.visual_grounding_top_k,
                )
            )
            boundary_relay = extract_selected_latent_kv(
                result["past_key_values"], latent_steps, selected_steps
            )
            print(
                f"[{'LatentKVRelay3' if config.visual_contrast_latent_kv_relay else 'LatentKVRelay2'}:{stage}] selected="
                f"{selected_steps.tolist()} next-role rows={relay_length(boundary_relay)}",
                flush=True,
            )
        probe_mass = result.get("probe_rv")
        self._last_probe_mass = (
            float(probe_mass)
            if isinstance(probe_mass, (float, int)) and not isinstance(probe_mass, bool)
            else None
        )
        if self._reallocation_enabled and self._last_probe_mass is not None:
            config = self._backbone._latent_reallocation
            next_alpha = config.alpha_for_probe(self._last_probe_mass)
            print(
                "[Reallocation] natural vision mass "
                f"{self._last_probe_mass:.4f}; next-role alpha {next_alpha:.4f}"
            )
        vision_columns = result.get("vis_cols")
        config = getattr(self._backbone, "_latent_reallocation", None)
        pathology_enabled = bool(
            config is not None
            and config.pathology_aware
            and stage == "reasoner"
            and question_profile.enabled
        )
        pathology_weights = (
            _pathology_reallocation_weights(
                columns=vision_columns,
                spans=result.get("spans"),
                grids=result.get("grids"),
                magnifications=self._backbone._prune_image_magnifications,
                parent_indices=self._backbone._prune_image_parent_indices,
                spatial_merge_size=int(getattr(
                    getattr(
                        getattr(getattr(self._backbone, "model", None), "config", None),
                        "vision_config",
                        None,
                    ),
                    "spatial_merge_size",
                    2,
                )),
                parent_anchor_strength=config.parent_anchor_strength,
                fine_evidence_boost=(
                    config.fine_evidence_boost * question_profile.fine_boost
                ),
                coarse_evidence_boost=(
                    config.fine_evidence_boost * question_profile.coarse_boost
                ),
            )
            if (
                pathology_enabled
                and config is not None
                and not config.hierarchy_context_relay
            )
            else torch.empty(0, dtype=torch.float32)
        )
        if pathology_weights.numel():
            print(
                "[PathologyReallocation] "
                f"survivors={pathology_weights.numel()} "
                f"weight_range={pathology_weights.min().item():.3f}-"
                f"{pathology_weights.max().item():.3f}",
                flush=True,
            )
        if self._reallocation_carry_enabled and isinstance(vision_columns, torch.Tensor):
            self._carried_vision_columns = torch.unique(
                torch.cat((
                    self._carried_vision_columns,
                    vision_columns.detach().cpu().to(dtype=torch.long),
                )),
                sorted=True,
            )
        if (
            stage == "reasoner"
            and (
                self._reallocation_enabled
                or os.environ.get("VLMAS_RELAY4", "").strip() == "1"
            )
        ):
            grounding_masses = result.get("probe_mass_steps") or ()
            reasoner_relay_enabled = bool(
                config is not None
                and stage in config.visual_grounded_latent_kv_relay_stages
            )
            if os.environ.get("VLMAS_RELAY4", "").strip() == "1":
                reasoner_relay_enabled = True
            if config is not None and config.pathology_context_latent_kv_relay:
                visual_columns = (
                    vision_columns
                    if isinstance(vision_columns, torch.Tensor)
                    else torch.empty(0, dtype=torch.long)
                )
                selected_steps = select_pathology_context_steps(
                    result["past_key_values"],
                    RelationalVisualContext(
                        latent_steps=latent_steps,
                        vision_columns=visual_columns,
                        text_start=previous_cache_length,
                        patch_ids=pathology_visual_patch_ids(
                            visual_columns.detach().cpu(), result.get("spans")
                        ),
                        parent_patch_indices=(
                            self._backbone._prune_image_parent_indices
                        ),
                    ),
                )
            elif config is not None and config.visual_contrast_latent_kv_relay:
                selected_steps = select_visual_contrast_steps(
                    result["past_key_values"],
                    latent_steps=latent_steps,
                    vision_columns=(
                        vision_columns
                        if isinstance(vision_columns, torch.Tensor)
                        else torch.empty(0, dtype=torch.long)
                    ),
                    text_start=previous_cache_length,
                )
            elif (
                config is not None
                and config.visual_grounded_latent_kv_relay
                and reasoner_relay_enabled
            ):
                selected_steps = select_visual_grounded_steps_from_masses(
                    grounding_masses,
                    threshold=config.visual_grounding_threshold,
                    top_k=config.visual_grounding_top_k,
                )
            else:
                selected_steps = select_visual_grounded_steps(
                    result.get("latent_to_vision_attn", {}),
                    fallback_step_count=latent_steps,
                )
            if os.environ.get("VLMAS_RELAY4", "").strip() == "1":
                # Relay4 is an all-instance handoff: do not collapse the
                # latent trajectory to the older step-level Relay2 subset.
                selected_steps = torch.arange(latent_steps, dtype=torch.long)
            self._answerer_latent_kv = (
                extract_selected_latent_kv(
                    result["past_key_values"], latent_steps, selected_steps
                )
                if config is not None and reasoner_relay_enabled and (
                    config.latent_kv_relay
                    or config.visual_grounded_latent_kv_relay
                    or config.visual_contrast_latent_kv_relay
                    or config.pathology_context_latent_kv_relay
                    or os.environ.get("VLMAS_RELAY4", "").strip() == "1"
                )
                else {}
            )
            if os.environ.get("VLMAS_RELAY4", "").strip() == "1":
                # Relay4: compare actual cached K rows.  A Qwen3-VL 4B cache
                # stores 8 KV heads, not 32 query heads; GQA's four query-head
                # aliases must not create four duplicate K-K decisions.
                # Per (latent-step, physical-KV-head), compare the top visual
                # cosine against the top text cosine and average over layers.
                # The resulting 10 × 8 mask gates FULR for the corresponding
                # GQA group at the Answerer.
                cache4 = result["past_key_values"]
                vis4 = vision_columns.to(self._backbone.device) if isinstance(vision_columns, torch.Tensor) else torch.empty(0, dtype=torch.long, device=self._backbone.device)
                L0_4 = int(previous_cache_length)
                L1_4 = int(cache4.layers[0].keys.shape[2])
                vis4 = vis4[(vis4 >= 0) & (vis4 < L0_4)]
                text4 = torch.ones(L0_4, dtype=torch.bool, device=self._backbone.device)
                if vis4.numel():
                    text4[vis4] = False
                text_idx4 = text4.nonzero(as_tuple=False).flatten()
                visual_scores4: list[torch.Tensor] = []
                text_scores4: list[torch.Tensor] = []
                for layer4 in cache4.layers:
                    key4 = layer4.keys[0].float()
                    if vis4.numel() and text_idx4.numel():
                        key4 = torch.nn.functional.normalize(key4, dim=-1)
                        lat4 = key4[:, L1_4 - latent_steps:L1_4, :]
                        vis_key4 = key4.index_select(1, vis4)
                        text_key4 = key4.index_select(1, text_idx4)
                        visual_scores4.append(
                            torch.einsum("hmd,hnd->hmn", lat4, vis_key4).amax(dim=-1)
                        )
                        text_scores4.append(
                            torch.einsum("hmd,hnd->hmn", lat4, text_key4).amax(dim=-1)
                        )
                valid_steps = torch.arange(latent_steps, dtype=torch.long)
                mask4 = (
                    torch.stack(visual_scores4).mean(dim=0)
                    > torch.stack(text_scores4).mean(dim=0)
                    if visual_scores4 and text_scores4
                    else torch.zeros((0, latent_steps), dtype=torch.bool, device=self._backbone.device)
                )
                self._backbone._relay4_visual_mask = mask4.detach()
                kept_heads4 = int(mask4.sum())
                total_heads4 = int(mask4.numel())
                kept_steps4 = (
                    mask4.any(dim=0).sum().item() if mask4.numel() else 0
                )
                print(
                    f"[Relay4/FULR] selected_steps={valid_steps.tolist()} "
                    f"visual_winning_kv_heads={kept_heads4}/{total_heads4} "
                    f"covering_steps={kept_steps4}/{latent_steps}",
                    flush=True,
                )
            selected_count = relay_length(self._answerer_latent_kv)
            if config is not None and (
                config.visual_grounded_latent_kv_relay
                or config.visual_contrast_latent_kv_relay
                or config.pathology_context_latent_kv_relay
            ):
                rounded_masses = ", ".join(
                    f"{float(mass):.3f}" for mass in grounding_masses
                )
                print(
                    f"[{'PathologyContextRelay3' if config.pathology_context_latent_kv_relay else 'LatentKVRelay3' if config.visual_contrast_latent_kv_relay else 'LatentKVRelay2'}] "
                    f"visual_mass=[{rounded_masses}] "
                    f"selected={selected_steps.tolist()}",
                    flush=True,
                )
            if selected_count:
                print(
                    f"[LatentKVRelay] selected {selected_count}/{latent_steps} visual-grounded latent steps",
                    flush=True,
                )
            self._answerer_reallocation_scale = (
                question_profile.alpha_scale
                if config is not None and config.pathology_aware
                else 1.0
            )
            self._answerer_vision_columns = (
                torch.empty(0, dtype=torch.long)
                if config is not None and config.pathology_aware and not pathology_enabled
                else
                self._carried_vision_columns.clone()
                if self._reallocation_carry_enabled
                else vision_columns.detach().cpu()
                if isinstance(vision_columns, torch.Tensor)
                else torch.empty(0, dtype=torch.long)
            )
            self._answerer_vision_weights = pathology_weights
            # Sender-relay experiment: adopt the surviving tokens' FINAL
            # pruning-decision scores (hierarchy prefill prune, reasoner stage)
            # as the sender priority, only when they align 1:1 with the stored
            # Answerer vision columns. Anything else → empty priority, and the
            # destination builder reports an explicit fallback.
            survivor_scores = getattr(
                self._backbone, "_prefill_prune_survivor_scores", None
            )
            relay_columns = self._answerer_vision_columns
            if (
                isinstance(survivor_scores, torch.Tensor)
                and relay_columns.numel() > 0
                and survivor_scores.numel() == relay_columns.numel()
            ):
                self._answerer_vision_priority = survivor_scores.clone()
            else:
                self._answerer_vision_priority = torch.empty(0, dtype=torch.float32)
            self._sender_audit = None
            if self._answerer_vision_priority.numel():
                raw_priority = self._answerer_vision_priority
                norm_priority = raw_priority.clamp_min(0.0)
                norm_priority = norm_priority / norm_priority.sum().clamp_min(1e-12)
                group_ids = getattr(
                    self._backbone, "_prefill_prune_survivor_group_ids", None
                )
                image_ids = getattr(
                    self._backbone, "_prefill_prune_survivor_image_ids", None
                )
                image_id_list = (
                    image_ids.tolist() if isinstance(image_ids, torch.Tensor) else []
                )
                magnifications = getattr(
                    self._backbone, "_prune_image_magnifications", ()
                )
                self._sender_audit = {
                    "prune_stage": "reasoner_prefill_hierarchy",
                    "retained_visual_columns": relay_columns.tolist(),
                    "retained_original_group_ids": (
                        group_ids.tolist()
                        if isinstance(group_ids, torch.Tensor)
                        else None
                    ),
                    "roi_ids": image_id_list or None,
                    "magnifications": [
                        int(magnifications[index])
                        if index < len(magnifications)
                        else 0
                        for index in image_id_list
                    ],
                    "sender_priority_raw": raw_priority.tolist(),
                    "sender_priority_norm": norm_priority.tolist(),
                }
            self._answerer_visual_patch_ids = (
                pathology_visual_patch_ids(vision_columns.detach().cpu(), result.get("spans"))
                if (
                    pathology_enabled
                    and config is not None
                    and config.hierarchy_context_relay
                    and isinstance(vision_columns, torch.Tensor)
                )
                else torch.empty(0, dtype=torch.long)
            )
            self._answerer_parent_patch_indices = (
                self._backbone._prune_image_parent_indices
                if self._answerer_visual_patch_ids.numel()
                else ()
            )
        # KV-swap carrier probe (Step 1, env-gated; off = no-op): remember the
        # reasoner's visual columns and freeze the FIRST case's visual K/V as
        # the cross-slide donor bank for the terminal swap decodes.
        import os as _os
        # KV re-staging: remember the reasoner's absolute visual columns for
        # the terminal-time move (positions are stashed by the backbone).
        if (
            (
                _os.environ.get("VLMAS_KV_RESTAGE") == "1"
                or _os.environ.get("VLMAS_VG_PREFILL") == "1"
            )
            and stage == "reasoner"
            and isinstance(vision_columns, torch.Tensor)
            and vision_columns.numel()
        ):
            self._restage_vision_columns = vision_columns.detach().cpu()
        # Pulse re-park (VLMAS_KV_PULSE=1): the reasoner ran with the visual
        # block boundary-restaged at recency; now that its latent steps are
        # done, bank the block (K already rotated to its restaged positions)
        # and remove it from the cache until the terminal restore rebinds it.
        if (
            _os.environ.get("VLMAS_KV_PULSE") == "1"
            and stage == "reasoner"
            and getattr(self._backbone, "_early_vis_cols", None) is not None
        ):
            import numpy as _np_pulse

            from memory.prune import apply_kv_prune as _pulse_prune
            _pcache = result["past_key_values"]
            _pcols = self._backbone._early_vis_cols
            self._backbone._park_bank = {
                "k": [layer.keys[:, :, _pcols, :].detach().clone()
                      for layer in _pcache.layers],
                "v": [layer.values[:, :, _pcols, :].detach().clone()
                      for layer in _pcache.layers],
                "pos": self._backbone._restage_vis_positions.detach().clone(),
            }
            _pcache, _plen, _, _ = _pulse_prune(
                _pcache, int(result["past_len"]), _pcols.tolist(),
                _np_pulse.zeros(int(_pcols.numel()), dtype=bool), [],
                self._backbone.device)
            result["past_key_values"] = _pcache
            result["past_len"] = _plen
            self._backbone._early_vis_cols = None
            print(f"[KVPulse] re-parked {int(_pcols.numel())} visual cols after "
                  f"{stage} (cache {_plen})", flush=True)
        if (
            _os.environ.get("VLMAS_KV_SWAP_PROBE") == "1"
            and stage == "reasoner"
            and isinstance(vision_columns, torch.Tensor)
            and vision_columns.numel()
        ):
            self._probe_vision_columns = vision_columns.detach().cpu()
            probe_cache = result["past_key_values"]
            if (
                getattr(self._backbone, "_kv_probe_donor", None) is None
                and hasattr(probe_cache, "layers")
            ):
                donor_cols = vision_columns.to(self._backbone.device)
                self._backbone._kv_probe_donor = {
                    "k": [
                        layer.keys[:, :, donor_cols, :].detach().clone()
                        for layer in probe_cache.layers
                    ],
                    "v": [
                        layer.values[:, :, donor_cols, :].detach().clone()
                        for layer in probe_cache.layers
                    ],
                }
                self._probe_donor_case = True
            else:
                self._probe_donor_case = False
        if os.environ.get("VLMAS_LATDEC", "").strip():
            # Decode this stage's latent steps straight out of the latent loop
            # (memory/latdec_diag.py). Reads result["latent_trajectory"], which the
            # backbone already returns; no extra forward, no decode change.
            from memory import latdec_diag as _latdec
            _latdec.run(self, stage=stage, result=result,
                        case_index=getattr(self, "_rpath_case_index", -1))
        full_cache_length = int(result["past_len"])
        # C2 13 RELH (memory/relh.py): remember this role's terminal latent column + coordinate.
        if _relh.enabled():
            _relh.record_endpoint(self._backbone, stage, full_cache_length,
                                  int(result["pos_cursor"]))
        if stage == "reasoner" and isinstance(vision_columns, torch.Tensor):
            # Pathways probe (memory/rpath_patch.py): the Reasoner's m latent
            # columns are the cache tail right here (before the turn is closed
            # and before the Answerer appends), so their absolute indices stay
            # valid at the terminal. Record them; pass W also dumps their K/V.
            from memory import rpath_patch as _rpatch
            self._rpath_latent_cols = torch.arange(
                full_cache_length - latent_steps, full_cache_length, dtype=torch.long)
            self._rpath_visual_cols = vision_columns.detach().cpu()
            if _rpatch.dump_dir():
                _rpatch.dump_reasoner_kv(
                    self, result["past_key_values"],
                    latent_cols=self._rpath_latent_cols,
                    visual_cols=self._rpath_visual_cols,
                    case_index=getattr(self, "_rpath_case_index", -1),
                    is_donor_case=getattr(self._backbone, "_rpath_is_donor_case", False))
        match self._transport_mode:
            case "latent_only":
                retained_tokens = latent_steps
            case "sequential_info_only":
                retained_tokens = full_cache_length - previous_cache_length
            case "upstream":
                # Upstream/0717-faithful: keep the FULL grown KV, do NOT close the
                # assistant turn. The template was already re-emitted per agent
                # (continuation_user_turn is False for any non-cumulative mode), so
                # the KV is a stacked [system,user,assistant+latent] transcript with
                # system repeated and turns left open — exactly what upstream
                # generate_latent_batch produces.
                relay_rows = relay_length(boundary_relay)
                append_relay(result["past_key_values"], boundary_relay)
                return LatentAppendResult(
                    cache=result["past_key_values"],
                    cache_length=full_cache_length + relay_rows,
                    position_cursor=int(result["pos_cursor"]),
                    added_tokens=full_cache_length + relay_rows - previous_cache_length,
                )
            case "cumulative" | "cumulative_nothink":
                # Both variants keep the FULL grown KV and close the assistant turn so
                # the next role starts at a clean boundary. They differ only in whether
                # the closed turn carried an OPEN think block: "cumulative" generated its
                # latent inside <think> (close_thinking=True emits </think><|im_end|>),
                # while "cumulative_nothink" already closed think in the prompt (empty
                # think), so it must NOT emit a second </think> — only <|im_end|>.
                if os.environ.get("VLMAS_NO_TURN_CLOSE", "").strip() == "1":
                    # 0915 step 2 (LatentMAS-faithful): append NOTHING after the latent
                    # block; the next role's prompt is prefilled directly behind it.
                    closed_cache = result["past_key_values"]
                    closed_length = int(full_cache_length)
                    closed_cursor = int(result["pos_cursor"])
                else:
                    closed_cache, closed_length, closed_cursor = (
                        self._backbone.close_assistant_turn(
                            result["past_key_values"],
                            int(result["pos_cursor"]),
                            close_thinking=keep_thinking_open,
                        )
                    )
                if stage == "reasoner" and isinstance(vision_columns, torch.Tensor):
                    # Pathways probe: the turn-closing tokens (</think><|im_end|>)
                    # are the last columns computed over the Reasoner's visual +
                    # latent context before the Answerer prompt. Record them so
                    # the patch ladder can test them as a carrier.
                    from memory import rpath_patch as _rpatch
                    self._rpath_close_cols = torch.arange(
                        full_cache_length, closed_length, dtype=torch.long)
                    if _rpatch.dump_dir():
                        _rpatch.dump_close_kv(
                            self, closed_cache, close_cols=self._rpath_close_cols,
                            case_index=getattr(self, "_rpath_case_index", -1))
                if _tc.enabled():
                    _tc.record_stage(
                        self, stage=stage, start=int(previous_cache_length), past_len=int(full_cache_length),
                        latent_steps=int(latent_steps), closed_length=int(closed_length),
                        vis_cols=vision_columns if isinstance(vision_columns, torch.Tensor) else None,
                        close_thinking=bool(keep_thinking_open), pos_cursor=int(result["pos_cursor"]))
                relay_rows = relay_length(boundary_relay)
                append_relay(closed_cache, boundary_relay)
                return LatentAppendResult(
                    cache=closed_cache,
                    cache_length=closed_length + relay_rows,
                    position_cursor=closed_cursor,
                    added_tokens=closed_length + relay_rows - previous_cache_length,
                )
            case unreachable:
                raise AssertionError(f"unsupported transport mode: {unreachable}")
        retained_cache = self._backbone.retain_last_kv(
            result["past_key_values"],
            retained_tokens,
        )
        relay_rows = relay_length(boundary_relay)
        append_relay(retained_cache, boundary_relay)
        return LatentAppendResult(
            cache=retained_cache,
            cache_length=retained_tokens + relay_rows,
            position_cursor=int(result["pos_cursor"]),
            added_tokens=retained_tokens + relay_rows,
        )

    def decode(
        self,
        *,
        cache: CacheT | None,
        position_cursor: int,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int,
        json_prefix: str | None,
        json_schema: str | None = None,
    ) -> str:
        """Decode on the cache copy supplied by `LatentBackendAdapter`."""
        # VLMAS_NAV_GRAMMAR=1 (opt-in, backbone-agnostic): compile the caller's
        # JSON schema into an xgrammar logits mask for this decode instead of
        # discarding it. Off (default) keeps every decode byte-identical.
        grammar_processor = None
        if (
            json_schema is not None
            and json_prefix is not None
            and (
                os.environ.get("VLMAS_NAV_GRAMMAR") == "1"
                or os.environ.get("VLMAS_TERMINAL_GRAMMAR", "").strip() == "1"
            )
        ):
            from vision_text_mas.json_constraint import build_json_logits_processor
            from vision_text_mas.qwen_backend import _model_vocab_size

            lm_model = getattr(self._backbone, "vlm", None) or getattr(
                self._backbone, "model", None
            )
            grammar_processor = build_json_logits_processor(
                tokenizer=self._backbone.processor.tokenizer,
                json_schema=json_schema,
                model_vocab_size=_model_vocab_size(lm_model),
            )
        if cache is None:
            raise LatentImageRequiredError(stage="decode-before-prefill")
        torch.manual_seed(self._seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self._seed)
        config = getattr(self._backbone, "_latent_reallocation", None)
        if config is not None and (
            config.latent_kv_relay
            or config.visual_grounded_latent_kv_relay
            or config.visual_contrast_latent_kv_relay
            or config.pathology_context_latent_kv_relay
        ):
            cache_length = int(cache.layers[0].keys.shape[-2])
            selected_count = relay_length(self._answerer_latent_kv)
            append_relay(cache, self._answerer_latent_kv)
            if selected_count:
                relay_columns = torch.arange(
                    cache_length, cache_length + selected_count, dtype=torch.long
                )
                self._answerer_vision_columns = torch.unique(
                    torch.cat((self._answerer_vision_columns, relay_columns)), sorted=True
                )
        if json_prefix is None:
            return self._backbone.generate_on_kv(
                system_prompt,
                user_prompt,
                cache,
                max_new_tokens,
                clone_cache=False,
                pos_cursor=position_cursor,
                do_sample=self._terminal_do_sample,
                rep_penalty=self._terminal_repetition_penalty,
                no_repeat=self._terminal_no_repeat_ngram_size,
                temperature=self._temperature,
                top_p=self._top_p,
                # Match original LatentMAS Judger: open the Thinking turn and
                # let the model reason before emitting the boxed answer.
                enable_thinking=self._answerer_thinking,
                continuation_user_turn=(
                    os.environ.get("VLMAS_NO_CONTINUATION", "").strip() != "1"
                    and self._transport_mode in ("cumulative", "cumulative_nothink")
                ),
            )
        # The terminal answerer decodes greedily per --answerer-greedy and stops as
        # soon as its envelope closes, so the raw output is a clean single answer
        # instead of sampling-drift + repeated/rambled output. The tag answerer
        # (json_prefix "<answer>") stops at "</answer>"; the JSON answerer
        # (json_prefix '{"answer":') stops at the first "}" that closes the flat
        # object. The intermediate "{" JSON agents (navigator) are neither and keep
        # their prior sampling.
        prefix = json_prefix.strip()
        is_answer_tag = prefix.startswith("<answer")
        is_terminal_json = prefix.startswith('{"answer"')
        is_terminal_boxed = prefix.startswith(r"\boxed{")
        is_terminal_choice = prefix == "CHOICE"
        is_terminal = (
            is_answer_tag
            or is_terminal_json
            or is_terminal_boxed
            or is_terminal_choice
        )
        # Forced-navigator prefix (VLMAS_NAV_FORCED_PREFIX): the object is flat,
        # so the first "}" closes it; stop there instead of running the budget.
        is_navigator_json = prefix.startswith('{"root_ids"')
        stop_strings: tuple[str, ...] = ()
        if is_answer_tag:
            stop_strings = ("</answer>",)
        elif is_terminal_json or is_terminal_boxed or is_navigator_json:
            stop_strings = ("}",)
        if grammar_processor is not None:
            # The grammar emits one complete flat object; its closing brace is
            # the whole-output terminator. The teacher-forced prefix must be
            # dropped — the matcher generates the object from "{" itself.
            stop_strings = ("}",)
        if is_terminal_json and os.environ.get("VLMAS_TERMINAL_EOS_STOP", "").strip() == "1":
            # 0915 (0717-style): let the model end its own turn; no '}' stop string.
            stop_strings = ()
        answerer_alpha = (
            config.alpha_for_role("answerer", self._last_probe_mass)
            if is_terminal and config is not None
            else 0.0
        )
        if config is not None and config.pathology_aware:
            answerer_alpha *= self._answerer_reallocation_scale
        answerer_config = (
            replace(config, alpha=answerer_alpha, adaptive=False)
            if config is not None and answerer_alpha > 0.0
            else None
        )
        # Sender-relay experiment: build the Answerer-handoff destination
        # distribution for the static relay sources. A failed build falls back
        # EXPLICITLY to the canonical destination (relay_source→key_norm, no
        # weight vector), and the fallback is recorded in the relay audit.
        relay_override_weights: torch.Tensor | None = None
        relay_build_audit: dict | None = None
        if (
            answerer_config is not None
            and answerer_config.relay_source in STATIC_RELAY_SOURCES
        ):
            relay_override_weights, relay_build_audit = build_destination_weights(
                relay_source=answerer_config.relay_source,
                columns=self._answerer_vision_columns,
                sender_priority=self._answerer_vision_priority,
                base_seed=self._seed,
            )
            if relay_override_weights is None:
                print(
                    "[SenderRelay] fallback → canonical destination "
                    f"({relay_build_audit['fallback_reason']})",
                    flush=True,
                )
                answerer_config = replace(answerer_config, relay_source="key_norm")
        if relay_override_weights is not None:
            assert relay_override_weights.numel() == self._answerer_vision_columns.numel(), (
                "sender-relay destination weights misaligned with visual columns"
            )
        effective_answerer_weights = (
            relay_override_weights
            if relay_override_weights is not None
            else (
                self._answerer_vision_weights
                if self._answerer_vision_weights.numel()
                else None
            )
        )
        if answerer_config is None:
            context = nullcontext()
        elif effective_answerer_weights is not None:
            if answerer_config.hierarchy_context_relay:
                context = terminal_reallocation(
                    self._backbone,
                    vision_mask=self._answerer_vision_columns,
                    vision_weights=effective_answerer_weights,
                    visual_patch_ids=self._answerer_visual_patch_ids,
                    parent_patch_indices=self._answerer_parent_patch_indices,
                    config=answerer_config,
                )
            else:
                context = terminal_reallocation(
                    self._backbone,
                    vision_mask=self._answerer_vision_columns,
                    vision_weights=effective_answerer_weights,
                    config=answerer_config,
                )
        else:
            if answerer_config.hierarchy_context_relay:
                context = terminal_reallocation(
                    self._backbone,
                    vision_mask=self._answerer_vision_columns,
                    visual_patch_ids=self._answerer_visual_patch_ids,
                    parent_patch_indices=self._answerer_parent_patch_indices,
                    config=answerer_config,
                )
            else:
                context = terminal_reallocation(
                    self._backbone,
                    vision_mask=self._answerer_vision_columns,
                    config=answerer_config,
                )
        # KV re-staging (env-gated, terminal only): move the reasoner's visual
        # K/V columns to recency with K re-rotated to the new positions —
        # position-only counterpart of refeed. Mutates this case's cache.
        _park_pending = getattr(self._backbone, "_park_bank", None) is not None
        # Self-arbitrated restage (VLMAS_KV_ARBITRATE=1, needs park+restage):
        # snapshot the PRIOR-ONLY cache (visual still parked) before the
        # restore, so the terminal can be decoded through both channels and
        # the model's own likelihood picks the winner — the unified law
        # (sign(readout quality - prior strength)) operationalized per case,
        # with zero dataset/model-specific knobs.
        self._arb_prior_cache = None
        self._arb_prior_cursor = None
        if (
            os.environ.get("VLMAS_KV_ARBITRATE") == "1"
            and is_terminal and _park_pending
        ):
            from copy import deepcopy as _dc
            self._arb_prior_cache = _dc(cache)
            self._arb_prior_cursor = position_cursor
        if os.environ.get("VLMAS_VG_PREFILL") == "1" and is_terminal:
            # Visual-Grounded Receiver Prefill: hand the terminal the exact
            # post-C1 visual columns so the Answerer prompt prefill can read
            # ONLY them (latent_terminal masks the rest during prefill).
            self._backbone._vg_prefill_cols = getattr(
                self, "_restage_vision_columns", None)
        if (
            os.environ.get("VLMAS_KV_RESTAGE") == "1"
            and is_terminal
            and (
                _park_pending  # bank restore ignores the column args entirely
                or (
                    os.environ.get("VLMAS_KV_RESTAGE_AT", "terminal") == "terminal"
                    and isinstance(getattr(self, "_restage_vision_columns", None), torch.Tensor)
                    and getattr(self, "_restage_vision_columns").numel() > 0
                    and getattr(self._backbone, "_restage_vis_positions", None) is not None
                )
            )
        ):
            if self._reallocation_enabled:
                print("[KVRestage] WARNING: reallocation enabled — vision_mask "
                      "columns go stale after the move", flush=True)
            position_cursor, _restaged = self._backbone.restage_visual_kv(
                cache,
                getattr(self, "_restage_vision_columns",
                        torch.empty(0, dtype=torch.long)),
                getattr(self._backbone, "_restage_vis_positions",
                        torch.empty(3, 0, dtype=torch.long)),
                position_cursor,
            )
        # Carrier probe (protocol [7]): DROP the early-restaged visual block at
        # the terminal — the answer must then come from what the latents carry.
        if (
            os.environ.get("VLMAS_KV_DROP_VISUAL") == "1"
            and is_terminal
            and getattr(self._backbone, "_early_vis_cols", None) is not None
        ):
            from memory.prune import apply_kv_prune as _drop_prune
            import numpy as _np_drop
            _cols = self._backbone._early_vis_cols
            cache, _new_len, _, _ = _drop_prune(
                cache, self._backbone._kv_len(cache), _cols.tolist(),
                _np_drop.zeros(int(_cols.numel()), dtype=bool), [],
                self._backbone.device)
            self._backbone._early_vis_cols = None
            print(f"[KVDrop] removed {int(_cols.numel())} early-restaged visual "
                  f"cols at terminal (cache {_new_len})", flush=True)
        # Cache length before the terminal decode grows it in place; the
        # post-hoc probes below must re-decode from THIS state, not from a
        # cache that already contains the generated answer.
        probe_base_len = int(self._backbone._kv_len(cache))
        # Level 1/2 at the Answerer (VLMAS_ACCESS_DIAG_TERMINAL=1): same
        # M_V / C_V decomposition as the Reasoner latent steps, hooked around
        # the terminal prompt prefill + generation. Off = no hooks.
        from memory import access_diag as _accdiag
        _acc_terminal = contextlib.nullcontext()
        _acc_vis = getattr(self, "_rpath_visual_cols", None)
        if is_terminal and _accdiag.terminal_enabled() and _acc_vis is not None:
            _acc_terminal = _accdiag.capture_access(
                self._backbone, stage="terminal",
                visual_cols=_acc_vis.to(self._backbone.device))
        # C2 Provenance-Factorized Re-Read (VLMAS_ANSWERER_PFR=1|identity):
        # Answerer attention re-reads the visual KV with a context-updated
        # query while keeping the native visual mass rho_V. Off = no hooks.
        from memory import pfr_attention as _pfr
        _pfr_ctx = contextlib.nullcontext()
        if is_terminal and _pfr.enabled() and getattr(self, "_rpath_visual_cols", None) is not None:
            _pfr_ctx = _pfr.install_pfr(
                self._backbone, self._rpath_visual_cols.to(self._backbone.device),
                case_index=int(getattr(self, "_rpath_case_index", -1)))
        # Visual-contribution cut at the Answerer (VLMAS_VCUT=zero|mask|identity,
        # VLMAS_VCUT_STAGE contains A): hooks on every layer for the prompt
        # prefill + generation (and again for VER score-only below). Off = none.
        from memory import visual_cut as _vcut
        _vcut_ctx = contextlib.nullcontext()
        # Retrieval kill test (VLMAS_RETR=context|token|random): keep one
        # reasoning-selected cross-scale group of visual columns, mask the rest
        # (mask mode of the visual cut, Answerer stage; VER scored the same way).
        self._vcut_a_cols = getattr(self, "_rpath_visual_cols", None)
        if is_terminal and self._vcut_a_cols is not None and os.environ.get("VLMAS_RETR", "").strip() not in ("", "igsweep"):
            from memory import retrieval_group as _rg
            _bb = self._backbone
            _cols_all = self._vcut_a_cols.to(_bb.device)
            _rc = getattr(_bb, "_route_vis_cols", None)
            _pos = getattr(_bb, "_route_vis_pos", None)
            _pages = getattr(_bb, "_route_pages", None)
            if _rc is not None and int(_rc.numel()) == int(_cols_all.numel()) and _pos is not None and _pages:
                _keep_local, _ = _rg.select(
                    _bb, cache, _rc.to(_bb.device), _pos.to(_bb.device), _pages,
                    tuple(getattr(_bb, "_prune_image_parent_indices", ()) or ()),
                    int(getattr(self, "_rpath_case_index", -1)))
                _mask = torch.ones(int(_cols_all.numel()), dtype=torch.bool, device=_bb.device)
                _mask[_keep_local.to(_bb.device)] = False
                self._vcut_a_cols = _cols_all[_mask].cpu()
            else:
                print("[Retr] SKIP: visual bookkeeping missing/mismatch", flush=True)
        if is_terminal and _vcut.enabled("A") and self._vcut_a_cols is not None:
            _vcut_ctx = _vcut.install_visual_cut(
                self._backbone, self._vcut_a_cols.to(self._backbone.device), stage="A")
        # C2 Cardinality-Invariant Visual Evidence Competition at the Answerer
        # (VLMAS_ANSWERER_GROUPC2=1): LME group competition over 5x-root groups,
        # native visual mass preserved, all Answerer rows (prefill+generation,
        # and VER scoring below). Off = no hooks.
        _gc2_ctx = contextlib.nullcontext()
        self._gc2_groups = None
        if is_terminal and os.environ.get("VLMAS_ANSWERER_GROUPC2", "").strip() == "1":
            from memory import retrieval_ig as _rig2
            from memory import group_attn as _ga
            _gg = _rig2.group_abs_cols(self)
            if _gg is not None:
                _dev = self._backbone.device
                self._gc2_groups = (_gg[0].to(_dev), [g.to(_dev) for g in _gg[1]])
                _gc2_ctx = _ga.install_group_attention(
                    self._backbone, self._gc2_groups[1], self._gc2_groups[0], "c2")
                print(f"[GroupC2] groups={[int(g.numel()) for g in self._gc2_groups[1]]} "
                      f"n_vis={int(self._gc2_groups[0].numel())}", flush=True)
            else:
                print("[GroupC2] SKIP: visual bookkeeping missing/mismatch", flush=True)
        if is_terminal and os.environ.get("VLMAS_OFH", "").strip():
            # C2 structure test (memory/ofh.py): Observation-Factorized Latent Handoff —
            # the same LME competition as GroupC2, but the partition is the Navigator
            # observation (crop) with shuffled / random-group falsifiers. Native total
            # visual mass is preserved by the "c2" rule. Off = unchanged.
            from memory import group_attn as _ga4
            from memory import ofh as _ofh
            _om = os.environ.get("VLMAS_OFH", "true").strip()
            _og = _ofh.observation_groups(self, _om)
            if _og is not None:
                _dev = self._backbone.device
                self._gc2_groups = (_og[0].to(_dev), [g.to(_dev) for g in _og[1]])
                _gc2_ctx = _ga4.install_group_attention(
                    self._backbone, self._gc2_groups[1], self._gc2_groups[0], "c2")
                print(f"[OFH] mode={_om} obs={[int(g.numel()) for g in self._gc2_groups[1]]} "
                      f"n_vis={int(self._gc2_groups[0].numel())}", flush=True)
            else:
                print(f"[OFH] mode={_om} SKIP: visual bookkeeping missing/mismatch", flush=True)
        # C2 Physical-Support Additive Readout (VLMAS_PSAR=1|identity, memory/psar.py): in the
        # formation window, read the visual memory per physical support (native q/K/V, softmax
        # inside each support) and average the support messages, scale-matched to the native
        # read. Support = C1 branch of every retained column from the backbone's visual
        # provenance (_visual_meta). Off = no hooks.
        _psar_ctx = contextlib.nullcontext()
        if is_terminal and os.environ.get("VLMAS_PSAR", "").strip():
            from memory import psar as _psar
            _pcfg = _psar.config()
            _pg = _psar.engine_groups(self, _pcfg["support"])
            if _pg["groups"] is None:
                print(f"[PSAR] SKIP: {_pg['note']}", flush=True)
            else:
                _plo, _phi = _psar.layer_window(_pcfg["layers"], len(self._backbone.lm.layers))
                _psar_ctx = _psar.install_psar(
                    self._backbone, _pg["cols"].to(self._backbone.device), _pg["groups"],
                    layers=(_plo, _phi), eta=_pcfg["eta"], target=_pcfg["target"],
                    rows=_pcfg["rows"], identity=(_pcfg["mode"] == "identity"))
                print(f"[PSAR] mode={_pcfg['mode']} support={_pcfg['support']} layers={_plo}-{_phi} "
                      f"eta={_pcfg['eta']} target={_pcfg['target']} rows={_pcfg['rows']} "
                      f"n_vis={int(_pg['cols'].numel())} supports={len(_pg['groups'])} "
                      f"sizes={[len(g) for g in _pg['groups']]} src={_pg['note']}", flush=True)
        # C2 12.6 Latent-Referenced Physical-Support Residual Readout (VLMAS_LRPSR=1|identity,
        # memory/lrpsr.py): at the fixed formation layers, keep the native visual component that
        # lies in the span of the Reasoner latent values and rebuild only the complementary
        # component from the per-physical-support reads, with the native norms conserved.
        _lrpsr_ctx = contextlib.nullcontext()
        _lrpsr = None
        if is_terminal and os.environ.get("VLMAS_LRPSR", "").strip():
            from memory import lrpsr as _lrpsr
            _lcfg = _lrpsr.config()
            _lg = _lrpsr.engine_groups(self, _lcfg["support"])
            if _lg["groups"] is None:
                print(f"[LRPSR] SKIP: {_lg['note']}", flush=True)
            else:
                _llayers = _lrpsr.layer_set(_lcfg["layers"], len(self._backbone.lm.layers))
                _lrpsr_ctx = _lrpsr.install_lrpsr(
                    self._backbone, _lg["cols"].to(self._backbone.device),
                    _lg["latent_cols"].to(self._backbone.device), _lg["groups"],
                    layers=_llayers, rows=_lcfg["rows"],
                    identity=(_lcfg["mode"] == "identity"))
                print(f"[LRPSR] mode={_lcfg['mode']} support={_lcfg['support']} layers={list(_llayers)} "
                      f"rows={_lcfg['rows']} n_vis={int(_lg['cols'].numel())} "
                      f"n_latent={int(_lg['latent_cols'].numel())} supports={len(_lg['groups'])} "
                      f"sizes={[len(g) for g in _lg['groups']]} src={_lg['note']}", flush=True)
        # C2 12.7 Physical-Support Fisher Projection (VLMAS_PSFP=1|identity|diag, memory/psfp.py):
        # read the physical-support visual direction at the fixed formation layers, map it and the
        # native visual term to direct logits, and project the decision remainder in the native
        # softmax Fisher metric only when it opposes that direction. Logits only; no hidden state.
        _psfp_ctx = contextlib.nullcontext()
        _psfp = None
        if is_terminal and os.environ.get("VLMAS_PSFP", "").strip():
            from memory import psfp as _psfp
            _fcfg = _psfp.config()
            _fg = _psfp.engine_groups(self, _fcfg["support"])
            if _fg["groups"] is None:
                print(f"[PSFP] SKIP: {_fg['note']}", flush=True)
            else:
                _flayers = _psfp.layer_set(_fcfg["layers"], len(self._backbone.lm.layers))
                _psfp_ctx = _psfp.install_psfp(
                    self._backbone, _fg["cols"].to(self._backbone.device), _fg["groups"],
                    layers=_flayers, rows=_fcfg["rows"], apply=_psfp.active())
                print(f"[PSFP] mode={_fcfg['mode']} support={_fcfg['support']} layers={list(_flayers)} "
                      f"rows={_fcfg['rows']} n_vis={int(_fg['cols'].numel())} "
                      f"supports={len(_fg['groups'])} sizes={[len(g) for g in _fg['groups']]} "
                      f"apply={_psfp.active()} src={_fg['note']}", flush=True)
        # C2 13 Role-Endpoint Latent Handoff (VLMAS_RELH, memory/relh.py): the Answerer reads the
        # Reasoner's terminal latent column at its own role boundary coordinate. One cached column,
        # key phase only, no new column, native values.
        _relh_ctx = contextlib.nullcontext()
        _relh_a = None
        if is_terminal and os.environ.get("VLMAS_RELH", "").strip():
            from memory import relh as _relh_a
            _acfg = _relh_a.config()
            _aeps = _relh_a.endpoints_for(self._backbone, "answerer")
            if not _aeps or not _relh_a.active_at("answerer"):
                print(f"[RELH] SKIP: endpoints={_aeps} mode={_acfg['mode']}", flush=True)
            else:
                _relh_ctx = _relh_a.install_relh(
                    self._backbone, _aeps, int(position_cursor) - 1, rows=_acfg["rows"],
                    identity=(_acfg["mode"] == "identity"))
                print(f"[RELH] mode={_acfg['mode']} endpoints={_aeps} "
                      f"handoff={int(position_cursor) - 1} rows={_acfg['rows']}", flush=True)
        # C2 12.8 Latent-Anchored Dual-Frame Read (VLMAS_LADR=1|identity, memory/ladr.py): keep the
        # visual columns' acquisition MRoPE address AND a role-local address re-anchored at the
        # Reasoner boundary, and marginalise the two address scores (log-mean-exp) inside every
        # Answerer attention. Same K/V payload, value read once, no layer choice.
        _ladr_ctx = contextlib.nullcontext()
        _ladr = None
        if is_terminal and os.environ.get("VLMAS_LADR", "").strip():
            from memory import ladr as _ladr
            _dcfg = _ladr.config()
            _dg = _ladr.engine_state(self)
            if _dg["cols"] is None:
                print(f"[LADR] SKIP: {_dg['note']}", flush=True)
            else:
                _ladr_ctx = _ladr.install_ladr(
                    self._backbone, _dg["cols"].to(self._backbone.device), _dg["positions"],
                    position_cursor, rows=_dcfg["rows"],
                    identity=(_dcfg["mode"] == "identity"),
                    mass_conserved=(_dcfg["mode"] == "mc"))
                print(f"[LADR] mode={_dcfg['mode']} rows={_dcfg['rows']} "
                      f"n_vis={int(_dg['cols'].numel())} cursor={position_cursor} "
                      f"src={_dg['note']}", flush=True)
        # C2 Latent-Provenance Visual Handoff (VLMAS_LPVH=1|identity, memory/lpvh.py): compose the
        # Answerer's native read of the Reasoner latent columns with the Reasoner's recorded support
        # provenance A and re-allocate the Answerer's visual mass between physical supports
        # (rho_V and the within-support distribution conserved). Off = no hooks.
        _lpvh_ctx = contextlib.nullcontext()
        _lpvh_state = None
        if is_terminal and os.environ.get("VLMAS_LPVH", "").strip():
            from memory import lpvh as _lpvh
            _lcfg = _lpvh.config()
            _lpvh_state = _lpvh.engine_state(self, _lcfg["prov"])
            if not _lpvh_state["ok"]:
                print(f"[LPVH] SKIP: {_lpvh_state['note']}", flush=True)
                _lpvh_state = None
            else:
                _llo, _lhi = _lpvh.layer_window(_lcfg["layers"], len(self._backbone.lm.layers))
                _lpvh_ctx = _lpvh.install_lpvh(
                    self._backbone, _lpvh_state["A"], _lpvh_state["latent_cols"], _lpvh_state["vis"],
                    _lpvh_state["gid"], _lpvh_state["sizes"], layers=(_llo, _lhi), rows=_lcfg["rows"],
                    identity=(_lcfg["mode"] == "identity"), log=_lcfg["log"])
                print(f"[LPVH] mode={_lcfg['mode']} support={_lcfg['support']} prov={_lcfg['prov']} "
                      f"layers={_llo}-{_lhi} rows={_lcfg['rows']} n_vis={int(_lpvh_state['vis'].numel())} "
                      f"supports={len(_lpvh_state['groups'])} sizes={_lpvh_state['sizes'].tolist()} "
                      f"T={int(_lpvh_state['A'].shape[0])} src={_lpvh_state['note']}", flush=True)
        # C2 Support-Resolved Visual Write Handoff (VLMAS_SRVW=1|identity, memory/srvw.py): at the Answerer
        # boundary row, a_g = [M_g . C_g / |M_g|^2]_+ per layer (Reasoner visual write C_g vs the Answerer's
        # native support message M_g); the visual mass is then re-weighted by a between physical supports
        # (rho_V and within-support ratios conserved). Off = no hooks. Not combined with LPVH (same hook site).
        _srvw_ctx = contextlib.nullcontext()
        _srvw_state = None
        if is_terminal and os.environ.get("VLMAS_SRVW", "").strip():
            from memory import srvw as _srvw
            _scfg = _srvw.config()
            _srvw_state = _srvw.engine_state(self)
            if os.environ.get("VLMAS_LPVH", "").strip():
                _srvw_state = {"ok": False, "note": "VLMAS_LPVH also set (same self_attn post-hook site)"}
            if not _srvw_state["ok"]:
                print(f"[SRVW] SKIP: {_srvw_state['note']}", flush=True)
                _srvw_state = None
            else:
                _slo, _shi = _srvw.layer_window(_scfg["layers"], len(self._backbone.lm.layers))
                _srvw_ctx = _srvw.install_srvw(
                    self._backbone, _srvw_state["C"], _srvw_state["vis"], _srvw_state["gid"], _srvw_state["sizes"],
                    layers=(_slo, _shi), boundary=_scfg["boundary"], identity=(_scfg["mode"] == "identity"),
                    log=_scfg["log"])
                print(f"[SRVW] mode={_scfg['mode']} boundary={_scfg['boundary']} layers={_slo}-{_shi} "
                      f"n_vis={int(_srvw_state['vis'].numel())} supports={len(_srvw_state['groups'])} "
                      f"sizes={_srvw_state['sizes'].tolist()} src={_srvw_state['note']}", flush=True)
        # Canonical addressing restricted to this Answerer call (VLMAS_KV_ROUTE_TERMINAL_ONLY=1,
        # read in latent_terminal): the flag is on only for the duration of the call.
        self._backbone._route_terminal_call = bool(is_terminal)
        # 0915 probe bookkeeping: expose the case index to latent_terminal (print-only use).
        self._backbone._route_case_index = getattr(self, "_rpath_case_index", None)
        # VLMAS_TERMINAL_NO_FORCE=1 (0915, off = byte-identical): do NOT teacher-force
        # the '{"answer":' opener. The assistant turn starts empty (0717-style free
        # generation) and the model writes the whole object itself; the '}' stop and
        # every is_terminal-gated hook still key off the original json_prefix.
        _eff_prefix = json_prefix
        if is_terminal_json and os.environ.get("VLMAS_TERMINAL_NO_FORCE", "").strip() == "1":
            _eff_prefix = ""
        # VELH (C2 #14) / Text-cut: in-place changes on the Answerer's cache + attention-mass measurement.
        _velh_ctx = contextlib.nullcontext()
        _velh_t = None
        if is_terminal and os.environ.get("VLMAS_VELH", "").strip():
            from memory import velh as _velh_t
            _velh_ctx = _velh_t.terminal(self, cache, int(position_cursor), case_index=int(getattr(self, "_rpath_case_index", -1) or 0))
        _tc_ctx = contextlib.nullcontext()
        _tc_t = None
        if is_terminal and os.environ.get("VLMAS_TEXTCUT", "").strip():
            from memory import textcut as _tc_t
            _tc_ctx = _tc_t.terminal(self, cache, int(position_cursor), case_index=int(getattr(self, "_rpath_case_index", -1) or 0))
        _rn_ctx = contextlib.nullcontext()
        _rn_t = None
        if is_terminal and os.environ.get("VLMAS_RNLCR", "").strip():
            from memory import rnlcr as _rn_t
            _rn_ctx = _rn_t.terminal(self, cache, int(position_cursor), case_index=int(getattr(self, "_rpath_case_index", -1) or 0),
                                     system_prompt=system_prompt, user_prompt=user_prompt,
                                     json_prefix=(None if grammar_processor is not None else _eff_prefix))
        try:
            with context as realloc_probe, _acc_terminal, _pfr_ctx, _vcut_ctx, _gc2_ctx, \
                    _psar_ctx as _psar_stats, _lpvh_ctx as _lpvh_stats, _srvw_ctx as _srvw_stats, \
                    _lrpsr_ctx as _lrpsr_stats, _psfp_ctx as _psfp_stats, \
                    _ladr_ctx as _ladr_stats, \
                    _relh_ctx as _relh_stats, \
                    _velh_ctx as _velh_stats, _tc_ctx as _tc_stats, _rn_ctx as _rn_stats:
                output = generate_terminal_json(
                    backbone=self._backbone,
                    cache=cache,
                    position_cursor=position_cursor,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    json_prefix=(
                        None if grammar_processor is not None else _eff_prefix
                    ),
                    max_new_tokens=max_new_tokens,
                    temperature=self._temperature,
                    top_p=self._top_p,
                    do_sample=(
                        self._terminal_do_sample
                        and not (is_terminal_boxed or is_terminal_choice)
                    ),
                    min_new_tokens=1 if is_terminal_choice else 0,
                    force_single_digit=is_terminal_choice,
                    stop_strings=stop_strings,
                    stop_at_balanced_box=is_terminal_boxed,
                    no_repeat_ngram_size=self._terminal_no_repeat_ngram_size,
                    repetition_penalty=self._terminal_repetition_penalty,
                    logits_processor=grammar_processor,
                )
        finally:
            self._backbone._route_terminal_call = False
        if _relh_a is not None and _relh_stats is not None:
            print(f"[RELH] case {getattr(self, '_rpath_case_index', -1)} " + _relh_a.summary(_relh_stats), flush=True)
        if _velh_t is not None and _velh_stats is not None:
            print(f"[VELH] case {getattr(self, '_rpath_case_index', -1)} " + _velh_t.summary(_velh_stats), flush=True)
        if _tc_t is not None and _tc_stats is not None:
            print(f"[TextCut] case {getattr(self, '_rpath_case_index', -1)} " + _tc_t.summary(_tc_stats), flush=True)
        if _rn_t is not None and _rn_stats is not None:
            print(f"[RNLCR] case {getattr(self, '_rpath_case_index', -1)} " + _rn_t.summary(_rn_stats), flush=True)
        if _ladr is not None and _ladr_stats is not None:
            print(f"[LADR] case {getattr(self, '_rpath_case_index', -1)} " + _ladr.summary(_ladr_stats), flush=True)
        if _psfp is not None and _psfp_stats is not None:
            print(f"[PSFP] case {getattr(self, '_rpath_case_index', -1)} " + _psfp.summary(_psfp_stats), flush=True)
        if _lrpsr is not None and _lrpsr_stats is not None:
            print(f"[LRPSR] case {getattr(self, '_rpath_case_index', -1)} " + _lrpsr.summary(_lrpsr_stats), flush=True)
        if _psar_stats is not None:
            print(f"[PSAR] case {getattr(self, '_rpath_case_index', -1)} " + _psar.summary(_psar_stats), flush=True)
        if _lpvh_state is not None and _lpvh_stats is not None:
            print(f"[LPVH] case {getattr(self, '_rpath_case_index', -1)} " + _lpvh.summary(_lpvh_stats, _lpvh_state), flush=True)
        if _srvw_state is not None and _srvw_stats is not None:
            print(f"[SRVW] case {getattr(self, '_rpath_case_index', -1)} " + _srvw.summary(_srvw_stats, _srvw_state), flush=True)
        if realloc_probe is not None and realloc_probe.mean_after is not None:
            print(
                "[reallocate] answerer handoff vision mass "
                f"natural={realloc_probe.mean_mass:.4f} "
                f"achieved={realloc_probe.mean_after:.4f}",
                flush=True,
            )
        # Sender-relay experiment: persist a per-case receiver/sender audit for
        # every Answerer decode that ran under reallocation. Written by the
        # case loop as relay_audit.json; absent for base/pruning-only runs.
        if answerer_config is not None and is_terminal:
            mass_before = realloc_probe.mean_mass if realloc_probe is not None else None
            mass_after = realloc_probe.mean_after if realloc_probe is not None else None
            diagnostics: dict = {}
            key_norm_vis = (
                getattr(realloc_probe, "key_norm_vis", None)
                if realloc_probe is not None
                else None
            )
            if isinstance(key_norm_vis, torch.Tensor) and key_norm_vis.numel():
                key_norm_dist = key_norm_vis.clamp_min(0.0)
                key_norm_dist = key_norm_dist / key_norm_dist.sum().clamp_min(1e-12)
                priority = self._answerer_vision_priority
                if priority.numel() == key_norm_dist.numel():
                    sender_dist = priority.clamp_min(0.0)
                    sender_dist = sender_dist / sender_dist.sum().clamp_min(1e-12)
                    diagnostics["cosine_key_norm_vs_sender"] = cosine_similarity(
                        key_norm_dist, sender_dist
                    )
                    diagnostics["spearman_key_norm_vs_sender"] = rank_correlation(
                        key_norm_dist, sender_dist
                    )
                if (
                    relay_override_weights is not None
                    and relay_override_weights.numel() == key_norm_dist.numel()
                ):
                    diagnostics["cosine_key_norm_vs_destination"] = cosine_similarity(
                        key_norm_dist, relay_override_weights
                    )
                    diagnostics["spearman_key_norm_vs_destination"] = rank_correlation(
                        key_norm_dist, relay_override_weights
                    )
            receiver: dict = {
                "relay_source": answerer_config.relay_source,
                "alpha": float(answerer_alpha),
                "vision_mass_before": mass_before,
                "vision_mass_after": mass_after,
                "total_reallocated_mass": (
                    mass_after - mass_before
                    if mass_after is not None and mass_before is not None
                    else None
                ),
                "visual_columns": self._answerer_vision_columns.tolist(),
                "destination_weights": (
                    relay_override_weights.tolist()
                    if relay_override_weights is not None
                    else None
                ),
                "fallback_used": bool(
                    relay_build_audit is not None
                    and relay_build_audit.get("fallback_used")
                ),
            }
            if relay_build_audit is not None:
                receiver["build"] = {
                    key: value
                    for key, value in relay_build_audit.items()
                    if key != "weights_norm"
                }
            self.last_relay_audit = {
                "receiver": receiver,
                "sender": self._sender_audit,
                "diagnostics": diagnostics,
            }
        if (
            is_terminal
            and getattr(self, "_arb_prior_cache", None) is not None
        ):
            from vision_text_mas.latent_terminal import score_terminal_continuation

            # Decode/scoring only APPEND K/V along the sequence dim, so instead
            # of multi-GB deepcopies (8B OOM) we record per-layer lengths and
            # truncate back afterwards — existing positions are never written.
            def _cache_lens(target_cache):
                return [
                    (0 if layer.keys is None else layer.keys.shape[2])
                    for layer in target_cache.layers
                ]

            def _trim_cache(target_cache, lens):
                for layer, n in zip(target_cache.layers, lens):
                    if layer.keys is not None and layer.keys.shape[2] > n:
                        layer.keys = layer.keys[:, :, :n, :]
                        layer.values = layer.values[:, :, :n, :]

            _prior_lens = _cache_lens(self._arb_prior_cache)
            prior_output = generate_terminal_json(
                backbone=self._backbone,
                cache=self._arb_prior_cache,
                position_cursor=self._arb_prior_cursor,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                json_prefix=json_prefix,
                max_new_tokens=max_new_tokens,
                temperature=self._temperature,
                top_p=self._top_p,
                do_sample=False,
                stop_strings=("}",),
            )
            _trim_cache(self._arb_prior_cache, _prior_lens)

            def _mean_logp(target_cache, cursor, text):
                if not text:
                    return float("-inf")
                lens = _cache_lens(target_cache)
                try:
                    total, count = score_terminal_continuation(
                        backbone=self._backbone,
                        cache=target_cache,
                        position_cursor=cursor,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        json_prefix=json_prefix,
                        continuation=text,
                    )
                    return total / max(count, 1)
                except Exception as error:  # noqa: BLE001
                    print(f"[Arbitrate] scoring failed: {error!r}", flush=True)
                    return float("-inf")
                finally:
                    _trim_cache(target_cache, lens)
            judge = os.environ.get("VLMAS_KV_ARBITRATE_JUDGE", "own")
            if judge == "contrast":
                # v3 rule: difference-in-differences. Each answer is scored
                # under BOTH caches; summing a candidate's scores across the
                # two caches makes the rule symmetric in caches (cancels each
                # cache's self-preference — v2's 8B failure mode) while still
                # comparing both answers under identical judges (cancels
                # per-channel fluency — v1's failure mode).
                lv_v = _mean_logp(cache, position_cursor, output)
                lv_p = _mean_logp(cache, position_cursor, prior_output)
                lp_v = _mean_logp(
                    self._arb_prior_cache, self._arb_prior_cursor, output)
                lp_p = _mean_logp(
                    self._arb_prior_cache, self._arb_prior_cursor,
                    prior_output)
                visual_score = lv_v + lp_v
                prior_score = lv_p + lp_p
            elif judge == "visual":
                # v2 rule: ONE judge — the evidence-bearing (restaged) cache —
                # scores BOTH candidate answers under identical conditions, so
                # channel-fluency bias cancels and only evidence support
                # separates them. (v1 "own" let each channel score its own
                # answer; that collapsed to prior 77:3 on gtex — fluent
                # no-evidence text out-scores evidence-wrangling text.)
                visual_score = _mean_logp(cache, position_cursor, output)
                prior_score = _mean_logp(
                    cache, position_cursor, prior_output)
            else:
                # v1: score each candidate under ITS OWN channel
                visual_score = _mean_logp(cache, position_cursor, output)
                prior_score = _mean_logp(
                    self._arb_prior_cache, self._arb_prior_cursor, prior_output)
            chosen = "visual" if visual_score >= prior_score else "prior"
            print(
                f"[Arbitrate] visual={visual_score:.4f} prior={prior_score:.4f}"
                f" -> {chosen}", flush=True)
            if chosen == "prior":
                output = prior_output
            self._arb_prior_cache = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if is_terminal and os.environ.get("VLMAS_LATENT_READOUT", "").strip() == "1":
            # What do the Reasoner latent columns carry? Clone-decode a short
            # summary from three copies of the pre-Answerer cache: full /
            # crop columns removed (latent must carry the visual content) /
            # latent columns removed (crops only). Print-only, off = nothing.
            from copy import deepcopy as _dc
            from memory import ver_decode as _verd
            _ro_cache = cache
            if int(self._backbone._kv_len(cache)) > probe_base_len:
                _ro_cache = _dc(cache); _ro_cache.crop(probe_base_len)
            _ro_prompt = ("Based on everything in context, state in two sentences what tissue "
                          "you have observed and which organ it indicates. Do not repeat the question.")
            _views = [("full", None)]
            if getattr(self, "_rpath_visual_cols", None) is not None:
                _views.append(("novis", self._rpath_visual_cols))
            if getattr(self, "_rpath_latent_cols", None) is not None:
                _views.append(("nolat", self._rpath_latent_cols))
            for _name, _cols in _views:
                try:
                    _c = _dc(_ro_cache) if _cols is None else _verd.drop_columns(
                        self._backbone, _ro_cache, _cols)
                    _txt = generate_terminal_json(
                        backbone=self._backbone, cache=_c, position_cursor=position_cursor,
                        system_prompt="You are the Answerer.", user_prompt=_ro_prompt,
                        json_prefix=None, max_new_tokens=90, temperature=self._temperature,
                        top_p=self._top_p, do_sample=False)
                    print(f"[LatentReadout:{_name}] case {getattr(self, '_rpath_case_index', -1)}: "
                          f"{_txt.strip()[:400]!r}", flush=True)
                    del _c
                except Exception as _e:  # noqa: BLE001
                    print(f"[LatentReadout:{_name}] failed: {_e!r}", flush=True)
        if is_terminal and os.environ.get("VLMAS_ANSWERER_VER", "").strip() == "1":
            # C2 Visual Evidence Ratio Decoding: pick the candidate whose
            # log-prob gains most from the visual KV being present. Scored on
            # the pre-Answerer cache (probe_base_len crop), normal output kept
            # for rationale/format. Off = unchanged.
            from memory import ver_decode as _ver
            _ver_cands = getattr(self, "_ver_candidates", None)
            _ver_cache = cache
            if int(self._backbone._kv_len(cache)) > probe_base_len:
                from copy import deepcopy as _dc
                _ver_cache = _dc(cache)
                _ver_cache.crop(probe_base_len)
            _vcut_ver = contextlib.nullcontext()
            if _vcut.enabled("A") and getattr(self, "_vcut_a_cols", None) is not None:
                _vcut_ver = _vcut.install_visual_cut(
                    self._backbone, self._vcut_a_cols.to(self._backbone.device), stage="A")
            _gc2_ver = contextlib.nullcontext()
            if getattr(self, "_gc2_groups", None) is not None:
                from memory import group_attn as _ga2
                _gc2_ver = _ga2.install_group_attention(
                    self._backbone, self._gc2_groups[1], self._gc2_groups[0], "c2")
            with _vcut_ver, _gc2_ver:
                output = _ver.visual_evidence_ratio(
                    self, cache=_ver_cache, position_cursor=position_cursor,
                    system_prompt=system_prompt, user_prompt=user_prompt,
                    json_prefix=json_prefix, candidates=_ver_cands or (),
                    visual_cols=getattr(self, "_rpath_visual_cols", None),
                    normal_output=output, case_index=getattr(self, "_rpath_case_index", -1))
        if is_terminal and os.environ.get("VLMAS_RETR", "").strip() == "igsweep":
            # Kill test: latent-query score vs per-group causal usefulness
            # (group-removal margins on the pre-Answerer cache). Off = unchanged.
            from memory import retrieval_ig as _rig
            _ig_cache = cache
            if int(self._backbone._kv_len(cache)) > probe_base_len:
                from copy import deepcopy as _dc2
                _ig_cache = _dc2(cache)
                _ig_cache.crop(probe_base_len)
            _rig.run(self, cache=_ig_cache, position_cursor=position_cursor,
                     system_prompt=system_prompt, user_prompt=user_prompt,
                     json_prefix=json_prefix, candidates=getattr(self, "_ver_candidates", None) or (),
                     case_index=getattr(self, "_rpath_case_index", -1))
            if _ig_cache is not cache:
                del _ig_cache
        if is_terminal and os.environ.get("VLMAS_CMODE", "").strip():
            # C2 structural diagnosis (memory/cmode_diag.py): decision-row hidden state
            # under full vs zeroA on the pre-Answerer cache, for the cross-dataset
            # common-mode / residual decomposition. Off = unchanged.
            from memory import cmode_diag as _cmode
            _cm_cache = cache
            if int(self._backbone._kv_len(cache)) > probe_base_len:
                from copy import deepcopy as _dc3
                _cm_cache = _dc3(cache)
                _cm_cache.crop(probe_base_len)
            _cmode.run(self, cache=_cm_cache, position_cursor=position_cursor,
                       system_prompt=system_prompt, user_prompt=user_prompt,
                       json_prefix=json_prefix,
                       candidates=getattr(self, "_ver_candidates", None) or (),
                       case_index=getattr(self, "_rpath_case_index", -1))
            if _cm_cache is not cache:
                del _cm_cache
        if is_terminal and os.environ.get("VLMAS_VCCA", "").strip():
            # C2 structural diagnosis (memory/vcca_diag.py): at every candidate-trie
            # branching node, how much of the full-vs-zeroA visual delta lands in the
            # candidate-contrast subspace, before and after the final norm. Off = unchanged.
            from memory import vcca_diag as _vcca
            _vc_cache = cache
            if int(self._backbone._kv_len(cache)) > probe_base_len:
                from copy import deepcopy as _dc4
                _vc_cache = _dc4(cache)
                _vc_cache.crop(probe_base_len)
            _vcca.run(self, cache=_vc_cache, position_cursor=position_cursor,
                      system_prompt=system_prompt, user_prompt=user_prompt,
                      json_prefix=json_prefix,
                      candidates=getattr(self, "_ver_candidates", None) or (),
                      case_index=getattr(self, "_rpath_case_index", -1))
            if _vc_cache is not cache:
                del _vc_cache
        if is_terminal and os.environ.get("VLMAS_VGAIN", "").strip():
            # C2 causal test (memory/vgain_diag.py): Answerer visual-path gain gamma on the
            # visual value term with native alpha, plus wrong-slide / nonvisual controls;
            # decodes with this call's own generation kwargs. Off = unchanged.
            from memory import vgain_diag as _vgain
            _vg_cache = cache
            if int(self._backbone._kv_len(cache)) > probe_base_len:
                from copy import deepcopy as _dcvg
                _vg_cache = _dcvg(cache)
                _vg_cache.crop(probe_base_len)
            _vgain.run(self, cache=_vg_cache, position_cursor=position_cursor,
                       system_prompt=system_prompt, user_prompt=user_prompt,
                       json_prefix=json_prefix,
                       candidates=getattr(self, "_ver_candidates", None) or (),
                       case_index=getattr(self, "_rpath_case_index", -1),
                       normal_output=output,
                       gen_kwargs=dict(
                           json_prefix=(None if grammar_processor is not None else json_prefix),
                           max_new_tokens=max_new_tokens,
                           temperature=self._temperature,
                           top_p=self._top_p,
                           do_sample=(self._terminal_do_sample
                                      and not (is_terminal_boxed or is_terminal_choice)),
                           min_new_tokens=1 if is_terminal_choice else 0,
                           force_single_digit=is_terminal_choice,
                           stop_strings=stop_strings,
                           stop_at_balanced_box=is_terminal_boxed,
                           no_repeat_ngram_size=self._terminal_no_repeat_ngram_size,
                           repetition_penalty=self._terminal_repetition_penalty,
                           logits_processor=grammar_processor,
                       ))
            if _vg_cache is not cache:
                del _vg_cache
        if is_terminal and os.environ.get("VLMAS_CANON_DIAG", "").strip():
            # Model-level diagnosis of canonical addressing (memory/canon_diag.py): native vs
            # canonical / canonical_t / shift on copies of the pre-Answerer cache. Off = unchanged.
            from memory import canon_diag as _cdiag
            _cd_cache = cache
            if int(self._backbone._kv_len(cache)) > probe_base_len:
                from copy import deepcopy as _dccd
                _cd_cache = _dccd(cache)
                _cd_cache.crop(probe_base_len)
            _cdiag.run(self, cache=_cd_cache, position_cursor=position_cursor,
                       system_prompt=system_prompt, user_prompt=user_prompt,
                       json_prefix=json_prefix,
                       candidates=getattr(self, "_ver_candidates", None) or (),
                       case_index=getattr(self, "_rpath_case_index", -1),
                       normal_output=output)
            if _cd_cache is not cache:
                del _cd_cache
        if is_terminal and os.environ.get("VLMAS_RSS", "").strip():
            # Efficient Role-Support Sensitivity (memory/rss_diag.py): validation-only
            # leave-one-support-out replays of the Reasoner latent suffix on copies of the
            # cache, after the normal decode. Off = unchanged.
            from memory import rss_diag as _rssd
            _rssd.run(self, cache=cache, probe_base_len=probe_base_len,
                      position_cursor=position_cursor, system_prompt=system_prompt,
                      user_prompt=user_prompt, json_prefix=json_prefix,
                      case_index=getattr(self, "_rpath_case_index", -1), normal_output=output)
        if is_terminal and os.environ.get("VLMAS_RMVR", "").strip():
            # C2 structural diagnosis (memory/rmvr_diag.py): at every generated answer
            # step, the receiver competition margin of the zeroA arm vs the differential
            # visual push of full-vs-zeroA, over the whole vocabulary (candidate trie
            # kept as a closed-set control). Off = unchanged.
            from memory import rmvr_diag as _rmvr
            _rm_cache = cache
            if int(self._backbone._kv_len(cache)) > probe_base_len:
                from copy import deepcopy as _dc5
                _rm_cache = _dc5(cache)
                _rm_cache.crop(probe_base_len)
            _rmvr.run(self, cache=_rm_cache, position_cursor=position_cursor,
                      system_prompt=system_prompt, user_prompt=user_prompt,
                      json_prefix=json_prefix,
                      candidates=getattr(self, "_ver_candidates", None) or (),
                      case_index=getattr(self, "_rpath_case_index", -1),
                      normal_output=output, max_new_tokens=max_new_tokens)
            if _rm_cache is not cache:
                del _rm_cache
        if is_terminal and os.environ.get("VLMAS_OFH_DIAG", "").strip():
            # C2 structure test (memory/ofh.py): within-observation token-density
            # sensitivity of the native readout vs the observation-factorized one
            # (duplicate / subsample falsifiers). Off = unchanged.
            from memory import ofh as _ofh2
            _of_cache = cache
            if int(self._backbone._kv_len(cache)) > probe_base_len:
                from copy import deepcopy as _dc6
                _of_cache = _dc6(cache)
                _of_cache.crop(probe_base_len)
            _ofh2.run(self, cache=_of_cache, position_cursor=position_cursor,
                      system_prompt=system_prompt, user_prompt=user_prompt,
                      json_prefix=json_prefix,
                      case_index=getattr(self, "_rpath_case_index", -1),
                      max_new_tokens=max_new_tokens)
            if _of_cache is not cache:
                del _of_cache
        if is_terminal and os.environ.get("VLMAS_OBSCAL", "").strip():
            # C2 feasibility probe (memory/obscal_diag.py): per-observation visual write
            # at a fixed receiver layer, its local logit influence, and the receiver
            # competition it would have to clear. Measurement only. Off = unchanged.
            from memory import obscal_diag as _obscal
            _oc_cache = cache
            if int(self._backbone._kv_len(cache)) > probe_base_len:
                from copy import deepcopy as _dc7
                _oc_cache = _dc7(cache)
                _oc_cache.crop(probe_base_len)
            _obscal.run(self, cache=_oc_cache, position_cursor=position_cursor,
                        system_prompt=system_prompt, user_prompt=user_prompt,
                        json_prefix=json_prefix,
                        case_index=getattr(self, "_rpath_case_index", -1),
                        max_new_tokens=max_new_tokens)
            if _oc_cache is not cache:
                del _oc_cache
        if is_terminal and os.environ.get("VLMAS_ALSI", "").strip():
            # C2 structural diagnosis (memory/alsi_diag.py): decompose the Answerer
            # attention output into persistent visual KV / Reasoner latent / static text
            # / generated prefix, propagate each source through the native suffix, and
            # measure visual-vs-nonvisual latent interference. Measurement only.
            from memory import alsi_diag as _alsi
            _al_cache = cache
            if int(self._backbone._kv_len(cache)) > probe_base_len:
                from copy import deepcopy as _dc8
                _al_cache = _dc8(cache)
                _al_cache.crop(probe_base_len)
            _alsi.run(self, cache=_al_cache, position_cursor=position_cursor,
                      system_prompt=system_prompt, user_prompt=user_prompt,
                      json_prefix=json_prefix,
                      case_index=getattr(self, "_rpath_case_index", -1),
                      max_new_tokens=max_new_tokens)
            if _al_cache is not cache:
                del _al_cache
        if is_terminal and os.environ.get("VLMAS_LENS", "").strip():
            # Where is the answer decided? (memory/lens_diag.py) Read every decoder
            # layer's residual through the model's own final norm + unembedding and
            # record when the argmax locks onto the emitted token. Measurement only.
            from memory import lens_diag as _lens
            _ln_cache = cache
            if int(self._backbone._kv_len(cache)) > probe_base_len:
                from copy import deepcopy as _dc9
                _ln_cache = _dc9(cache)
                _ln_cache.crop(probe_base_len)
            _lens.run(self, cache=_ln_cache, position_cursor=position_cursor,
                      system_prompt=system_prompt, user_prompt=user_prompt,
                      json_prefix=json_prefix,
                      case_index=getattr(self, "_rpath_case_index", -1),
                      max_new_tokens=max_new_tokens)
            if _ln_cache is not cache:
                del _ln_cache
        if is_terminal and os.environ.get("VLMAS_GPFX", "").strip():
            # How much of the answer margin g is the JSON prefill worth?
            # (memory/gpfx_diag.py) Same generated tokens, teacher-forced under four
            # prompt conditions: full / no-prefix / prefix-only / cache-only.
            from memory import gpfx_diag as _gpfx
            _gp_cache = cache
            if int(self._backbone._kv_len(cache)) > probe_base_len:
                from copy import deepcopy as _dc10
                _gp_cache = _dc10(cache)
                _gp_cache.crop(probe_base_len)
            _gpfx.run(self, cache=_gp_cache, position_cursor=position_cursor,
                      system_prompt=system_prompt, user_prompt=user_prompt,
                      json_prefix=json_prefix,
                      case_index=getattr(self, "_rpath_case_index", -1),
                      max_new_tokens=max_new_tokens)
            if _gp_cache is not cache:
                del _gp_cache
        if is_terminal and os.environ.get("VLMAS_NLOI", "").strip():
            # C2 structural diagnosis (memory/nloi_diag.py): exact single / pair
            # observation value-cut forwards at the Answerer, final-state interaction
            # Psi_ij and complementarity C_ij, true vs token-shuffled grouping.
            from memory import nloi_diag as _nloi
            _nl_cache = cache
            if int(self._backbone._kv_len(cache)) > probe_base_len:
                from copy import deepcopy as _dc11
                _nl_cache = _dc11(cache)
                _nl_cache.crop(probe_base_len)
            _nloi.run(self, cache=_nl_cache, position_cursor=position_cursor,
                      system_prompt=system_prompt, user_prompt=user_prompt,
                      json_prefix=json_prefix,
                      case_index=getattr(self, "_rpath_case_index", -1),
                      max_new_tokens=max_new_tokens)
            if _nl_cache is not cache:
                del _nl_cache
        if is_terminal and os.environ.get("VLMAS_RPATH_PATCH", "").strip():
            from memory import rpath_patch as _rpatch
            _rpatch.run_patch_probe(
                self, cache=cache, position_cursor=position_cursor,
                system_prompt=system_prompt, user_prompt=user_prompt,
                json_prefix=json_prefix, max_new_tokens=max_new_tokens,
                normal_output=output, pre_answerer_len=probe_base_len)
        if is_terminal and os.environ.get("VLMAS_KV_SWAP_PROBE") == "1":
            self._run_kv_swap_probe(
                cache=cache,
                pre_answerer_len=probe_base_len,
                position_cursor=position_cursor,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                json_prefix=json_prefix,
                max_new_tokens=max_new_tokens,
                normal_output=output,
            )
        return output

    def _run_kv_swap_probe(
        self,
        *,
        cache: CacheT,
        position_cursor: int,
        system_prompt: str,
        user_prompt: str,
        json_prefix: str | None,
        max_new_tokens: int,
        normal_output: str,
        pre_answerer_len: int | None = None,
    ) -> None:
        """Step-1 carrier probe: re-decode the SAME terminal with the visual
        K / V / K+V swapped to a frozen cross-slide donor, on cache copies.
        Appends one JSONL row per case; never touches the real decode.

        VLMAS_KV_SWAP_PROBE_BANDS="10,18,26,33" adds Step-3 coarse layer
        resolution: one extra V-only swap PER listed layer (all other layers
        keep matched V), logged as V_L{i}.
        VLMAS_KV_SWAP_PROBE_SCORE=1 additionally teacher-forces the NORMAL
        output under every cache variant and logs its sum log-probability
        (score_normal + score_{mode}), so Δ = score_normal − score_mode is a
        graded slide-specificity signal even when the argmax answer is stable.
        """
        from copy import deepcopy

        from vision_text_mas.latent_terminal import score_terminal_continuation

        columns = getattr(self, "_probe_vision_columns", None)
        donor = getattr(self._backbone, "_kv_probe_donor", None)
        live_len = int(self._backbone._kv_len(cache))
        if pre_answerer_len is not None and live_len > int(pre_answerer_len):
            # generate_terminal_json grew `cache` in place with the Answerer
            # prompt + answer; crop a copy back so the probe re-decodes from
            # the pre-Answerer state instead of copying the answer in context.
            cache = deepcopy(cache)
            cache.crop(int(pre_answerer_len))
            print(f"[KVSwapProbe] cache {live_len} -> cropped to pre-Answerer "
                  f"{int(pre_answerer_len)}", flush=True)
        row: dict = {
            "n_vis": int(columns.numel()) if isinstance(columns, torch.Tensor) else 0,
            "donor_case": bool(getattr(self, "_probe_donor_case", False)),
            "normal": normal_output,
            "cache_len_live": live_len,
            "cache_len_probe": int(pre_answerer_len) if pre_answerer_len is not None else live_len,
        }
        want_score = os.environ.get("VLMAS_KV_SWAP_PROBE_SCORE") == "1"

        def score_on_copy(source_cache) -> float | None:
            # Scores are a supplementary graded signal; a scoring failure must
            # never take down the probe's primary flip measurements.
            if not (want_score and normal_output):
                return None
            score_cache = deepcopy(source_cache)
            try:
                total, _ = score_terminal_continuation(
                    backbone=self._backbone,
                    cache=score_cache,
                    position_cursor=position_cursor,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    json_prefix=json_prefix,
                    continuation=normal_output,
                )
            except Exception as error:  # noqa: BLE001
                if not getattr(self, "_probe_score_error_logged", False):
                    print(f"[KVSwapProbe] scoring disabled: {error!r}", flush=True)
                    self._probe_score_error_logged = True
                return None
            finally:
                del score_cache
            return total

        if (
            isinstance(columns, torch.Tensor)
            and columns.numel()
            and donor is not None
            and not row["donor_case"]
            and hasattr(cache, "layers")
        ):
            device_cols = columns.to(self._backbone.device)
            n = min(int(donor["v"][0].shape[2]), int(device_cols.numel()))
            swap_cols = device_cols[:n]
            row["swapped_cols"] = n
            normal_score = score_on_copy(cache)
            if normal_score is not None:
                row["score_normal"] = normal_score
            num_layers = len(cache.layers)
            modes: list[tuple[str, bool, bool, set[int] | None]] = [
                ("K", True, False, None),
                ("V", False, True, None),
                ("KV", True, True, None),
            ]
            for token in os.environ.get("VLMAS_KV_SWAP_PROBE_BANDS", "").split(","):
                token = token.strip()
                if not token:
                    continue
                try:
                    band_layer = int(token)
                except ValueError:
                    continue
                if 0 <= band_layer < num_layers:
                    modes.append((f"V_L{band_layer}", False, True, {band_layer}))
            for mode, swap_k, swap_v, layer_subset in modes:
                probe_cache = deepcopy(cache)
                for layer_index, layer in enumerate(probe_cache.layers):
                    if layer_subset is not None and layer_index not in layer_subset:
                        continue
                    if swap_k:
                        layer.keys[:, :, swap_cols, :] = (
                            donor["k"][layer_index][:, :, :n, :].to(layer.keys.dtype)
                        )
                    if swap_v:
                        layer.values[:, :, swap_cols, :] = (
                            donor["v"][layer_index][:, :, :n, :].to(layer.values.dtype)
                        )
                mode_score = score_on_copy(probe_cache)
                if mode_score is not None:
                    row[f"score_{mode}"] = mode_score
                row[mode] = generate_terminal_json(
                    backbone=self._backbone,
                    cache=probe_cache,
                    position_cursor=position_cursor,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    json_prefix=json_prefix,
                    max_new_tokens=max_new_tokens,
                    temperature=self._temperature,
                    top_p=self._top_p,
                    do_sample=False,
                    stop_strings=("}",),
                )
                del probe_cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        out_path = os.environ.get(
            "VLMAS_KV_SWAP_PROBE_OUT", "kv_swap_probe.jsonl"
        )
        import json as _json
        with open(out_path, "a", encoding="utf-8") as handle:
            handle.write(_json.dumps(row, ensure_ascii=False) + "\n")
        print(
            f"[KVSwapProbe] logged (n_vis={row['n_vis']}, donor_case={row['donor_case']})",
            flush=True,
        )

    def release(self, cache: CacheT | None) -> None:
        """Release allocator-held CUDA blocks after the case loses its cache."""
        _ = cache
        self._last_probe_mass = None
        self._answerer_vision_columns = torch.empty(0, dtype=torch.long)
        self._answerer_vision_weights = torch.empty(0, dtype=torch.float32)
        self._answerer_visual_patch_ids = torch.empty(0, dtype=torch.long)
        self._answerer_parent_patch_indices = ()
        self._answerer_reallocation_scale = 1.0
        self._answerer_latent_kv = {}
        self._answerer_vision_priority = torch.empty(0, dtype=torch.float32)
        self._sender_audit = None
        self._carried_vision_columns = torch.empty(0, dtype=torch.long)
        self._backbone._reallocation_carried_vision_columns = torch.empty(
            0, dtype=torch.long
        )
        self._backbone._reallocation_vision_weights = torch.empty(0, dtype=torch.float32)
        if self._reallocation_enabled and hasattr(
            self._backbone, "_latent_reallocation_probe"
        ):
            del self._backbone._latent_reallocation_probe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
