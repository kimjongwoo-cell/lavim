"""Physical Qwen KV engine for Latent VL-MAS without enabled pruning."""

from __future__ import annotations

import importlib
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
from vision_text_mas.latent_context_relay import (
    append_latent_kv_relay,
    extract_selected_terminal_latent_kv,
    latent_relay_length,
    select_visual_grounded_latent_steps,
)
from vision_text_mas.latent_terminal import generate_terminal_json
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
        question_end = len(user_prompt)
    question = user_prompt[question_start:question_end]
    background = "Describe every patch using patch_id, quality, architecture, cellularity"
    return [question], [background]


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

    def _kv_len(self, cache: CacheT) -> int: ...

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
            (prune_config is not None and not prune_config.prefill_hierarchy_aware)
            or (reallocation is not None and reallocation.latent_kv_relay)
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
        self._backbone._prune_image_magnifications = tuple(
            int(image.info.get("latent_magnification", 0)) for image in images
        )
        self._backbone._prune_image_parent_indices = tuple(
            patch_index.get(str(image.info.get("latent_parent_patch_id", "")), -1)
            if int(image.info.get("latent_magnification", 0)) > 5 else -1
            for image in images
        )
        # Planner and Navigator retain complete visual control context for the
        # root-grid contract. Only pathology evidence enters morphology pruning.
        self._backbone._prune_morphology_enabled = stage == "reasoner"
        prune_args = getattr(self._backbone, "_prune_args", None)
        reasoner_only_pruning = bool(
            prune_args is not None and getattr(prune_args, "prune_reasoner_only", False)
        )
        prune_this_stage = not reasoner_only_pruning or stage == "reasoner"
        reasoner_targets, reasoner_background = (
            _reasoner_attention_targets(user_prompt)
            if stage == "reasoner" else (None, None)
        )
        layers = list(self._latent_probe_layers)
        pruning_probe_layer = (
            getattr(prune_args, "prune_probe_layer", None)
            if prune_this_stage and stage != "navigator"
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
            result = self._backbone.grounded_prefill_and_latent(
                images=list(images),
                system_prompt=system_prompt,
                user_text=user_prompt,
                m=latent_steps,
                l_mid_layers=layers,
                enable_thinking=keep_thinking_open,
                want_attn=self._capture_attention and prune_this_stage,
                role_targets=reasoner_targets,
                bg_targets=reasoner_background,
            )
        else:
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
                role_targets=reasoner_targets,
                bg_targets=reasoner_background,
                continuation_user_turn=self._transport_mode
                in ("cumulative", "cumulative_nothink"),
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
                and not config.latent_kv_relay
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
        if self._reallocation_enabled and stage == "reasoner":
            selected_latent_steps = select_visual_grounded_latent_steps(
                result.get("latent_to_vision_attn", {})
            )
            self._answerer_latent_kv = (
                extract_selected_terminal_latent_kv(
                    result["past_key_values"],
                    latent_steps=latent_steps,
                    selected_steps=selected_latent_steps,
                )
                if config is not None and config.latent_kv_relay
                else {}
            )
            relay_count = latent_relay_length(self._answerer_latent_kv)
            if relay_count:
                print(
                    f"[LatentKVRelay] selected {relay_count}/{latent_steps} "
                    "visual-grounded Reasoner latent K/V entries",
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
        full_cache_length = int(result["past_len"])
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
                return LatentAppendResult(
                    cache=result["past_key_values"],
                    cache_length=full_cache_length,
                    position_cursor=int(result["pos_cursor"]),
                    added_tokens=full_cache_length - previous_cache_length,
                )
            case "cumulative" | "cumulative_nothink":
                # Both variants keep the FULL grown KV and close the assistant turn so
                # the next role starts at a clean boundary. They differ only in whether
                # the closed turn carried an OPEN think block: "cumulative" generated its
                # latent inside <think> (close_thinking=True emits </think><|im_end|>),
                # while "cumulative_nothink" already closed think in the prompt (empty
                # think), so it must NOT emit a second </think> — only <|im_end|>.
                closed_cache, closed_length, closed_cursor = (
                    self._backbone.close_assistant_turn(
                        result["past_key_values"],
                        int(result["pos_cursor"]),
                        close_thinking=keep_thinking_open,
                    )
                )
                return LatentAppendResult(
                    cache=closed_cache,
                    cache_length=closed_length,
                    position_cursor=closed_cursor,
                    added_tokens=closed_length - previous_cache_length,
                )
            case unreachable:
                raise AssertionError(f"unsupported transport mode: {unreachable}")
        retained_cache = self._backbone.retain_last_kv(
            result["past_key_values"],
            retained_tokens,
        )
        return LatentAppendResult(
            cache=retained_cache,
            cache_length=retained_tokens,
            position_cursor=int(result["pos_cursor"]),
            added_tokens=retained_tokens,
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
        _ = json_schema
        if cache is None:
            raise LatentImageRequiredError(stage="decode-before-prefill")
        torch.manual_seed(self._seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self._seed)
        config = getattr(self._backbone, "_latent_reallocation", None)
        if config is not None and config.latent_kv_relay:
            cache_length = self._backbone._kv_len(cache)
            relay_count = latent_relay_length(self._answerer_latent_kv)
            append_latent_kv_relay(cache, self._answerer_latent_kv)
            if relay_count:
                relay_columns = torch.arange(
                    cache_length,
                    cache_length + relay_count,
                    dtype=torch.long,
                )
                self._answerer_vision_columns = torch.unique(
                    torch.cat((self._answerer_vision_columns, relay_columns)),
                    sorted=True,
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
                continuation_user_turn=self._transport_mode
                in ("cumulative", "cumulative_nothink"),
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
        stop_strings: tuple[str, ...] = ()
        if is_answer_tag:
            stop_strings = ("</answer>",)
        elif is_terminal_json or is_terminal_boxed:
            stop_strings = ("}",)
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
        if answerer_config is None:
            context = nullcontext()
        elif self._answerer_vision_weights.numel():
            if answerer_config.hierarchy_context_relay:
                context = terminal_reallocation(
                    self._backbone,
                    vision_mask=self._answerer_vision_columns,
                    vision_weights=self._answerer_vision_weights,
                    visual_patch_ids=self._answerer_visual_patch_ids,
                    parent_patch_indices=self._answerer_parent_patch_indices,
                    config=answerer_config,
                )
            else:
                context = terminal_reallocation(
                    self._backbone,
                    vision_mask=self._answerer_vision_columns,
                    vision_weights=self._answerer_vision_weights,
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
        with context as realloc_probe:
            output = generate_terminal_json(
                backbone=self._backbone,
                cache=cache,
                position_cursor=position_cursor,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                json_prefix=json_prefix,
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
            )
        if realloc_probe is not None and realloc_probe.mean_after is not None:
            print(
                "[reallocate] answerer handoff vision mass "
                f"natural={realloc_probe.mean_mass:.4f} "
                f"achieved={realloc_probe.mean_after:.4f}",
                flush=True,
            )
        return output

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
