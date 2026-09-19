"""Install the same actual-latent-KV insertion at both native role handoffs."""

import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path

import torch
from PIL import Image
from pvcr.errors import LayoutError
from transformers.cache_utils import Cache

from latent_visual_relay.attention import AttentionState, installed_attention
from latent_visual_relay.cache import InsertionAudit, insert_selected
from latent_visual_relay.capture import Bank, Capture, Mode
from vision_text_mas import latent_hf_ablation_cli as cli
from vision_text_mas import latent_hybrid_variants as hybrid
from vision_text_mas.contracts import CaseInput, RunResult
from vision_text_mas.latent_qwen_engine import LatentAppendResult, LatentQwenEngine


class LatentVisualRelay:
    """Own per-case captured addresses and appended-head masks, never model weights."""

    def __init__(self, output: Path, mode: Mode, relay_log_gain: float = 0.0) -> None:
        self.output = output
        self.mode: Mode = mode
        self.attention = AttentionState(relay_log_gain=relay_log_gain)
        self.receiver_relation_gain = 0.0
        self.banks: dict[str, Bank] = {}
        self.captures: dict[str, Capture] = {}
        self.insertions: dict[str, InsertionAudit] = {}

    def attach(self, cache: Cache | None, sender: str) -> None:
        if self.mode == "off":
            return
        if cache is None or sender not in self.banks:
            raise LayoutError(f"Missing {sender} latent cache at receiver handoff")
        bank = self.banks[sender]
        inserted = insert_selected(cache, bank.source, bank.selected)
        inserted = replace(inserted, sender=sender)
        self.insertions[sender] = inserted.audit
        count = int(inserted.columns.numel())
        insertion = inserted.audit.insertion_index
        self.attention.visual_columns = bank.source.visual_columns + (
            bank.source.visual_columns >= insertion
        ) * count
        if count:
            self.attention.blocks = [
                replace(
                    block, columns=block.columns + (block.columns >= insertion) * count
                )
                for block in self.attention.blocks
            ]
            self.attention.blocks.append(inserted)

    @contextmanager
    def installed(self) -> Iterator[None]:
        native_append = LatentQwenEngine.append
        native_decode = LatentQwenEngine.decode
        native_step = hybrid.latent_reallocation
        native_case = cli.run_case_with_retries
        experiment = self

        def append(
            self: LatentQwenEngine[Cache],
            *,
            cache: Cache | None,
            cache_length: int,
            position_cursor: int,
            images: tuple[Image.Image, ...],
            system_prompt: str,
            user_prompt: str,
            latent_steps: int,
            stage: str = "",
        ) -> LatentAppendResult[Cache]:
            # Exact existing engine ABI; MRoPE cursor is independent of cache length.
            if stage == "reasoner":
                experiment.attach(cache, "navigator")
                if cache is not None:
                    cache_length = int(cache.get_seq_length())
            if stage not in ("navigator", "reasoner"):
                return native_append(
                    self,
                    cache=cache,
                    cache_length=cache_length,
                    position_cursor=position_cursor,
                    images=images,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    latent_steps=latent_steps,
                    stage=stage,
                )
            if latent_steps != 10 or self._transport_mode != "cumulative":
                raise LayoutError("Latent visual relay requires cumulative10-step Base")
            capture = Capture(stage, images, experiment.mode)
            experiment.attention.capture = capture
            visual = getattr(self._backbone, "visual", None)
            if not isinstance(visual, torch.nn.Module):
                raise LayoutError("Missing Qwen vision module")
            handle = visual.register_forward_pre_hook(
                capture.grid_hook, with_kwargs=True
            )
            try:
                result = native_append(
                    self,
                    cache=cache,
                    cache_length=cache_length,
                    position_cursor=position_cursor,
                    images=images,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    latent_steps=latent_steps,
                    stage=stage,
                )
                if result.cache is None:
                    raise LayoutError("Missing sender cache after latent generation")
                bank = capture.finish(result.cache)
                if bank.source.latent_columns[-1] >= result.cache_length:
                    raise LayoutError("Latent source columns exceed cache")
                experiment.banks[stage] = bank
                experiment.captures[stage] = capture
                return result
            finally:
                handle.remove()
                experiment.attention.capture = None

        @contextmanager
        def step(
            backbone: hybrid._VariantBackbone,
            vision_columns: torch.Tensor,
            visual_patch_ids: torch.Tensor | None = None,
            parent_patch_indices: tuple[int, ...] = (),
        ) -> Iterator[None]:
            # Exact native latent-reallocation context ABI; no new latent forwards.
            capture = experiment.attention.capture
            with native_step(
                backbone, vision_columns, visual_patch_ids, parent_patch_indices
            ):
                if capture is None:
                    yield None
                    return
                capture.prepare(vision_columns)
                capture.observing = True
                try:
                    yield None
                finally:
                    capture.observing = False
                capture.step += 1

        def decode(
            self: LatentQwenEngine[Cache],
            *,
            cache: Cache | None,
            position_cursor: int,
            system_prompt: str,
            user_prompt: str,
            max_new_tokens: int,
            json_prefix: str | None,
            json_schema: str | None = None,
        ) -> str:
            # The runner requires the native structured-JSON Answerer protocol.
            saved_blocks = experiment.attention.blocks
            saved_relation_gain = experiment.attention.receiver_relation_gain
            if json_prefix is not None and json_prefix.strip().startswith('{"answer"'):
                experiment.attach(cache, "reasoner")
                if experiment.mode == "attention_context_read":
                    experiment.attention.receiver_relation_gain = (
                        experiment.receiver_relation_gain
                    )
                insertion = experiment.insertions.get("reasoner")
                if insertion is not None and insertion.after > insertion.before:
                    bank = experiment.banks["reasoner"]
                    self._answerer_vision_columns = (
                        torch.cat(
                            (
                                bank.source.visual_columns,
                                experiment.attention.blocks[-1].columns,
                            )
                        )
                        .sort()
                        .values
                    )
            try:
                return native_decode(
                    self,
                    cache=cache,
                    position_cursor=position_cursor,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_new_tokens=max_new_tokens,
                    json_prefix=json_prefix,
                    json_schema=json_schema,
                )
            finally:
                experiment.attention.blocks = saved_blocks
                experiment.attention.receiver_relation_gain = saved_relation_gain

        def run_case(
            *,
            case: CaseInput,
            max_attempts: int,
            operation: Callable[[], RunResult],
            metrics_path: Path,
        ) -> RunResult:
            experiment.banks.clear()
            experiment.captures.clear()
            experiment.insertions.clear()
            experiment.attention.blocks.clear()
            experiment.attention.receiver_calls = 0
            experiment.attention.receiver_mass.clear()
            experiment.attention.receiver_relation_calls = 0
            experiment.attention.receiver_relation_bias_mass = 0.0
            result = native_case(
                case=case,
                max_attempts=max_attempts,
                operation=operation,
                metrics_path=metrics_path,
            )
            experiment.save_audit(case.dataset_index)
            experiment.attention.blocks.clear()
            experiment.banks.clear()
            experiment.captures.clear()
            return result

        LatentQwenEngine.append = append
        LatentQwenEngine.decode = decode
        hybrid.latent_reallocation = step
        cli.run_case_with_retries = run_case
        try:
            with installed_attention(self.attention):
                yield None
        finally:
            LatentQwenEngine.append = native_append
            LatentQwenEngine.decode = native_decode
            hybrid.latent_reallocation = native_step
            cli.run_case_with_retries = native_case

    def save_audit(self, index: int) -> None:
        """Persist source selection, measured insertion and actual receiver calls."""
        senders = []
        for stage, bank in self.banks.items():
            capture = self.captures[stage]
            senders.append(
                {
                    "stage": stage,
                    "source": "original_latent_cache_kv",
                    "latent_columns": bank.source.latent_columns.tolist(),
                    "query_heads": capture.query_heads,
                    "kv_heads": next(iter(bank.selected.values())).shape[1],
                    "selected_heads": {
                        str(layer): selected.cpu().tolist()
                        for layer, selected in bank.selected.items()
                    },
                    "selection_scores": {
                        str(layer): score.cpu().tolist()
                        for layer, score in (
                            capture.selection_scores.items()
                            if capture.selection_scores
                            else {
                                layer: torch.stack(scores, dim=-1)
                                for layer, scores in capture.scores.items()
                            }.items()
                        )
                    },
                }
            )
        payload = {
            "dataset_index": index,
            "mode": self.mode,
            "relay_log_gain": self.attention.relay_log_gain,
            "senders": senders,
            "insertions": {
                stage: asdict(audit) for stage, audit in self.insertions.items()
            },
            "receiver_layer_calls": self.attention.receiver_calls,
            "receiver_relation_calls": self.attention.receiver_relation_calls,
            "receiver_relation_bias_mass": self.attention.receiver_relation_bias_mass,
            "receiver_attention_mass": {
                name: sum(values) / len(values)
                for name, values in self.attention.receiver_mass.items()
                if values
            },
        }
        path = self.output / "latent_visual_relay" / f"{index:03d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(
            f"[latent-visual-relay] id={index} inserted_rows={[(stage, audit.after - audit.before) for stage, audit in self.insertions.items()]}",
            flush=True,
        )
