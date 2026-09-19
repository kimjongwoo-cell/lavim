"""Shared Navigator→Reasoner and Reasoner→Answerer relay at existing boundaries."""

import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import torch
from PIL import Image
from transformers.cache_utils import Cache

from pvcr.attention import AttentionState, attention_adapter
from pvcr.capture import Capture, SenderAudit
from pvcr.errors import LayoutError
from pvcr.layout import ImageMetadata, Variant
from pvcr.receiver import CacheAudit, RelayBank, substitute_values
from vision_text_mas import latent_hf_ablation_cli as cli
from vision_text_mas import latent_hybrid_variants as hybrid
from vision_text_mas.contracts import CaseInput, RunResult
from vision_text_mas.latent_qwen_engine import LatentAppendResult, LatentQwenEngine


class PVCR:
    """Own only per-case relay state; patches are scoped to this standalone process."""

    def __init__(
        self,
        output_root: Path,
        variant: Variant,
        boundary: Literal["both", "navigator", "reasoner"] = "both",
    ) -> None:
        self.output_root = output_root
        self.variant: Variant = variant
        self.boundary: Literal["both", "navigator", "reasoner"] = boundary
        self.attention = AttentionState()
        self.banks: dict[str, RelayBank] = {}
        self.audits: list[SenderAudit] = []
        self.receiver_calls: dict[str, int] = {}
        self.cache_audits: dict[str, CacheAudit] = {}

    @contextmanager
    def receiving(self, cache: Cache | None, sender: str) -> Iterator[None]:
        enabled = self.variant != "off" and self.boundary in ("both", sender)
        if not enabled:
            yield None
            return
        if cache is None or sender not in self.banks:
            raise LayoutError(f"Missing {sender} PVCR bank/cache at receiver boundary")
        bank = self.banks[sender]
        before = self.attention.receiver_calls
        self.attention.receiver = bank
        try:
            with substitute_values(cache, bank) as audit:
                self.cache_audits[sender] = audit
                yield None
        finally:
            self.attention.receiver = None
            self.receiver_calls[sender] = self.attention.receiver_calls - before
        if self.receiver_calls[sender] == 0:
            raise LayoutError("PVCR receiver attention adapter did not fire")

    @contextmanager
    def installed(self) -> Iterator[None]:
        original_append = LatentQwenEngine.append
        original_decode = LatentQwenEngine.decode
        original_step = hybrid.latent_reallocation
        original_case = cli.run_case_with_retries
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
            def native_append() -> LatentAppendResult[Cache]:
                return original_append(
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

            if stage not in ("navigator", "reasoner"):
                return native_append()
            if latent_steps != 10 or self._transport_mode != "cumulative":
                raise LayoutError(
                    "PVCR requires cumulative transport and 10 sender steps"
                )
            metadata = tuple(
                ImageMetadata.model_validate(
                    {
                        "patch_id": image.info.get("latent_patch_id", ""),
                        "parent_id": image.info.get("latent_parent_patch_id", ""),
                        "magnification": image.info.get("latent_magnification", 0),
                        "box": image.info.get("latent_box"),
                    }
                )
                for image in images
            )
            capture = Capture(stage, metadata, experiment.variant)
            experiment.attention.capture = capture

            def grid_hook(
                module: torch.nn.Module,
                args: tuple[torch.Tensor, ...],
                kwargs: dict[str, torch.Tensor],
            ) -> None:
                del module
                grid = kwargs.get("grid_thw")
                if grid is None and len(args) > 1:
                    grid = args[1]
                if grid is None:
                    raise LayoutError("PVCR could not observe vision grid_thw")
                capture.grids = grid.detach().cpu().clone()

            visual = getattr(self._backbone, "visual", None)
            if not isinstance(visual, torch.nn.Module):
                raise LayoutError("PVCR requires a Qwen vision module")
            hook = visual.register_forward_pre_hook(grid_hook, with_kwargs=True)
            try:
                if stage == "reasoner":
                    with experiment.receiving(cache, "navigator"):
                        result = native_append()
                else:
                    result = native_append()
                bank, audit = capture.finish()
                if audit.columns[-1] >= result.cache_length:
                    raise LayoutError(
                        "Captured latent columns are outside the closed sender cache"
                    )
                if stage == "reasoner":
                    native_columns = getattr(self, "_rpath_latent_cols", None)
                    if (
                        not isinstance(native_columns, torch.Tensor)
                        or native_columns.tolist() != audit.columns
                    ):
                        raise LayoutError(
                            "Observed/native Reasoner latent columns differ"
                        )
                experiment.banks[stage] = bank
                experiment.audits.append(audit)
                return result
            finally:
                hook.remove()
                experiment.attention.capture = None

        @contextmanager
        def step(
            backbone: hybrid._VariantBackbone,
            vision_columns: torch.Tensor,
            visual_patch_ids: torch.Tensor | None = None,
            parent_patch_indices: tuple[int, ...] = (),
        ) -> Iterator[None]:
            capture = experiment.attention.capture
            with original_step(
                backbone, vision_columns, visual_patch_ids, parent_patch_indices
            ):
                if capture is None:
                    yield None
                else:
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
            def native_decode() -> str:
                return original_decode(
                    self,
                    cache=cache,
                    position_cursor=position_cursor,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_new_tokens=max_new_tokens,
                    json_prefix=json_prefix,
                    json_schema=json_schema,
                )

            if json_prefix is not None and json_prefix.strip().startswith('{"answer"'):
                with experiment.receiving(cache, "reasoner"):
                    return native_decode()
            return native_decode()

        def run_case(
            *,
            case: CaseInput,
            max_attempts: int,
            operation: Callable[[], RunResult],
            metrics_path: Path,
        ) -> RunResult:
            experiment.banks.clear()
            experiment.audits.clear()
            experiment.receiver_calls.clear()
            experiment.cache_audits.clear()
            experiment.attention.receiver_calls = 0
            result = original_case(
                case=case,
                max_attempts=max_attempts,
                operation=operation,
                metrics_path=metrics_path,
            )
            path = experiment.output_root / "pvcr" / f"{case.dataset_index:03d}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "dataset_index": case.dataset_index,
                        "variant": experiment.variant,
                        "boundary": experiment.boundary,
                        "senders": [audit.model_dump() for audit in experiment.audits],
                        "receiver_layer_calls": experiment.receiver_calls,
                        "receiver_cache_lengths": {
                            stage: asdict(audit)
                            for stage, audit in experiment.cache_audits.items()
                        },
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(
                f"[PVCR] completed_id={case.dataset_index} boundaries={experiment.receiver_calls}",
                flush=True,
            )
            experiment.banks.clear()
            return result

        LatentQwenEngine.append = append
        LatentQwenEngine.decode = decode
        hybrid.latent_reallocation = step
        cli.run_case_with_retries = run_case
        try:
            with attention_adapter(self.attention):
                yield None
        finally:
            LatentQwenEngine.append = original_append
            LatentQwenEngine.decode = original_decode
            hybrid.latent_reallocation = original_step
            cli.run_case_with_retries = original_case
