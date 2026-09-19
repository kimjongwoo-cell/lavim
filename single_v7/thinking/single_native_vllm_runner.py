"""In-process vLLM runner for the fair Single v7 adapter."""

from __future__ import annotations

import os
import base64
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Protocol

from PIL import Image
from vllm import LLM, SamplingParams


class AdapterModule(Protocol):
    """Stock adapter helpers used by the runner replacement."""

    SYSTEM_PROMPT: str

    def clean_generation(self, text: str) -> str: ...


@dataclass(frozen=True, slots=True)
class NativeSingleVllmError(RuntimeError):
    """Direct Single generation failed before returning a completion."""

    detail: str

    def __str__(self) -> str:
        return self.detail


def _engine_kwargs(model: str) -> dict[str, str | int | float | bool]:
    """Build the in-process engine configuration from explicit environment knobs."""
    return {
        "model": model,
        "tensor_parallel_size": int(os.environ.get("SINGLE_VLLM_TP", "1")),
        "gpu_memory_utilization": float(os.environ.get("SINGLE_VLLM_GPU_UTIL", "0.9")),
        "max_model_len": int(os.environ.get("MAX_MODEL_LEN", "8192")),
        "enable_prefix_caching": True,
    }


def _data_url(image: Image.Image) -> str:
    """Render an offline vLLM image message without an HTTP request."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def build_native_inprocess_vllm_runner(adapter: AdapterModule):
    """Build the stock-compatible runner over one locally created vLLM LLM."""

    class NativeInprocessVllmRunner:
        """Keep the existing Single prompt/output contract without HTTP."""

        def __init__(self, args) -> None:
            self.args = args
            if not args.llm_model:
                raise NativeSingleVllmError("--llm-model must name the local checkpoint")
            self._engine = LLM(**_engine_kwargs(str(Path(args.llm_model))))

        def _close_thinking(self, text: str) -> str:
            return text + "\n</think>\n\n"

        def generate(self, image_path, prompt: str) -> tuple[str, int]:
            image = Image.open(image_path).convert("RGB")
            content = [
                {"type": "image_url", "image_url": {"url": _data_url(image)}},
                {"type": "text", "text": prompt},
            ]
            messages = [
                {"role": "system", "content": adapter.SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ]
            closed = self._close_thinking("")
            continuation = bool(closed)
            if continuation:
                messages.append({"role": "assistant", "content": closed})
            sampling = SamplingParams(
                temperature=self.args.temperature if self.args.do_sample else 0.0,
                top_p=self.args.top_p if self.args.do_sample and self.args.top_p else 1.0,
                seed=self.args.seed,
                max_tokens=self.args.max_new_tokens,
            )
            outputs = self._engine.chat(
                messages,
                sampling,
                use_tqdm=False,
                add_generation_prompt=not continuation,
                continue_final_message=continuation,
            )
            if not outputs or not outputs[0].outputs:
                raise NativeSingleVllmError("direct vLLM returned no completion")
            text = outputs[0].outputs[0].text
            return adapter.clean_generation(text), len(outputs[0].outputs[0].token_ids)

    return NativeInprocessVllmRunner
