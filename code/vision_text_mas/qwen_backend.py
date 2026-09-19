"""Physical Transformers generation backend for Qwen3-VL."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, TypedDict, cast

from PIL import Image

if TYPE_CHECKING:
    import torch


class ImageItem(TypedDict):
    """One image item in a Transformers chat message."""

    type: Literal["image"]
    image: Image.Image


class TextItem(TypedDict):
    """One text item in a Transformers chat message."""

    type: Literal["text"]
    text: str


class ChatMessage(TypedDict):
    """One system or multimodal user chat message."""

    role: Literal["system", "user"]
    content: str | list[ImageItem | TextItem]


class ProcessorBatch(Protocol):
    """Tensor mapping returned by the multimodal processor."""

    def to(self, device: torch.device) -> ProcessorBatch: ...

    def __getitem__(self, key: str) -> torch.Tensor: ...


class ProcessorProtocol(Protocol):
    """Subset of the Qwen processor used by this backend."""

    def apply_chat_template(
        self,
        conversation: list[ChatMessage],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str: ...

    def __call__(
        self,
        *,
        text: list[str],
        images: list[Image.Image],
        return_tensors: str,
    ) -> ProcessorBatch: ...

    def batch_decode(
        self,
        sequences: torch.Tensor,
        *,
        skip_special_tokens: bool,
    ) -> list[str]: ...


class ModelProtocol(Protocol):
    """Subset of the Qwen model used by this backend."""

    @property
    def device(self) -> torch.device: ...

    def eval(self) -> ModelProtocol: ...

    def generate(
        self,
        **inputs: torch.Tensor | int | float | bool,
    ) -> torch.Tensor: ...


class GenerationBackend(Protocol):
    """One physical text generation over zero or more images."""

    def generate(
        self,
        *,
        images: tuple[Image.Image, ...],
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int,
        json_prefix: str | None,
        json_schema: str | None = None,
    ) -> str: ...


def _model_vocab_size(model: object) -> int | None:
    """Return the LM-head vocabulary size, including multimodal config layouts."""
    output_embeddings = getattr(model, "get_output_embeddings", None)
    if callable(output_embeddings):
        weight = getattr(output_embeddings(), "weight", None)
        shape = getattr(weight, "shape", ())
        if len(shape) == 2 and isinstance(shape[0], int):
            return int(shape[0])
    config_vocab_size = getattr(getattr(model, "config", None), "vocab_size", None)
    return int(config_vocab_size) if isinstance(config_vocab_size, int) else None


def _json_logits_processor(
    *,
    processor: ProcessorProtocol,
    json_schema: str,
    model_vocab_size: int | None,
) -> object:
    """Build a fresh grammar mask for one JSON generation call."""
    from vision_text_mas.json_constraint import build_json_logits_processor

    return build_json_logits_processor(
        tokenizer=processor.tokenizer,
        json_schema=json_schema,
        model_vocab_size=model_vocab_size,
    )


class TransformersQwenBackend:
    """One loaded Qwen checkpoint shared by every runtime role."""

    def __init__(
        self,
        processor: ProcessorProtocol,
        model: ModelProtocol,
        *,
        temperature: float = 0.6,
        top_p: float = 0.95,
        seed: int = 42,
        max_model_len: int = 8_192,
        constrained_json: bool = False,
    ) -> None:
        self._processor = processor
        self._model = model
        self._temperature = temperature
        self._top_p = top_p
        self._seed = seed
        self._max_model_len = max_model_len
        self._constrained_json = constrained_json

    @classmethod
    def from_pretrained(
        cls,
        model_path: Path,
        device: str,
        *,
        temperature: float = 0.6,
        top_p: float = 0.95,
        seed: int = 42,
        max_model_len: int = 8_192,
        constrained_json: bool = False,
    ) -> TransformersQwenBackend:
        """Load one bfloat16 checkpoint on the requested device."""
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        processor = cast(
            ProcessorProtocol,
            AutoProcessor.from_pretrained(model_path),
        )
        model = cast(
            ModelProtocol,
            cast(
                object,
                AutoModelForImageTextToText.from_pretrained(
                    model_path,
                    dtype=torch.bfloat16,
                    device_map=device,
                ).eval(),
            ),
        )
        return cls(
            processor=processor,
            model=model,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            max_model_len=max_model_len,
            constrained_json=constrained_json,
        )

    def generate(
        self,
        *,
        images: tuple[Image.Image, ...],
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int,
        json_prefix: str | None,
        json_schema: str | None = None,
    ) -> str:
        """Render one chat and decode only newly generated tokens."""
        import torch

        use_json_constraint = self._constrained_json and json_schema is not None

        content: list[ImageItem | TextItem] = [
            ImageItem(type="image", image=image) for image in images
        ]
        content.append(TextItem(type="text", text=user_prompt))
        messages: list[ChatMessage] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        chat = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if json_prefix is not None and not use_json_constraint:
            if chat.endswith("<think>\n"):
                chat += f"</think>\n\n{json_prefix}"
            else:
                chat += json_prefix
        processor_kwargs: dict[str, object] = {
            "text": [chat],
            "return_tensors": "pt",
        }
        if images:
            processor_kwargs["images"] = list(images)
        batch = self._processor(**processor_kwargs).to(self._model.device)
        inputs = cast(Mapping[str, "torch.Tensor"], cast(object, batch))
        request_tokens = inputs["input_ids"].shape[1] + max_new_tokens
        if request_tokens > self._max_model_len:
            raise RuntimeError(
                "request exceeds model context limit: "
                f"{request_tokens} > {self._max_model_len}"
            )
        generation_inputs: dict[str, object] = {
            **dict(inputs),
            "max_new_tokens": max_new_tokens,
            "do_sample": not use_json_constraint,
            "use_cache": True,
        }
        if not use_json_constraint:
            generation_inputs["temperature"] = self._temperature
            generation_inputs["top_p"] = self._top_p
        if use_json_constraint:
            generation_inputs["logits_processor"] = [
                _json_logits_processor(
                    processor=self._processor,
                    json_schema=json_schema,
                    model_vocab_size=_model_vocab_size(self._model),
                )
            ]
        with torch.inference_mode():
            torch.manual_seed(self._seed)
            generated = self._model.generate(**generation_inputs)
        new_ids = generated[:, inputs["input_ids"].shape[1] :]
        return self._processor.batch_decode(
            new_ids,
            skip_special_tokens=True,
        )[0].strip()
