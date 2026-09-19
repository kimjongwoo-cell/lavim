"""Native-thinking vLLM runner for the fair Single v7 experiment."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol, Self


class AdapterModule(Protocol):
    """Stock adapter helpers consumed by the replacement runner."""

    SYSTEM_PROMPT: str

    def image_to_data_url(self, image_path) -> str: ...

    def clean_generation(self, text: str) -> str: ...


class HttpResponse(Protocol):
    """Context-managed HTTP response subset used by the transport."""

    def __enter__(self) -> Self: ...

    def __exit__(self, *args) -> None: ...

    def read(self) -> bytes: ...


class UrlOpen(Protocol):
    """Injectable urllib boundary for payload-level tests."""

    def __call__(
        self,
        request: urllib.request.Request,
        timeout: float,
    ) -> HttpResponse: ...


@dataclass(frozen=True, slots=True)
class VllmSingleResponseError(RuntimeError):
    """The shared server returned an unusable completion response."""

    detail: str

    def __str__(self) -> str:
        return self.detail


def build_native_thinking_vllm_runner(
    adapter: AdapterModule,
    *,
    urlopen: UrlOpen = urllib.request.urlopen,
):
    """Build the stock-compatible runner while retaining adapter utilities."""

    class NativeThinkingVllmRunner:
        """Send native-thinking or explicitly closed-thinking chat requests."""

        def __init__(self, args) -> None:
            self.args = args
            self.base_url = args.llm_url.rstrip("/")
            self.model = args.llm_model
            if not self.model:
                raise VllmSingleResponseError(
                    "--llm-model is required when --llm-url is used"
                )

        def _close_thinking(self, text: str) -> str:
            return text + "\n</think>\n\n"

        def generate(self, image_path, prompt: str) -> tuple[str, int]:
            messages = [
                {"role": "system", "content": adapter.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": adapter.image_to_data_url(image_path)
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
            closed_prefix = self._close_thinking("")
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": self.args.max_new_tokens,
                "temperature": (
                    self.args.temperature if self.args.do_sample else 0.0
                ),
                "seed": self.args.seed,
            }
            if self.args.do_sample and self.args.top_p is not None:
                payload["top_p"] = self.args.top_p
            if closed_prefix:
                messages.append({"role": "assistant", "content": closed_prefix})
                payload["add_generation_prompt"] = False
                payload["continue_final_message"] = True

            request = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            if self.args.api_key:
                request.add_header("Authorization", f"Bearer {self.args.api_key}")
            try:
                with urlopen(request, timeout=self.args.request_timeout) as response:
                    result = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as error:
                body = error.read().decode("utf-8", errors="replace")
                raise VllmSingleResponseError(
                    f"vLLM request failed with HTTP {error.code}: {body}"
                ) from error
            except urllib.error.URLError as error:
                raise VllmSingleResponseError(
                    f"could not connect to vLLM at {self.base_url}: {error}"
                ) from error
            try:
                message = result["choices"][0]["message"]
                content = message.get("content") or ""
                tokens = int((result.get("usage") or {}).get("completion_tokens") or 0)
            except (AttributeError, KeyError, IndexError, TypeError) as error:
                raise VllmSingleResponseError(
                    f"unexpected vLLM response: {result}"
                ) from error
            return adapter.clean_generation(content), tokens

    return NativeThinkingVllmRunner
