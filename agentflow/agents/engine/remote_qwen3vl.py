"""OpenAI-compatible client for a Qwen3-VL served by vLLM.

A drop-in for :class:`LocalQwen3VL` on the agent side: the same ``content``
list of text and PNG bytes, the same ``system_prompt`` / ``max_tokens`` /
``temperature`` keywords, and the same opt-in per-call image pixel budget.
The budget is applied client-side with Qwen's own ``smart_resize`` rule so
the server sees the frame at the size the local processor would have used,
and the number of vision tokens per frame stays identical across backends.
"""

from __future__ import annotations

import io
import math
import os
from typing import Any, List, Optional, Union

from PIL import Image

from .base import EngineLM

# Qwen3-VL: 16 px patches merged 2x2, so every side is a multiple of 32.
QWEN3_VL_SIZE_FACTOR = 32
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_JPEG_MAGIC = b"\xff\xd8\xff"


def smart_resize(
    height: int,
    width: int,
    *,
    factor: int = QWEN3_VL_SIZE_FACTOR,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    """Qwen-VL's resize rule: sides multiple of ``factor`` within the budget."""
    if min_pixels <= 0 or max_pixels < min_pixels:
        raise ValueError("image pixel budget must satisfy 0 < min <= max")
    if max(height, width) / max(1, min(height, width)) > 200:
        raise ValueError("absolute aspect ratio must be smaller than 200")
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def fit_pixel_budget(
    encoded: bytes,
    *,
    min_pixels: Optional[int],
    max_pixels: Optional[int],
) -> bytes:
    """Re-encode ``encoded`` at the size Qwen's processor would pick."""
    if min_pixels is None and max_pixels is None:
        return encoded
    if min_pixels is None or max_pixels is None:
        raise ValueError(
            "image_min_pixels and image_max_pixels must be provided together"
        )
    image = Image.open(io.BytesIO(encoded))
    width, height = image.size
    new_height, new_width = smart_resize(
        height, width, min_pixels=int(min_pixels), max_pixels=int(max_pixels)
    )
    if (new_width, new_height) == (width, height):
        return encoded
    resized = image.convert("RGB").resize(
        (new_width, new_height), Image.Resampling.BICUBIC
    )
    buffer = io.BytesIO()
    resized.save(buffer, format="PNG")
    return buffer.getvalue()


def _data_url(encoded: bytes) -> str:
    import base64

    if encoded.startswith(_PNG_MAGIC):
        mime = "image/png"
    elif encoded.startswith(_JPEG_MAGIC):
        mime = "image/jpeg"
    else:
        raise ValueError("image bytes must be PNG or JPEG")
    return f"data:{mime};base64,{base64.b64encode(encoded).decode('ascii')}"


class RemoteQwen3VL(EngineLM):
    """Chat-completions client with LocalQwen3VL's calling convention."""

    DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
    supports_image_pixel_budget = True
    is_multimodal = True

    def __init__(
        self,
        model_string: str,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_s: float = 300.0,
        max_retries: int = 2,
        client: Any = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self.model_string = model_string
        self.system_prompt = system_prompt
        self.base_url = base_url or os.environ.get(
            "VLLM_BASE_URL", "http://127.0.0.1:8000/v1"
        )
        if client is None:
            from openai import OpenAI

            client = OpenAI(
                base_url=self.base_url,
                api_key=api_key or os.environ.get("VLLM_API_KEY", "EMPTY"),
                timeout=timeout_s,
                max_retries=max_retries,
            )
        self.client = client
        # Usage of the most recent call, including the prefix-cache hit count
        # vLLM reports as ``prompt_tokens_details.cached_tokens``.
        self.last_usage: dict[str, Any] = {}
        self.last_latency_ms: float = 0.0

    def generate(
        self,
        content: Union[str, List[Union[str, bytes]]],
        system_prompt: Optional[str] = None,
        *,
        temperature: float = 0,
        max_tokens: int = 2048,
        top_p: float = 1.0,
        image_min_pixels: Optional[int] = None,
        image_max_pixels: Optional[int] = None,
        response_format: Any = None,
        **_ignored: Any,
    ) -> str:
        import time

        messages = self.build_messages(
            content,
            system_prompt or self.system_prompt,
            image_min_pixels=image_min_pixels,
            image_max_pixels=image_max_pixels,
        )
        started = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model_string,
            messages=messages,
            temperature=float(temperature or 0.0),
            max_tokens=int(max_tokens),
            top_p=float(top_p),
        )
        self.last_latency_ms = (time.perf_counter() - started) * 1000
        usage = getattr(response, "usage", None)
        self.last_usage = {}
        if usage is not None:
            details = getattr(usage, "prompt_tokens_details", None)
            self.last_usage = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "cached_tokens": getattr(details, "cached_tokens", None)
                if details is not None
                else None,
            }
        text = response.choices[0].message.content
        return "" if text is None else str(text)

    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)

    @staticmethod
    def build_messages(
        content: Union[str, List[Union[str, bytes]]],
        system_prompt: str,
        *,
        image_min_pixels: Optional[int] = None,
        image_max_pixels: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Mirror LocalQwen3VL: one system text, one user turn, in order."""
        if isinstance(content, str):
            content = [content]
        user_content: list[dict[str, Any]] = []
        for item in content:
            if isinstance(item, str):
                user_content.append({"type": "text", "text": item})
            elif isinstance(item, bytes):
                encoded = fit_pixel_budget(
                    item,
                    min_pixels=image_min_pixels,
                    max_pixels=image_max_pixels,
                )
                user_content.append(
                    {"type": "image_url", "image_url": {"url": _data_url(encoded)}}
                )
            else:
                raise ValueError(f"Unsupported input type: {type(item)}")
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]


__all__ = ("RemoteQwen3VL", "fit_pixel_budget", "smart_resize")
