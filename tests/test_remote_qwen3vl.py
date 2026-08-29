from __future__ import annotations

import base64
import io
from types import SimpleNamespace

import numpy as np
from PIL import Image

from agentflow.agents.engine.remote_qwen3vl import (
    RemoteQwen3VL,
    fit_pixel_budget,
    smart_resize,
)


def _png(width, height):
    buffer = io.BytesIO()
    Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8)).save(
        buffer, format="PNG"
    )
    return buffer.getvalue()


class _StubClient:
    def __init__(self, text="{}"):
        self.calls = []
        self.text = text
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.text))],
            usage=SimpleNamespace(
                prompt_tokens=1500,
                completion_tokens=40,
                prompt_tokens_details=SimpleNamespace(cached_tokens=1200),
            ),
        )


def test_smart_resize_matches_qwen_rule_for_the_actor_frame():
    # 640x480 under the waypoint budget of 448**2 becomes 512x384 (both
    # multiples of 32, 196608 <= 200704), exactly what the local processor
    # produces.
    assert smart_resize(480, 640, min_pixels=64**2, max_pixels=448**2) == (384, 512)
    # Already within budget and aligned: unchanged.
    assert smart_resize(224, 320, min_pixels=64**2, max_pixels=320**2) == (224, 320)


def test_fit_pixel_budget_reencodes_only_when_needed():
    original = _png(640, 480)
    assert fit_pixel_budget(original, min_pixels=None, max_pixels=None) is original
    shrunk = fit_pixel_budget(original, min_pixels=64**2, max_pixels=448**2)
    assert Image.open(io.BytesIO(shrunk)).size == (512, 384)
    aligned = _png(320, 224)
    assert fit_pixel_budget(aligned, min_pixels=64**2, max_pixels=320**2) is aligned


def test_generate_keeps_text_and_images_in_order_and_reads_cached_tokens():
    client = _StubClient('{"ok":true}')
    engine = RemoteQwen3VL("qwen3-vl-8b", client=client)

    text = engine(
        ["frame=1", _png(64, 64), "frame=2", _png(64, 64)],
        system_prompt="SYSTEM",
        max_tokens=96,
        temperature=0,
        image_min_pixels=64**2,
        image_max_pixels=320**2,
    )

    assert text == '{"ok":true}'
    call = client.calls[0]
    assert call["model"] == "qwen3-vl-8b"
    assert call["max_tokens"] == 96 and call["temperature"] == 0.0
    system, user = call["messages"]
    assert system == {"role": "system", "content": "SYSTEM"}
    kinds = [part["type"] for part in user["content"]]
    assert kinds == ["text", "image_url", "text", "image_url"]
    url = user["content"][1]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1])[:4] == b"\x89PNG"
    assert engine.last_usage["cached_tokens"] == 1200


def test_actor_routes_vllm_prefix_to_the_remote_engine(monkeypatch):
    from agentflow.agents.models_embodied_v2.actor import Actor

    monkeypatch.setenv("VLLM_BASE_URL", "http://127.0.0.1:9/v1")
    actor = Actor("vllm-qwen3-vl-8b")
    assert isinstance(actor.llm, RemoteQwen3VL)
    assert actor.llm.model_string == "qwen3-vl-8b"
    assert actor.llm.base_url == "http://127.0.0.1:9/v1"
