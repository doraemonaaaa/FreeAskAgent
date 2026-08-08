from __future__ import annotations

import io
from dataclasses import replace

import pytest
from PIL import Image

from agentflow.agents.models_embodied_v2.TemporalCaptioner import (
    Subgoal,
    TemporalAnalysisRequest,
    TemporalCaptioner,
    TemporalFrameInput,
    TemporalInputError,
    TemporalOutputError,
)


np = pytest.importorskip("numpy")


class FakeEngine:
    supports_image_pixel_budget = True

    def __init__(self, response):
        self.response = response
        self.calls = []

    def __call__(self, content, **kwargs):
        self.calls.append((content, kwargs))
        return self.response


def _request() -> TemporalAnalysisRequest:
    return TemporalAnalysisRequest(
        subgoal=Subgoal(
            "1",
            "Cross the doorway",
            "The final view is inside the pool room.",
        ),
        frames=tuple(
            TemporalFrameInput(
                frame_id=index,
                image=np.full((12, 16, 3), index, dtype=np.uint8),
            )
            for index in range(1, 9)
        ),
    )


def test_sends_subgoal_and_eight_images_without_actions():
    response = '{"completed":true}'
    engine = FakeEngine(response)
    result = TemporalCaptioner(engine=engine).analyze(_request())

    content, kwargs = engine.calls[0]
    assert content[0] == (
        "Subgoal: Cross the doorway\n"
        "Completion proof: The final view is inside the pool room.\n"
        "Images: 8, ordered oldest first."
    )
    assert sum(isinstance(item, bytes) for item in content) == 8
    assert all("action" not in item.lower() for item in content if isinstance(item, str))
    assert kwargs["max_tokens"] == 48
    assert kwargs["image_max_pixels"] == 224**2
    assert result.completed is True
    assert result.error is False
    assert result.error_mode == "NONE"
    assert result.raw_response == response


def test_error_detection_is_disabled():
    result = TemporalCaptioner(
        engine=FakeEngine('{"completed":false}')
    ).analyze(_request())
    assert result.completed is False
    assert result.error is False
    assert result.error_mode == "NONE"


def test_float_image_is_normalized():
    request = _request()
    first = replace(request.frames[0], image=np.ones((8, 10, 3)))
    request = replace(request, frames=(first, *request.frames[1:]))
    engine = FakeEngine(
        '{"completed":false}'
    )

    TemporalCaptioner(engine=engine).analyze(request)

    content, _ = engine.calls[0]
    decoded = np.asarray(Image.open(io.BytesIO(content[1])).convert("RGB"))
    assert int(decoded.min()) == 255


def test_request_allows_one_to_eight_ordered_frames():
    request = _request()
    one_frame = replace(request, frames=request.frames[:1])
    assert len(one_frame.frames) == 1
    with pytest.raises(TemporalInputError, match="at most eight"):
        replace(
            request,
            frames=(
                *request.frames,
                TemporalFrameInput(
                    frame_id=9,
                    image=np.zeros((8, 10, 3), dtype=np.uint8),
                ),
            ),
        )
    with pytest.raises(TemporalInputError, match="unique and increasing"):
        replace(
            request,
            frames=(request.frames[1], request.frames[0], *request.frames[2:]),
        )


@pytest.mark.parametrize(
    "response",
    (
        "true",
        '{"complete":false}',
        '{"completed":false,"error":false,"error_mode":"WALL_STUCK"}',
    ),
)
def test_invalid_or_inconsistent_output_is_rejected(response):
    captioner = TemporalCaptioner(engine=FakeEngine(response))
    with pytest.raises(TemporalOutputError):
        captioner.analyze(_request())
    assert captioner.last_raw_response == response


def test_engine_is_reused_between_windows():
    engine = FakeEngine(
        '{"completed":false}'
    )
    captioner = TemporalCaptioner(engine=engine)
    captioner.analyze(_request())
    captioner.analyze(_request())
    assert len(engine.calls) == 2


@pytest.mark.parametrize(
    "response",
    (
        '```json\n{"completed":false}\n```',
        'Result: {"completed":false} done.',
    ),
)
def test_complete_json_object_is_recovered_from_wrapping_text(response):
    result = TemporalCaptioner(engine=FakeEngine(response)).analyze(_request())
    assert result.completed is False
