from __future__ import annotations

import io
from dataclasses import replace

import pytest
from PIL import Image

from agentflow.agents.models_embodied_v2.TemporalCaptioner import (
    Subgoal,
    TemporalAnalysisRequest,
    TemporalCaptioner,
    TemporalInputError,
    TemporalOutputError,
    TemporalStepInput,
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
        steps=tuple(
            TemporalStepInput(
                step_id=index,
                action=("FORWARD" if index < 5 else "TURN_LEFT"),
                image=np.full((12, 16, 3), index, dtype=np.uint8),
            )
            for index in range(1, 9)
        ),
    )


def test_fast_mode_sends_eight_pairs_and_returns_one_boolean():
    engine = FakeEngine("true")
    result = TemporalCaptioner(engine=engine).analyze(_request())

    content, kwargs = engine.calls[0]
    assert content[0] == (
        "Subgoal: Cross the doorway\n"
        "Completion proof: The final view is inside the pool room."
    )
    markers = [
        index
        for index, item in enumerate(content)
        if isinstance(item, str) and item.startswith("[STEP ")
    ]
    assert len(markers) == 8
    assert all(isinstance(content[index + 1], bytes) for index in markers)
    assert kwargs["max_tokens"] == 1
    assert kwargs["image_max_pixels"] == 224**2
    assert result.completed is True
    assert result.raw_response == "true"


def test_false_is_the_only_other_valid_model_output():
    result = TemporalCaptioner(
        engine=FakeEngine("false")
    ).analyze(_request())
    assert result.completed is False


def test_float_images_and_action_aliases_are_normalized():
    request = _request()
    first = replace(
        request.steps[0],
        action="MOVE_FORWARD",
        image=np.ones((8, 10, 3)),
    )
    request = replace(request, steps=(first, *request.steps[1:]))
    engine = FakeEngine("false")

    TemporalCaptioner(engine=engine).analyze(request)

    content, _ = engine.calls[0]
    marker = next(
        index
        for index, item in enumerate(content)
        if isinstance(item, str) and item.startswith("[STEP 1 ")
    )
    assert "action=FORWARD" in content[marker]
    decoded = np.asarray(
        Image.open(io.BytesIO(content[marker + 1])).convert("RGB")
    )
    assert int(decoded.min()) == 255


def test_request_requires_exactly_eight_ordered_action_steps():
    request = _request()
    with pytest.raises(TemporalInputError, match="exactly eight"):
        replace(request, steps=request.steps[:7])
    with pytest.raises(TemporalInputError, match="unique and increasing"):
        replace(
            request,
            steps=(request.steps[1], request.steps[0], *request.steps[2:]),
        )
    with pytest.raises(TemporalInputError, match="Unsupported action"):
        replace(request.steps[0], action="GO_BACK")


def test_non_boolean_model_output_is_rejected():
    captioner = TemporalCaptioner(
        engine=FakeEngine('{"completed": false}')
    )
    with pytest.raises(TemporalOutputError, match="true or false"):
        captioner.analyze(_request())
    assert captioner.last_raw_response == '{"completed": false}'


def test_engine_is_reused_between_windows():
    engine = FakeEngine("false")
    captioner = TemporalCaptioner(engine=engine)
    captioner.analyze(_request())
    captioner.analyze(_request())
    assert len(engine.calls) == 2
