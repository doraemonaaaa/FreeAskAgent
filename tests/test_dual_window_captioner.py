from __future__ import annotations

import json

import pytest

from agentflow.agents.models_embodied_v2.memory.temporal_memory.temporal_captioner import (
    Subgoal,
    TemporalCaptioner,
    TemporalCaptionerConfig,
    TemporalFrameInput,
    TemporalInputError,
)


np = pytest.importorskip("numpy")


class SequenceEngine:
    supports_image_pixel_budget = True

    def __init__(self, *responses: str):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, content, **kwargs):
        self.calls.append((content, kwargs))
        return self.responses.pop(0)


def _frames(count: int) -> tuple[TemporalFrameInput, ...]:
    return tuple(
        TemporalFrameInput(
            frame_id=index,
            image=np.full((8, 10, 3), index, dtype=np.uint8),
        )
        for index in range(1, count + 1)
    )


def test_dual_window_uses_selected_evidence_and_skips_disabled_error():
    engine = SequenceEngine('{"completed":true}')
    captioner = TemporalCaptioner(engine=engine)
    completion_frames = _frames(12)

    result = captioner.analyze_dual_window(
        subgoal=Subgoal(
            "1",
            "Exit the room",
            "The final view is outside the starting room.",
        ),
        completion_frames=completion_frames,
        error_frames=completion_frames[-8:],
    )

    completion_content, completion_kwargs = engine.calls[0]
    assert sum(isinstance(item, bytes) for item in completion_content) == 12
    assert "selected ordered visual evidence" in (
        completion_kwargs["system_prompt"]
    )
    assert "landmark fields are auxiliary" in completion_content[0]
    assert len(engine.calls) == 1
    assert result.completed is True
    assert result.error is False
    assert result.error_mode == "NONE"
    assert result.completion_window_size == 12
    assert result.error_window_size == 8
    assert result.completion_raw_response == '{"completed":true}'
    assert result.error_raw_response == "disabled"
    assert result.error_latency_ms == 0.0


def test_dual_window_requires_error_frames_to_be_recent_suffix():
    frames = _frames(10)
    with pytest.raises(TemporalInputError, match="recent suffix"):
        TemporalCaptioner(engine=SequenceEngine()).analyze_dual_window(
            subgoal=Subgoal("1", "Exit", "The camera is outside."),
            completion_frames=frames,
            error_frames=frames[:8],
        )


def test_dual_window_runs_independent_strict_error_judgement_when_enabled():
    engine = SequenceEngine(
        '{"completed":false}',
        '{"error":true,"error_mode":"TURN_OSCILLATION",'
        '"confidence":0.93,"evidence":"yaw alternates; no translation"}',
    )
    captioner = TemporalCaptioner(
        engine=engine,
        config=TemporalCaptionerConfig(enable_error_detection=True),
    )
    completion_frames = _frames(12)

    result = captioner.analyze_dual_window(
        subgoal=Subgoal(
            "1",
            "Exit the room",
            "The final view is outside the starting room.",
        ),
        completion_frames=completion_frames,
        error_frames=completion_frames[-8:],
    )

    assert len(engine.calls) == 2
    error_content, error_kwargs = engine.calls[1]
    assert sum(isinstance(item, bytes) for item in error_content) == 8
    assert "recent ordered" in error_kwargs["system_prompt"]
    assert result.completed is False
    assert result.error is True
    assert result.error_mode == "TURN_OSCILLATION"
    assert result.error_confidence == pytest.approx(0.93)
    assert result.error_evidence == "yaw alternates; no translation"
    assert result.error_raw_response == (
        '{"error":true,"error_mode":"TURN_OSCILLATION",'
        '"confidence":0.93,"evidence":"yaw alternates; no translation"}'
    )
    assert "translation_m=0.000" in error_content[1]
    assert "yaw_delta_deg=+0.0" in error_content[1]


def test_dual_window_skips_error_model_until_four_frames():
    engine = SequenceEngine('{"completed":false}')
    captioner = TemporalCaptioner(
        engine=engine,
        config=TemporalCaptionerConfig(enable_error_detection=True),
    )
    frames = _frames(3)

    result = captioner.analyze_dual_window(
        subgoal=Subgoal("1", "Exit", "The camera is outside."),
        completion_frames=frames,
        error_frames=frames,
    )

    assert len(engine.calls) == 1
    assert result.error is False
    assert result.error_mode == "NONE"
    assert result.error_raw_response == "insufficient_frames"


@pytest.mark.parametrize(
    "response",
    [
        '{"error":false,"error_mode":"TURN_OSCILLATION",'
        '"confidence":0.9,"evidence":"bad consistency"}',
        '{"error":true,"error_mode":"NONE","confidence":0.9,'
        '"evidence":"bad consistency"}',
        '{"error":1,"error_mode":"GET_NOWHERE","confidence":0.9,'
        '"evidence":"wrong boolean type"}',
        '{"error":true,"error_mode":"UNKNOWN","confidence":0.9,'
        '"evidence":"unknown mode"}',
        '{"error":true,"error_mode":"GET_NOWHERE","confidence":1.1,'
        '"evidence":"bad confidence"}',
        '{"error":true,"error_mode":"GET_NOWHERE","confidence":0.9,'
        '"evidence":"  "}',
    ],
)
def test_dual_window_rejects_invalid_error_schema(response):
    captioner = TemporalCaptioner(
        engine=SequenceEngine('{"completed":false}', response),
        config=TemporalCaptionerConfig(enable_error_detection=True),
    )
    frames = _frames(8)

    with pytest.raises(ValueError, match="wrong schema"):
        captioner.analyze_dual_window(
            subgoal=Subgoal("1", "Exit", "The camera is outside."),
            completion_frames=frames,
            error_frames=frames,
        )


def test_dual_window_retries_truncated_error_json_once():
    engine = SequenceEngine(
        '{"completed":false}',
        '{"error":true,"error_mode":"WALL_STUCK"',
        '{"error":true,"error_mode":"WALL_STUCK","confidence":0.9,'
        '"evidence":"no translation across repeated views"}',
    )
    captioner = TemporalCaptioner(
        engine=engine,
        config=TemporalCaptionerConfig(enable_error_detection=True),
    )
    frames = _frames(8)

    result = captioner.analyze_dual_window(
        subgoal=Subgoal("1", "Exit", "The camera is outside."),
        completion_frames=frames,
        error_frames=frames,
    )

    assert len(engine.calls) == 3
    assert result.error is True
    assert result.error_mode == "WALL_STUCK"
    assert captioner.last_failed_raw_response is None


def test_dual_window_preserves_failed_raw_response_after_retry():
    engine = SequenceEngine(
        '{"completed":false}',
        '{"error":true',
        '{"error":still invalid',
    )
    captioner = TemporalCaptioner(
        engine=engine,
        config=TemporalCaptionerConfig(enable_error_detection=True),
    )
    frames = _frames(8)

    with pytest.raises(ValueError, match="invalid JSON"):
        captioner.analyze_dual_window(
            subgoal=Subgoal("1", "Exit", "The camera is outside."),
            completion_frames=frames,
            error_frames=frames,
        )

    failed = json.loads(captioner.last_failed_raw_response)
    assert failed["stage"] == "error"
    assert failed["response"] == '{"error":still invalid'
