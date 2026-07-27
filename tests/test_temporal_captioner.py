from __future__ import annotations

import io
import json
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


def _image(value=0):
    return np.full((20, 28, 3), value, dtype=np.uint8)


def _subgoals():
    return (
        Subgoal("1", "Exit the room", "Cross the doorway threshold."),
        Subgoal("2", "Stop before the pool", "Reach the pool edge and stop."),
    )


def _request(count=8, *, actions=True, image=None):
    action_cycle = ("FORWARD", "TURN_LEFT", "TURN_RIGHT")
    return TemporalAnalysisRequest(
        episode_id="632",
        task="Exit into the pool room and stop before the pool.",
        task_guidance=(
            "A commanded action is not completion evidence. "
            "Judge completion from post-action images."
        ),
        subgoals=_subgoals(),
        steps=tuple(
            TemporalStepInput(
                step_id=index,
                action=action_cycle[(index - 1) % len(action_cycle)]
                if actions
                else None,
                image=_image(index) if image is None else image,
                timestamp_seconds=float(index),
            )
            for index in range(1, count + 1)
        ),
    )


def _payload(
    step_ids=range(1, 9),
    *,
    completed=(True, False),
    persistent_error=False,
    error_mode="NONE",
    error_ids=None,
):
    ids = list(step_ids)
    return {
        "steps": [
            {
                "step_id": step_id,
                "caption": f"scene {step_id}",
                "visual_change": "FORWARD",
                "error_clue": None,
            }
            for step_id in ids
        ],
        "subgoals": [
            {
                "subgoal_id": str(index),
                "completed": value,
                "evidence": "visible evidence",
                "evidence_step_ids": [ids[-1]],
            }
            for index, value in enumerate(completed, start=1)
        ],
        "persistent_error": persistent_error,
        "error_mode": error_mode,
        "error_evidence": (
            "same obstacle across steps" if persistent_error else "none"
        ),
        "error_evidence_step_ids": list(error_ids or []),
        "confidence": 0.8,
    }


def test_prompt_contains_task_guidance_subgoals_and_eight_interleaved_frames():
    engine = FakeEngine(json.dumps(_payload()))
    result = TemporalCaptioner(engine=engine).analyze(_request())

    content, kwargs = engine.calls[0]
    assert "[TASK]\nExit into the pool room" in content[0]
    assert "[TASK GUIDANCE]" in content[0]
    assert "Completion criteria: Cross the doorway threshold." in content[0]
    markers = [
        index
        for index, item in enumerate(content)
        if isinstance(item, str) and item.startswith("\n[STEP step_id=")
    ]
    assert len(markers) == 8
    assert all(isinstance(content[index + 1], bytes) for index in markers)
    assert "step_id=1 action=FORWARD post_t=1.000s" in content[markers[0]]
    assert "step_id=8 action=TURN_LEFT post_t=8.000s" in content[markers[-1]]
    assert "最近走的8步发生了什么" in content[-1]
    prompt_text = "\n".join(item for item in content if isinstance(item, str))
    assert "observed_action=" not in prompt_text
    assert "action_match=" not in prompt_text
    assert "collision=" not in prompt_text
    assert kwargs["max_tokens"] == 384
    assert kwargs["image_min_pixels"] == 64**2
    assert kwargs["image_max_pixels"] == 448**2
    assert [item.step_id for item in result.steps] == list(range(1, 9))
    assert result.status_for("1").completed is True
    assert result.status_for("2").completed is False


def test_unknown_action_is_explicit_and_float_unit_images_stay_bright():
    engine = FakeEngine(
        json.dumps(_payload([1], completed=(False, False)))
    )
    request = _request(1, actions=False, image=np.ones((8, 10, 3)))

    TemporalCaptioner(engine=engine).analyze(request)

    content, _ = engine.calls[0]
    marker_index = next(
        index
        for index, item in enumerate(content)
        if isinstance(item, str) and item.startswith("\n[STEP")
    )
    assert "action=UNKNOWN" in content[marker_index]
    encoded = content[marker_index + 1]
    decoded = np.asarray(Image.open(io.BytesIO(encoded)).convert("RGB"))
    assert int(decoded.min()) == 255


def test_completion_only_mode_uses_frames_and_returns_one_boolean():
    engine = FakeEngine("true")
    request = replace(
        _request(actions=False),
        subgoals=(_subgoals()[0],),
    )

    result = TemporalCaptioner(engine=engine).evaluate_subgoal(request)

    content, kwargs = engine.calls[0]
    text_items = [item for item in content if isinstance(item, str)]
    assert text_items == [
        "Subgoal: Exit the room\n"
        "Completion proof: Cross the doorway threshold."
    ]
    assert len([item for item in content if isinstance(item, bytes)]) == 8
    prompt = "\n".join(text_items)
    assert "Exit into the pool room" not in prompt
    assert "commanded action" not in prompt
    assert "frame_id=" not in prompt
    assert "timestamp" not in prompt
    assert result.completed is True
    assert result.subgoal_id == "1"
    assert result.raw_response == "true"
    assert kwargs["max_tokens"] == 1
    assert kwargs["system_prompt"] == (
        "Use only the ordered images; the subgoal text is not evidence. "
        "Reply true only if they visibly prove completion by the final image, "
        "otherwise false. The transition need not occur inside this window "
        "when the final visible state itself proves completion. Reply only "
        "true or false."
    )


def test_completion_only_mode_rejects_actions_and_multiple_subgoals():
    captioner = TemporalCaptioner(engine=FakeEngine("{}"))
    with pytest.raises(TemporalInputError, match="exactly one subgoal"):
        captioner.evaluate_subgoal(_request(actions=False))
    with pytest.raises(TemporalInputError, match="action=None"):
        captioner.evaluate_subgoal(
            replace(_request(), subgoals=(_subgoals()[0],))
        )
    with pytest.raises(TemporalInputError, match="exactly eight frames"):
        captioner.evaluate_subgoal(
            replace(
                _request(7, actions=False),
                subgoals=(_subgoals()[0],),
            )
        )


def test_completion_only_mode_rejects_non_boolean_output():
    with pytest.raises(TemporalOutputError, match="exactly true or false"):
        TemporalCaptioner(engine=FakeEngine('{"completed": false}')).evaluate_subgoal(
            replace(
                _request(actions=False),
                subgoals=(_subgoals()[0],),
            )
        )


def test_request_validates_count_order_timestamps_and_actions():
    request = _request()
    with pytest.raises(TemporalInputError, match="between 1 and 8"):
        replace(request, steps=())
    with pytest.raises(TemporalInputError, match="strictly increasing"):
        replace(
            request,
            steps=(request.steps[1], request.steps[0], *request.steps[2:]),
        )
    with pytest.raises(TemporalInputError, match="timestamps"):
        replace(
            request,
            steps=(
                request.steps[0],
                replace(request.steps[1], timestamp_seconds=1.0),
                *request.steps[2:],
            ),
        )
    with pytest.raises(TemporalInputError, match="Unsupported action"):
        replace(request.steps[0], action="GO_BACK")


def test_output_ids_must_match_input_and_evidence_must_be_grounded():
    wrong_order = _payload([2, 1, 3, 4, 5, 6, 7, 8])
    with pytest.raises(TemporalOutputError, match="step IDs"):
        TemporalCaptioner(
            engine=FakeEngine(json.dumps(wrong_order))
        ).analyze(_request())

    wrong_subgoal = _payload()
    wrong_subgoal["subgoals"][0]["subgoal_id"] = "99"
    with pytest.raises(TemporalOutputError, match="subgoal IDs"):
        TemporalCaptioner(
            engine=FakeEngine(json.dumps(wrong_subgoal))
        ).analyze(_request())

    unknown_evidence = _payload()
    unknown_evidence["subgoals"][0]["evidence_step_ids"] = [99]
    with pytest.raises(TemporalOutputError, match="evidence step IDs"):
        TemporalCaptioner(
            engine=FakeEngine(json.dumps(unknown_evidence))
        ).analyze(_request())


def test_persistent_error_requires_mode_and_multistep_evidence():
    no_mode = _payload(
        persistent_error=True,
        error_mode="NONE",
        error_ids=[1, 2],
    )
    with pytest.raises(TemporalOutputError, match="non-NONE"):
        TemporalCaptioner(
            engine=FakeEngine(json.dumps(no_mode))
        ).analyze(_request())

    two_steps = _payload(
        persistent_error=True,
        error_mode="WALL_STUCK",
        error_ids=[1, 2],
    )
    with pytest.raises(TemporalOutputError, match="at least three"):
        TemporalCaptioner(
            engine=FakeEngine(json.dumps(two_steps))
        ).analyze(_request())

    inconsistent_false = _payload()
    inconsistent_false["error_mode"] = "GET_NOWHERE"
    with pytest.raises(TemporalOutputError, match="error_mode=NONE"):
        TemporalCaptioner(
            engine=FakeEngine(json.dumps(inconsistent_false))
        ).analyze(_request())


def test_invalid_or_multiple_json_objects_are_rejected_and_counted():
    captioner = TemporalCaptioner(engine=FakeEngine("not json"))
    with pytest.raises(TemporalOutputError):
        captioner.analyze(_request())
    assert captioner.last_raw_response == "not json"
    assert captioner.performance_summary()["failure_count"] == 1

    captioner = TemporalCaptioner(
        engine=FakeEngine(json.dumps(_payload()) + "\n{}")
    )
    with pytest.raises(TemporalOutputError, match="trailing content"):
        captioner.analyze(_request())


def test_engine_factory_is_lazy(monkeypatch):
    from agentflow.agents.engine import factory

    engine = FakeEngine(json.dumps(_payload()))
    calls = []

    def create(model_string, **kwargs):
        calls.append((model_string, kwargs))
        return engine

    monkeypatch.setattr(factory, "create_llm_engine", create)
    captioner = TemporalCaptioner(model_path="/models/test")
    assert calls == []

    captioner.analyze(_request())

    assert calls[0][0] == "local-qwen3vl-/models/test"
    assert calls[0][1]["is_multimodal"] is True
    assert captioner.performance_summary()["success_count"] == 1
