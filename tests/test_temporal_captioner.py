from __future__ import annotations

import io
from dataclasses import replace

import pytest
from PIL import Image

from agentflow.agents.models_embodied_v2.memory.temporal_memory.temporal_captioner import (
    SceneAnalysisRequest,
    Subgoal,
    TemporalAnalysisRequest,
    TemporalCaptioner,
    TemporalFrameInput,
    TemporalInputError,
    TemporalOutputError,
)
from agentflow.agents.models_embodied_v2.data_models import PreviewView


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


def _scene_request() -> SceneAnalysisRequest:
    return SceneAnalysisRequest(
        subgoal=Subgoal(
            "2",
            "Walk forward to the pool area",
            "The camera is directly beside the pool.",
        ),
        frames=tuple(
            TemporalFrameInput(
                frame_id=index,
                image=np.full((12, 16, 3), index, dtype=np.uint8),
                translation_m=0.25,
                subgoal_path_length_m=index * 0.25,
            )
            for index in range(1, 10)
        ),
        is_final_subgoal=True,
    )


SCENE_RESPONSE = (
    '{"landmark":{"visible":true,"direction":"LEFT",'
    '"proximity":"NEAR","passed":false,'
    '"destination_dominant":true,"u":512,"v":430,'
    '"confidence":0.95},"door_state":"NOT_APPLICABLE",'
    '"door_camera_side":"NOT_APPLICABLE",'
    '"door_transition":"NOT_APPLICABLE",'
    '"current_room_side":"NOT_APPLICABLE","completed":true,'
    '"completion_confidence":0.91,"error_mode":"NONE",'
    '"error_confidence":0.0,"final_target":{"visible":true,'
    '"proximity":"AT","confidence":0.93},'
    '"evidence":"camera is beside the pool"}'
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


def test_scene_analysis_combines_all_perception_in_one_bounded_call():
    engine = FakeEngine(SCENE_RESPONSE)
    captioner = TemporalCaptioner(engine=engine)
    request = _scene_request()

    result = captioner.analyze_scene(request)

    assert len(engine.calls) == 1
    content, kwargs = engine.calls[0]
    assert "Is final subgoal: True" in content[0]
    assert "Next route stage" not in content[0]
    assert sum(isinstance(item, bytes) for item in content) == 9
    assert kwargs["image_max_pixels"] == 224**2
    assert result.completed is True
    assert result.error_mode == "NONE"
    assert result.final_target.visible is True
    assert result.door_state == "NOT_APPLICABLE"
    # Normalized coordinates resolve contradictory free-form directions at
    # the perception boundary.
    assert result.landmark.direction == "CENTER"


def test_final_completion_rejects_model_near_as_not_yet_at_target():
    response = SCENE_RESPONSE.replace(
        '"proximity":"AT"', '"proximity":"NEAR"'
    ).replace(
        '"evidence":"camera is beside the pool"',
        '"evidence":"pool is visible in the distance through the doorway; '
        'camera has not reached it"',
    )
    result = TemporalCaptioner(engine=FakeEngine(response)).analyze_scene(
        _scene_request()
    )

    assert result.final_target.visible is True
    assert result.final_target.proximity == "NEAR"
    assert result.completed is False


def test_final_completion_normalizes_positive_model_evidence_to_at():
    response = SCENE_RESPONSE.replace(
        '"proximity":"AT"', '"proximity":"FAR"'
    ).replace(
        '"evidence":"camera is beside the pool"',
        '"evidence":"camera has reached the pool; its edge is in the '
        'foreground and no source room is visible"',
    )
    result = TemporalCaptioner(engine=FakeEngine(response)).analyze_scene(
        _scene_request()
    )

    assert result.final_target.proximity == "AT"
    assert result.completed is True


def test_scene_prompt_marks_current_frame_and_rejects_false_crossing_cues():
    doorway_request = replace(
        _scene_request(),
        subgoal=Subgoal(
            "1",
            "Exit through the doorway into the pool room",
            "The camera has crossed the threshold.",
        ),
        is_final_subgoal=False,
    )
    response = SCENE_RESPONSE.replace(
        '"door_state":"NOT_APPLICABLE"',
        '"door_state":"AT_THRESHOLD"',
    ).replace(
        '"door_camera_side":"NOT_APPLICABLE"',
        '"door_camera_side":"BEFORE_DOOR"',
    ).replace(
        '"door_transition":"NOT_APPLICABLE"',
        '"door_transition":"APPROACHED"',
    ).replace(
        '"current_room_side":"NOT_APPLICABLE"',
        '"current_room_side":"ORIGINAL_SIDE"',
    ).replace('"completed":true', '"completed":false')
    engine = FakeEngine(response)

    TemporalCaptioner(engine=engine).analyze_scene(doorway_request)

    content, kwargs = engine.calls[0]
    metadata = [item for item in content if isinstance(item, str)]
    assert "role=HISTORICAL" in metadata[1]
    assert "role=CURRENT" in metadata[-1]
    system_prompt = kwargs["system_prompt"]
    assert "door disappearing" in system_prompt
    assert "TURNED_AWAY, not PASSED_THROUGH" in system_prompt
    assert "bathtub, sink" in system_prompt


def test_door_completion_requires_model_visible_passage_and_far_side():
    doorway_request = replace(
        _scene_request(),
        subgoal=Subgoal(
            "1",
            "Exit through the doorway",
            "The camera has crossed the threshold.",
        ),
        is_final_subgoal=False,
    )
    contradictory = SCENE_RESPONSE.replace(
        '"door_state":"NOT_APPLICABLE"',
        '"door_state":"CROSSED"',
    ).replace(
        '"door_camera_side":"NOT_APPLICABLE"',
        '"door_camera_side":"AFTER_DOOR"',
    ).replace(
        '"door_transition":"NOT_APPLICABLE"',
        '"door_transition":"APPROACHED"',
    ).replace(
        '"current_room_side":"NOT_APPLICABLE"',
        '"current_room_side":"ORIGINAL_SIDE"',
    )

    result = TemporalCaptioner(
        engine=FakeEngine(contradictory)
    ).analyze_scene(doorway_request)

    assert result.completed is False


def test_scene_analysis_reuses_encoded_retained_frames():
    engine = FakeEngine(SCENE_RESPONSE)
    captioner = TemporalCaptioner(engine=engine)
    request = _scene_request()

    captioner.analyze_scene(request)
    first_encoded = [
        item for item in engine.calls[0][0] if isinstance(item, bytes)
    ]
    captioner.analyze_scene(request)
    second_encoded = [
        item for item in engine.calls[1][0] if isinstance(item, bytes)
    ]

    assert len(engine.calls) == 2
    assert all(
        first is second
        for first, second in zip(first_encoded, second_encoded, strict=True)
    )
    assert len(captioner._png_cache) == 9


def test_preview_selector_compares_views_without_evicting_temporal_cache():
    engine = FakeEngine(
        '{"view_index":2,"u":320,"v":780,"confidence":0.93,'
        '"evidence":"doorway and floor path are centered"}'
    )
    captioner = TemporalCaptioner(engine=engine)
    captioner._png(np.zeros((12, 16, 3), dtype=np.uint8))
    cached_before = tuple(captioner._png_cache.items())
    views = tuple(
        PreviewView(
            yaw_deg=yaw,
            rgb=np.full((12, 16, 3), index, dtype=np.uint8),
        )
        for index, yaw in enumerate((-45.0, 0.0, 45.0))
    )

    selection = captioner.select(
        subgoal=Subgoal(
            "1",
            "Exit through the doorway",
            "The camera crosses the threshold.",
        ),
        views=views,
    )

    assert selection is not None
    assert selection.view_index == 2
    assert (selection.u, selection.v) == (320, 780)
    content, kwargs = engine.calls[-1]
    assert "structural doorway" in content[1]
    assert "view_index=2; yaw_deg=+45.0" in content[-2]
    assert "negative is to the left, positive is\nto the right" in (
        engine.calls[-1][1]["system_prompt"]
    )
    assert sum(isinstance(item, bytes) for item in content) == 3
    assert kwargs["max_tokens"] == 48
    assert tuple(captioner._png_cache.items()) == cached_before


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


TRUNCATED_PRETTY_SCENE = """{
  "door_state": "NOT_APPLICABLE",
  "door_camera_side": "NOT_APPLICABLE",
  "door_transition": "NOT_APPLICABLE",
  "current_room_side": "NOT_APPLICABLE",
  "landmark": {
    "visible": true,
    "direction": "RIGHT",
    "proximity": "NEAR",
    "passed": false,
    "destination_dominant": false,
    "u": 720,
    "v": 640,
    "confidence": 0.8
  },
  "completed": false,
  "completion_confidence": 0.7,
  "error_mode": "NONE",
  "error_confidence": 0.6,
  "final_target": {
    "visible": false,
    "proximity": "UNKNOWN",
    "confidence": 0.5
  },
  "evidence": "The pool edge is visible to the right of the camera and the walk"""


def test_scene_analysis_repairs_a_pretty_printed_reply_cut_by_the_token_budget():
    """Qwen3-VL-8B sometimes indents the object, which doubles its length
    and lets the token budget cut it inside the trailing evidence string."""
    captioner = TemporalCaptioner(engine=FakeEngine(TRUNCATED_PRETTY_SCENE))

    result = captioner.analyze_scene(_scene_request())

    assert result.completed is False
    assert result.landmark.visible is True
    assert (result.landmark.u, result.landmark.v) == (720, 640)
    assert result.landmark.proximity == "NEAR"
    assert result.completion_evidence.startswith("The pool edge is visible")
    assert captioner.last_failed_raw_response is None


def test_scene_analysis_repair_backs_up_to_the_last_complete_field():
    cut_inside_number = TRUNCATED_PRETTY_SCENE.split('"completion_confidence"')[0]
    cut_inside_number += '"completion_confidence": 0.'
    captioner = TemporalCaptioner(engine=FakeEngine(cut_inside_number))
    with pytest.raises(TemporalOutputError, match="invalid JSON"):
        # The fields before the cut are recovered, but required keys are
        # missing: that is reported as the truncation it is, so retry logic
        # and step logs keep telling it apart from a wrong schema.
        captioner.analyze_scene(_scene_request())
    assert captioner.last_failed_raw_response is not None


def test_scene_schema_mismatch_is_an_output_error_with_the_raw_text_kept():
    wrong_schema = '{"completed": false, "evidence": "only two keys"}'
    captioner = TemporalCaptioner(engine=FakeEngine(wrong_schema))

    with pytest.raises(TemporalOutputError, match="wrong schema"):
        captioner.analyze_scene(_scene_request())

    assert captioner.last_failed_raw_response is not None
    assert "only two keys" in captioner.last_failed_raw_response


def test_scene_prompts_describe_value_types_instead_of_placeholder_values():
    from agentflow.agents.models_embodied_v2.memory.temporal_memory.temporal_captioner import (
        DOOR_SCENE_SYSTEM_PROMPT,
        SCENE_SYSTEM_PROMPT,
    )
    from agentflow.agents.models_embodied_v2.skiils.protocol import POINT_PROMPT

    for prompt in (SCENE_SYSTEM_PROMPT, DOOR_SCENE_SYSTEM_PROMPT, POINT_PROMPT):
        # A literal 0.0 or 500/750 in the schema line was copied verbatim by
        # Qwen3-VL-8B on every step of a full run.
        assert '"confidence":0.0' not in prompt
        assert '"u":500' not in prompt and '"v":750' not in prompt
        assert "<float 0-1>" in prompt
        assert "single-line" in prompt
