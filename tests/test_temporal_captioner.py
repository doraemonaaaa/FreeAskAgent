import json
from dataclasses import replace
from pathlib import Path

import pytest

from agentflow.agents.models_embodied_v2.TemporalCaptioner import (
    CameraMotion,
    ErrorAssessment,
    MotionSignal,
    ProgressSignals,
    TemporalAnalysisRequest,
    TemporalCaptionPayload,
    TemporalCaptioner,
    TemporalInputError,
    TemporalOutputError,
    TemporalStepCaptionPayload,
    TemporalStepInput,
    TemporalStepModelPayload,
    TemporalWindow,
    TimedAction,
    TimestampedFrame,
    TopologySignal,
)


class FakeEngine:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def __call__(self, content, **kwargs):
        self.calls.append((content, kwargs))
        return self.response


def _png_bytes(value):
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    image = np.full((24, 32, 3), value, dtype=np.uint8)
    from io import BytesIO

    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


def _valid_payload():
    return {
        "latest_scene": "最后看到条纹墙。",
        "scene_summary": "相机在同一房间内扫过床、墙面和门口。",
        "activity_summary": "先左转，随后反向右转。",
        "overall_progress": "STALLED",
        "phases": [
            {
                "interval": {"start_seconds": 5.0, "end_seconds": 7.6},
                "scene": "从床扫向门口。",
                "commanded_activity": "TURN_LEFT",
                "observed_motion": "TURN_LEFT",
                "progress": "STALLED",
                "evidence_timestamps_seconds": [5.0, 7.6],
                "confidence": 0.9,
            },
            {
                "interval": {"start_seconds": 7.6, "end_seconds": 10.0},
                "scene": "沿相反方向扫回。",
                "commanded_activity": "TURN_RIGHT",
                "observed_motion": "TURN_RIGHT",
                "progress": "REGRESSING",
                "evidence_timestamps_seconds": [7.6, 10.0],
                "confidence": 0.9,
            },
        ],
        "errors": {
            mode: {
                "verdict": "ABSENT",
                "confidence": 0.6,
                "interval": None,
                "evidence_timestamps_seconds": [],
                "reason": "没有足够证据。",
            }
            for mode in (
                "collision",
                "repeated_visit",
                "motion_oscillation",
                "get_nowhere",
            )
        },
    }


def _window():
    return TemporalWindow(
        start_seconds=5.0,
        end_seconds=10.0,
        frames=(
            TimestampedFrame(5.0, _png_bytes(30), step_id=50),
            TimestampedFrame(7.5, _png_bytes(80), step_id=75),
            TimestampedFrame(9.5, _png_bytes(120), step_id=95),
        ),
        actions=(
            TimedAction(5.0, "TURN_LEFT", step_id=50),
            TimedAction(7.6, "TURN_RIGHT", step_id=76),
        ),
        motion=(
            MotionSignal(
                5.0,
                7.6,
                CameraMotion.TURN_LEFT,
                scene_flow_dx_fraction=0.02,
                confidence=0.9,
                source="test",
            ),
            MotionSignal(
                7.6,
                10.0,
                CameraMotion.TURN_RIGHT,
                scene_flow_dx_fraction=-0.02,
                confidence=0.9,
                source="test",
            ),
        ),
        reverse_retrace_similarity=0.99,
    )


def _step_request(*, mismatch_step_id=None):
    steps = []
    actions = (
        "FORWARD",
        "TURN_LEFT",
        "TURN_RIGHT",
        "FORWARD",
        "FORWARD",
        "TURN_LEFT",
        "TURN_RIGHT",
        "STOP",
    )
    observed = (
        CameraMotion.FORWARD,
        CameraMotion.TURN_LEFT,
        CameraMotion.TURN_RIGHT,
        CameraMotion.FORWARD,
        CameraMotion.FORWARD,
        CameraMotion.TURN_LEFT,
        CameraMotion.TURN_RIGHT,
        CameraMotion.STATIONARY,
    )
    for index, (action, motion) in enumerate(zip(actions, observed), start=1):
        is_mismatch = index == mismatch_step_id
        effective_motion = (
            CameraMotion.STATIONARY if is_mismatch else motion
        )
        steps.append(
            TemporalStepInput(
                step_id=index,
                commanded_action=action,
                post_timestamp_seconds=float(index),
                image=_png_bytes(index * 20),
                motion=MotionSignal(
                    float(index - 1),
                    float(index),
                    effective_motion,
                    scene_flow_magnitude_fraction=0.01,
                    collision=False,
                    confidence=0.9,
                    source="test",
                ),
                observed_motion=effective_motion,
                action_match="MISMATCH" if is_mismatch else "MATCH",
                collision=False,
                topology_node_id=f"node-{index}",
                is_new_node=True,
                is_revisit=False,
                distance_to_goal_meters=float(9 - index),
            )
        )
    return TemporalAnalysisRequest(
        episode_id="episode-1",
        goal="走到门口",
        steps=tuple(steps),
        progress=ProgressSignals(
            net_displacement_meters=2.0,
            new_landmarks_count=2,
            new_topological_nodes_count=8,
            completed_subgoals_count=0,
            no_progress_steps=0,
        ),
    )


def _three_step_request():
    base = _step_request()
    return replace(
        base,
        steps=base.steps[:3],
        progress=ProgressSignals(
            net_displacement_meters=0.8,
            new_landmarks_count=2,
            new_topological_nodes_count=3,
            completed_subgoals_count=0,
            no_progress_steps=0,
        ),
    )


def _valid_step_payload(*, fake_alignment=False):
    errors = {
        mode: {
            "verdict": "ABSENT",
            "confidence": 0.9,
            "interval": None,
            "evidence_timestamps_seconds": [],
            "reason": "没有发现该错误。",
        }
        for mode in (
            "collision",
            "repeated_visit",
            "motion_oscillation",
            "get_nowhere",
            "action_execution_mismatch",
        )
    }
    captions = []
    for step_id in range(1, 9):
        caption = {
            "step_id": step_id,
            "scene_after_action": f"看到房间地标{step_id}",
            "visible_landmarks": [f"地标{step_id}"],
            "visual_perceived_action": "UNKNOWN",
            "visual_error_clues": [],
            "confidence": 0.8,
        }
        if fake_alignment:
            caption.update(
                commanded_action="TURN_RIGHT",
                post_timestamp_seconds=100.0 + step_id,
                observed_motion="TURN_RIGHT",
                action_match="MISMATCH",
                collision=True,
            )
        captions.append(caption)
    return {
        "latest_scene": "最后看到门口。",
        "scene_summary": "依次经过房间内的多个地标。",
        "overall_progress": "PROGRESSING",
        "step_captions": captions,
        "errors": errors,
    }


def _valid_compact_three_step_payload():
    return {
        "p": "PROGRESSING",
        "s": [
            {
                "i": 1,
                "c": "床铺位于房间一侧",
                "l": ["床铺"],
                "m": "FORWARD",
                "e": [],
            },
            {
                "i": 2,
                "c": "浅色墙面和木门可见",
                "l": ["木门", "墙面"],
                "m": "TURN_LEFT",
                "e": [],
            },
            {
                "i": 3,
                "c": "门口和浅色墙面",
                "l": ["门口", "墙面"],
                "m": "TURN_RIGHT",
                "e": [],
            },
        ],
        "x": [],
    }


def test_step_mode_keeps_eight_step_compatibility_payload_and_images():
    fake = FakeEngine(
        json.dumps(_valid_step_payload(), ensure_ascii=False)
    )
    record = TemporalCaptioner(engine=fake).analyze_steps(_step_request())

    assert len(fake.calls) == 1
    content, kwargs = fake.calls[0]
    markers = [
        index
        for index, item in enumerate(content)
        if isinstance(item, str) and item.startswith("\n[STEP step_id=")
    ]
    assert len(markers) == 8
    assert all(isinstance(content[index + 1], bytes) for index in markers)
    assert "step_id=1 command=FORWARD post_t=1.000s" in content[markers[0]]
    assert "step_id=8 command=STOP post_t=8.000s" in content[markers[-1]]
    assert (
        "最近走的八步发生了什么？按照每步 action 后的画面进行描述。"
        in content[-1]
    )
    assert "response_format" not in kwargs
    assert kwargs["max_tokens"] == 128
    assert record.frame_timestamps_seconds == [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0
    ]
    assert [caption.step_id for caption in record.step_captions] == list(
        range(1, 9)
    )
    memory_text = record.to_memory_text()
    assert "Step scenes:" in memory_text
    assert "1:看到房间地标1" in memory_text
    assert "8:看到房间地标8" in memory_text


def test_default_three_step_compact_json_has_three_markers_and_128_token_cap():
    request = _three_step_request()
    fake = FakeEngine(
        json.dumps(_valid_compact_three_step_payload(), ensure_ascii=False)
    )
    fake.supports_image_pixel_budget = True
    captioner = TemporalCaptioner(engine=fake)

    record = captioner.analyze_steps(request)

    assert len(fake.calls) == 1
    content, kwargs = fake.calls[0]
    markers = [
        index
        for index, item in enumerate(content)
        if isinstance(item, str) and item.startswith("\n[STEP step_id=")
    ]
    assert len(markers) == 3
    assert all(isinstance(content[index + 1], bytes) for index in markers)
    assert content[0].startswith("[BEGIN 3 STEP POST-ACTION STORYBOARD]")
    assert "step_id=1 command=FORWARD post_t=1.000s" in content[markers[0]]
    assert "step_id=3 command=TURN_RIGHT post_t=3.000s" in content[markers[-1]]
    assert "[END 3 STEP POST-ACTION STORYBOARD]" in content[-2]
    assert "最近走的三步发生了什么" in content[-1]
    assert "s 必须依次包含这些ID：[1, 2, 3]" in content[-1]
    assert kwargs["max_tokens"] == 128
    assert "response_format" not in kwargs
    assert kwargs["image_min_pixels"] == 64**2
    assert kwargs["image_max_pixels"] == 128**2
    assert captioner.config.max_image_edge == 128
    assert captioner.config.inference_latency_budget_ms == 5000.0

    assert record.frame_timestamps_seconds == [1.0, 2.0, 3.0]
    assert [caption.step_id for caption in record.step_captions] == [1, 2, 3]
    assert [caption.confidence for caption in record.step_captions] == [0.6] * 3
    assert record.latest_scene == "门口和浅色墙面"
    assert record.latency_budget_ms == 5000.0
    assert json.loads(record.raw_response) == _valid_compact_three_step_payload()


def test_step_mode_typed_alignment_overrides_model_fields():
    request = _step_request()
    fake = FakeEngine(json.dumps(_valid_step_payload(fake_alignment=True)))

    record = TemporalCaptioner(engine=fake).analyze_steps(request)

    for caption, step in zip(record.step_captions, request.steps):
        assert caption.commanded_action == step.commanded_action
        assert caption.post_timestamp_seconds == step.post_timestamp_seconds
        assert caption.observed_motion == step.observed_motion
        # Preserve what the video model visually perceived as a separate,
        # lower-priority signal; only caller-owned observed_motion is forced.
        assert caption.visual_perceived_action == CameraMotion.UNKNOWN
        assert caption.action_match == step.action_match
        assert caption.collision is step.collision
    assert record.action_timeline[0]["action"] == "FORWARD"
    assert record.action_timeline[0]["post_timestamp_seconds"] == 1.0
    assert record.motion_evidence[0]["camera_motion"] == "FORWARD"


def test_missing_step_confidence_defaults_to_uncertain_confidence():
    payload = _valid_step_payload()
    for caption in payload["step_captions"]:
        caption.pop("confidence")

    record = TemporalCaptioner(
        engine=FakeEngine(json.dumps(payload))
    ).analyze_steps(_step_request())

    assert [caption.confidence for caption in record.step_captions] == [0.5] * 8


def test_step_mode_rejects_missing_reordered_or_duplicate_step_ids():
    payload = _valid_step_payload()
    payload["step_captions"][4]["step_id"] = 3
    fake = FakeEngine(json.dumps(payload))

    with pytest.raises(TemporalOutputError, match="one-to-one"):
        TemporalCaptioner(engine=fake).analyze_steps(_step_request())


def test_step_request_accepts_two_steps_for_compatibility():
    valid = _step_request()
    request = TemporalAnalysisRequest(
        episode_id=valid.episode_id,
        goal=valid.goal,
        steps=valid.steps[:2],
    )
    assert len(request.steps) == 2


@pytest.mark.parametrize("step_count", [1, 9])
def test_step_request_rejects_fewer_than_two_or_more_than_eight_steps(
    step_count,
):
    valid = _step_request()
    steps = valid.steps
    if step_count == 9:
        last = steps[-1]
        ninth = replace(
            last,
            step_id=9,
            post_timestamp_seconds=9.0,
            motion=replace(
                last.motion,
                start_seconds=8.0,
                end_seconds=9.0,
            ),
        )
        steps = (*steps, ninth)
    else:
        steps = steps[:step_count]

    with pytest.raises(TemporalInputError, match="between 2 and 8"):
        TemporalAnalysisRequest(
            episode_id=valid.episode_id,
            goal=valid.goal,
            steps=steps,
        )


@pytest.mark.parametrize("invalid_step_id", [True, 1.0, 1.9, "1"])
def test_step_input_rejects_non_integer_step_ids(invalid_step_id):
    step = _step_request().steps[0]
    with pytest.raises(TemporalInputError, match="step_id"):
        replace(step, step_id=invalid_step_id)


def test_typed_action_execution_mismatch_overrides_model_absent():
    request = _step_request(mismatch_step_id=4)
    record = TemporalCaptioner(
        engine=FakeEngine(json.dumps(_valid_step_payload()))
    ).analyze_steps(request)

    mismatch = record.errors.action_execution_mismatch
    assert mismatch.verdict == "PRESENT"
    assert mismatch.source == "FUSED"
    assert mismatch.evidence_timestamps_seconds == [4.0]
    assert record.step_captions[3].observed_motion == CameraMotion.STATIONARY
    assert record.step_captions[3].action_match == "MISMATCH"


def test_typed_mismatch_repairs_incomplete_model_error_grounding():
    request = _step_request(mismatch_step_id=4)
    payload = _valid_step_payload()
    payload["errors"]["action_execution_mismatch"].update(
        verdict="PRESENT",
        confidence=0.9,
        interval=None,
        evidence_timestamps_seconds=[],
        reason="视觉上疑似命令与运动不一致。",
    )

    record = TemporalCaptioner(
        engine=FakeEngine(json.dumps(payload))
    ).analyze_steps(request)

    mismatch = record.errors.action_execution_mismatch
    assert mismatch.verdict == "PRESENT"
    assert mismatch.source == "FUSED"
    assert mismatch.interval is not None
    assert mismatch.interval.start_seconds == 3.0
    assert mismatch.interval.end_seconds == 4.0
    assert mismatch.evidence_timestamps_seconds == [4.0]


def test_typed_matches_replace_incomplete_model_mismatch_claim():
    payload = _valid_step_payload()
    payload["errors"]["action_execution_mismatch"].update(
        verdict="PRESENT",
        confidence=0.9,
        interval=None,
        evidence_timestamps_seconds=[],
        reason="视觉上疑似命令与运动不一致。",
    )

    record = TemporalCaptioner(
        engine=FakeEngine(json.dumps(payload))
    ).analyze_steps(_step_request())

    mismatch = record.errors.action_execution_mismatch
    assert mismatch.verdict == "ABSENT"
    assert mismatch.source == "FUSED"
    assert mismatch.interval is None
    assert mismatch.evidence_timestamps_seconds == []


def test_incomplete_model_mismatch_without_typed_coverage_is_uncertain():
    request = _step_request()
    uncertain_step = replace(
        request.steps[3],
        action_match="UNCERTAIN",
    )
    request = replace(
        request,
        steps=(*request.steps[:3], uncertain_step, *request.steps[4:]),
    )
    payload = _valid_step_payload()
    payload["errors"]["action_execution_mismatch"].update(
        verdict="PRESENT",
        confidence=0.9,
        interval=None,
        evidence_timestamps_seconds=[],
        reason="视觉上疑似命令与运动不一致。",
    )

    record = TemporalCaptioner(
        engine=FakeEngine(json.dumps(payload))
    ).analyze_steps(request)

    mismatch = record.errors.action_execution_mismatch
    assert mismatch.verdict == "UNCERTAIN"
    assert mismatch.source == "MODEL"
    assert mismatch.interval is None
    assert mismatch.evidence_timestamps_seconds == []
    assert "缺少完整时间定位或证据" in mismatch.reason


def test_uncovered_model_error_with_forged_evidence_time_is_rejected():
    request = _step_request()
    uncertain_step = replace(
        request.steps[3],
        action_match="UNCERTAIN",
    )
    request = replace(
        request,
        steps=(*request.steps[:3], uncertain_step, *request.steps[4:]),
    )
    payload = _valid_step_payload()
    payload["errors"]["action_execution_mismatch"].update(
        verdict="PRESENT",
        confidence=0.9,
        interval={"start_seconds": 0.0, "end_seconds": 8.0},
        evidence_timestamps_seconds=[123.0],
        reason="视觉上疑似命令与运动不一致。",
    )

    with pytest.raises(TemporalOutputError, match="does not match"):
        TemporalCaptioner(
            engine=FakeEngine(json.dumps(payload))
        ).analyze_steps(request)


def test_final_step_revisit_interval_stays_inside_public_record_window():
    request = _step_request()
    final_step = replace(
        request.steps[-1],
        topology_node_id=request.steps[0].topology_node_id,
        is_new_node=False,
        is_revisit=True,
    )
    request = replace(
        request,
        steps=(*request.steps[:-1], final_step),
    )

    record = TemporalCaptioner(
        engine=FakeEngine(json.dumps(_valid_step_payload()))
    ).analyze_steps(request)

    assessment = record.errors.repeated_visit
    assert assessment.verdict == "PRESENT"
    assert assessment.interval is not None
    assert assessment.interval.start_seconds < assessment.interval.end_seconds
    assert assessment.interval.end_seconds <= record.window.end_seconds


def test_all_typed_matches_override_model_mismatch_claim():
    payload = _valid_step_payload()
    payload["errors"]["action_execution_mismatch"].update(
        verdict="PRESENT",
        confidence=1.0,
        interval={"start_seconds": 0.0, "end_seconds": 8.0},
        evidence_timestamps_seconds=[0.0, 8.0],
        reason="模型声称命令和运动不一致。",
    )
    record = TemporalCaptioner(
        engine=FakeEngine(json.dumps(payload))
    ).analyze_steps(_step_request())

    mismatch = record.errors.action_execution_mismatch
    assert mismatch.verdict == "ABSENT"
    assert mismatch.source == "FUSED"
    assert "均匹配" in mismatch.reason


def test_step_get_nowhere_uses_quarter_meter_threshold():
    base = _step_request()
    steps = tuple(
        replace(
            step,
            distance_to_goal_meters=5.0,
            is_new_node=False,
        )
        for step in base.steps
    )
    request = replace(
        base,
        steps=steps,
        progress=ProgressSignals(
            net_displacement_meters=0.2,
            new_landmarks_count=0,
            new_topological_nodes_count=0,
            completed_subgoals_count=0,
            no_progress_steps=8,
        ),
    )

    record = TemporalCaptioner(
        engine=FakeEngine(json.dumps(_valid_step_payload()))
    ).analyze_steps(request)

    assert record.errors.get_nowhere.verdict == "PRESENT"
    assert record.errors.get_nowhere.source == "FUSED"
    assert record.overall_progress == "STALLED"


def test_step_mode_does_not_call_one_turn_correction_oscillation():
    base = _step_request()
    motions = (
        CameraMotion.TURN_LEFT,
        CameraMotion.TURN_LEFT,
        CameraMotion.TURN_LEFT,
        CameraMotion.TURN_RIGHT,
        CameraMotion.TURN_RIGHT,
        CameraMotion.TURN_RIGHT,
        CameraMotion.STATIONARY,
        CameraMotion.STATIONARY,
    )
    steps = tuple(
        replace(
            step,
            observed_motion=motion,
            motion=replace(step.motion, camera_motion=motion),
        )
        for step, motion in zip(base.steps, motions)
    )
    request = replace(
        base,
        steps=steps,
        reverse_retrace_similarity=0.99,
        progress=ProgressSignals(net_displacement_meters=0.0),
    )
    payload = _valid_step_payload()
    payload["errors"]["motion_oscillation"].update(
        verdict="PRESENT",
        confidence=0.9,
        interval={"start_seconds": 0.0, "end_seconds": 8.0},
        evidence_timestamps_seconds=[0.0, 4.0, 8.0],
        reason="模型把一次纠正判断成振荡。",
    )

    record = TemporalCaptioner(
        engine=FakeEngine(json.dumps(payload))
    ).analyze_steps(request)

    assert record.errors.motion_oscillation.verdict != "PRESENT"


def test_fake_engine_receives_absolute_timestamped_storyboard():
    fake = FakeEngine(json.dumps(_valid_payload(), ensure_ascii=False))
    captioner = TemporalCaptioner(engine=fake)

    record = captioner.analyze(_window())

    assert record.frame_timestamps_seconds == [5.0, 7.5, 9.5]
    assert len(fake.calls) == 1
    content, kwargs = fake.calls[0]
    markers = [
        index
        for index, item in enumerate(content)
        if isinstance(item, str) and "[FRAME id=" in item
    ]
    assert len(markers) == 3
    assert "absolute_t=5.000s" in content[markers[0]]
    assert "absolute_t=7.500s" in content[markers[1]]
    assert "absolute_t=9.500s" in content[markers[2]]
    assert all(isinstance(content[index + 1], bytes) for index in markers)
    assert kwargs["response_format"] is TemporalCaptionPayload
    assert '"issued_at_seconds":5.0' in content[-1]
    assert '"active_until_seconds":7.6' in content[-1]
    assert record.action_timeline[0]["active_until_seconds"] == 7.6


def test_injected_engine_does_not_call_factory():
    fake = FakeEngine(json.dumps(_valid_payload()))

    def fail_factory():
        raise AssertionError("factory must not be called")

    captioner = TemporalCaptioner(engine=fake)
    captioner._engine_factory = fail_factory
    captioner.analyze(_window())


def test_default_factory_is_lazy_and_receives_local_model_path(monkeypatch):
    from agentflow.agents.engine import factory

    fake = FakeEngine(json.dumps(_valid_payload()))
    calls = []

    def fake_create(model_string, **kwargs):
        calls.append((model_string, kwargs))
        return fake

    monkeypatch.setattr(factory, "create_llm_engine", fake_create)
    captioner = TemporalCaptioner(model_path="/models/Qwen3-VL-8B-Instruct")
    assert calls == []

    captioner.analyze(_window())

    assert calls == [
        (
            "local-qwen3vl-/models/Qwen3-VL-8B-Instruct",
            {
                "is_multimodal": True,
                "use_cache": False,
                "debug_performance": False,
            },
        )
    ]


def test_rule_evidence_preserves_motion_oscillation():
    payload = _valid_payload()
    payload["overall_progress"] = "PROGRESSING"
    payload["errors"]["motion_oscillation"]["verdict"] = "ABSENT"
    fake = FakeEngine(json.dumps(payload))

    record = TemporalCaptioner(engine=fake).analyze(_window())

    assessment = record.errors.motion_oscillation
    assert assessment.verdict == "PRESENT"
    assert assessment.source == "FUSED"
    assert assessment.confidence >= 0.99
    assert assessment.evidence_timestamps_seconds == [5.0, 7.6, 10.0]
    assert "模型补充" not in assessment.reason
    assert record.overall_progress == "UNCERTAIN"
    assert record.phases[1].progress == "REGRESSING"


def test_commands_are_derived_from_actions_not_images():
    payload = _valid_payload()
    payload["phases"][0]["commanded_activity"] = "TURN_RIGHT"
    payload["phases"][1]["commanded_activity"] = "FORWARD"

    record = TemporalCaptioner(
        engine=FakeEngine(json.dumps(payload))
    ).analyze(_window())

    assert [phase.commanded_activity for phase in record.phases] == [
        "TURN_LEFT",
        "TURN_RIGHT",
    ]

    no_action_record = TemporalCaptioner(
        engine=FakeEngine(json.dumps(payload))
    ).analyze(replace(_window(), actions=()))
    assert [phase.commanded_activity for phase in no_action_record.phases] == [
        "NONE",
        "NONE",
    ]


def test_action_interval_overlaps_phase_that_starts_after_issue_time():
    payload = _valid_payload()
    payload["phases"][0]["interval"]["start_seconds"] = 7.5
    payload["phases"][0]["evidence_timestamps_seconds"] = [7.5, 7.6]

    record = TemporalCaptioner(
        engine=FakeEngine(json.dumps(payload))
    ).analyze(_window())

    assert record.phases[0].commanded_activity == "TURN_LEFT"


def test_low_confidence_absence_becomes_uncertain():
    payload = _valid_payload()
    for assessment in payload["errors"].values():
        assessment["confidence"] = 0.0
        assessment["source"] = "RULE"

    record = TemporalCaptioner(
        engine=FakeEngine(json.dumps(payload))
    ).analyze(_window())

    assert record.errors.collision.verdict == "UNCERTAIN"
    assert record.errors.repeated_visit.verdict == "UNCERTAIN"
    assert record.errors.get_nowhere.verdict == "UNCERTAIN"
    assert record.errors.collision.confidence == 0.5
    assert record.errors.collision.source == "MODEL"
    assert record.errors.motion_oscillation.verdict == "PRESENT"


def test_activity_summary_is_grounded_with_motion_boundaries():
    payload = _valid_payload()
    payload["activity_summary"] = "机器人执行FORWARD命令并前进。"

    record = TemporalCaptioner(
        engine=FakeEngine(json.dumps(payload, ensure_ascii=False))
    ).analyze(_window())

    assert "5.0–7.6秒相机向左转" in record.activity_summary
    assert "7.6–10.0秒相机向右转" in record.activity_summary
    assert "FORWARD" not in record.activity_summary
    assert "命令" not in record.activity_summary


def test_activity_drops_model_action_claim_without_reliable_motion():
    payload = _valid_payload()
    payload["activity_summary"] = "机器人执行FORWARD命令并前进。"
    payload["phases"][0]["interval"]["end_seconds"] = 7.5
    payload["phases"][0]["evidence_timestamps_seconds"] = [5.0, 7.5]
    payload["phases"][1]["interval"]["start_seconds"] = 7.5
    payload["phases"][1]["evidence_timestamps_seconds"] = [7.5, 10.0]
    window = replace(_window(), actions=(), motion=(), reverse_retrace_similarity=None)

    record = TemporalCaptioner(
        engine=FakeEngine(json.dumps(payload))
    ).analyze(window)

    assert record.activity_summary == "没有可靠的运动或动作证据"
    assert all(phase.commanded_activity == "NONE" for phase in record.phases)
    assert all(
        phase.observed_motion == CameraMotion.UNKNOWN
        for phase in record.phases
    )
    assert record.observed_motion_sequence == []


def test_phase_motion_is_derived_from_typed_motion_evidence():
    payload = _valid_payload()
    payload["phases"][0]["observed_motion"] = "TURN_RIGHT"
    payload["phases"][1]["observed_motion"] = "TURN_LEFT"

    record = TemporalCaptioner(
        engine=FakeEngine(json.dumps(payload))
    ).analyze(_window())

    assert [phase.observed_motion for phase in record.phases] == [
        CameraMotion.TURN_LEFT,
        CameraMotion.TURN_RIGHT,
    ]


def test_unreliable_typed_motion_is_not_written_as_activity_fact():
    payload = _valid_payload()
    unreliable = tuple(
        replace(signal, confidence=0.0) for signal in _window().motion
    )

    record = TemporalCaptioner(
        engine=FakeEngine(json.dumps(payload))
    ).analyze(replace(_window(), motion=unreliable))

    assert record.activity_summary == "5.0秒发出TURN_LEFT；7.6秒发出TURN_RIGHT；没有可靠的执行运动证据"
    assert all(
        phase.observed_motion == CameraMotion.UNKNOWN
        for phase in record.phases
    )
    assert record.observed_motion_sequence == []


def test_positive_progress_suppresses_hard_oscillation_override():
    payload = _valid_payload()
    payload["overall_progress"] = "PROGRESSING"
    payload["errors"]["motion_oscillation"].update(
        verdict="PRESENT",
        confidence=0.9,
        interval={"start_seconds": 5.0, "end_seconds": 10.0},
        evidence_timestamps_seconds=[5.0, 7.6, 10.0],
        reason="视觉上发生回扫。",
    )
    window = replace(
        _window(),
        progress=ProgressSignals(
            net_displacement_meters=2.0,
            new_landmarks_count=1,
            new_topological_nodes_count=1,
            completed_subgoals_count=1,
            no_progress_steps=0,
        ),
    )

    record = TemporalCaptioner(
        engine=FakeEngine(json.dumps(payload))
    ).analyze(window)

    assert record.errors.motion_oscillation.verdict == "UNCERTAIN"
    assert record.overall_progress == "PROGRESSING"


def test_typed_collision_false_conflicts_with_model_present():
    payload = _valid_payload()
    payload["errors"]["collision"].update(
        verdict="PRESENT",
        confidence=0.9,
        interval={"start_seconds": 5.0, "end_seconds": 10.0},
        evidence_timestamps_seconds=[5.0, 10.0],
        reason="模型认为发生碰撞。",
    )
    motion = tuple(
        replace(signal, collision=False) for signal in _window().motion
    )

    record = TemporalCaptioner(
        engine=FakeEngine(json.dumps(payload))
    ).analyze(replace(_window(), motion=motion))

    assert record.errors.collision.verdict == "UNCERTAIN"
    assert "结构化证据" in record.errors.collision.reason


def test_non_repeated_topology_conflicts_with_model_revisit():
    payload = _valid_payload()
    payload["errors"]["repeated_visit"].update(
        verdict="PRESENT",
        confidence=0.9,
        interval={"start_seconds": 5.0, "end_seconds": 9.5},
        evidence_timestamps_seconds=[5.0, 9.5],
        reason="模型认为重复访问。",
    )
    topology = (
        TopologySignal(5.0, node_id="a", visit_count=1),
        TopologySignal(9.5, node_id="b", visit_count=1),
    )

    record = TemporalCaptioner(
        engine=FakeEngine(json.dumps(payload))
    ).analyze(replace(_window(), topology=topology))

    assert record.errors.repeated_visit.verdict == "UNCERTAIN"


def test_completed_subgoal_conflicts_with_model_get_nowhere():
    payload = _valid_payload()
    payload["errors"]["get_nowhere"].update(
        verdict="PRESENT",
        confidence=0.9,
        interval={"start_seconds": 5.0, "end_seconds": 10.0},
        evidence_timestamps_seconds=[5.0, 10.0],
        reason="模型认为没有进展。",
    )
    window = replace(
        _window(),
        progress=ProgressSignals(completed_subgoals_count=1),
    )

    record = TemporalCaptioner(
        engine=FakeEngine(json.dumps(payload))
    ).analyze(window)

    assert record.errors.get_nowhere.verdict == "UNCERTAIN"
    assert record.overall_progress == "PROGRESSING"


def test_stationary_motion_conflicts_with_model_oscillation():
    payload = _valid_payload()
    payload["errors"]["motion_oscillation"].update(
        verdict="PRESENT",
        confidence=0.9,
        interval={"start_seconds": 5.0, "end_seconds": 10.0},
        evidence_timestamps_seconds=[5.0, 10.0],
        reason="模型认为发生振荡。",
    )
    stationary = (
        MotionSignal(
            5.0,
            10.0,
            CameraMotion.STATIONARY,
            scene_flow_magnitude_fraction=0.0,
            confidence=0.9,
            source="test",
        ),
    )

    record = TemporalCaptioner(
        engine=FakeEngine(json.dumps(payload))
    ).analyze(
        replace(
            _window(),
            motion=stationary,
            reverse_retrace_similarity=None,
        )
    )

    assert record.errors.motion_oscillation.verdict == "UNCERTAIN"
    assert all(
        phase.observed_motion == CameraMotion.STATIONARY
        for phase in record.phases
    )


def test_scene_fields_drop_dynamic_action_claims():
    payload = _valid_payload()
    payload["latest_scene"] = "机器人执行FORWARD并持续前进"
    payload["scene_summary"] = "相机前进，看到一张床"
    payload["phases"][0]["scene"] = "TURN_LEFT后看到条纹墙"
    payload["phases"][1]["scene"] = "机器人右转，可见蓝花墙"
    stationary = (
        MotionSignal(
            5.0,
            7.6,
            CameraMotion.STATIONARY,
            scene_flow_magnitude_fraction=0.0,
            confidence=0.9,
            source="test",
        ),
        MotionSignal(
            7.6,
            10.0,
            CameraMotion.STATIONARY,
            scene_flow_magnitude_fraction=0.0,
            confidence=0.9,
            source="test",
        ),
    )
    window = replace(
        _window(),
        actions=(),
        motion=stationary,
        reverse_retrace_similarity=None,
    )

    record = TemporalCaptioner(
        engine=FakeEngine(json.dumps(payload))
    ).analyze(window)

    scene_text = " ".join(
        [
            record.latest_scene,
            record.scene_summary,
            *(phase.scene for phase in record.phases),
        ]
    )
    assert "FORWARD" not in scene_text
    assert "前进" not in scene_text
    assert "TURN_LEFT" not in scene_text
    assert "右转" not in scene_text
    assert record.scene_summary == "看到一张床"
    assert record.phases[0].scene == "条纹墙"
    assert record.phases[1].scene == "可见蓝花墙"
    assert all(
        phase.observed_motion == CameraMotion.STATIONARY
        for phase in record.phases
    )


def test_scene_contract_drops_motion_synonyms_and_salvages_entities():
    payload = _valid_payload()
    payload["latest_scene"] = "相机扫向门口"
    payload["scene_summary"] = "机器人朝门口驶去，撞到墙后看到床"
    payload["phases"][0]["scene"] = "相机扫过条纹墙"
    payload["phases"][1]["scene"] = "机器人返回，视野中可见蓝花墙"
    stationary = (
        MotionSignal(
            5.0,
            7.6,
            CameraMotion.STATIONARY,
            scene_flow_magnitude_fraction=0.0,
            collision=False,
            confidence=0.9,
            source="test",
        ),
        MotionSignal(
            7.6,
            10.0,
            CameraMotion.STATIONARY,
            scene_flow_magnitude_fraction=0.0,
            collision=False,
            confidence=0.9,
            source="test",
        ),
    )

    record = TemporalCaptioner(
        engine=FakeEngine(json.dumps(payload))
    ).analyze(
        replace(
            _window(),
            actions=(),
            motion=stationary,
            reverse_retrace_similarity=None,
        )
    )

    scene_text = " ".join(
        [
            record.latest_scene,
            record.scene_summary,
            *(phase.scene for phase in record.phases),
        ]
    )
    for dynamic_term in (
        "相机",
        "机器人",
        "扫向",
        "驶去",
        "撞到",
        "返回",
        "视野",
    ):
        assert dynamic_term not in scene_text
    assert record.scene_summary == "床"
    assert record.phases[0].scene == "床"
    assert record.phases[1].scene == "蓝花墙"


def test_scene_filter_keeps_static_camera_relative_objects():
    captioner = TemporalCaptioner()

    assert (
        captioner._static_scene_text("画面显示前方有一扇木门")
        == "前方有一扇木门"
    )
    assert (
        captioner._static_scene_text("机器人前方有一扇木门")
        == "前方有一扇木门"
    )
    assert captioner._static_scene_text("相机扫向门口") is None
    assert captioner._static_scene_text("机器人朝门口驶去") is None
    assert captioner._static_scene_text("撞到墙后看到床") == "床"


def test_goal_distance_reduction_is_authoritative_progress():
    payload = _valid_payload()
    payload["overall_progress"] = "REGRESSING"
    for phase in payload["phases"]:
        phase["progress"] = "REGRESSING"
    topology = (
        TopologySignal(5.0, distance_to_goal_meters=10.0),
        TopologySignal(9.5, distance_to_goal_meters=2.0),
    )

    record = TemporalCaptioner(
        engine=FakeEngine(json.dumps(payload))
    ).analyze(replace(_window(), topology=topology))

    assert record.overall_progress == "PROGRESSING"
    assert all(phase.progress == "UNCERTAIN" for phase in record.phases)


def test_goal_distance_regression_overrides_nonzero_displacement():
    payload = _valid_payload()
    payload["overall_progress"] = "PROGRESSING"
    topology = (
        TopologySignal(5.0, distance_to_goal_meters=2.0),
        TopologySignal(9.5, distance_to_goal_meters=10.0),
    )
    window = replace(
        _window(),
        topology=topology,
        progress=ProgressSignals(net_displacement_meters=2.0),
    )

    record = TemporalCaptioner(
        engine=FakeEngine(json.dumps(payload))
    ).analyze(window)

    assert record.overall_progress == "REGRESSING"


def test_zero_displacement_does_not_leave_progressing_oscillation():
    payload = _valid_payload()
    payload["overall_progress"] = "PROGRESSING"
    for phase in payload["phases"]:
        phase["progress"] = "PROGRESSING"
    window = replace(
        _window(),
        progress=ProgressSignals(net_displacement_meters=0.0),
    )

    record = TemporalCaptioner(
        engine=FakeEngine(json.dumps(payload))
    ).analyze(window)

    assert record.errors.motion_oscillation.verdict == "PRESENT"
    assert record.overall_progress == "UNCERTAIN"


def test_cross_field_progress_conflict_is_downgraded():
    payload = _valid_payload()
    payload["overall_progress"] = "PROGRESSING"
    for phase in payload["phases"]:
        phase["progress"] = "STALLED"
    window = replace(_window(), motion=(), reverse_retrace_similarity=None)

    record = TemporalCaptioner(
        engine=FakeEngine(json.dumps(payload))
    ).analyze(window)

    assert record.overall_progress == "UNCERTAIN"


def test_completed_subgoal_overrides_model_stalled_overall():
    payload = _valid_payload()
    payload["overall_progress"] = "STALLED"
    for phase in payload["phases"]:
        phase["progress"] = "STALLED"
    window = replace(
        _window(),
        progress=ProgressSignals(completed_subgoals_count=1),
    )

    record = TemporalCaptioner(
        engine=FakeEngine(json.dumps(payload))
    ).analyze(window)

    assert record.overall_progress == "PROGRESSING"
    assert all(phase.progress == "UNCERTAIN" for phase in record.phases)


def test_rule_confidence_is_not_inflated_by_conflicting_model():
    payload = _valid_payload()
    payload["errors"]["motion_oscillation"].update(
        verdict="ABSENT",
        confidence=1.0,
        interval=None,
        evidence_timestamps_seconds=[],
        reason="模型认为没有振荡。",
    )

    record = TemporalCaptioner(
        engine=FakeEngine(json.dumps(payload))
    ).analyze(_window())

    assessment = record.errors.motion_oscillation
    assert assessment.verdict == "PRESENT"
    assert assessment.confidence == pytest.approx(0.99)
    assert assessment.source == "FUSED"


def test_partial_progress_metrics_do_not_prove_get_nowhere():
    payload = _valid_payload()
    payload["errors"]["get_nowhere"].update(
        verdict="PRESENT",
        confidence=0.4,
        interval={"start_seconds": 5.0, "end_seconds": 10.0},
        evidence_timestamps_seconds=[5.0, 10.0],
        reason="可能没有进展。",
    )
    window = replace(
        _window(),
        progress=ProgressSignals(
            net_displacement_meters=0.0,
            new_landmarks_count=0,
            no_progress_steps=10,
        ),
    )

    record = TemporalCaptioner(
        engine=FakeEngine(json.dumps(payload))
    ).analyze(window)

    assert record.errors.get_nowhere.verdict == "UNCERTAIN"


def test_low_confidence_turns_do_not_trigger_oscillation_rule():
    payload = _valid_payload()
    low_confidence_motion = tuple(
        replace(signal, confidence=0.4) for signal in _window().motion
    )
    window = replace(_window(), motion=low_confidence_motion)

    record = TemporalCaptioner(
        engine=FakeEngine(json.dumps(payload))
    ).analyze(window)

    assert record.errors.motion_oscillation.verdict != "PRESENT"


def test_non_monotonic_frames_fail_before_inference():
    fake = FakeEngine(json.dumps(_valid_payload()))
    window = TemporalWindow(
        start_seconds=5.0,
        end_seconds=10.0,
        frames=(
            TimestampedFrame(7.5, _png_bytes(1)),
            TimestampedFrame(5.0, _png_bytes(2)),
        ),
    )

    with pytest.raises(TemporalInputError, match="strictly increasing"):
        TemporalCaptioner(engine=fake).analyze(window)

    assert fake.calls == []


def test_window_is_half_open():
    fake = FakeEngine(json.dumps(_valid_payload()))
    window = TemporalWindow(
        start_seconds=5.0,
        end_seconds=10.0,
        frames=(
            TimestampedFrame(5.0, _png_bytes(1)),
            TimestampedFrame(10.0, _png_bytes(2)),
        ),
    )

    with pytest.raises(TemporalInputError, match="outside"):
        TemporalCaptioner(engine=fake).analyze(window)


def test_uniform_cap_keeps_first_and_last():
    values = list(range(10))
    assert TemporalCaptioner._uniform_cap(values, 4) == [0, 3, 6, 9]
    assert TemporalCaptioner._uniform_cap(values, 12) == values


def test_video_sampler_preserves_absolute_half_open_times(tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    path = tmp_path / "timeline.avi"
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (32, 24),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV MJPG writer unavailable")
    for frame_index in range(100):
        writer.write(np.full((24, 32, 3), frame_index, dtype=np.uint8))
    writer.release()

    captioner = TemporalCaptioner()
    frames = captioner.sample_video_frames(
        path,
        start_seconds=5.0,
        end_seconds=10.0,
        fps=2.0,
        max_frames=12,
    )
    capped = captioner.sample_video_frames(
        path,
        start_seconds=5.0,
        end_seconds=10.0,
        fps=2.0,
        max_frames=4,
    )

    assert [frame.timestamp_seconds for frame in frames] == [
        5.0,
        5.5,
        6.0,
        6.5,
        7.0,
        7.5,
        8.0,
        8.5,
        9.0,
        9.5,
    ]
    assert [frame.timestamp_seconds for frame in capped] == [5.0, 6.5, 8.0, 9.5]


def test_sparse_frames_are_not_used_as_reliable_flow():
    np = pytest.importorskip("numpy")
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    frames = (
        TimestampedFrame(5.0, image),
        TimestampedFrame(5.5, image),
    )

    motion = TemporalCaptioner().estimate_motion_signals(frames)

    assert len(motion) == 1
    assert motion[0].camera_motion == CameraMotion.UNKNOWN
    assert motion[0].confidence == 0


def test_invalid_json_and_window_external_phase_are_rejected():
    invalid_json = TemporalCaptioner(engine=FakeEngine("not json"))
    with pytest.raises(TemporalOutputError, match="invalid structured JSON"):
        invalid_json.analyze(_window())
    assert invalid_json.last_raw_response == "not json"

    payload = _valid_payload()
    payload["phases"][0]["interval"]["start_seconds"] = 4.5
    invalid_time = TemporalCaptioner(engine=FakeEngine(json.dumps(payload)))
    with pytest.raises(TemporalOutputError, match="outside"):
        invalid_time.analyze(_window())


def test_memory_text_contains_activity_and_confirmed_error():
    fake = FakeEngine(json.dumps(_valid_payload(), ensure_ascii=False))
    record = TemporalCaptioner(engine=fake).analyze(_window())

    text = record.to_memory_text()
    assert "5.0–7.6秒相机向左转" in text
    assert "7.6–10.0秒相机向右转" in text
    assert "motion_oscillation" in text
    assert "5.000-10.000s" in text
    assert record.latency_budget_ms == 5000.0
    assert record.latency_budget_met is True


def test_foundation_model_timing_is_weighted_and_resettable():
    captioner = TemporalCaptioner(
        engine=FakeEngine(json.dumps(_valid_payload(), ensure_ascii=False))
    )

    captioner.analyze(_window())
    captioner.analyze(_window())

    timing = captioner.performance_summary()
    assert timing["inference_count"] == 2
    assert timing["success_count"] == 2
    assert timing["failure_count"] == 0
    assert timing["total_inference_ms"] >= 0
    assert timing["average_inference_ms"] == pytest.approx(
        timing["total_inference_ms"] / 2
    )
    captioner.reset_performance_stats()
    assert captioner.performance_summary()["inference_count"] == 0


def test_error_assessment_rejects_invalid_confidence():
    with pytest.raises(Exception):
        ErrorAssessment(
            verdict="UNCERTAIN",
            confidence=1.5,
            reason="bad",
        )


def test_632_motion_regression_when_video_is_available():
    video = Path("/data/pengyh/workspace/FreeAskAgent_R2R/videos/632.mp4")
    if not video.is_file():
        pytest.skip("632 integration video unavailable")
    cv2 = pytest.importorskip("cv2")
    capture = cv2.VideoCapture(str(video))
    try:
        native_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        capture.release()
    duration = total_frames / native_fps if native_fps > 0 else 0.0
    if duration < 10.0:
        pytest.skip(
            f"632 video metadata exposes only {duration:.3f}s; need >=10s"
        )
    captioner = TemporalCaptioner()
    frames = captioner.sample_video_frames(
        video,
        start_seconds=5.0,
        end_seconds=10.0,
        fps=10.0,
        max_frames=60,
        crop=(0, 0, 272, 256),
    )
    motion = captioner.estimate_motion_signals(frames)
    retrace = captioner.reverse_retrace_similarity(frames, motion)

    reliable_turns = [
        signal.camera_motion
        for signal in motion
        if signal.camera_motion
        in {CameraMotion.TURN_LEFT, CameraMotion.TURN_RIGHT}
    ]
    assert reliable_turns == [CameraMotion.TURN_LEFT, CameraMotion.TURN_RIGHT]
    assert motion[0].end_seconds == pytest.approx(7.6, abs=0.3)
    assert retrace is not None and retrace > 0.98
