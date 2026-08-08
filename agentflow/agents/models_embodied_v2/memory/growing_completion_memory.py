"""Growing temporal evidence and motion-grounded completion guards."""

from __future__ import annotations

from collections import deque
from dataclasses import replace
import json
import re
from typing import Any, Deque, Optional

from agentflow.agents.models_embodied_v2.TemporalCaptioner import (
    TemporalCaptioner,
)
from agentflow.agents.models_embodied_v2.data_models import (
    CaptionResult,
    MemoryFrame,
    TemporalFrameInput,
    TemporalMemoryConfig,
)
from agentflow.agents.models_embodied_v2.memory.task_memory import TaskMemory
from agentflow.agents.models_embodied_v2.memory.temporal_memory import (
    TemporalMemory,
    TemporalStateError,
)
from agentflow.agents.vln_agent_3_protocol import (
    DECISION_POINT_MIN_PATH_M,
    DECISION_POINT_TURN_EVIDENCE_DEG,
    DOORWAY_LANDMARK_CONFIDENCE,
    DOORWAY_MIN_PATH_M,
    MAX_COMPLETION_EVIDENCE_FRAMES,
    RECENT_COMPLETION_EVIDENCE_FRAMES,
    TURN_ALIGNMENT_DEG,
    LandmarkOutput as _LandmarkOutput,
)


class GrowingCompletionMemory(TemporalMemory):
    """Retain full history and send bounded evidence to completion."""

    def __init__(
        self,
        *,
        captioner: TemporalCaptioner,
        task_memory: TaskMemory,
    ) -> None:
        super().__init__(
            captioner=captioner,
            task_memory=task_memory,
            config=TemporalMemoryConfig(enable_error_detection=True),
        )
        # The base class uses deque(maxlen=8). Keep the active subgoal's full
        # history here, but select a bounded evidence set for each model call.
        self._frames: Deque[MemoryFrame] = deque()
        self._pending_translation_m = 0.0
        self._pending_yaw_delta_deg = 0.0
        self._pending_landmark = self._unknown_landmark("tracker not run")
        self._last_completion_frame_ids: tuple[int, ...] = ()
        self._last_completion_guard: Optional[str] = None

    def reset(self) -> None:
        super().reset()
        self._last_completion_guard = None

    @staticmethod
    def _unknown_landmark(evidence: str) -> _LandmarkOutput:
        return _LandmarkOutput(
            visible=False,
            direction="UNKNOWN",
            proximity="UNKNOWN",
            passed=False,
            confidence=0.0,
            evidence=evidence,
        )

    def set_motion_evidence(
        self,
        *,
        translation_m: float,
        yaw_delta_deg: float,
    ) -> None:
        """Attach measured motion to the next RGB observation."""
        self._pending_translation_m = float(translation_m)
        self._pending_yaw_delta_deg = float(yaw_delta_deg)

    def set_landmark_evidence(
        self,
        landmark: _LandmarkOutput,
    ) -> None:
        """Attach a validated landmark judgement to the next observation."""
        if not isinstance(landmark, _LandmarkOutput):
            raise TypeError("landmark must be a validated _LandmarkOutput")
        self._pending_landmark = landmark

    def append_observation(self, image: Any) -> MemoryFrame:
        frame = super().append_observation(image)
        annotated = replace(
            frame,
            translation_m=self._pending_translation_m,
            yaw_delta_deg=self._pending_yaw_delta_deg,
            subgoal_path_length_m=(
                (
                    self._frames[-2].subgoal_path_length_m
                    if len(self._frames) > 1
                    else 0.0
                )
                + self._pending_translation_m
            ),
            landmark_visible=self._pending_landmark.visible,
            landmark_direction=self._pending_landmark.direction,
            landmark_proximity=self._pending_landmark.proximity,
            landmark_passed=self._pending_landmark.passed,
            landmark_confidence=self._pending_landmark.confidence,
            landmark_evidence=self._pending_landmark.evidence,
        )
        self._frames[-1] = annotated
        self._pending_translation_m = 0.0
        self._pending_yaw_delta_deg = 0.0
        self._pending_landmark = self._unknown_landmark("tracker not run")
        return annotated

    @staticmethod
    def _landmark_signature(frame: MemoryFrame) -> tuple[Any, ...]:
        return (
            frame.landmark_visible,
            frame.landmark_direction,
            frame.landmark_proximity,
            frame.landmark_passed,
        )

    def _select_completion_frames(self) -> tuple[MemoryFrame, ...]:
        """Keep first, state changes, uniform history, and recent frames."""
        frames = tuple(self._frames)
        if len(frames) <= MAX_COMPLETION_EVIDENCE_FRAMES:
            return frames

        recent_start = (
            len(frames) - RECENT_COMPLETION_EVIDENCE_FRAMES
        )
        selected = {0, *range(recent_start, len(frames))}

        transitions: list[int] = []
        for index in range(1, len(frames)):
            if (
                self._landmark_signature(frames[index - 1])
                != self._landmark_signature(frames[index])
            ):
                transitions.extend((index - 1, index))
        for index in reversed(transitions):
            if len(selected) >= MAX_COMPLETION_EVIDENCE_FRAMES:
                break
            selected.add(index)

        candidates = [
            index
            for index in range(1, recent_start)
            if index not in selected
        ]
        slots = MAX_COMPLETION_EVIDENCE_FRAMES - len(selected)
        if slots > 0 and candidates:
            if len(candidates) <= slots:
                selected.update(candidates)
            elif slots == 1:
                selected.add(candidates[len(candidates) // 2])
            else:
                positions = {
                    round(step * (len(candidates) - 1) / (slots - 1))
                    for step in range(slots)
                }
                selected.update(candidates[position] for position in positions)

        return tuple(frames[index] for index in sorted(selected))

    @staticmethod
    def _temporal_input(frame: MemoryFrame) -> TemporalFrameInput:
        return TemporalFrameInput(
            frame.frame_id,
            frame.image,
            translation_m=frame.translation_m,
            yaw_delta_deg=frame.yaw_delta_deg,
            subgoal_path_length_m=frame.subgoal_path_length_m,
            landmark_visible=frame.landmark_visible,
            landmark_direction=frame.landmark_direction,
            landmark_proximity=frame.landmark_proximity,
            landmark_passed=frame.landmark_passed,
            landmark_confidence=frame.landmark_confidence,
            landmark_evidence=frame.landmark_evidence,
        )

    def analyze(self) -> CaptionResult:
        self._sync_task_state()
        if self._subgoal is None:
            raise TemporalStateError("current subgoal is not set")
        if not self._frames:
            raise TemporalStateError("at least one frame is required")

        selected_frames = self._select_completion_frames()
        completion_frames = tuple(
            self._temporal_input(frame) for frame in selected_frames
        )
        self._last_completion_frame_ids = tuple(
            frame.frame_id for frame in selected_frames
        )
        dual_result = self.captioner.analyze_dual_window(
            subgoal=self._subgoal,
            completion_frames=completion_frames,
            error_frames=completion_frames[-8:],
        )
        completed = dual_result.completed
        self._last_completion_guard = None
        doorway_stage = self._is_doorway_stage()
        if doorway_stage:
            grounded, guard = self._motion_proves_doorway_completion()
            if grounded:
                completed = True
                self._last_completion_guard = guard
            elif completed:
                path_m = self._frames[-1].subgoal_path_length_m
                if path_m >= 0.75:
                    self._last_completion_guard = (
                        "accepted temporal doorway completion after "
                        f"{path_m:.2f}m measured progress"
                    )
                else:
                    completed = False
                    self._last_completion_guard = (
                        "rejected doorway completion before sufficient "
                        f"measured progress: path={path_m:.2f}m"
                    )
        if not completed and self._last_completion_guard is None:
            completed, self._last_completion_guard = (
                self._motion_proves_decision_point_completion()
            )
        if not completed and self._last_completion_guard is None:
            completed, self._last_completion_guard = (
                self._motion_proves_turn_completion()
            )
        raw_payload = {
            "completion": dual_result.completion_raw_response,
            "error": dual_result.error_raw_response,
        }
        if self._last_completion_guard is not None:
            raw_payload["completion_guard"] = (
                self._last_completion_guard
            )
        result = CaptionResult(
            subgoal_id=dual_result.subgoal_id,
            completed=completed,
            error=dual_result.error,
            error_mode=dual_result.error_mode,
            raw_response=json.dumps(raw_payload, ensure_ascii=False),
            latency_ms=dual_result.latency_ms,
            error_confidence=dual_result.error_confidence,
            error_evidence=dual_result.error_evidence,
        )
        return self._store(result)

    def _is_doorway_stage(self) -> bool:
        assert self._subgoal is not None
        description = (
            f"{self._subgoal.description} "
            f"{self._subgoal.completion_criteria}"
        )
        return bool(
            re.search(
                r"\b(?:exit|leave|doorway|threshold|cross(?:ed|ing)?)\b",
                description,
                flags=re.IGNORECASE,
            )
        )

    def _motion_proves_doorway_completion(
        self,
    ) -> tuple[bool, Optional[str]]:
        """Close an exit stage only with tracker and measured crossing proof."""
        if not self._is_doorway_stage():
            return False, None
        if (
            not self._frames
            or self._frames[-1].subgoal_path_length_m
            < DOORWAY_MIN_PATH_M
        ):
            return False, None

        for frame in reversed(self._frames):
            if (
                frame.landmark_passed
                and frame.landmark_confidence
                >= DOORWAY_LANDMARK_CONFIDENCE
            ):
                return (
                    True,
                    (
                        "motion-grounded doorway crossing: tracker passed "
                        f"with confidence={frame.landmark_confidence:.2f} "
                        f"after path={frame.subgoal_path_length_m:.2f}m"
                    ),
                )

        # The tracker can miss the exact crossing frame. Accept an ordered
        # near-threshold approach followed by substantial, mostly forward
        # measured motion, even if the doorway is no longer classified.
        frames = tuple(self._frames)
        for index, approach in enumerate(frames[:-1]):
            if (
                not approach.landmark_visible
                or approach.landmark_proximity not in {"NEAR", "AT"}
                or approach.landmark_confidence
                < DOORWAY_LANDMARK_CONFIDENCE
            ):
                continue
            later = frames[index + 1 :]
            translation = sum(frame.translation_m for frame in later)
            yaw = sum(abs(frame.yaw_delta_deg) for frame in later)
            if translation >= 0.75 and yaw <= 45.0:
                return (
                    True,
                    (
                        "motion-grounded doorway crossing fallback: "
                        f"advanced {translation:.2f}m after a verified "
                        f"{approach.landmark_proximity.lower()} threshold "
                        f"with {yaw:.1f}deg accumulated yaw"
                    ),
                )

        return False, None

    def _motion_proves_decision_point_completion(
        self,
    ) -> tuple[bool, Optional[str]]:
        """Close a pre-turn stage after measured execution of its next turn."""
        assert self._subgoal is not None
        match = re.search(
            r"\bnext\s+(left|right)-turn\s+decision\s+point\b",
            self._subgoal.description,
            flags=re.IGNORECASE,
        )
        if match is None or not self._frames:
            return False, None
        direction = match.group(1).lower()
        for frame in reversed(self._frames):
            if frame.subgoal_path_length_m < DECISION_POINT_MIN_PATH_M:
                continue
            yaw = frame.yaw_delta_deg
            matching_turn = (
                yaw <= -DECISION_POINT_TURN_EVIDENCE_DEG
                if direction == "left"
                else yaw >= DECISION_POINT_TURN_EVIDENCE_DEG
            )
            if matching_turn:
                return (
                    True,
                    (
                        "motion-grounded decision point: "
                        f"path={frame.subgoal_path_length_m:.2f}m and "
                        f"measured {direction} yaw={yaw:+.1f}deg"
                    ),
                )
        return False, None

    def _motion_proves_turn_completion(
        self,
    ) -> tuple[bool, Optional[str]]:
        """Close a pure turn stage once its measured net rotation is enough."""
        assert self._subgoal is not None
        if re.search(
            r"\bnext\s+(?:left|right)-turn\s+decision\s+point\b",
            self._subgoal.description,
            flags=re.IGNORECASE,
        ):
            return False, None
        match = re.search(
            r"\bturn\s+(left|right)\b",
            self._subgoal.description,
            flags=re.IGNORECASE,
        )
        if match is None:
            return False, None
        direction = match.group(1).lower()
        net_yaw = sum(frame.yaw_delta_deg for frame in self._frames)
        turn_progress = -net_yaw if direction == "left" else net_yaw
        if turn_progress < TURN_ALIGNMENT_DEG:
            return False, None
        return (
            True,
            (
                "motion-grounded turn alignment: "
                f"measured net {direction} yaw={turn_progress:.1f}deg"
            ),
        )

    def context(self) -> str:
        self._sync_task_state()
        if self._latest_result is not None:
            return self._latest_result.to_memory_text()
        return (
            f"Completion history: {len(self._frames)} frames; selected "
            f"evidence: "
            f"{min(len(self._frames), MAX_COMPLETION_EVIDENCE_FRAMES)}; "
            f"error window: {min(len(self._frames), 8)}/8 frames"
        )

    def diagnostics(
        self,
        *,
        include_raw_response: bool = False,
    ) -> dict[str, Any]:
        values = super().diagnostics(
            include_raw_response=include_raw_response
        )
        values.update(
            {
                "completion_history_size": len(self._frames),
                "completion_window_size": min(
                    len(self._frames), MAX_COMPLETION_EVIDENCE_FRAMES
                ),
                "completion_frame_ids": list(
                    self._last_completion_frame_ids
                    if self._frames
                    else ()
                ),
                "error_window_size": min(len(self._frames), 8),
                "error_detection_enabled": True,
                "completion_guard": self._last_completion_guard,
                "recent_motion": [
                    {
                        "frame_id": frame.frame_id,
                        "translation_m": frame.translation_m,
                        "yaw_delta_deg": frame.yaw_delta_deg,
                        "subgoal_path_length_m": (
                            frame.subgoal_path_length_m
                        ),
                    }
                    for frame in tuple(self._frames)[-8:]
                ],
            }
        )
        return values
