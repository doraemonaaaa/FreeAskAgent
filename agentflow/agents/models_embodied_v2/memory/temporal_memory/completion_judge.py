"""Bounded temporal evidence and model-owned completion judgement."""

from __future__ import annotations

from collections import deque
from dataclasses import replace
import json
from typing import Any, Deque, Optional

from agentflow.agents.models_embodied_v2.data_models import (
    CaptionResult,
    MemoryFrame,
    PreviewSelection,
    SceneAnalysisRequest,
    SceneAnalysisResult,
    TemporalFrameInput,
    TemporalMemoryConfig,
)
from .interfaces import TaskMemoryPort, TemporalCaptionerPort
from .temporal_memory import TemporalStateError
from .preview_store import PreviewStore
from agentflow.agents.models_embodied_v2.skiils.preview import (
    PreviewSelector,
    UnimplementedPreviewSelector,
)
from agentflow.agents.models_embodied_v2.skiils.protocol import (
    MAX_COMPLETION_EVIDENCE_FRAMES,
    LandmarkOutput as _LandmarkOutput,
)


class CompletionMemoryMixin:
    """Keep bounded evidence and publish the Captioner's completion result."""

    def __init__(
        self,
        *,
        captioner: TemporalCaptionerPort,
        task_memory: TaskMemoryPort,
        preview_selector: Optional[PreviewSelector] = None,
        config: Optional[TemporalMemoryConfig] = None,
    ) -> None:
        super().__init__(
            captioner=captioner,
            task_memory=task_memory,
            config=config or TemporalMemoryConfig(enable_error_detection=True),
        )
        # RGB history is intentionally bounded. Path length and completion
        # votes are scalar state, so a long subgoal cannot make later steps
        # scan or retain every source frame.
        self._frames: Deque[MemoryFrame] = deque(
            maxlen=MAX_COMPLETION_EVIDENCE_FRAMES
        )
        self._pending_translation_m = 0.0
        self._pending_yaw_delta_deg = 0.0
        self._pending_landmark = self._unknown_landmark("tracker not run")
        self._subgoal_path_length_m = 0.0
        self._latest_scene: Optional[SceneAnalysisResult] = None
        self._last_completion_frame_ids: tuple[int, ...] = ()
        self._last_completion_guard: Optional[str] = None
        self._doorway_approach_seen = False
        self._doorway_target_distance_m: Optional[float] = None
        self._final_target_at_streak = 0
        if preview_selector is not None:
            self.preview_selector: PreviewSelector = preview_selector
        elif hasattr(captioner, "select"):
            # Production TemporalCaptioner implements both ports with one
            # shared engine. Lightweight test fakes can omit preview support.
            self.preview_selector = captioner
        else:
            self.preview_selector = UnimplementedPreviewSelector()
        self._preview_store = PreviewStore()

    @property
    def _preview_views(self) -> tuple[Any, ...]:
        return self._preview_store.views

    @_preview_views.setter
    def _preview_views(self, value: tuple[Any, ...]) -> None:
        self._preview_store.views = value

    @property
    def _preview_selection(self) -> Optional[PreviewSelection]:
        return self._preview_store.selection

    @_preview_selection.setter
    def _preview_selection(self, value: Optional[PreviewSelection]) -> None:
        self._preview_store.selection = value

    @property
    def _preview_error(self) -> Optional[str]:
        return self._preview_store.error

    @_preview_error.setter
    def _preview_error(self, value: Optional[str]) -> None:
        self._preview_store.error = value

    def reset(self) -> None:
        super().reset()
        self._last_completion_guard = None
        if hasattr(self, "_subgoal_path_length_m"):
            self._subgoal_path_length_m = 0.0
        if hasattr(self, "_latest_scene"):
            self._latest_scene = None
        if hasattr(self, "_doorway_approach_seen"):
            self._doorway_approach_seen = False
        if hasattr(self, "_doorway_target_distance_m"):
            self._doorway_target_distance_m = None
        if hasattr(self, "_final_target_at_streak"):
            self._final_target_at_streak = 0
        if hasattr(self, "_pending_translation_m"):
            self._pending_translation_m = 0.0
            self._pending_yaw_delta_deg = 0.0
            self._pending_landmark = self._unknown_landmark("tracker not run")
        self.clear_preview()

    def _sync_task_state(self) -> None:
        old_id = self._subgoal.subgoal_id if self._subgoal else None
        super()._sync_task_state()
        new_id = self._subgoal.subgoal_id if self._subgoal else None
        if old_id != new_id:
            self._subgoal_path_length_m = 0.0
            self._latest_scene = None
            self._doorway_approach_seen = False
            self._doorway_target_distance_m = None
            self._final_target_at_streak = 0
            self._pending_translation_m = 0.0
            self._pending_yaw_delta_deg = 0.0
            self._pending_landmark = self._unknown_landmark("tracker not run")

    def set_preview_views(self, views: Any) -> None:
        """Hold the surrounding views a PREVIEW asked for, and judge them.

        These are simultaneous views from one standing position, not a temporal
        sequence, so they are deliberately kept out of ``_frames``: adding
        several images for a single step would corrupt both the measured path
        length and the completion evidence.

        Selection runs here, on arrival, because that is the moment the
        Captioner has everything it needs. A selector that declines or raises
        leaves the selection unset rather than guessing a heading, so the
        actor's fallback stays distinguishable from a real judgement.
        """
        # Synchronised first: an episode rollover inside ``_sync_task_state``
        # calls ``reset``, which clears the preview slot, and storing before
        # that would silently drop the views we were just handed.
        self._sync_task_state()
        self._preview_views = tuple(views)
        self._preview_selection = None
        self._preview_error = None
        if not self._preview_views:
            return
        try:
            selection = self.preview_selector.select(
                subgoal=self._subgoal,
                views=self._preview_views,
            )
        except Exception as exc:
            self._preview_error = f"{type(exc).__name__}: {exc}"
            return
        if selection is not None:
            self.set_preview_selection(selection)

    def preview_views(self) -> tuple[Any, ...]:
        """Return the views held for this step, empty when none were taken."""
        return self._preview_views

    def set_preview_selection(self, selection: PreviewSelection) -> None:
        """Record which held view the actor should act on.

        The index is bounded against the held views here because a waypoint
        back-projected through the wrong view's depth and camera transform
        yields a plausible world coordinate pointing the wrong way.
        """
        if not isinstance(selection, PreviewSelection):
            raise TypeError("selection must be a PreviewSelection")
        if not self._preview_views:
            raise TemporalStateError("no preview views are held")
        if selection.view_index >= len(self._preview_views):
            raise TemporalStateError(
                "view_index {} addresses no held view (have {})".format(
                    selection.view_index, len(self._preview_views)
                )
            )
        self._preview_selection = selection

    def preview_selection(self) -> Optional[PreviewSelection]:
        """Return the Captioner's choice, or None while it has not answered."""
        return self._preview_selection

    def last_preview_error(self) -> Optional[str]:
        """Return why the selector failed on this step, if it did."""
        return self._preview_error

    def clear_preview(self) -> None:
        if hasattr(self, "_preview_store"):
            self._preview_store.clear()

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

    def set_doorway_target_distance(
        self,
        distance_m: Optional[float],
    ) -> None:
        """Attach distance to the model-localized structural doorway.

        ``None`` means that the actor has not localized a stable doorway for
        this subgoal. This is geometric confirmation of a model-selected
        object, not accumulated path length: walking around inside the source
        room cannot satisfy it unless the camera actually reaches that target.
        """
        if distance_m is None:
            self._doorway_target_distance_m = None
            return
        value = float(distance_m)
        if value < 0.0:
            raise ValueError("doorway target distance must not be negative")
        self._doorway_target_distance_m = value

    def append_observation(self, image: Any) -> MemoryFrame:
        frame = super().append_observation(image)
        self._subgoal_path_length_m += self._pending_translation_m
        annotated = replace(
            frame,
            translation_m=self._pending_translation_m,
            yaw_delta_deg=self._pending_yaw_delta_deg,
            subgoal_path_length_m=self._subgoal_path_length_m,
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
        """Return the already bounded active-subgoal evidence window."""
        return tuple(self._frames)

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

        # Lightweight legacy fakes remain usable outside the production agent.
        if not hasattr(self.captioner, "analyze_scene"):
            return super().analyze()

        selected_frames = self._select_completion_frames()
        completion_frames = tuple(
            self._temporal_input(frame) for frame in selected_frames
        )
        self._last_completion_frame_ids = tuple(
            frame.frame_id for frame in selected_frames
        )
        is_final = bool(
            getattr(self.task_memory, "is_current_subgoal_final", lambda: False)()
        )
        scene = self.captioner.analyze_scene(
            SceneAnalysisRequest(
                subgoal=self._subgoal,
                frames=completion_frames,
                is_final_subgoal=is_final,
            )
        )
        self._latest_scene = scene
        self._annotate_latest_landmark(scene)

        final_at = bool(
            is_final
            and scene.completed
            and scene.final_target.visible
            and scene.final_target.proximity == "AT"
            and scene.final_target.confidence >= 0.60
        )
        self._final_target_at_streak = (
            self._final_target_at_streak + 1 if final_at else 0
        )

        is_doorway = scene.door_state != "NOT_APPLICABLE"
        approached_before = self._doorway_approach_seen
        if (
            scene.door_state in {"APPROACHING", "AT_THRESHOLD", "CROSSING"}
            or scene.door_camera_side in {"BEFORE_DOOR", "AT_DOOR"}
        ):
            self._doorway_approach_seen = True
        completed = bool(
            scene.completed and scene.completion_confidence >= 0.60
        )
        reached_model_doorway = bool(
            self._doorway_target_distance_m is not None
            and self._doorway_target_distance_m <= 0.35
        )
        if (
            completed
            and is_doorway
            and not (approached_before or reached_model_doorway)
        ):
            completed = False
            self._last_completion_guard = (
                "rejected doorway completion: camera has not reached the "
                "model-localized structural doorway and no earlier model "
                "observation placed it before/at the doorway"
            )
        elif completed and is_final and self._final_target_at_streak < 2:
            completed = False
            self._last_completion_guard = (
                "rejected final completion: model-owned AT evidence must "
                "remain stable for two consecutive observations"
            )
        else:
            self._last_completion_guard = (
                "accepted model-owned temporal completion"
                if completed
                else "model reports active subgoal is not complete"
            )
        raw_payload = {
            "scene": scene.raw_response,
        }
        if self._last_completion_guard is not None:
            raw_payload["completion_guard"] = (
                self._last_completion_guard
            )
        result = CaptionResult(
            subgoal_id=scene.subgoal_id,
            completed=completed,
            error=scene.error,
            error_mode=scene.error_mode,
            raw_response=json.dumps(raw_payload, ensure_ascii=False),
            latency_ms=scene.latency_ms,
            error_confidence=scene.error_confidence,
            error_evidence=scene.error_evidence,
            completion_confidence=scene.completion_confidence,
            completion_evidence=scene.completion_evidence,
            door_state=scene.door_state,
            door_camera_side=scene.door_camera_side,
            landmark=scene.landmark,
            final_target=scene.final_target,
        )
        return self._store(result)

    def _annotate_latest_landmark(self, scene: SceneAnalysisResult) -> None:
        landmark = scene.landmark
        self._frames[-1] = replace(
            self._frames[-1],
            landmark_visible=landmark.visible,
            landmark_direction=landmark.direction,
            landmark_proximity=landmark.proximity,
            landmark_passed=landmark.passed,
            landmark_confidence=landmark.confidence,
            landmark_evidence=landmark.evidence,
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
                "doorway_target_distance_m": (
                    self._doorway_target_distance_m
                ),
                "final_target_at_streak": self._final_target_at_streak,
                # How the preview seam behaved this step: whether views were
                # held at all, and whether the selector answered for them.
                "preview_view_count": len(self._preview_views),
                "preview_selected_view": (
                    None
                    if self._preview_selection is None
                    else self._preview_selection.view_index
                ),
                "preview_error": self._preview_error,
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
