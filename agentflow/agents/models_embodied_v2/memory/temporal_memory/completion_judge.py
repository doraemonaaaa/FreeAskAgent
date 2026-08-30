"""Bounded temporal evidence and model-owned completion judgement."""

from __future__ import annotations

from collections import deque
from dataclasses import replace
import json
import re
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
from agentflow.agents.models_embodied_v2.data_models import Subgoal
from .interfaces import TaskMemoryPort, TemporalCaptionerPort
from .temporal_memory import TemporalStateError
from .preview_store import PreviewStore
from agentflow.agents.models_embodied_v2.skiils.preview import (
    PreviewSelector,
    UnimplementedPreviewSelector,
)
from agentflow.agents.models_embodied_v2.skiils.protocol import (
    DOORWAY_CROSSED_MIN_PATH_M,
    DOORWAY_CROSSED_STREAK_ACCEPT,
    COMMITTED_TARGET_REACHED_M,
    LANDMARK_RANGE_VETO_M,
    STAGE_SKIP_AT_STREAK,
    STAGE_SKIP_MAX_REMAINING_STAGES,
    STAGE_SKIP_MIN_CONFIDENCE,
    STAGE_SKIP_MIN_TASK_PATH_M,
    STAGE_SKIP_STALL_OBSERVATIONS,
    STAIRS_LEVEL_TOLERANCE_M,
    STAIRS_MIN_RISE_M,
    TURN_AROUND_MIN_PROGRESS_DEG,
    TURN_AROUND_PATTERN,
    TURN_MIN_PROGRESS_DEG,
    TURN_TARGET_CENTRED_U,
    TURN_TARGET_MIN_PROGRESS_DEG,
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
        self._doorway_crossed_streak = 0
        self._doorway_reached = False
        self._doorway_reach_tolerance_m = COMMITTED_TARGET_REACHED_M
        self._committed_target_seen = False
        self._turn_progress_deg = 0.0
        self._elevation_history: Deque[float] = deque(maxlen=3)
        self._final_target_at_streak = 0
        self._destination_at_streak = 0
        self._subgoal_observations = 0
        self._task_path_length_m = 0.0
        self._last_stage_skip: Optional[dict[str, Any]] = None
        self._stop_proposed_last_step = False
        self._depth_m: Any = None
        self._last_landmark_range_m: Optional[float] = None
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
        if hasattr(self, "_doorway_crossed_streak"):
            self._doorway_crossed_streak = 0
            self._doorway_reached = False
            self._doorway_reach_tolerance_m = COMMITTED_TARGET_REACHED_M
            self._committed_target_seen = False
            self._turn_progress_deg = 0.0
            self._elevation_history.clear()
        if hasattr(self, "_final_target_at_streak"):
            self._final_target_at_streak = 0
            self._destination_at_streak = 0
            self._subgoal_observations = 0
            self._task_path_length_m = 0.0
            self._last_stage_skip = None
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
            self._doorway_crossed_streak = 0
            self._doorway_reached = False
            self._doorway_reach_tolerance_m = COMMITTED_TARGET_REACHED_M
            self._committed_target_seen = False
            self._turn_progress_deg = 0.0
            self._elevation_history.clear()
            self._final_target_at_streak = 0
            self._destination_at_streak = 0
            self._subgoal_observations = 0
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
        *,
        reach_tolerance_m: Optional[float] = None,
    ) -> None:
        """Attach distance to the actor's committed target for this subgoal.

        The target is the model-localized doorway or landmark point the actor
        is walking to. ``None`` means no such point is held. This is geometric
        confirmation of a model-selected object, not accumulated path length:
        walking around inside the source room cannot satisfy it unless the
        camera actually reaches that target, and no completion is accepted
        while the camera is still on its way there.
        """
        if distance_m is None:
            self._doorway_target_distance_m = None
            return
        value = float(distance_m)
        if value < 0.0:
            raise ValueError("doorway target distance must not be negative")
        self._doorway_target_distance_m = value
        self._committed_target_seen = True
        if reach_tolerance_m is not None:
            self._doorway_reach_tolerance_m = max(
                COMMITTED_TARGET_REACHED_M, float(reach_tolerance_m)
            )

    set_committed_target_distance = set_doorway_target_distance

    def set_elevation_progress(self, rise_m: float) -> None:
        """Attach the camera's height change since the subgoal began."""
        self._elevation_history.append(float(rise_m))

    @staticmethod
    def _stairs_direction(subgoal: Any) -> Optional[str]:
        description = (getattr(subgoal, "description", "") or "").lower()
        if not re.search(r"\b(?:stairs?|staircase|stairway|steps)\b", description):
            return None
        if re.search(r"\b(?:up|ascend|climb)\b", description):
            return "up"
        if re.search(r"\b(?:down|descend)\b", description):
            return "down"
        return None

    def set_depth_observation(self, depth_m: Any) -> None:
        """Attach the current depth map (metres, HxW) for range checks."""
        self._depth_m = depth_m

    def _landmark_range(self, landmark: Any) -> Optional[float]:
        """Range to the model's landmark pixel through the depth map."""
        depth = self._depth_m
        u = getattr(landmark, "u", None)
        v = getattr(landmark, "v", None)
        if depth is None or u is None or v is None:
            return None
        try:
            import numpy as np

            values = np.asarray(depth, dtype=np.float64)
            if values.ndim == 3:
                values = values[..., 0]
            height, width = values.shape
            column = int(round(float(u) * (width - 1) / 1000.0))
            row = int(round(float(v) * (height - 1) / 1000.0))
            patch = values[
                max(row - 2, 0):row + 3, max(column - 2, 0):column + 3
            ]
            patch = patch[np.isfinite(patch) & (patch > 0.0)]
            if patch.size == 0:
                return None
            return float(np.median(patch))
        except Exception:
            return None

    def set_stop_proposed(self, proposed: bool) -> None:
        """Record whether the waypoint model proposed STOP on the last step.

        The scene observer and the waypoint model are independent calls; the
        destination is accepted when both say the camera is there.
        """
        self._stop_proposed_last_step = bool(proposed)

    def set_turn_progress(self, degrees: float) -> None:
        """Attach the measured rotation in the subgoal's requested direction.

        The model judges a turn from single frames and cannot tell how far
        the camera has rotated, so a turn subgoal's completion is gated on
        odometry here rather than on what the frames look like.
        """
        self._turn_progress_deg = float(degrees)

    @staticmethod
    def _turn_direction(subgoal: Any) -> Optional[str]:
        description = getattr(subgoal, "description", "") or ""
        if re.search(TURN_AROUND_PATTERN, description, re.IGNORECASE):
            return "around"
        match = re.search(r"\bturn\s+(left|right)\b", description, re.IGNORECASE)
        return match.group(1).lower() if match else None

    @staticmethod
    def _turn_target_deg(subgoal: Any) -> float:
        """Rotation that completes the stage: ~180 for "turn around"."""
        description = getattr(subgoal, "description", "") or ""
        if re.search(TURN_AROUND_PATTERN, description, re.IGNORECASE):
            return TURN_AROUND_MIN_PROGRESS_DEG
        return TURN_MIN_PROGRESS_DEG

    def append_observation(self, image: Any) -> MemoryFrame:
        frame = super().append_observation(image)
        self._subgoal_path_length_m += self._pending_translation_m
        self._task_path_length_m += self._pending_translation_m
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
        final_subgoal = getattr(
            self.task_memory, "get_final_subgoal", lambda: None
        )()
        self._subgoal_observations += 1
        scene = self.captioner.analyze_scene(
            SceneAnalysisRequest(
                subgoal=self._subgoal,
                frames=completion_frames,
                is_final_subgoal=is_final,
                final_subgoal=(
                    final_subgoal if isinstance(final_subgoal, Subgoal) else None
                ),
            )
        )
        self._latest_scene = scene
        self._annotate_latest_landmark(scene)
        landmark_range = (
            self._landmark_range(scene.landmark)
            if scene.landmark.visible
            else None
        )
        self._last_landmark_range_m = landmark_range
        doorway_distance = self._doorway_target_distance_m
        reach_tolerance = self._doorway_reach_tolerance_m
        if doorway_distance is not None and doorway_distance <= reach_tolerance:
            self._doorway_reached = True
        # The model located its landmark in this very image; the depth map
        # says how far away that pixel is. A crossing or an arrival claimed
        # while it is still metres ahead is contradicted by measurement.
        # Once the camera has measurably reached the stage's committed point
        # the veto no longer applies: after a crossing the model tends to
        # attach the "landmark" to something else deeper in the room.
        landmark_far = bool(
            landmark_range is not None
            and landmark_range > LANDMARK_RANGE_VETO_M
            and not self._doorway_reached
        )

        # The model's own ``completed`` flag is not required here: it
        # contradicts its AT report often enough that waiting for both left
        # the agent standing at the destination until the step budget ran
        # out. AT is the structured fact; the streak and the waypoint model's
        # independent STOP proposal supply the confirmation.
        # Structural fields only: the model's confidence values are advisory
        # (Qwen3-VL-8B returned the prompt's placeholder 0.0 on every step of
        # a full run), and the streak, the depth veto and the committed-point
        # odometry below are the actual evidence.
        final_at = bool(
            is_final
            and scene.final_target.visible
            and scene.final_target.proximity == "AT"
        )
        self._final_target_at_streak = (
            self._final_target_at_streak + 1 if final_at else 0
        )
        # Judged on arrival, not en route: a committed point still ahead
        # of the camera means the stage's own walk has not finished, and
        # the AT streak this checkpoint reports from several metres out
        # cannot override that measurement.
        committed_target_ahead = bool(
            doorway_distance is not None
            and doorway_distance > reach_tolerance
            and not self._doorway_reached
        )
        final_confirmed = bool(
            is_final
            and not landmark_far
            and not committed_target_ahead
            and (
                (
                    self._final_target_at_streak >= 2
                    and self._stop_proposed_last_step
                )
                or self._final_target_at_streak >= 3
            )
        )
        destination_at = bool(
            scene.final_target.visible
            and scene.final_target.proximity == "AT"
            and scene.final_target.confidence >= STAGE_SKIP_MIN_CONFIDENCE
        )
        self._destination_at_streak = (
            self._destination_at_streak + 1 if destination_at else 0
        )

        is_doorway = scene.door_state != "NOT_APPLICABLE"
        approached_before = self._doorway_approach_seen
        if (
            scene.door_state in {"APPROACHING", "AT_THRESHOLD", "CROSSING"}
            or scene.door_camera_side in {"BEFORE_DOOR", "AT_DOOR"}
        ):
            self._doorway_approach_seen = True
        # The Captioner already folds the structural door/target fields into
        # ``completed``; the measured guards below decide whether to accept it.
        completed = bool(scene.completed)
        reached_model_doorway = bool(
            self._doorway_reached
            or (doorway_distance is not None and doorway_distance <= 0.35)
        )
        target_still_ahead = bool(
            doorway_distance is not None
            and doorway_distance > reach_tolerance
            and not self._doorway_reached
        )
        # Arriving at the point the actor committed to for a plain landmark
        # stage is the stage's endpoint; doorway and final stages keep their
        # own crossing and stop protocols.
        landmark_arrival = bool(
            self._doorway_reached
            and self._committed_target_seen
            and not is_doorway
            and not is_final
        )
        stairs_direction = self._stairs_direction(self._subgoal)
        rise = self._elevation_history[-1] if self._elevation_history else 0.0
        levelled = bool(
            len(self._elevation_history) == self._elevation_history.maxlen
            and max(self._elevation_history) - min(self._elevation_history)
            <= STAIRS_LEVEL_TOLERANCE_M
        )
        stairs_done = bool(
            stairs_direction is not None
            and levelled
            and (rise >= STAIRS_MIN_RISE_M if stairs_direction == "up" else rise <= -STAIRS_MIN_RISE_M)
        )
        stairs_incomplete = stairs_direction is not None and not stairs_done
        turn_direction = self._turn_direction(self._subgoal)
        turn_target = self._turn_target_deg(self._subgoal)
        landmark_centred = bool(
            scene.landmark.visible
            and scene.landmark.u is not None
            and abs(scene.landmark.u - 500) <= TURN_TARGET_CENTRED_U
        )
        turn_incomplete = bool(
            turn_direction is not None
            and self._turn_progress_deg < turn_target
            and not (
                landmark_centred
                and self._turn_progress_deg >= TURN_TARGET_MIN_PROGRESS_DEG
            )
        )
        # A turn is a rotation, not a place: once the camera has measurably
        # turned far enough the stage is done, whatever a single frame looks
        # like to the model. Its criterion ("X is centred after the turn")
        # is only true for an instant and the model rarely catches it.
        turn_done = bool(
            turn_direction is not None
            and (
                self._turn_progress_deg >= turn_target
                or (
                    landmark_centred
                    and self._turn_progress_deg >= TURN_TARGET_MIN_PROGRESS_DEG
                )
            )
        )
        # ``scene.completed`` on a doorway stage already requires CROSSED,
        # AFTER_DOOR, PASSED_THROUGH, FAR_SIDE and destination_dominant to
        # agree; the streak and walked distance supply the confirmation.
        crossed_now = bool(
            is_doorway
            and scene.completed
            and scene.door_state == "CROSSED"
            and scene.door_camera_side == "AFTER_DOOR"
        )
        self._doorway_crossed_streak = (
            self._doorway_crossed_streak + 1 if crossed_now else 0
        )
        sustained_crossing = bool(
            self._doorway_crossed_streak >= DOORWAY_CROSSED_STREAK_ACCEPT
            and self._subgoal_path_length_m >= DOORWAY_CROSSED_MIN_PATH_M
        )
        # A turn stage ends when the rotation is done; the point committed
        # to during it belongs to the walk that follows, not to the turn.
        if turn_done:
            completed = True
            self._last_completion_guard = (
                f"accepted measured {turn_direction} turn of "
                f"{self._turn_progress_deg:.0f} deg"
                + (
                    f"; landmark centred at u={scene.landmark.u}"
                    if landmark_centred
                    else ""
                )
            )
        elif final_confirmed:
            completed = True
            self._last_completion_guard = (
                "accepted destination: AT for "
                f"{self._final_target_at_streak} consecutive observations"
                + (
                    " with the waypoint model proposing STOP"
                    if self._stop_proposed_last_step
                    else ""
                )
            )
        elif stairs_done:
            completed = True
            self._last_completion_guard = (
                f"accepted measured stairs {stairs_direction}: height changed "
                f"{rise:+.2f} m and levelled off"
            )
        elif landmark_arrival:
            completed = True
            self._last_completion_guard = (
                "accepted arrival at the committed landmark point "
                f"(within {reach_tolerance:.2f} m)"
            )
        elif completed and stairs_incomplete:
            completed = False
            self._last_completion_guard = (
                f"deferred completion: stairs {stairs_direction} not measured "
                f"(height change {rise:+.2f} m, levelled={levelled})"
            )
        elif completed and target_still_ahead:
            # Judged on arrival, not en route: the actor is still walking to
            # the point it committed to for this subgoal, so whatever the
            # frames look like, the subgoal's own action has not finished.
            # Measurement wins over any reported stage or streak.
            completed = False
            self._last_completion_guard = (
                "deferred completion: committed waypoint is still "
                f"{doorway_distance:.2f} m ahead and was never reached"
            )
        elif completed and landmark_far:
            completed = False
            self._last_completion_guard = (
                "rejected completion: the model's own landmark is still "
                f"{landmark_range:.2f} m away in the depth map"
            )
        elif completed and turn_incomplete:
            completed = False
            self._last_completion_guard = (
                f"deferred completion: measured {turn_direction} turn is "
                f"{self._turn_progress_deg:.0f} deg, below "
                f"{turn_target:.0f} deg"
            )
        elif (
            completed
            and is_doorway
            and not (
                approached_before
                or reached_model_doorway
                or sustained_crossing
            )
        ):
            completed = False
            self._last_completion_guard = (
                "rejected doorway completion: camera has not reached the "
                "model-localized structural doorway and no earlier model "
                "observation placed it before/at the doorway "
                f"(crossed streak {self._doorway_crossed_streak}/"
                f"{DOORWAY_CROSSED_STREAK_ACCEPT}, walked "
                f"{self._subgoal_path_length_m:.2f} m)"
            )
        elif (
            completed
            and is_doorway
            and sustained_crossing
            and not (approached_before or reached_model_doorway)
        ):
            self._last_completion_guard = (
                "accepted sustained doorway crossing: "
                f"{self._doorway_crossed_streak} consecutive confident "
                "CROSSED/AFTER_DOOR judgements after "
                f"{self._subgoal_path_length_m:.2f} m of walking"
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
        skip = None
        if not completed and not is_final:
            skip = self._maybe_skip_to_final(destination_at)
        raw_payload = {
            "scene": scene.raw_response,
        }
        if self._last_completion_guard is not None:
            raw_payload["completion_guard"] = (
                self._last_completion_guard
            )
        if skip is not None:
            raw_payload["stage_skip"] = skip
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

    def _maybe_skip_to_final(self, destination_at: bool) -> Optional[str]:
        """Jump to the final stage when the destination itself is verified.

        Intermediate stages are navigation cues; the destination is what the
        task is judged on. When an intermediate stage cannot be completed but
        the destination is repeatedly reported AT, the plan is stuck on a
        cue it no longer needs. This is deliberately hard to trigger: a
        streak of confident AT reports, a plausible position in the plan
        (near its end, or stuck for a long time), and enough distance walked
        that a look-alike right after the start cannot qualify. The final
        stage still has to pass its own stop protocol afterwards.
        """
        skip_to_final = getattr(self.task_memory, "skip_to_final", None)
        if skip_to_final is None or not destination_at:
            return None
        if self._destination_at_streak < STAGE_SKIP_AT_STREAK:
            return None
        if self._task_path_length_m < STAGE_SKIP_MIN_TASK_PATH_M:
            return None
        remaining = self._remaining_stages()
        stalled = self._subgoal_observations >= STAGE_SKIP_STALL_OBSERVATIONS
        if remaining is None or (
            remaining > STAGE_SKIP_MAX_REMAINING_STAGES and not stalled
        ):
            return None
        reason = (
            f"destination reported AT for {self._destination_at_streak} "
            f"consecutive observations after {self._task_path_length_m:.1f} m; "
            f"{remaining} stage(s) remained and the active stage had "
            f"{self._subgoal_observations} observations"
        )
        skipped = skip_to_final(reason)
        if not skipped:
            return None
        self._last_stage_skip = {"skipped": list(skipped), "reason": reason}
        self._last_completion_guard = (
            f"skipped stages {', '.join(skipped)} to the final stage: {reason}"
        )
        return self._last_completion_guard

    def _remaining_stages(self) -> Optional[int]:
        """Stages after the active one, or None when the plan is unknown."""
        current = self._subgoal
        final = getattr(self.task_memory, "get_final_subgoal", lambda: None)()
        if current is None or final is None:
            return None
        try:
            return int(final.subgoal_id) - int(current.subgoal_id)
        except (TypeError, ValueError):
            return None

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
                "stop_proposed_last_step": self._stop_proposed_last_step,
                "landmark_range_m": self._last_landmark_range_m,
                "destination_at_streak": self._destination_at_streak,
                "subgoal_observations": self._subgoal_observations,
                "task_path_length_m": self._task_path_length_m,
                "stage_skip": self._last_stage_skip,
                "doorway_crossed_streak": self._doorway_crossed_streak,
                "doorway_reached": self._doorway_reached,
                "turn_progress_deg": self._turn_progress_deg,
                "doorway_reach_tolerance_m": self._doorway_reach_tolerance_m,
                "elevation_rise_m": (
                    self._elevation_history[-1]
                    if self._elevation_history
                    else 0.0
                ),
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
