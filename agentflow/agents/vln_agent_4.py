"""VLN RGB-D waypoint actor with unified temporal scene understanding."""


from __future__ import annotations

from collections import deque
from dataclasses import replace
import re
import time
from typing import Any, Deque, Optional, Sequence

import numpy as np

from agentflow.agents.models_embodied_v2.actor import Actor
from agentflow.agents.models_embodied_v2.memory.temporal_memory import (
    TemporalCaptioner,
)
from agentflow.agents.models_embodied_v2.data_models import (
    ActionMode,
    Subgoal,
    TemporalCaptionerConfig,
    CameraIntrinsics,
    NavigationDecision,
    NavigationPoint,
    PreviewSelection,
    PreviewView,
)

from agentflow.agents.models_embodied_v2.memory.task_memory import TaskMemory

from agentflow.agents.models_embodied_v2.skiils.protocol import (
    BEHAVIOR_HISTORY_SIZE,
    CORRIDOR_LOCK_FORWARD_STEPS,
    DEFAULT_MODEL_PATH,
    ERROR_CONFIRMATION_WINDOW,
    FINAL_STOP_EVIDENCE_WINDOW,
    LANDMARK_HISTORY_SIZE,
    LANDMARK_STEER_MIN_CONFIDENCE,
    NavigationIntent,
    POINT_PROMPT,
    PREVIEW_REARM_TRANSLATION_M,
    PREVIEW_SELECTION_MIN_CONFIDENCE,
    RECOVERY_LATERAL_DISTANCE_M,
    SUBGOAL_GENERATION_ATTEMPTS,
    STRUCTURED_VLM_MAX_TOKENS,
    TEMPORAL_MAX_IMAGE_EDGE,
    SUBGOAL_PROMPT,
    TURN_ALIGNMENT_DEG,
    TURN_EVIDENCE_DEG,
    LandmarkOutput as _LandmarkOutput,
    WaypointOutput as _WaypointOutput,
)
from agentflow.agents.models_embodied_v2.skiils.planning import parse_subgoal_plan
from agentflow.agents.models_embodied_v2.skiils.landmark import LandmarkTrackerMixin
from agentflow.agents.models_embodied_v2.skiils.waypoint import WaypointPolicyMixin

from agentflow.agents.models_embodied_v2.memory.temporal_memory import (
    TemporalMemory,
)
class VLNAgent(LandmarkTrackerMixin, WaypointPolicyMixin):
    """Version 3 actor used by the Habitat waypoint worker."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        *,
        engine=None,
        debug_performance=False,
        use_cache=False,
        min_depth_m=0.25,
        max_depth_m=10.0,
        patch_radius_px=3,
        max_patch_depth_spread_m=0.35,
        camera_height_m=None,
        max_floor_offset_m=0.30,
        task_memory=None,
        temporal_memory=None,
        **kwargs,
    ) -> None:
        # The actor owns observation validation, walkable-pixel snapping, and
        # back-projection.  This agent owns everything with task state: the
        # plan, both memories, the navigation phase, and the guards.
        self.actor = Actor(
            model_path,
            engine=engine,
            debug_performance=debug_performance,
            use_cache=use_cache,
            min_depth_m=min_depth_m,
            max_depth_m=max_depth_m,
            patch_radius_px=patch_radius_px,
            max_patch_depth_spread_m=max_patch_depth_spread_m,
            camera_height_m=camera_height_m,
            max_floor_offset_m=max_floor_offset_m,
        )
        # The landmark and waypoint policies prompt the same checkpoint.
        self.llm = self.actor.llm

        self.last_model_response: Optional[str] = None
        # Per-``act`` wall-clock breakdown, in milliseconds, for latency
        # debugging.
        self.last_timings: dict[str, float] = {}
        # The analysis produced during the last ``act``, or None when Temporal
        # Memory did not analyze. Held here because reading Temporal Memory's
        # own ``latest_result`` clears it as soon as the subgoal advances.
        self.last_caption: Optional[Any] = None

        self.task_instruction: Optional[str] = None
        self.subgoals: list[Subgoal] = []
        self.last_subgoal_response: Optional[str] = None

        self.task_memory = task_memory
        if temporal_memory is not None and task_memory is None:
            raise ValueError(
                "temporal_memory requires its associated task_memory."
            )
        self.temporal_memory = temporal_memory

        self._landmark_history: Deque[dict[str, Any]] = deque(
            maxlen=LANDMARK_HISTORY_SIZE
        )
        self._behavior_history: Deque[dict[str, Any]] = deque(
            maxlen=BEHAVIOR_HISTORY_SIZE
        )
        self._error_candidates: Deque[str] = deque(
            maxlen=ERROR_CONFIRMATION_WINDOW
        )
        self._final_stop_evidence: Deque[bool] = deque(
            maxlen=FINAL_STOP_EVIDENCE_WINDOW
        )
        self._reset_runtime_state()

    def _reset_runtime_state(self) -> None:
        """Clear all episode-local navigation and debug state."""
        self.last_subgoal_before: Optional[str] = None
        self.last_subgoal_after: Optional[str] = None
        self.last_requested_pixel: Optional[tuple[int, int]] = None
        self.last_requested_normalized: Optional[tuple[int, int]] = None
        # Set instead of the pixel pair when a step rotates in place.
        self.last_requested_turn_deg: Optional[int] = None
        self.last_waypoint_raw_response: Optional[str] = None
        self.last_waypoint_stop_disposition: Optional[str] = None
        self.last_waypoint_model_intent: Optional[str] = None
        self.last_waypoint_applied_intent: Optional[str] = None
        self.last_waypoint_model_action_mode: Optional[ActionMode] = None
        self.last_waypoint_applied_action_mode: Optional[ActionMode] = None
        # Which surrounding view resolved the last PREVIEW, its heading offset,
        # the Captioner's own judgement, and why a fallback was used instead.
        # All stay None on a step that never previewed.
        self.last_preview_view_index: Optional[int] = None
        self.last_preview_yaw_deg: Optional[float] = None
        self.last_preview_selection: Optional[PreviewSelection] = None
        self.last_preview_guard_reason: Optional[str] = None
        # A preview stays consumed at this physical position until the camera
        # has made real translational progress. Rotation alone must not re-arm
        # another relative-heading preview.
        self._preview_requires_progress = False
        self._preview_anchor_position_xz: Optional[np.ndarray] = None
        self.last_waypoint_guard_reason: Optional[str] = None
        self.last_waypoint_evidence: Optional[str] = None
        self.last_waypoint_confidence: Optional[float] = None
        self.last_error_candidate: Optional[str] = None
        self.last_error_guard_reason: Optional[str] = None
        self.last_recovery_mode: Optional[str] = None
        self.last_landmark: Optional[_LandmarkOutput] = None
        self.last_landmark_raw_response: Optional[str] = None
        self.last_landmark_error: Optional[str] = None
        self.last_landmark_normalized: Optional[tuple[int, int]] = None
        self.last_landmark_pixel: Optional[tuple[int, int]] = None
        self._landmark_subgoal_id: Optional[str] = None
        self._error_candidate_subgoal: Optional[str] = None
        self._active_recovery_mode: Optional[str] = None
        self._recovery_progress_m = 0.0
        self._recovery_attempt_steps = 0
        self._recovery_anchor_position_xz: Optional[np.ndarray] = None
        self._force_forward_this_step = False
        self._force_left_turn_this_step = False
        self._previous_position: Optional[np.ndarray] = None
        self._previous_yaw_deg: Optional[float] = None
        self._navigation_subgoal_id: Optional[str] = None
        self._navigation_phase: NavigationIntent = "FOLLOW_CORRIDOR"
        self._corridor_forward_streak = 0
        self._corridor_heading_yaw_deg: Optional[float] = None
        self._subgoal_net_yaw_deg = 0.0
        self._turn_follow_phase_started = False
        # A model-localized doorway is a stable physical target. Keep its
        # world-space waypoint across steps instead of letting independent VLM
        # calls move the target around while the follower is routing around
        # furniture toward the opening.
        self._doorway_waypoint: Optional[NavigationPoint] = None
        self._doorway_waypoint_subgoal_id: Optional[str] = None
        self._doorway_waypoint_best_distance_m: Optional[float] = None
        self._doorway_waypoint_stagnant_steps = 0
        self._landmark_history.clear()
        self._behavior_history.clear()
        self._error_candidates.clear()
        self._final_stop_evidence.clear()

    def act(
        self,
        rgb: Any,
        depth: Any,
        instruction: str,
        intrinsics: CameraIntrinsics | Any,
        camera_to_world: Any,
        *,
        normalized_depth: bool = False,
        depth_min_m: Optional[float] = None,
        depth_max_m: Optional[float] = None,
    ) -> NavigationDecision:
        """Run one step while retaining state transitions for debug output."""
        act_started = time.perf_counter()
        # These flags describe only the waypoint selected during this call.
        # A recovery selection may set one of them below; clear both before
        # delegating to the RGB-D layer so a completed recovery cannot leak
        # into later model-selected waypoints.
        self._force_forward_this_step = False
        self._force_left_turn_this_step = False
        # Preview views belong to the step that asked for them. Holding them
        # past it would keep several images alive for the rest of the episode
        # and let a later reader mistake them for this step's.
        if isinstance(self.temporal_memory, TemporalMemory):
            self.temporal_memory.clear_preview()
        current = (
            self.task_memory.get_current_subgoal()
            if self.task_memory is not None
            else None
        )
        self.last_subgoal_before = (
            current.subgoal_id if current is not None else None
        )
        self._sync_navigation_phase(current)
        had_previous_pose = self._previous_position is not None
        previous_waypoint = self.last_requested_normalized
        translation_m, yaw_delta_deg = self._measure_motion(
            camera_to_world
        )
        if had_previous_pose:
            self._update_preview_progress()
            self._update_recovery_progress(translation_m)
            self._record_behavior(
                subgoal_id=self.last_subgoal_before,
                translation_m=translation_m,
                yaw_delta_deg=yaw_delta_deg,
                requested_waypoint=previous_waypoint,
            )
            self._update_navigation_progress(
                current,
                yaw_delta_deg=yaw_delta_deg,
            )
            self._update_corridor_lock(
                translation_m=translation_m,
                yaw_delta_deg=yaw_delta_deg,
            )
        if isinstance(self.temporal_memory, TemporalMemory):
            self.temporal_memory.set_motion_evidence(
                translation_m=translation_m,
                yaw_delta_deg=yaw_delta_deg,
            )
            self.temporal_memory.set_doorway_target_distance(
                self._doorway_target_distance(
                    current,
                    camera_to_world=camera_to_world,
                )
            )
        try:
            decision = self._waypoint_decision(
                rgb,
                depth,
                instruction,
                intrinsics,
                camera_to_world,
                normalized_depth=normalized_depth,
                depth_min_m=depth_min_m,
                depth_max_m=depth_max_m,
            )
        except ValueError as exc:
            if str(exc) != (
                "Depth observation contains no valid walkable waypoint."
            ):
                raise
            # At very close range every lower-image depth value can be
            # invalid. That is an action-level recovery condition, not a
            # reason to abort the episode. Return a synthetic lateral target;
            # the Habitat runner turns when its follower rejects this point.
            image = self.actor.as_rgb_array(rgb)
            transform = np.asarray(camera_to_world, dtype=np.float64)
            position = transform[:3, 3]
            self._force_left_turn_this_step = True
            self.last_recovery_mode = "NO_VALID_DEPTH"
            # The model may have asked to preview, but the near-field obstacle
            # has to be cleared before any surrounding view is worth taking.
            self.last_waypoint_applied_action_mode = "EXECUTION"
            self.last_waypoint_guard_reason = (
                "forced turn because depth has no valid walkable waypoint"
            )
            self.last_waypoint_evidence = (
                "action-level NO_VALID_DEPTH recovery: turn away from the "
                "near-field obstacle"
            )
            self.last_requested_normalized = (250, 500)
            self.last_requested_pixel = (
                int(image.shape[1] * 0.25),
                int(image.shape[0] * 0.50),
            )
            decision = NavigationDecision(
                stop=False,
                point=NavigationPoint(
                    pixel_uv=self.last_requested_pixel,
                    depth_m=0.0,
                    camera_xyz=(0.0, 0.0, -1.0),
                    world_xyz=tuple(float(value) for value in position),
                ),
                raw_response=(
                    f"{self.last_model_response}; no-valid-depth recovery"
                ),
                action_mode="EXECUTION",
            )
        if (
            self._force_forward_this_step
            and not decision.stop
            and decision.point is not None
        ):
            transform = np.asarray(camera_to_world, dtype=np.float64)
            decision = replace(
                decision,
                point=replace(
                    decision.point,
                    world_xyz=(
                        float(transform[0, 3]),
                        decision.point.world_xyz[1],
                        float(transform[2, 3]),
                    ),
                ),
                raw_response=(
                    f"{decision.raw_response}; force-forward waypoint"
                ),
            )
        elif (
            self._force_left_turn_this_step
            and not decision.stop
            and decision.point is not None
        ):
            transform = np.asarray(camera_to_world, dtype=np.float64)
            camera_right = transform[:3, 0]
            lateral = camera_right[[0, 2]]
            lateral_norm = float(np.linalg.norm(lateral))
            if lateral_norm > 1e-6:
                lateral = lateral / lateral_norm
            position = transform[:3, 3]
            decision = replace(
                decision,
                point=replace(
                    decision.point,
                    world_xyz=(
                        float(
                            position[0]
                            - RECOVERY_LATERAL_DISTANCE_M * lateral[0]
                        ),
                        decision.point.world_xyz[1],
                        float(
                            position[2]
                            - RECOVERY_LATERAL_DISTANCE_M * lateral[1]
                        ),
                    ),
                ),
                raw_response=(
                    f"{decision.raw_response}; force-left-turn waypoint"
                ),
            )
        current = (
            self.task_memory.get_current_subgoal()
            if self.task_memory is not None
            else None
        )
        self.last_subgoal_after = (
            current.subgoal_id if current is not None else None
        )
        self.last_timings["total_ms"] = (
            time.perf_counter() - act_started
        ) * 1000
        return decision

    def act_on_preview(
        self,
        views: Sequence[PreviewView],
        instruction: str,
        *,
        normalized_depth: bool = False,
        depth_min_m: Optional[float] = None,
        depth_max_m: Optional[float] = None,
    ) -> NavigationDecision:
        """Act on the surrounding views the controller rendered for a PREVIEW.

        This is the second half of the step ``act`` began, not a new one.
        Motion, the landmark tracker, and Temporal Memory were all advanced
        there and deliberately do not run again: re-entering ``act`` would count
        the same frame twice and record a stationary step as a stall.

        The views are deposited in working memory, which is where the Captioner
        judges which heading to take.  This agent only picks the floor point
        inside whichever view comes back, with its ordinary waypoint policy.
        """
        started = time.perf_counter()
        views = tuple(views)
        if not views:
            raise ValueError("resolving a preview requires at least one view")
        unusable = [
            index
            for index, view in enumerate(views)
            if not view.is_navigable
        ]
        if unusable:
            raise ValueError(
                "preview views {} lack depth, intrinsics, or a camera "
                "transform, so a waypoint chosen in them could not be "
                "back-projected".format(unusable)
            )
        # A previewed step selects a fresh action, so no forced primitive from
        # the first half may leak into it.
        self._force_forward_this_step = False
        self._force_left_turn_this_step = False
        view_transform = np.asarray(views[0].camera_to_world, dtype=np.float64)
        self._preview_requires_progress = True
        self._preview_anchor_position_xz = view_transform[[0, 2], 3].copy()

        preview_select_started = time.perf_counter()
        view_index = self._selected_preview_view(views)
        timings = {
            "preview_select_ms": (
                time.perf_counter() - preview_select_started
            ) * 1000
        }
        view = views[view_index]
        self.last_preview_view_index = view_index
        self.last_preview_yaw_deg = view.yaw_deg

        image = self.actor.as_rgb_array(view.rgb)
        select_started = time.perf_counter()
        selection = self.last_preview_selection
        if (
            selection is not None
            and selection.confidence >= PREVIEW_SELECTION_MIN_CONFIDENCE
        ):
            # The selector already spent the one VLM call needed to resolve
            # this PREVIEW. Commit to its selected floor point instead of
            # asking the waypoint model to interpret the same heading again
            # (which could request PREVIEW in a loop).
            self._commit_selected_preview(
                selection,
                width=image.shape[1],
                height=image.shape[0],
            )
        else:
            self._select_pixel(
                image,
                instruction,
                subgoal_context=(
                    self.task_memory.current_subgoal_context()
                    if self.task_memory is not None
                    else ""
                ),
                # All three constrain this second pass within one step: the
                # error votes and recovery hold were already consumed by
                # ``act``, the surrounding views are already in hand, and a
                # turn would be measured against the agent's real facing
                # rather than the rotated preview view.
                evaluate_recovery=False,
                allow_preview=False,
                allow_turn=False,
            )
        timings["select_pixel_ms"] = (
            time.perf_counter() - select_started
        ) * 1000
        # With previews and turns both refused above, the only outcome left is
        # a waypoint inside the chosen view.
        requested_uv = self.last_requested_pixel
        assert requested_uv is not None

        depth_m = self.actor.depth_in_meters(
            view.depth,
            image.shape[:2],
            normalized=normalized_depth,
            depth_min_m=depth_min_m,
            depth_max_m=depth_max_m,
        )
        waypoint_started = time.perf_counter()
        try:
            point = self.actor.waypoint_from_pixel(
                requested_uv,
                depth_m,
                view.intrinsics,
                view.camera_to_world,
            )
        except ValueError as exc:
            if str(exc) != (
                "Depth observation contains no valid walkable waypoint."
            ):
                raise
            # The chosen heading is blocked at very close range. Returning no
            # point leaves the controller on its own recovery primitive, which
            # is the same degradation a PREVIEW with no renderer gets.
            self.last_recovery_mode = "NO_VALID_DEPTH"
            self.last_waypoint_guard_reason = (
                "previewed view {} has no valid walkable waypoint".format(
                    view_index
                )
            )
            self._record_timings(timings, started)
            return NavigationDecision(
                stop=False,
                raw_response=(
                    f"{self.last_model_response}; previewed view has no "
                    "valid depth"
                ),
                action_mode=self._applied_action_mode(),
            )
        self._maybe_lock_doorway_waypoint(
            point,
            camera_to_world=view.camera_to_world,
        )
        timings["waypoint_ms"] = (
            time.perf_counter() - waypoint_started
        ) * 1000
        self._record_timings(timings, started)
        return NavigationDecision(
            stop=False,
            point=point,
            raw_response=self.last_model_response,
            action_mode=self._applied_action_mode(),
        )

    def _commit_selected_preview(
        self,
        selection: PreviewSelection,
        *,
        width: int,
        height: int,
    ) -> None:
        """Turn one high-confidence preview target into a floor waypoint."""
        intent: NavigationIntent = (
            self._navigation_phase
            if self._navigation_phase != "STOP"
            else "FINAL_APPROACH"
        )
        raw_response = getattr(
            getattr(self.temporal_memory, "captioner", None),
            "last_preview_raw_response",
            None,
        )
        self.last_model_response = raw_response or selection.evidence
        self.last_waypoint_raw_response = self.last_model_response
        self.last_waypoint_model_intent = intent
        self.last_waypoint_applied_intent = intent
        self.last_waypoint_model_action_mode = "EXECUTION"
        self.last_waypoint_applied_action_mode = "EXECUTION"
        self.last_waypoint_evidence = selection.evidence
        self.last_waypoint_confidence = selection.confidence
        self.last_waypoint_guard_reason = (
            "committed directly to the Captioner-selected preview heading; "
            "skipped redundant waypoint VLM"
        )
        self.last_requested_turn_deg = None
        self.last_requested_normalized = (selection.u, selection.v)
        self.last_requested_pixel = (
            self._scale_normalized(selection.u, width),
            self._scale_normalized(selection.v, height),
        )

    def _waypoint_decision(
        self,
        rgb: Any,
        depth: Any,
        instruction: str,
        intrinsics: CameraIntrinsics | Any,
        camera_to_world: Any,
        *,
        normalized_depth: bool = False,
        depth_min_m: Optional[float] = None,
        depth_max_m: Optional[float] = None,
    ) -> NavigationDecision:
        """Return the stop decision or the world-space RGB-D waypoint.

        Habitat's default depth sensor emits meters.  Set ``normalized_depth``
        only when its configuration uses normalized [0, 1] observations, and
        provide that sensor's ``depth_min_m`` and ``depth_max_m`` bounds.
        """
        started = time.perf_counter()
        image = self.actor.as_rgb_array(rgb)
        timings = {"rgb_ms": (time.perf_counter() - started) * 1000}

        memory_started = time.perf_counter()
        caption = None
        if self.task_memory is not None:
            self.task_memory.record_input(image)
            if self.temporal_memory is not None:
                current_before_analysis = (
                    self.task_memory.get_current_subgoal()
                )
                doorway_distance = self._doorway_target_distance(
                    current_before_analysis,
                    camera_to_world=camera_to_world,
                )
                # Completion cannot pass the structural-door guard while the
                # camera is still far from the model-localized doorway. Keep
                # every RGB/motion frame but avoid an expensive, guaranteed
                # in-progress Captioner call until the threshold is near.
                defer_scene_analysis = bool(
                    doorway_distance is not None
                    and doorway_distance > 1.25
                )
                caption = self.temporal_memory.update_from_task_memory(
                    analyze=not defer_scene_analysis,
                )
        self.last_caption = caption
        if caption is not None:
            self._record_scene_landmark(image, caption)
        timings.update({
            # Whole memory phase: frame copy, rule-based error detection, and
            # the Captioner call when Temporal Memory analyzes this step.
            "memory_ms": (time.perf_counter() - memory_started) * 1000,
            # The Captioner's own model time, so the rule-based visual
            # bookkeeping around it can be read as the difference.
            "captioner_ms": (
                caption.latency_ms if caption is not None else 0.0
            ),
        })

        # Every planned subgoal is done, so the task itself is done. Stop here
        # rather than asking the model to steer towards a subgoal that no
        # longer exists.
        if self.task_memory is not None and self.task_memory.is_task_complete():
            self.last_model_response = "all subgoals complete"
            # This returns before ``_select_pixel`` clears the per-step waypoint
            # state, so set the mode here rather than reporting the previous
            # step's value alongside a terminal decision.
            self.last_waypoint_applied_action_mode = "EXECUTION"
            self._record_timings(timings, started)
            return NavigationDecision(
                stop=True,
                raw_response=self.last_model_response,
            )

        current = (
            self.task_memory.get_current_subgoal()
            if self.task_memory is not None
            else None
        )
        self._release_doorway_waypoint_for_motion(
            current,
            camera_to_world=camera_to_world,
        )
        locked = self._locked_doorway_decision(
            current,
            camera_to_world=camera_to_world,
        )
        if locked is not None:
            timings["depth_ms"] = 0.0
            timings["select_pixel_ms"] = 0.0
            timings["waypoint_ms"] = 0.0
            self._record_timings(timings, started)
            return locked

        depth_started = time.perf_counter()
        depth_m = self.actor.depth_in_meters(
            depth,
            image.shape[:2],
            normalized=normalized_depth,
            depth_min_m=depth_min_m,
            depth_max_m=depth_max_m,
        )
        timings["depth_ms"] = (time.perf_counter() - depth_started) * 1000

        # Read Task Memory only after Temporal Memory has published this
        # step's analysis, so waypoint selection steers by the subgoal this
        # step established rather than the previous step's.
        select_started = time.perf_counter()
        selected = self._select_pixel(
            image,
            instruction,
            subgoal_context=(
                self.task_memory.current_subgoal_context()
                if self.task_memory is not None
                else ""
            ),
            depth_m=depth_m,
        )
        timings["select_pixel_ms"] = (
            time.perf_counter() - select_started
        ) * 1000
        selected = self._steer_to_visible_landmark(
            selected,
            current,
            width=image.shape[1],
            height=image.shape[0],
        )
        if selected.action_mode == "PREVIEW":
            # PREVIEW carries no action by design: the controller renders the
            # surrounding views and calls back, so this is not a stop.
            self._record_timings(timings, started)
            return NavigationDecision(
                stop=False,
                raw_response=self.last_model_response,
                action_mode="PREVIEW",
            )
        if selected.stop:
            self._record_timings(timings, started)
            return NavigationDecision(
                stop=True,
                raw_response=self.last_model_response,
                action_mode="EXECUTION",
            )
        if selected.is_turn:
            # An in-place turn never reaches the RGB-D layer: there is no pixel
            # to snap and no point to back-project.
            self._record_timings(timings, started)
            return NavigationDecision(
                stop=False,
                raw_response=self.last_model_response,
                action_mode=self._applied_action_mode(),
                turn_deg=selected.turn_deg,
            )
        requested_uv = self.last_requested_pixel
        assert requested_uv is not None
        waypoint_started = time.perf_counter()
        point = self.actor.waypoint_from_pixel(
            requested_uv,
            depth_m,
            intrinsics,
            camera_to_world,
        )
        self._maybe_lock_doorway_waypoint(
            point,
            camera_to_world=camera_to_world,
        )
        timings["waypoint_ms"] = (time.perf_counter() - waypoint_started) * 1000
        self._record_timings(timings, started)
        return NavigationDecision(
            stop=False,
            point=point,
            raw_response=self.last_model_response,
            action_mode=self._applied_action_mode(),
        )

    def _steer_to_visible_landmark(
        self,
        selected: _WaypointOutput,
        current: Optional[Subgoal],
        *,
        width: int,
        height: int,
    ) -> _WaypointOutput:
        """Walk toward a landmark the Captioner has already located.

        The waypoint model reliably asks to TURN or PREVIEW when the doorway
        sits off-centre in the image, even though its prompt tells it to
        steer to the floor beneath a visible landmark. When the Captioner
        reported that landmark in this same image, the geometry is already
        known: the requested pixel becomes the landmark's own position, and
        the RGB-D layer snaps it down to the floor in front of it. A waypoint
        or STOP the model produced itself is left alone.
        """
        if selected.stop or (
            selected.action_mode != "PREVIEW" and not selected.is_turn
        ):
            return selected
        landmark = self.last_landmark
        if (
            current is None
            or landmark is None
            or self._landmark_subgoal_id != current.subgoal_id
            or not landmark.visible
            or landmark.u is None
            or landmark.v is None
            or landmark.confidence < LANDMARK_STEER_MIN_CONFIDENCE
        ):
            return selected
        intent: NavigationIntent = (
            self._navigation_phase
            if self._navigation_phase != "STOP"
            else "APPROACH_LANDMARK"
        )
        asked = "PREVIEW" if selected.action_mode == "PREVIEW" else "TURN"
        evidence = (
            "steered to the floor beneath the Captioner-localized landmark "
            f"at ({landmark.u},{landmark.v}) instead of the model's {asked}: "
            f"{landmark.evidence}"
        )
        override = _WaypointOutput(
            stop=False,
            intent=intent,
            action_mode="EXECUTION",
            u=landmark.u,
            v=landmark.v,
            confidence=landmark.confidence,
            evidence=evidence,
        )
        self.last_waypoint_guard_reason = (
            f"overrode model {asked}: landmark is localized in the current "
            "image"
        )
        self.last_waypoint_applied_intent = intent
        self.last_waypoint_applied_action_mode = "EXECUTION"
        self.last_waypoint_evidence = evidence
        self.last_waypoint_confidence = landmark.confidence
        self.last_requested_turn_deg = None
        self.last_requested_normalized = (landmark.u, landmark.v)
        self.last_requested_pixel = (
            self._scale_normalized(landmark.u, width),
            self._scale_normalized(landmark.v, height),
        )
        self._force_forward_this_step = False
        self._force_left_turn_this_step = False
        return override

    @staticmethod
    def _doorway_subgoal(subgoal: Optional[Subgoal]) -> bool:
        if subgoal is None:
            return False
        return bool(
            re.search(
                r"\b(?:door|doorway|exit|threshold|cross)\b",
                f"{subgoal.description} {subgoal.completion_criteria}",
                flags=re.IGNORECASE,
            )
        )

    def _maybe_lock_doorway_waypoint(
        self,
        point: NavigationPoint,
        *,
        camera_to_world: Any,
    ) -> None:
        current = (
            self.task_memory.get_current_subgoal()
            if self.task_memory is not None
            else None
        )
        evidence = self.last_waypoint_evidence or ""
        if not (
            self._doorway_subgoal(current)
            and (self.last_waypoint_confidence or 0.0) >= 0.85
            and re.search(
                r"\b(?:door|doorway|threshold|opening|exit)\b",
                evidence,
                flags=re.IGNORECASE,
            )
        ):
            return
        # A point the actor could not place on the floor is a wall, a far
        # window, or a clamped depth reading. Locking it would make the agent
        # chase a target it can never reach and would also hold the doorway
        # completion guard shut, so it steers this step only.
        if self.actor.camera_height_m is not None and not point.on_floor:
            self.last_waypoint_guard_reason = (
                (self.last_waypoint_guard_reason or "")
                + "; doorway waypoint not locked: point is off the floor"
            ).lstrip("; ")
            return
        # Do not replace a still-active structural target with a fresh VLM
        # projection. Near a doorway, small view changes can move the chosen
        # pixel onto the far wall and produce a plausible but very distant
        # world point. The existing target is released only by measured
        # stagnation or by advancing to another subgoal.
        if (
            self._doorway_waypoint is not None
            and self._doorway_waypoint_subgoal_id == current.subgoal_id
        ):
            return
        self._doorway_waypoint = point
        self._doorway_waypoint_subgoal_id = current.subgoal_id
        distance_m = self._planar_waypoint_distance(
            point,
            camera_to_world=camera_to_world,
        )
        self._doorway_waypoint_best_distance_m = distance_m
        self._doorway_waypoint_stagnant_steps = 0

    def _clear_doorway_waypoint(self) -> None:
        self._doorway_waypoint = None
        self._doorway_waypoint_subgoal_id = None
        self._doorway_waypoint_best_distance_m = None
        self._doorway_waypoint_stagnant_steps = 0

    def _release_doorway_waypoint_for_motion(
        self,
        current: Optional[Subgoal],
        *,
        camera_to_world: Any,
    ) -> None:
        """Drop a locked door target as soon as measured motion loops.

        A locked waypoint bypasses the waypoint VLM, so its motion recovery
        must run before the lock's early return. The ordinary waypoint path
        then consumes the same grounded mode and emits its deterministic
        recovery action during this step.
        """
        if self._doorway_target_distance(
            current,
            camera_to_world=camera_to_world,
        ) is None:
            return
        mode, reason = self._motion_grounded_error_candidate("NONE")
        # Four stalled motion intervals distinguish a legitimate short doorway
        # alignment from a same-sign spin, so both loop modes invalidate the
        # locked target.
        if mode not in {"TURN_OSCILLATION", "IN_PLACE_SPIN"}:
            return
        self._clear_doorway_waypoint()
        self.last_error_candidate = mode
        self.last_error_guard_reason = (
            f"released locked doorway waypoint: {reason}"
        )

    @staticmethod
    def _planar_waypoint_distance(
        point: NavigationPoint,
        *,
        camera_to_world: Any,
    ) -> float:
        transform = np.asarray(camera_to_world, dtype=np.float64)
        position_xz = transform[[0, 2], 3]
        target_xz = np.asarray(point.world_xyz, dtype=np.float64)[[0, 2]]
        return float(np.linalg.norm(target_xz - position_xz))

    def _doorway_target_distance(
        self,
        current: Optional[Subgoal],
        *,
        camera_to_world: Any,
    ) -> Optional[float]:
        point = self._doorway_waypoint
        current_id = current.subgoal_id if current is not None else None
        if (
            point is None
            or current_id != self._doorway_waypoint_subgoal_id
            or not self._doorway_subgoal(current)
        ):
            return None
        return self._planar_waypoint_distance(
            point,
            camera_to_world=camera_to_world,
        )

    def _locked_doorway_decision(
        self,
        current: Optional[Subgoal],
        *,
        camera_to_world: Any,
    ) -> Optional[NavigationDecision]:
        point = self._doorway_waypoint
        current_id = current.subgoal_id if current is not None else None
        if (
            point is None
            or current_id != self._doorway_waypoint_subgoal_id
            or not self._doorway_subgoal(current)
        ):
            if point is not None:
                self._clear_doorway_waypoint()
            return None
        distance_m = self._planar_waypoint_distance(
            point,
            camera_to_world=camera_to_world,
        )
        # Reaching the localized threshold releases control to the policy, but
        # keeps the target as completion evidence until Task Memory advances.
        if distance_m <= 0.35:
            return None
        best = self._doorway_waypoint_best_distance_m
        if best is None or distance_m < best - 0.10:
            self._doorway_waypoint_best_distance_m = distance_m
            self._doorway_waypoint_stagnant_steps = 0
        else:
            self._doorway_waypoint_stagnant_steps += 1
        if self._doorway_waypoint_stagnant_steps >= 16:
            self._clear_doorway_waypoint()
            return None
        intent: NavigationIntent = (
            self._navigation_phase
            if self._navigation_phase != "STOP"
            else "APPROACH_LANDMARK"
        )
        self.last_model_response = "reusing locked doorway waypoint"
        self.last_waypoint_raw_response = self.last_model_response
        self.last_waypoint_model_intent = intent
        self.last_waypoint_applied_intent = intent
        self.last_waypoint_model_action_mode = "EXECUTION"
        self.last_waypoint_applied_action_mode = "EXECUTION"
        self.last_waypoint_confidence = 1.0
        self.last_waypoint_evidence = (
            "continuing toward the previously localized structural doorway"
        )
        self.last_waypoint_guard_reason = (
            "reused stable doorway world waypoint; skipped waypoint VLM"
        )
        self.last_requested_turn_deg = None
        return NavigationDecision(
            stop=False,
            point=point,
            raw_response=self.last_model_response,
            action_mode="EXECUTION",
        )

    def _applied_action_mode(self) -> ActionMode:
        """Return the mode this step actually asks the controller to take."""
        return self.last_waypoint_applied_action_mode or "EXECUTION"

    def _selected_preview_view(
        self,
        views: Sequence[PreviewView],
    ) -> int:
        """Deposit the views in working memory and read back the chosen one.

        Depositing triggers the Captioner, so the selection is read
        immediately afterwards. If a replacement selector declines, fall back
        to the most forward view and record that fallback in the guard reason.
        """
        forward_index = min(
            range(len(views)),
            key=lambda index: abs(views[index].yaw_deg),
        )
        self.last_preview_selection = None
        # Kept apart from ``last_waypoint_guard_reason``: ``_select_pixel``
        # clears that one on entry, and these are different guards anyway.
        self.last_preview_guard_reason = None
        if not isinstance(self.temporal_memory, TemporalMemory):
            self.last_preview_guard_reason = (
                "preview views discarded: working memory cannot hold them"
            )
            return forward_index

        self.temporal_memory.set_preview_views(views)
        selection = self.temporal_memory.preview_selection()
        self.last_preview_selection = selection
        if selection is not None:
            # Direction is semantic: the widest free space may be the room we
            # are meant to leave. Depth is still validated when the selected
            # floor point is back-projected, but it must not replace the
            # Captioner's chosen doorway or landmark view.
            return selection.view_index

        error = self.temporal_memory.last_preview_error()
        self.last_preview_guard_reason = (
            "preview view selector failed ({}); used the most forward "
            "view".format(error)
            if error is not None
            else "preview view selector declined; used the most forward view"
        )
        return forward_index

    def _record_timings(
        self,
        timings: dict[str, float],
        started: float,
    ) -> None:
        timings["total_ms"] = (time.perf_counter() - started) * 1000
        self.last_timings = timings

    def _measure_motion(
        self,
        camera_to_world: Any,
    ) -> tuple[float, float]:
        transform = np.asarray(camera_to_world, dtype=np.float64)
        if transform.shape != (4, 4):
            raise ValueError(
                "camera_to_world must be a 4x4 camera transform."
            )
        position = transform[:3, 3].copy()
        forward = -transform[:3, 2]
        yaw_deg = float(
            np.degrees(np.arctan2(forward[0], -forward[2]))
        )
        if self._previous_position is None:
            translation_m = 0.0
            yaw_delta_deg = 0.0
        else:
            translation_m = float(
                np.linalg.norm(
                    position[[0, 2]]
                    - self._previous_position[[0, 2]]
                )
            )
            assert self._previous_yaw_deg is not None
            yaw_delta_deg = (
                yaw_deg - self._previous_yaw_deg + 180.0
            ) % 360.0 - 180.0
        self._previous_position = position
        self._previous_yaw_deg = yaw_deg
        return translation_m, yaw_delta_deg

    def _update_preview_progress(self) -> None:
        """Re-arm PREVIEW only after net displacement from its anchor."""
        if (
            not self._preview_requires_progress
            or self._preview_anchor_position_xz is None
            or self._previous_position is None
        ):
            return
        displacement = float(np.linalg.norm(
            self._previous_position[[0, 2]] - self._preview_anchor_position_xz
        ))
        if displacement < PREVIEW_REARM_TRANSLATION_M:
            return
        self._preview_requires_progress = False
        self._preview_anchor_position_xz = None

    def _phase_for_subgoal(
        self,
        subgoal: Optional[Subgoal],
    ) -> NavigationIntent:
        if subgoal is None:
            return "STOP"
        description = subgoal.description.lower()
        if re.search(r"\bturn\s+left\b", description):
            return "TURN_LEFT"
        if re.search(r"\bturn\s+right\b", description):
            return "TURN_RIGHT"
        arrival_text = (
            f"{subgoal.description} {subgoal.completion_criteria}"
        ).lower()
        if re.search(
            r"\b(stop|final|destination|arrive|arrival|beside)\b|"
            r"\b(?:next to|in front of)\b",
            arrival_text,
        ):
            return "FINAL_APPROACH"
        if re.search(r"\b(straight|hallway|corridor)\b", description):
            return "FOLLOW_CORRIDOR"
        return "APPROACH_LANDMARK"

    def _sync_navigation_phase(
        self,
        subgoal: Optional[Subgoal],
    ) -> None:
        subgoal_id = subgoal.subgoal_id if subgoal is not None else None
        if subgoal_id != self._navigation_subgoal_id:
            self._navigation_subgoal_id = subgoal_id
            # A new stage can introduce a new turn or landmark at the same
            # physical position, so an earlier stage's preview must not block
            # it from looking around.
            self._preview_requires_progress = False
            self._preview_anchor_position_xz = None
            self._final_stop_evidence.clear()
            self._corridor_forward_streak = 0
            self._corridor_heading_yaw_deg = None
            self._subgoal_net_yaw_deg = 0.0
            self._turn_follow_phase_started = False
        self._navigation_phase = (
            "FOLLOW_CORRIDOR"
            if self._turn_follow_phase_started
            else self._phase_for_subgoal(subgoal)
        )

    def _update_navigation_progress(
        self,
        subgoal: Optional[Subgoal],
        *,
        yaw_delta_deg: float,
    ) -> None:
        if subgoal is None:
            return
        self._subgoal_net_yaw_deg += yaw_delta_deg
        initial_phase = self._phase_for_subgoal(subgoal)
        if initial_phase not in ("TURN_LEFT", "TURN_RIGHT"):
            return
        if not re.search(
            r"\bnext\s+(?:left|right)-turn\s+decision\s+point\b",
            subgoal.description,
            flags=re.IGNORECASE,
        ):
            return
        turn_progress = (
            -self._subgoal_net_yaw_deg
            if initial_phase == "TURN_LEFT"
            else self._subgoal_net_yaw_deg
        )
        if turn_progress < TURN_ALIGNMENT_DEG:
            return
        if not self._turn_follow_phase_started:
            self._turn_follow_phase_started = True
            self._corridor_forward_streak = 0
            self._corridor_heading_yaw_deg = None
        self._navigation_phase = "FOLLOW_CORRIDOR"

    def _update_corridor_lock(
        self,
        *,
        translation_m: float,
        yaw_delta_deg: float,
    ) -> None:
        if self._navigation_phase != "FOLLOW_CORRIDOR":
            self._corridor_forward_streak = 0
            return
        if (
            translation_m >= 0.10
            and abs(yaw_delta_deg) < TURN_EVIDENCE_DEG
        ):
            self._corridor_forward_streak += 1
        elif self._corridor_heading_yaw_deg is None:
            self._corridor_forward_streak = 0
        if (
            self._corridor_heading_yaw_deg is None
            and self._corridor_forward_streak
            >= CORRIDOR_LOCK_FORWARD_STEPS
            and self._previous_yaw_deg is not None
        ):
            self._corridor_heading_yaw_deg = self._previous_yaw_deg

    def _next_subgoal(
        self,
        current: Optional[Subgoal],
    ) -> Optional[Subgoal]:
        if current is None:
            return None
        for index, item in enumerate(self.subgoals):
            if item.subgoal_id == current.subgoal_id:
                next_index = index + 1
                return (
                    self.subgoals[next_index]
                    if next_index < len(self.subgoals)
                    else None
                )
        return None

    def _landmark_is_near_for_current_subgoal(
        self,
        current: Optional[Subgoal],
    ) -> bool:
        return bool(
            current is not None
            and self.last_landmark is not None
            and self._landmark_subgoal_id == current.subgoal_id
            and self.last_landmark.visible
            and self.last_landmark.proximity in ("NEAR", "AT")
            and self.last_landmark.confidence >= 0.6
        )

    def _record_behavior(
        self,
        *,
        subgoal_id: Optional[str],
        translation_m: float,
        yaw_delta_deg: float,
        requested_waypoint: Optional[tuple[int, int]],
    ) -> None:
        if translation_m >= 0.10:
            behavior = "MOVE_FORWARD"
        elif yaw_delta_deg <= -TURN_EVIDENCE_DEG:
            behavior = "TURN_LEFT"
        elif yaw_delta_deg >= TURN_EVIDENCE_DEG:
            behavior = "TURN_RIGHT"
        else:
            behavior = "NO_MOTION"
        self._behavior_history.append(
            {
                "subgoal_id": subgoal_id,
                "behavior": behavior,
                "translation_m": translation_m,
                "yaw_delta_deg": yaw_delta_deg,
                "requested_waypoint": requested_waypoint,
            }
        )

    def behavior_history(self) -> tuple[dict[str, Any], ...]:
        """Return recent measured outcomes of previously requested waypoints."""
        return tuple(dict(item) for item in self._behavior_history)

    def _behavior_context(self) -> str:
        if not self._behavior_history:
            return "Recent waypoint behavior: none (first observation)."
        lines = ["Recent waypoint behavior, oldest first:"]
        for item in self._behavior_history:
            waypoint = item["requested_waypoint"]
            waypoint_text = (
                "none"
                if waypoint is None
                else f"({waypoint[0]},{waypoint[1]})"
            )
            lines.append(
                f"- subgoal={item['subgoal_id']}; "
                f"behavior={item['behavior']}; "
                f"translation_m={item['translation_m']:.3f}; "
                f"yaw_delta_deg={item['yaw_delta_deg']:+.1f}; "
                f"previous_normalized_waypoint={waypoint_text}"
            )
        return "\n".join(lines)

    def prepare_task(self, instruction: str) -> tuple[Subgoal, ...]:
        """Generate and strictly validate an ordered subgoal plan."""
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be a non-empty string.")

        normalized_instruction = instruction.strip()
        last_error: Optional[Exception] = None
        response_text = ""
        for attempt in range(SUBGOAL_GENERATION_ATTEMPTS):
            if attempt == 0:
                content = [
                    f"Navigation instruction: {normalized_instruction}"
                ]
            else:
                content = [
                    f"Navigation instruction: {normalized_instruction}\n"
                    "The previous response failed strict validation "
                    f"({last_error}). Write one stage per line as "
                    "id|description|completion criterion, with IDs counting "
                    "1, 2, 3 and no gaps. Output only those lines: no JSON, "
                    "no brackets, no commentary."
                ]
            response = self.llm(
                content,
                system_prompt=SUBGOAL_PROMPT,
                # A plan that nests its stages outgrows the budget it would
                # need when flat, and a response truncated before its closing
                # brackets cannot be parsed at all.
                max_tokens=1024,
                # Greedy decoding first; a retry has to be allowed to diverge,
                # or it reproduces the response that just failed.
                temperature=0.0 if attempt == 0 else 0.7,
            )
            response_text = str(response)
            self.last_subgoal_response = response_text
            try:
                subgoals = parse_subgoal_plan(
                    response_text,
                    instruction=normalized_instruction,
                )
                break
            except Exception as exc:
                last_error = exc
        else:
            raise ValueError(
                "Actor returned invalid subgoal JSON after "
                f"{SUBGOAL_GENERATION_ATTEMPTS} attempts "
                # Without the underlying exception a plain AttributeError or
                # NameError in the parsing path is indistinguishable from a
                # genuinely malformed model response.
                f"({type(last_error).__name__}: {last_error}): "
                f"{response_text!r}"
            ) from last_error

        self.task_instruction = normalized_instruction
        self.subgoals = subgoals
        self.reset_memory(normalized_instruction, subgoals)
        return tuple(self.subgoals)

    def reset_memory(
        self,
        instruction: str,
        subgoals: Sequence[Subgoal],
    ) -> None:
        """Initialize task state and the growing completion memory."""
        if self.task_memory is None:
            self.task_memory = TaskMemory(instruction, subgoals=subgoals)
        else:
            self.task_memory.reset(goal=instruction, subgoals=subgoals)

        if self.temporal_memory is None:
            self.temporal_memory = TemporalMemory(
                captioner=TemporalCaptioner(
                    engine=self.llm,
                    config=TemporalCaptionerConfig(
                        enable_error_detection=True,
                        min_error_detection_frames=8,
                        max_image_edge=TEMPORAL_MAX_IMAGE_EDGE,
                        # One compact response now carries landmark,
                        # completion, error, and final-target evidence.
                        max_tokens=max(256, STRUCTURED_VLM_MAX_TOKENS),
                    ),
                ),
                task_memory=self.task_memory,
            )
        else:
            self.temporal_memory.reset()
        self._reset_runtime_state()

    # The landmark and waypoint policies prompt the shared checkpoint through
    # the agent, so both model helpers stay reachable under their mixin names.
    def _rgb_to_png(self, rgb: np.ndarray) -> bytes:
        return self.actor.rgb_to_png(rgb)

    def _as_rgb_array(self, rgb: Any) -> np.ndarray:
        return self.actor.as_rgb_array(rgb)

    def _extract_json_object(self, response: str) -> dict[str, Any]:
        return self.actor.extract_json_object(response)


__all__ = (
    "CameraIntrinsics",
    "DEFAULT_MODEL_PATH",
    "TemporalMemory",
    "NavigationDecision",
    "NavigationPoint",
    "POINT_PROMPT",
    "PreviewView",
    "Subgoal",
    "SUBGOAL_PROMPT",
    "VLNAgent",
)
