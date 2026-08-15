"""Compact Qwen judgement for up to eight consecutive VLN observations."""

from __future__ import annotations

import io
import json
import re
import time
from collections import OrderedDict
from typing import Any, Mapping, Optional, Sequence

from pydantic import BaseModel, ConfigDict

from ...data_models import (
    CaptionResult,
    DualWindowCaptionResult,
    ErrorMode,
    FinalTargetEvidence,
    PreviewSelection,
    SceneAnalysisRequest,
    SceneAnalysisResult,
    SceneLandmark,
    Subgoal,
    TemporalAnalysisRequest,
    TemporalCaptionerConfig,
    TemporalFrameInput,
    TemporalInputError,
)
from agentflow.agents.models_embodied_v2.skiils.preview import (
    PREVIEW_SELECTION_PROMPT,
    parse_preview_selection,
)


DEFAULT_MODEL_PATH = "models/Qwen3-VL-8B-Instruct"


class TemporalCaptionerError(RuntimeError):
    pass


class TemporalInferenceError(TemporalCaptionerError):
    pass


class TemporalOutputError(TemporalCaptionerError, ValueError):
    pass


class _ModelResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    completed: bool
    error: bool
    error_mode: ErrorMode


class _CompletionModelResult(BaseModel):
    """Strict result schema for the completion-only visual judgement."""

    model_config = ConfigDict(extra="forbid", strict=True)

    completed: bool


class _ErrorModelResult(BaseModel):
    """Strict result schema for the recent-window error judgement."""

    model_config = ConfigDict(extra="forbid", strict=True)

    error: bool
    error_mode: ErrorMode
    confidence: float
    evidence: str


class _SceneLandmarkModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    visible: bool
    direction: str
    proximity: str
    passed: bool
    destination_dominant: bool
    u: Optional[int] = None
    v: Optional[int] = None
    confidence: float


class _FinalTargetModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    visible: bool
    proximity: str
    confidence: float


class _SceneModelResult(BaseModel):
    """Compact wire schema shared by all temporal perception tasks."""

    model_config = ConfigDict(extra="forbid")

    landmark: _SceneLandmarkModel
    completed: bool
    completion_confidence: float
    door_state: str
    door_camera_side: str
    door_transition: str
    current_room_side: str
    error_mode: str
    error_confidence: float
    final_target: _FinalTargetModel
    evidence: str


SYSTEM_PROMPT = """Inspect the ordered first-person navigation images, oldest
first. There may be as few as one image early in a subgoal.
Judge whether the final image visibly proves the current subgoal is complete.
Also detect a cumulative visual error:
- WALL_STUCK: the camera stays at the same nearby obstacle.
- TURN_OSCILLATION: views repeatedly alternate left and right.
- IN_PLACE_SPIN: views rotate and return to earlier views without progress.
- GET_NOWHERE: the sequence shows no meaningful visual progress.

Set error=false and error_mode=NONE when no error is visible. A cumulative
error needs several images as evidence, so report NONE when too few are given.
Output only the requested compact JSON."""


COMPLETION_SYSTEM_PROMPT = """Inspect the selected ordered visual evidence
for one active navigation subgoal. Judge only whether the final observation,
viewed together with the ordered evidence, visibly proves that subgoal's
completion criterion. Motion and landmark fields are auxiliary evidence, not
substitutes for visual confirmation. Output only:
{"completed":false}"""


ERROR_SYSTEM_PROMPT = """Inspect the recent ordered first-person observations
and their measured motion metadata. Diagnose a cumulative execution error only
when the recent ordered evidence is clear:
- WALL_STUCK: repeated negligible translation at an obstacle;
- TURN_OSCILLATION: alternating left/right turns without progress;
- IN_PLACE_SPIN: rotation returns to earlier views without progress;
- GET_NOWHERE: no meaningful progress despite attempted navigation.

Set error=false and error_mode=NONE when there is no clear error. Confidence
must be in [0,1], and evidence must briefly ground the judgement in the recent
views and motion. Output only:
{"error":false,"error_mode":"NONE","confidence":0.0,"evidence":"..."}"""


DOOR_SCENE_SYSTEM_PROMPT = """You are the temporal scene observer specialized
in verifying physical doorway passage from ordered first-person video frames.
Answer one primary question: did the camera itself pass through the active
subgoal's structural doorway? Do not infer passage from the route instruction,
from seeing the destination through an opening, or from the door disappearing.

Use this evidence protocol before producing JSON:
1. Treat the first frame as the ORIGINAL_SIDE visual reference. Compare its
   walls, fixtures, and room layout with the CURRENT (last) frame.
2. Inspect adjacent recent frames for a literal passage transition: the same
   doorway grows toward the camera, its jambs/threshold move to and beyond the
   image boundaries, then the far-side space fills the view while forward
   translation occurs.
3. If a doorway moves toward an image edge while yaw changes and then vanishes,
   that is TURNED_AWAY, not PASSED_THROUGH. If the CURRENT frame still shows
   original-side walls, bathtub, sink, bed, or decorations, current_room_side
   is ORIGINAL_SIDE even when a blue room is visible through the opening.
4. Never reconstruct a crossing in an unobserved interval. Ambiguous evidence
   means completed=false.

Use door_state NOT_VISIBLE, APPROACHING, AT_THRESHOLD, CROSSING, or CROSSED;
door_camera_side UNKNOWN, BEFORE_DOOR, AT_DOOR, or AFTER_DOOR;
door_transition NONE, TURNED_AWAY, APPROACHED, or PASSED_THROUGH; and
current_room_side ORIGINAL_SIDE, FAR_SIDE, or AMBIGUOUS. landmark refers only
to the structural opening in the CURRENT frame. Its direction is LEFT, CENTER,
RIGHT, or UNKNOWN. destination_dominant means the CURRENT frame is dominated
by the far-side space, not that it is merely visible through the doorway.

completed=true is allowed only when all of these independent outputs agree:
door_state=CROSSED, door_camera_side=AFTER_DOOR,
door_transition=PASSED_THROUGH, current_room_side=FAR_SIDE, and
landmark.destination_dominant=true. Otherwise return completed=false. Diagnose
an execution error only when the multi-frame evidence is clear; otherwise use
NONE. This doorway subgoal is not the final target, so final_target is always
invisible/UNKNOWN.

Return only this exact compact JSON shape:
{"door_state":"NOT_VISIBLE","door_camera_side":"UNKNOWN","door_transition":"NONE","current_room_side":"AMBIGUOUS","landmark":{"visible":false,"direction":"UNKNOWN","proximity":"UNKNOWN","passed":false,"destination_dominant":false,"u":null,"v":null,"confidence":0.0},"completed":false,"completion_confidence":0.0,"error_mode":"NONE","error_confidence":0.0,"final_target":{"visible":false,"proximity":"UNKNOWN","confidence":0.0},"evidence":"brief frame-grounded reason"}"""


SCENE_SYSTEM_PROMPT = """You are the temporal scene observer for indoor
navigation. Inspect all ordered first-person frames for the active subgoal. The
last image is the current view. Judge only the active subgoal; do not infer its
completion from objects belonging to a later route stage.

For an active doorway/exit/crossing subgoal, landmark means the structural
doorway itself: a traversable opening with jambs/frame and threshold. A sink,
bathtub, window, wall decoration, tiled panel, blue wall, or room seen through
an opening is not the doorway. Track that same doorway across the sequence.
Set door_state to exactly one of:
- NOT_VISIBLE: the structural doorway cannot be located;
- APPROACHING: it is visible and the camera remains on the original side;
- AT_THRESHOLD: the camera is at its opening but has not passed through;
- CROSSING: the ordered frames show the camera moving through the opening;
- CROSSED: the camera is clearly in the space on the far side.
Set door_camera_side to UNKNOWN, BEFORE_DOOR, AT_DOOR, or AFTER_DOOR. A room
visible through a doorway never proves CROSSED/AFTER_DOOR. If the doorway is
still ahead or beside the camera, use BEFORE_DOOR. For a doorway subgoal,
completed=true only for CROSSED plus AFTER_DOOR; otherwise completed=false.

Independently report two doorway proof facts. door_transition is exactly one
of NONE, TURNED_AWAY, APPROACHED, or PASSED_THROUGH. current_room_side is
exactly ORIGINAL_SIDE, FAR_SIDE, or AMBIGUOUS. The first frame establishes the
visual appearance of ORIGINAL_SIDE. PASSED_THROUGH means the supplied images
actually show the jamb/threshold pass the camera, not merely that a door was
seen and later disappeared. FAR_SIDE means the CURRENT frame no longer shows
the first frame's room surfaces or fixtures and is dominated by the space that
was beyond the opening. For completed=true both must be PASSED_THROUGH and
FAR_SIDE, and landmark.destination_dominant must be true.

Decide a doorway crossing from image flow, not from the doorway disappearing.
CROSSED/AFTER_DOOR requires the most recent consecutive frames to visibly show
this sequence: the same opening grows to fill the view, its jambs/threshold
move past the camera, and the current frame is clearly inside the far-side
space. There must be forward translation during that visible transition. If
the doorway instead moves left/right and leaves the image while yaw changes,
the camera merely turned away: use BEFORE_DOOR and completed=false. If the
current frame still contains recognizable surfaces or fixtures from the
original room, it is not AFTER_DOOR. Missing or ambiguous crossing evidence
always means completed=false; never reconstruct an unseen crossing.

For a non-doorway subgoal use door_state=NOT_APPLICABLE and
door_camera_side=NOT_APPLICABLE, door_transition=NOT_APPLICABLE, and
current_room_side=NOT_APPLICABLE. Track only that subgoal's named landmark or
final destination. Seeing a target at a distance is not reaching it. On the
final subgoal, final_target reports visible and FAR/NEAR/AT; otherwise it must
be invisible/UNKNOWN. For a final subgoal, completed=true requires
final_target.visible=true and proximity=AT; NEAR still means continue moving.
A final-target proximity must use these visual meanings consistently:
- FAR: the target is distant or is only seen through an opening;
- NEAR: the camera is in the destination room and approaching, but the target
  boundary is not yet in the immediate foreground;
- AT: the target edge, rail, or surrounding floor boundary is in the immediate
  foreground and the camera is directly beside it.
Never return completed=true together with FAR or NEAR. If the evidence says
"reached", "directly beside", or "in the foreground", use AT only when that
claim is visibly true; otherwise keep completed=false and use FAR or NEAR.
A swimming pool is a large built-in pool and surrounding
pool-room floor; a bathtub, sink, tiled wash area, blue wall, picture, or a
glimpse through a doorway is not a swimming pool. If the source-room doorway
frame, sink, bathtub, or other source-room fixture remains visible, the camera
has not reached AT even when the pool itself is large in the image; use FAR or
NEAR and completed=false. AT means the camera has entered the destination
space and is directly beside the requested target. A newly activated movement
subgoal with only one current frame has no temporal proof that the requested
walk or approach occurred, so return completed=false when the destination
identity or proximity is not visually certain.

Also diagnose a cumulative execution error only with clear multi-frame
evidence: WALL_STUCK, TURN_OSCILLATION, IN_PLACE_SPIN, GET_NOWHERE, or NONE.
direction and u/v always refer to the current image. Evidence must identify
what structural landmark is visible and describe the temporal change; when
completion is uncertain return completed=false.

Return only this exact compact JSON shape:
{"door_state":"NOT_VISIBLE","door_camera_side":"UNKNOWN","door_transition":"NONE","current_room_side":"AMBIGUOUS","landmark":{"visible":false,"direction":"UNKNOWN","proximity":"UNKNOWN","passed":false,"destination_dominant":false,"u":null,"v":null,"confidence":0.0},"completed":false,"completion_confidence":0.0,"error_mode":"NONE","error_confidence":0.0,"final_target":{"visible":false,"proximity":"UNKNOWN","confidence":0.0},"evidence":"brief frame-grounded reason"}"""


class TemporalCaptioner:
    """Judge subgoal completion and visual error in one compact call."""

    def __init__(
        self,
        *,
        engine: Optional[Any] = None,
        model_path: str = DEFAULT_MODEL_PATH,
        config: Optional[TemporalCaptionerConfig] = None,
        use_cache: bool = False,
        debug_performance: bool = False,
        engine_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.model_path = model_path
        self.config = config or TemporalCaptionerConfig()
        self._engine = engine
        self._use_cache = use_cache
        self._debug_performance = debug_performance
        self._engine_kwargs = dict(engine_kwargs or {})
        self.last_raw_response: Optional[str] = None
        self.last_failed_raw_response: Optional[str] = None
        self.last_preview_raw_response: Optional[str] = None
        # A bounded identity cache avoids re-resizing and re-encoding the same
        # retained temporal frames on every subsequent step. The source object
        # is kept beside the bytes so a recycled id can never hit stale data.
        self._png_cache: OrderedDict[int, tuple[Any, bytes]] = OrderedDict()

    def analyze_scene(
        self,
        request: SceneAnalysisRequest,
    ) -> SceneAnalysisResult:
        """Understand landmark, completion, error, and target in one call."""
        if not isinstance(request, SceneAnalysisRequest):
            raise TemporalInputError(
                "analyze_scene expects SceneAnalysisRequest"
            )
        engine = self._get_engine()
        kwargs = self._inference_kwargs()
        started = time.perf_counter()
        try:
            system_prompt = (
                DOOR_SCENE_SYSTEM_PROMPT
                if self._is_doorway_subgoal(request.subgoal)
                else SCENE_SYSTEM_PROMPT
            )
            response = engine(
                self._scene_content(request),
                system_prompt=system_prompt,
                **kwargs,
            )
            raw_response = str(response)
            self.last_raw_response = raw_response
            parsed = _SceneModelResult.model_validate(
                self._json_value(response)
            )
            result = self._scene_result(
                request,
                parsed,
                raw_response,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        except TemporalOutputError:
            self._remember_failed_response("scene", self.last_raw_response)
            raise
        except Exception as exc:
            raise TemporalInferenceError("Scene inference failed") from exc
        return result

    def select(
        self,
        *,
        subgoal: Optional[Subgoal],
        views: Sequence[Any],
    ) -> Optional[PreviewSelection]:
        """Select one simultaneous preview heading with the shared VLM.

        Preview images bypass the temporal PNG cache because they are
        short-lived views from one position and must not evict retained
        temporal frames. This selection replaces the old second waypoint
        inference, keeping PREVIEW at one additional VLM request.
        """
        values = tuple(views)
        if subgoal is None or not values:
            return None
        content: list[Any] = [
            f"Active subgoal: {subgoal.description}\n"
            f"Completion criterion: {subgoal.completion_criteria}\n"
            f"Available simultaneous views: {len(values)}."
        ]
        if self._is_doorway_subgoal(subgoal):
            content.append(
                "Select the view that most clearly centers the structural "
                "doorway and its traversable floor approach. A bathtub, "
                "sink, tiled panel, window, or blue wall is not a doorway."
            )
        for index, view in enumerate(values):
            yaw_deg = getattr(view, "yaw_deg", None)
            image = getattr(view, "rgb", view)
            if yaw_deg is None:
                raise TemporalInputError(
                    "preview views must provide yaw_deg"
                )
            content.extend(
                (
                    f"view_index={index}; yaw_deg={float(yaw_deg):+.1f}",
                    self._encode_png(image),
                )
            )
        kwargs = self._inference_kwargs()
        kwargs["max_tokens"] = min(96, self.config.max_tokens)
        try:
            response = self._get_engine()(
                content,
                system_prompt=PREVIEW_SELECTION_PROMPT,
                **kwargs,
            )
            self.last_preview_raw_response = str(response)
            return parse_preview_selection(
                dict(self._json_value(response)),
                view_count=len(values),
            )
        except Exception as exc:
            raise TemporalInferenceError(
                "Preview view selection failed"
            ) from exc

    def analyze(self, request: TemporalAnalysisRequest) -> CaptionResult:
        if not isinstance(request, TemporalAnalysisRequest):
            raise TemporalInputError("analyze expects TemporalAnalysisRequest")
        engine = self._get_engine()
        kwargs = {
            "system_prompt": SYSTEM_PROMPT,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if getattr(engine, "supports_image_pixel_budget", False):
            kwargs.update(
                image_min_pixels=min(64**2, self.config.max_image_edge**2),
                image_max_pixels=self.config.max_image_edge**2,
            )

        started = time.perf_counter()
        try:
            response = engine(self._content(request), **kwargs)
            self.last_raw_response = str(response)
            result = self._parse_single(response)
        except TemporalOutputError:
            raise
        except Exception as exc:
            raise TemporalInferenceError("Temporal inference failed") from exc
        return CaptionResult(
            subgoal_id=request.subgoal.subgoal_id,
            completed=result.completed,
            error=result.error,
            error_mode=result.error_mode,
            raw_response=self.last_raw_response,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    def analyze_dual_window(
        self,
        *,
        subgoal: Subgoal,
        completion_frames: Sequence[TemporalFrameInput],
        error_frames: Sequence[TemporalFrameInput],
    ) -> DualWindowCaptionResult:
        """Independently judge completion and recent execution errors.

        Completion can use a compact representative history, whereas error
        diagnosis is intentionally limited to the recent suffix. This keeps
        error evidence local while retaining an anchor frame for completion.
        """
        completion = self._validate_window(
            completion_frames, label="completion"
        )
        error = self._validate_window(error_frames, label="error", max_size=8)
        error_ids = tuple(frame.frame_id for frame in error)
        completion_suffix = tuple(
            frame.frame_id for frame in completion[-len(error) :]
        )
        if error_ids != completion_suffix:
            raise TemporalInputError(
                "error frames must be a recent suffix of completion frames"
            )
        if not isinstance(subgoal, Subgoal):
            raise TemporalInputError("subgoal must be a Subgoal")

        self.last_failed_raw_response = None
        engine = self._get_engine()
        kwargs = self._inference_kwargs()
        total_started = time.perf_counter()
        completion_started = time.perf_counter()
        try:
            completion_response = engine(
                self._completion_content(subgoal, completion),
                system_prompt=COMPLETION_SYSTEM_PROMPT,
                **kwargs,
            )
            completion_raw = str(completion_response)
            completion_result = self._parse_completion(completion_response)
        except TemporalOutputError:
            self._remember_failed_response("completion", self.last_raw_response)
            raise
        except Exception as exc:
            raise TemporalInferenceError("Temporal completion inference failed") from exc
        completion_latency_ms = (time.perf_counter() - completion_started) * 1000

        error_result = None
        error_raw = "disabled"
        error_latency_ms = 0.0
        if self.config.enable_error_detection:
            if len(error) < self.config.min_error_detection_frames:
                error_raw = "insufficient_frames"
            else:
                error_started = time.perf_counter()
                error_result, error_raw = self._analyze_error_window(
                    engine=engine,
                    subgoal=subgoal,
                    frames=error,
                    kwargs=kwargs,
                )
                error_latency_ms = (time.perf_counter() - error_started) * 1000

        self.last_raw_response = error_raw if error_result is not None else completion_raw
        return DualWindowCaptionResult(
            subgoal_id=subgoal.subgoal_id,
            completed=completion_result.completed,
            error=False if error_result is None else error_result.error,
            error_mode="NONE" if error_result is None else error_result.error_mode,
            completion_window_size=len(completion),
            error_window_size=len(error),
            completion_raw_response=completion_raw,
            error_raw_response=error_raw,
            completion_latency_ms=completion_latency_ms,
            error_latency_ms=error_latency_ms,
            latency_ms=(time.perf_counter() - total_started) * 1000,
            error_confidence=(
                0.0 if error_result is None else error_result.confidence
            ),
            error_evidence=(
                "" if error_result is None else error_result.evidence.strip()
            ),
        )

    def _content(self, request: TemporalAnalysisRequest) -> list[Any]:
        content: list[Any] = [
            f"Subgoal: {request.subgoal.description}\n"
            f"Completion proof: {request.subgoal.completion_criteria}\n"
            f"Images: {len(request.frames)}, ordered oldest first."
        ]
        content.extend(self._png(frame.image) for frame in request.frames)
        content.append(
            'Return exactly: {"completed":false,"error":false,'
            '"error_mode":"NONE"}'
        )
        return content

    def _scene_content(self, request: SceneAnalysisRequest) -> list[Any]:
        doorway_stage = self._is_doorway_subgoal(request.subgoal)
        door_contract = (
            "This is a DOORWAY subgoal. Door fields must not be "
            "NOT_APPLICABLE. When the structural opening is visible, "
            "landmark must locate that opening in the current image."
            if doorway_stage
            else (
                "This is not a doorway subgoal. Use NOT_APPLICABLE for all "
                "four door fields."
            )
        )
        content: list[Any] = [
            f"Active subgoal: {request.subgoal.description}\n"
            f"Completion criterion: {request.subgoal.completion_criteria}\n"
            f"Active subgoal type: "
            f"{'DOORWAY' if doorway_stage else 'NON_DOORWAY'}\n"
            f"Door output contract: {door_contract}\n"
            f"Is final subgoal: {request.is_final_subgoal}\n"
            f"Frames: {len(request.frames)}, oldest first. The image after "
            f"the final metadata line is the CURRENT observation."
        ]
        last_index = len(request.frames) - 1
        for index, frame in enumerate(request.frames):
            role = "CURRENT" if index == last_index else "HISTORICAL"
            content.append(
                f"frame={frame.frame_id}; role={role}; translation_m="
                f"{frame.translation_m:.3f}; yaw_delta_deg="
                f"{frame.yaw_delta_deg:+.1f}; path_m="
                f"{frame.subgoal_path_length_m:.3f}"
            )
            content.append(self._png(frame.image))
        return content

    @staticmethod
    def _is_doorway_subgoal(subgoal: Subgoal) -> bool:
        text = f"{subgoal.description} {subgoal.completion_criteria}"
        return bool(
            re.search(
                r"\b(?:exit|leave|doorway|threshold|cross(?:ed|ing)?)\b",
                text,
                flags=re.IGNORECASE,
            )
        )

    @staticmethod
    def _bounded_confidence(value: Any) -> float:
        try:
            return min(1.0, max(0.0, float(value)))
        except (TypeError, ValueError):
            return 0.0

    def _scene_result(
        self,
        request: SceneAnalysisRequest,
        parsed: _SceneModelResult,
        raw_response: str,
        *,
        latency_ms: float,
    ) -> SceneAnalysisResult:
        evidence = str(parsed.evidence or "scene evidence unavailable").strip()
        landmark = parsed.landmark
        direction = landmark.direction.upper()
        proximity = landmark.proximity.upper()
        if direction not in {"LEFT", "CENTER", "RIGHT", "UNKNOWN"}:
            direction = "UNKNOWN"
        if proximity not in {"FAR", "NEAR", "AT", "UNKNOWN"}:
            proximity = "UNKNOWN"
        visible = bool(landmark.visible)
        passed = bool(landmark.passed)
        if not visible and not passed:
            direction = "UNKNOWN"
            proximity = "UNKNOWN"
        if visible and (direction == "UNKNOWN" or proximity == "UNKNOWN"):
            visible = False
            direction = "UNKNOWN"
            proximity = "UNKNOWN"
        u = landmark.u if visible else None
        v = landmark.v if visible else None
        if u is not None and not 0 <= u <= 1000:
            u = None
        if v is not None and not 0 <= v <= 1000:
            v = None
        if visible and u is not None:
            # Coordinates are less ambiguous than a free enum and the model
            # occasionally returns direction=LEFT with a centered u. Resolve
            # the contradiction once at the perception boundary.
            direction = "LEFT" if u < 400 else "RIGHT" if u > 600 else "CENTER"

        target = parsed.final_target
        target_visible = bool(target.visible) and request.is_final_subgoal
        target_proximity = target.proximity.upper()
        if target_proximity not in {"FAR", "NEAR", "AT", "UNKNOWN"}:
            target_proximity = "UNKNOWN"
        if not target_visible:
            target_proximity = "UNKNOWN"
        # Qwen occasionally returns a self-contradictory final-target object:
        # completed=true and evidence such as "directly beside" or
        # "foreground", but proximity=FAR. Reconcile only from the same
        # model response when its positive visual statement is unambiguous,
        # high-confidence, and not accompanied by an approach/distance cue.
        final_evidence_positive = bool(
            re.search(
                r"\b(?:directly beside|positioned beside|has reached|"
                r"have reached|in the (?:immediate )?foreground|foreground)\b",
                evidence,
                flags=re.IGNORECASE,
            )
        )
        final_evidence_negative = bool(
            re.search(
                r"\b(?:not yet|still approaching|at a distance|in the "
                r"distance|through (?:an|the) (?:door|doorway|opening)|"
                r"has not reached|have not reached)\b",
                evidence,
                flags=re.IGNORECASE,
            )
        )
        if (
            request.is_final_subgoal
            and target_visible
            and bool(parsed.completed)
            and bool(landmark.destination_dominant)
            and self._bounded_confidence(target.confidence) >= 0.60
            and final_evidence_positive
            and not final_evidence_negative
        ):
            target_proximity = "AT"

        error_mode = parsed.error_mode.upper()
        if error_mode not in {
            "NONE",
            "WALL_STUCK",
            "TURN_OSCILLATION",
            "IN_PLACE_SPIN",
            "GET_NOWHERE",
        }:
            error_mode = "NONE"
        door_state = parsed.door_state.upper()
        if door_state not in {
            "NOT_APPLICABLE",
            "NOT_VISIBLE",
            "APPROACHING",
            "AT_THRESHOLD",
            "CROSSING",
            "CROSSED",
        }:
            door_state = "NOT_VISIBLE"
        door_camera_side = parsed.door_camera_side.upper()
        if door_camera_side not in {
            "NOT_APPLICABLE",
            "UNKNOWN",
            "BEFORE_DOOR",
            "AT_DOOR",
            "AFTER_DOOR",
        }:
            door_camera_side = "UNKNOWN"
        doorway_stage = self._is_doorway_subgoal(request.subgoal)
        if doorway_stage and door_state == "NOT_APPLICABLE":
            door_state = "NOT_VISIBLE"
        if doorway_stage and door_camera_side == "NOT_APPLICABLE":
            door_camera_side = "UNKNOWN"
        if not doorway_stage:
            door_state = "NOT_APPLICABLE"
            door_camera_side = "NOT_APPLICABLE"
        door_transition = parsed.door_transition.upper()
        if door_transition not in {
            "NOT_APPLICABLE",
            "NONE",
            "TURNED_AWAY",
            "APPROACHED",
            "PASSED_THROUGH",
        }:
            door_transition = "NONE"
        current_room_side = parsed.current_room_side.upper()
        if current_room_side not in {
            "NOT_APPLICABLE",
            "ORIGINAL_SIDE",
            "FAR_SIDE",
            "AMBIGUOUS",
        }:
            current_room_side = "AMBIGUOUS"
        if doorway_stage and door_transition == "NOT_APPLICABLE":
            door_transition = "NONE"
        if doorway_stage and current_room_side == "NOT_APPLICABLE":
            current_room_side = "AMBIGUOUS"
        if not doorway_stage:
            door_transition = "NOT_APPLICABLE"
            current_room_side = "NOT_APPLICABLE"
        completed = bool(parsed.completed)
        if doorway_stage:
            completed = bool(
                completed
                and door_state == "CROSSED"
                and door_camera_side == "AFTER_DOOR"
                and door_transition == "PASSED_THROUGH"
                and current_room_side == "FAR_SIDE"
                and landmark.destination_dominant
            )
        elif request.is_final_subgoal:
            # Keep final completion model-owned while enforcing consistency
            # among the model's own structured facts and evidence.
            completed = bool(
                completed
                and target_visible
                and target_proximity == "AT"
            )
        passed = door_state == "CROSSED" or passed
        return SceneAnalysisResult(
            subgoal_id=request.subgoal.subgoal_id,
            landmark=SceneLandmark(
                visible=visible,
                direction=direction,
                proximity=proximity,
                passed=passed,
                destination_dominant=bool(
                    landmark.destination_dominant
                ),
                confidence=self._bounded_confidence(landmark.confidence),
                evidence=evidence,
                u=u,
                v=v,
            ),
            completed=completed,
            completion_confidence=self._bounded_confidence(
                parsed.completion_confidence
            ),
            completion_evidence=evidence,
            door_state=door_state,
            door_camera_side=door_camera_side,
            error=error_mode != "NONE",
            error_mode=error_mode,
            error_confidence=self._bounded_confidence(
                parsed.error_confidence
            ),
            error_evidence=evidence,
            final_target=FinalTargetEvidence(
                visible=target_visible,
                proximity=target_proximity,
                confidence=self._bounded_confidence(target.confidence),
                evidence=evidence,
            ),
            raw_response=raw_response,
            latency_ms=latency_ms,
        )

    def _parse_single(self, response: Any) -> _ModelResult:
        value = self._json_value(response)
        if set(value) == {"completed"}:
            completion = self._validate_model(
                _CompletionModelResult,
                value,
            )
            return _ModelResult(
                completed=completion.completed,
                error=False,
                error_mode="NONE",
            )
        try:
            result = _ModelResult.model_validate(value)
        except Exception as exc:
            raise TemporalOutputError("model returned the wrong schema") from exc
        if result.error != (result.error_mode != "NONE"):
            raise TemporalOutputError("error and error_mode disagree")
        return result

    def _parse_completion(self, response: Any) -> _CompletionModelResult:
        return self._validate_model(
            _CompletionModelResult,
            self._json_value(response),
        )

    def _parse_error(self, response: Any) -> _ErrorModelResult:
        result = self._validate_model(
            _ErrorModelResult,
            self._json_value(response),
        )
        if result.error != (result.error_mode != "NONE"):
            raise TemporalOutputError("model returned the wrong schema")
        if not 0.0 <= result.confidence <= 1.0:
            raise TemporalOutputError("model returned the wrong schema")
        if not result.evidence.strip():
            raise TemporalOutputError("model returned the wrong schema")
        return result

    @staticmethod
    def _validate_model(model: type[BaseModel], value: Mapping[str, Any]) -> Any:
        try:
            return model.model_validate(value)
        except Exception as exc:
            raise TemporalOutputError("model returned the wrong schema") from exc

    @staticmethod
    def _json_value(response: Any) -> Mapping[str, Any]:
        if isinstance(response, Mapping):
            return response
        text = str(response or "").strip()
        if text.startswith("```") and text.endswith("```"):
            text = "\n".join(text.splitlines()[1:-1]).strip()
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if start < 0 or end <= start:
                raise TemporalOutputError("model returned invalid JSON")
            try:
                value = json.loads(text[start : end + 1])
            except json.JSONDecodeError as exc:
                raise TemporalOutputError("model returned invalid JSON") from exc
        if not isinstance(value, Mapping):
            raise TemporalOutputError("model returned the wrong schema")
        return value

    @staticmethod
    def _validate_window(
        frames: Sequence[TemporalFrameInput],
        *,
        label: str,
        max_size: Optional[int] = None,
    ) -> tuple[TemporalFrameInput, ...]:
        values = tuple(frames)
        if not values:
            raise TemporalInputError(f"{label} frames must not be empty")
        if max_size is not None and len(values) > max_size:
            raise TemporalInputError(
                f"{label} frames allow at most {max_size} images"
            )
        if any(not isinstance(frame, TemporalFrameInput) for frame in values):
            raise TemporalInputError(
                f"{label} frames must contain TemporalFrameInput values"
            )
        ids = [frame.frame_id for frame in values]
        if ids != sorted(set(ids)):
            raise TemporalInputError(
                f"{label} frame IDs must be unique and increasing"
            )
        return values

    def _inference_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        engine = self._get_engine()
        if getattr(engine, "supports_image_pixel_budget", False):
            kwargs.update(
                image_min_pixels=min(64**2, self.config.max_image_edge**2),
                image_max_pixels=self.config.max_image_edge**2,
            )
        return kwargs

    def _completion_content(
        self,
        subgoal: Subgoal,
        frames: Sequence[TemporalFrameInput],
    ) -> list[Any]:
        content: list[Any] = [
            f"Subgoal: {subgoal.description}\n"
            f"Completion proof: {subgoal.completion_criteria}\n"
            f"Images: {len(frames)}, selected ordered visual evidence; "
            "landmark fields are auxiliary."
        ]
        for frame in frames:
            content.extend(
                (
                    "frame_id={}; translation_m={:.3f}; "
                    "yaw_delta_deg={:+.1f}; path_length_m={:.3f}; "
                    "landmark_visible={}; landmark_direction={}; "
                    "landmark_proximity={}; landmark_passed={}; "
                    "landmark_confidence={:.2f}".format(
                        frame.frame_id,
                        frame.translation_m,
                        frame.yaw_delta_deg,
                        frame.subgoal_path_length_m,
                        frame.landmark_visible,
                        frame.landmark_direction,
                        frame.landmark_proximity,
                        frame.landmark_passed,
                        frame.landmark_confidence,
                    ),
                    self._png(frame.image),
                )
            )
        return content

    def _error_content(
        self,
        subgoal: Subgoal,
        frames: Sequence[TemporalFrameInput],
    ) -> list[Any]:
        content: list[Any] = [
            f"Subgoal: {subgoal.description}\n"
            f"Completion proof: {subgoal.completion_criteria}\n"
            f"Recent ordered frames: {len(frames)}."
        ]
        for frame in frames:
            content.extend(
                (
                    "frame_id={}; translation_m={:.3f}; "
                    "yaw_delta_deg={:+.1f}; path_length_m={:.3f}; "
                    "landmark_visible={}; landmark_direction={}; "
                    "landmark_proximity={}; landmark_passed={}; "
                    "landmark_confidence={:.2f}".format(
                        frame.frame_id,
                        frame.translation_m,
                        frame.yaw_delta_deg,
                        frame.subgoal_path_length_m,
                        frame.landmark_visible,
                        frame.landmark_direction,
                        frame.landmark_proximity,
                        frame.landmark_passed,
                        frame.landmark_confidence,
                    ),
                    self._png(frame.image),
                )
            )
        return content

    def _analyze_error_window(
        self,
        *,
        engine: Any,
        subgoal: Subgoal,
        frames: Sequence[TemporalFrameInput],
        kwargs: Mapping[str, Any],
    ) -> tuple[_ErrorModelResult, str]:
        response = None
        for attempt in range(2):
            try:
                response = engine(
                    self._error_content(subgoal, frames),
                    system_prompt=ERROR_SYSTEM_PROMPT,
                    **kwargs,
                )
                raw = str(response)
                return self._parse_error(response), raw
            except TemporalOutputError as exc:
                # Retry only malformed/truncated JSON. A structurally wrong
                # response cannot be repaired by asking the identical schema
                # again and must remain visible to callers immediately.
                if "invalid JSON" not in str(exc) or attempt == 1:
                    self._remember_failed_response("error", response)
                    raise
        raise AssertionError("unreachable error-inference state")

    def _remember_failed_response(self, stage: str, response: Any) -> None:
        self.last_failed_raw_response = json.dumps(
            {"stage": stage, "response": str(response)},
            ensure_ascii=False,
        )

    def _get_engine(self) -> Any:
        if self._engine is None:
            try:
                from agentflow.agents.engine.factory import create_llm_engine

                self._engine = create_llm_engine(
                    model_string=f"local-qwen3vl-{self.model_path}",
                    is_multimodal=True,
                    use_cache=self._use_cache,
                    debug_performance=self._debug_performance,
                    **self._engine_kwargs,
                )
            except Exception as exc:
                raise TemporalInferenceError("Could not create Qwen engine") from exc
        return self._engine

    def _png(self, image: Any) -> bytes:
        cache_key = id(image)
        cached = self._png_cache.get(cache_key)
        if cached is not None and cached[0] is image:
            self._png_cache.move_to_end(cache_key)
            return cached[1]
        encoded = self._encode_png(image)
        self._png_cache[cache_key] = (image, encoded)
        self._png_cache.move_to_end(cache_key)
        while len(self._png_cache) > 17:
            self._png_cache.popitem(last=False)
        return encoded

    def _encode_png(self, image: Any) -> bytes:
        """Encode an image without adding it to the temporal frame cache."""
        try:
            import numpy as np
            from PIL import Image

            if isinstance(image, bytes):
                pil = Image.open(io.BytesIO(image)).convert("RGB")
            elif isinstance(image, Image.Image):
                pil = image.convert("RGB")
            else:
                array = np.asarray(image)
                if array.ndim != 3 or array.shape[2] not in (3, 4):
                    raise ValueError
                if np.issubdtype(array.dtype, np.floating) and array.size:
                    if float(np.nanmax(array)) <= 1:
                        array = array * 255
                array = np.clip(array, 0, 255).astype(np.uint8)
                pil = Image.fromarray(array).convert("RGB")
            pil.thumbnail(
                (self.config.max_image_edge,) * 2,
                Image.Resampling.LANCZOS,
            )
            buffer = io.BytesIO()
            pil.save(buffer, format="PNG")
            return buffer.getvalue()
        except Exception as exc:
            raise TemporalInputError("unsupported image") from exc


__all__ = (
    "CaptionResult",
    "DualWindowCaptionResult",
    "ErrorMode",
    "Subgoal",
    "TemporalAnalysisRequest",
    "TemporalCaptioner",
    "TemporalCaptionerConfig",
    "TemporalCaptionerError",
    "TemporalFrameInput",
    "TemporalInferenceError",
    "TemporalInputError",
    "TemporalOutputError",
)
