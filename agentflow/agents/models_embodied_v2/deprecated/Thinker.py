"""ModelA: asynchronous long/short-horizon multimodal thinker."""

import threading
from typing import Any, Callable, Optional, Tuple

from agentflow.agents.engine.factory import create_llm_engine
from .Actor import (
    ACTION_TOKENS,
    MOVE_TOKENS,
    TURN_LEFT,
    TURN_RIGHT,
    Actor,
    DEFAULT_MODEL_PATH,
)
from .freespace_gate import EscapeDirection
from ..memory import CompositeMemory
from .subtask_tracker import SubtaskTracker, coerce_distance, extract_json
from .stop_gate import StopDecision, StopGate


# These run once (decomposition) or every step (the rest), so anything the
# tracker's invariants already reject is dead weight here. What stays is what no
# invariant can repair: the shape of the reply and the quality of the distance.
SUBTASK_DECOMPOSITION_PROMPT = """Task planner for a visual navigation robot.
Split the task into an ordered, minimal list of observable subtasks. The last one
must be the arrival step -- reaching the destination, not just seeing it.

JSON only:
{"target_phrase": "<the destination, 1-4 words, e.g. 'the rug'>",
 "subtasks": [{"text": "<subtask 1>"}, {"text": "<subtask 2>"}]}"""


COMPLETION_STATUS_PROMPT = """You update one entry of a visual-navigation subtask tracker.
The active subtask's index is given to you; report on that one only.

JSON only:
{"updates": [{"index": <active index>, "status": "<NOT_STARTED|IN_PROGRESS|COMPLETE|BLOCKED>",
              "evidence": "<what in the image supports this>",
              "distance_estimate_m": <meters to this subtask's target, or null>,
              "blocked_reason": "<only when BLOCKED>"}]}

Estimate `distance_estimate_m` honestly; never shrink it to justify a status.
A separate system applies the arrival threshold to your number."""


ARRIVAL_CHECK_PROMPT = """Range estimator for a navigation robot.

JSON only:
{"target_visible": <true|false>, "distance_m": <meters, or null if not visible>,
 "reason": "<one short sentence>"}

Estimate the straight-line distance from the camera to the named target. Scale
references: doorway ~2 m tall, couch ~2 m wide, floor tile ~0.3 m. A target seen
through a doorway or window lies beyond that opening -- report the full distance,
not the distance to the opening. When torn between two values, report the larger."""


DIRECTIVE_PROMPT = """Visual thinker for a navigation robot. From the image, task
memory, and recent actions, output one concise directive for ModelB: which way to
turn, what to move toward, what to avoid, and any blocking obstacle.
ModelB can only move forward 0.25 m or turn left/right 15 degrees.

Every directive must name a heading the robot can physically drive along -- open
floor, a doorway, a corridor, a gap between furniture. Seeing the target is not
the same as having a path to it: when the straight line to the target runs into a
wall or furniture, say which side to turn toward to get around it.

A separate gate owns the stop decision and already knows the measured distance to
the target. Never write "stop" unless the gate reported stopping is permitted.
No action tokens."""


# Asked only when the path forward has already failed, so the question is not
# "where is the target" -- it is the narrower and far more answerable "where is
# there floor I can drive on".
ESCAPE_DIRECTION_PROMPT = """Free-space scout for a navigation robot whose path
straight ahead is physically blocked -- it commanded FORWARD and did not move.

Ignore the destination for a moment. Look at the image and name the nearest
direction holding traversable floor the robot could actually drive through: a
doorway, a corridor, a gap between furniture, or plain open floor. A surface
behind glass, a closed door, or a raised obstacle is not traversable.

JSON only:
{"direction": "<left|right>", "bearing_deg": <degrees to turn, 15-180>,
 "opening": "<doorway|corridor|gap|open floor|none>",
 "reason": "<one short sentence naming what you see there>"}

The camera sees roughly 90 degrees, so a bearing beyond that means "keep turning
that way and look again". Prefer the side showing more uninterrupted floor; if
neither side shows any, answer the side with the larger bearing."""


def _coerce_bearing(raw: Any) -> Optional[float]:
    """Read a turn magnitude in degrees, whatever sign or units slipped in."""
    value = coerce_distance(raw)
    if value is None:
        value = coerce_distance(str(raw).lstrip("-+") if raw is not None else None)
    if value is None:
        return None
    return min(180.0, abs(value))


class Thinker:
    """Keep ModelA planning asynchronously while ModelB continues acting."""

    def __init__(
        self,
        goal: str,
        actor: Actor,
        *,
        planner_model_path: str = DEFAULT_MODEL_PATH,
        bootstrap_instruction: Optional[str] = None,
        debug_performance: bool = False,
        use_cache: bool = False,
        show_output: bool = True,
        memory: Optional[Any] = None,
        planner_engine: Optional[Any] = None,
        arrival_radius_m: float = 2.0,
        confirmations_required: int = 3,
        min_steps_since_target_seen: int = 4,
    ):
        self.goal, self.actor, self.show_output = goal, actor, show_output
        self._llm = planner_engine or create_llm_engine(
            model_string=f"local-qwen3vl-{planner_model_path}",
            is_multimodal=True,
            use_cache=use_cache,
            debug_performance=debug_performance,
        )
        self.memory = (
            memory
            if memory is not None
            else CompositeMemory(goal=goal, mode="temporal")
        )
        # Compatibility alias for older callers; this is the coordinator.
        self.task_memory = self.memory
        self.arrival_radius_m = arrival_radius_m
        self.confirmations_required = confirmations_required
        self.min_steps_since_target_seen = min_steps_since_target_seen
        self.stop_gate = StopGate(
            arrival_radius_m=arrival_radius_m,
            min_steps_since_target_seen=min_steps_since_target_seen,
        )
        # The tracker is a structured object now; `subtask_tracker` still hands
        # the memory layer the same rendered text it has always consumed.
        self._subtask_tracker: Optional[SubtaskTracker] = None
        self._stop_decision = StopDecision(
            False, "the planner has not produced a subtask plan yet"
        )
        self._actor_context = f"Original goal: {goal}"
        self._directive = bootstrap_instruction or goal
        self._lock, self._condition = threading.Lock(), threading.Condition()
        self._pending: Optional[
            Tuple[
                int,
                bytes,
                Tuple[str, ...],
                Optional[Callable[[str], None]],
            ]
        ] = None
        self._submitted_count = self._completed_count = 0
        self._closed, self._error, self._thought_count = False, None, 0
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    @property
    def directive(self) -> str:
        with self._lock:
            return self._directive

    @property
    def subtask_tracker(self) -> Optional[str]:
        """The fixed subtask list with its latest statuses, rendered as text."""
        with self._lock:
            tracker = self._subtask_tracker
        return None if tracker is None else tracker.render()

    @property
    def tracker(self) -> Optional[SubtaskTracker]:
        """The structured tracker; statuses only move under its own invariants."""
        with self._lock:
            return self._subtask_tracker

    @property
    def stop_decision(self) -> StopDecision:
        """Whether STOP is legal right now, and why."""
        with self._lock:
            return self._stop_decision

    def allowed_actions(self) -> Tuple[str, ...]:
        """ModelB's action space this step, STOP masked out until it is earned."""
        return ACTION_TOKENS if self.stop_decision.allowed else MOVE_TOKENS

    def actor_context(self) -> str:
        """Goal plus subtask progress, so ModelB never acts on a bare directive."""
        with self._lock:
            return self._actor_context

    @property
    def planner_engine(self) -> Any:
        """Expose the loaded planner engine to the temporal captioner."""
        return self._llm

    def submit_observation(
        self,
        rgb_image: Any,
        *,
        wait_for_completion: bool = False,
        tracker_updated_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Submit an observation and optionally wait for its new directive.

        The callback runs after the tracker is updated and before directive
        inference, allowing the agent to close and analyze the previous
        action/post-frame pair synchronously.
        """
        image_bytes = self.actor.rgb_to_bytes(rgb_image)
        actions = self._recent_actions()
        with self._lock:
            directive = self._directive
        with self._condition:
            self._submitted_count += 1
            request_id = self._submitted_count
            self._pending = (
                request_id,
                image_bytes,
                actions,
                tracker_updated_callback,
            )
            self._condition.notify()
            if wait_for_completion:
                while self._completed_count < request_id and not self._closed:
                    self._condition.wait()
                if self._error is not None:
                    raise RuntimeError("Thinker failed to process the observation.") from self._error
                with self._lock:
                    return self._directive
        return directive

    def record_action(self, action: str) -> None:
        """Compatibility hook; live action history is owned by memory."""
        with self._lock:
            record_event = getattr(self.memory, "record_event", None)
            if callable(record_event):
                record_event("Actor action", action)

    def update_tracker_only(self, rgb_image: Any) -> str:
        """Update the tracker for a terminal frame without a new directive.

        Terminal-only, so it runs after the worker thread has gone idle.
        """
        image_bytes = self.actor.rgb_to_bytes(rgb_image)
        actions = self._recent_actions()
        with self._lock:
            tracker = self._subtask_tracker
        tracker, target_visible, distance = self._updated_tracker(
            tracker,
            actions,
            image_bytes,
        )
        self.stop_gate.observe(target_visible=target_visible, distance_m=distance)
        decision = self.stop_gate.evaluate(tracker)
        rendered = tracker.render()
        update_status = getattr(self.memory, "update_subgoal_status", None)
        if callable(update_status):
            update_status(rendered)
        with self._lock:
            self._subtask_tracker = tracker
            self._stop_decision = decision
            self._actor_context = self._build_actor_context(tracker, decision)
        return rendered

    def reset(self, goal: str) -> None:
        """Reset episode-scoped planner state without reloading weights."""
        with self._condition:
            self._pending = None
            self._completed_count = self._submitted_count
            self._condition.notify_all()
        self.stop_gate.reset()
        with self._lock:
            self.goal = goal
            self._subtask_tracker = None
            self._directive = goal
            self._error = None
            self._stop_decision = StopDecision(
                False, "the planner has not produced a subtask plan yet"
            )
            self._actor_context = f"Original goal: {goal}"

    def close(self, timeout: Optional[float] = None) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify()
        self._thread.join(timeout)

    def _loop(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._closed:
                    self._condition.wait()
                if self._closed:
                    return
                request_id, image_bytes, actions, tracker_updated_callback = (
                    self._pending
                )
                self._pending = None
            try:
                with self._lock:
                    tracker = self._subtask_tracker
                was_uninitialized = tracker is None
                tracker, target_visible, distance = self._updated_tracker(
                    tracker,
                    actions,
                    image_bytes,
                )
                rendered = tracker.render()
                if self.show_output:
                    label = (
                        "subtasks"
                        if was_uninitialized
                        else "completion status"
                    )
                    print(f"[ModelA {label}]\n{rendered}", flush=True)

                # Arrival evidence accrues over time; the gate is the only place
                # that turns it into permission to stop.
                self.stop_gate.observe(
                    target_visible=target_visible,
                    distance_m=distance,
                )
                decision = self.stop_gate.evaluate(tracker)

                update_status = getattr(
                    self.memory,
                    "update_subgoal_status",
                    None,
                )
                if callable(update_status):
                    update_status(rendered)
                if tracker_updated_callback is not None:
                    tracker_updated_callback(rendered)
                memory_context = self.memory.context()
                directive = self._infer_directive(
                    image_bytes,
                    memory_context,
                    actions,
                    decision,
                )
                self._thought_count += 1
                with self._lock:
                    self._subtask_tracker, self._directive, self._error = tracker, directive, None
                    self._stop_decision = decision
                    self._actor_context = self._build_actor_context(
                        tracker,
                        decision,
                    )
                    record_event = getattr(
                        self.memory,
                        "record_event",
                        None,
                    )
                    if callable(record_event):
                        record_event("Thinker directive", directive)
                with self._condition:
                    self._completed_count = request_id
                    self._condition.notify_all()
                if self.show_output:
                    print(f"[ModelA directive] {directive}", flush=True)
                    verdict = "ALLOWED" if decision.allowed else "DENIED"
                    print(f"[StopGate] {verdict}: {decision.reason}", flush=True)
            except Exception as exc:
                with self._lock:
                    self._error = exc
                with self._condition:
                    self._completed_count = request_id
                    self._condition.notify_all()
                if self.show_output:
                    print(f"[ModelA thinker error] {exc}", flush=True)

    def _recent_actions(self) -> Tuple[str, ...]:
        getter = getattr(self.memory, "recent_actions", None)
        if not callable(getter):
            return ()
        return tuple(getter())

    def _build_actor_context(
        self,
        tracker: SubtaskTracker,
        decision: StopDecision,
    ) -> str:
        permission = (
            "STOP is ALLOWED this step."
            if decision.allowed
            else f"STOP is NOT allowed this step ({decision.reason})."
        )
        return f"{tracker.render_for_actor()}\n{permission}"

    def _updated_tracker(
        self,
        tracker: Optional[SubtaskTracker],
        actions: Tuple[str, ...],
        image_bytes: bytes,
    ) -> Tuple[SubtaskTracker, bool, Optional[float]]:
        """Return the tracker plus this step's arrival evidence."""
        if tracker is None:
            return self._decompose_original_task(), False, None
        target_visible, distance = self._check_arrival(tracker, image_bytes)
        self._update_completion_status(
            tracker,
            actions,
            image_bytes,
            distance,
        )
        return tracker, target_visible, distance

    def _decompose_original_task(self) -> SubtaskTracker:
        """Create the tracker once, using only the task supplied at construction."""
        reply = self._llm(
            f"Original task: {self.goal}",
            system_prompt=SUBTASK_DECOMPOSITION_PROMPT,
            max_tokens=320,
            temperature=0,
        ).strip()
        if not reply:
            raise ValueError("Subtask planner returned an empty tracker.")
        return SubtaskTracker.from_plan(
            self.goal,
            reply,
            arrival_radius_m=self.arrival_radius_m,
            confirmations_required=self.confirmations_required,
        )

    def _check_arrival(
        self,
        tracker: SubtaskTracker,
        image_bytes: bytes,
    ) -> Tuple[bool, Optional[float]]:
        """Ask one narrow range question about the final target.

        Kept separate from the status update on purpose: a call that only has to
        estimate one distance is far more reliable than the same judgement buried
        in a prompt that is also juggling statuses and evidence.
        """
        if not tracker.awaiting_arrival:
            return False, None
        prompt = (
            f"Target: {tracker.target_phrase}\n"
            "How far is the camera from this target right now?"
        )
        try:
            payload = extract_json(
                self._llm(
                    [prompt, image_bytes],
                    system_prompt=ARRIVAL_CHECK_PROMPT,
                    max_tokens=128,
                    temperature=0,
                )
            )
        except ValueError as exc:
            if self.show_output:
                print(f"[ModelA arrival check unparsed] {exc}", flush=True)
            return False, None
        if not isinstance(payload, dict):
            return False, None
        visible = bool(payload.get("target_visible"))
        distance = coerce_distance(payload.get("distance_m")) if visible else None
        if self.show_output:
            print(
                f"[ModelA arrival check] visible={visible} distance={distance} "
                f"reason={payload.get('reason', '')!r}",
                flush=True,
            )
        return visible, distance

    def propose_escape(
        self,
        rgb_image: Any,
        *,
        blocked_note: str = "",
        turn_degrees: float = 15.0,
        max_turn_steps: int = 12,
    ) -> Optional[EscapeDirection]:
        """Scout one traversable heading from the current observation.

        Runs on the caller's thread, like :meth:`update_tracker_only`: the agent
        only asks once the worker has returned this step's directive, and the
        answer is needed before the action is chosen.
        """
        image_bytes = self.actor.rgb_to_bytes(rgb_image)
        prompt = "\n".join(
            part
            for part in (
                "The robot commanded FORWARD and did not move.",
                blocked_note,
                f"Goal for context (do not chase it into a wall): {self.goal}",
                "Where is the nearest traversable opening?",
            )
            if part
        )
        try:
            payload = extract_json(
                self._llm(
                    [prompt, image_bytes],
                    system_prompt=ESCAPE_DIRECTION_PROMPT,
                    max_tokens=128,
                    temperature=0,
                )
            )
        except ValueError as exc:
            if self.show_output:
                print(f"[ModelA escape scout unparsed] {exc}", flush=True)
            return None
        if not isinstance(payload, dict):
            return None
        direction = str(payload.get("direction", "")).strip().lower()
        if direction not in {"left", "right"}:
            return None
        # `direction` already carries the sign, so a signed bearing is just the
        # same information written twice; magnitude is all that is left to read.
        bearing = _coerce_bearing(payload.get("bearing_deg"))
        steps = (
            max(1, min(max_turn_steps, round(bearing / turn_degrees)))
            if bearing is not None
            else min(max_turn_steps, round(90.0 / turn_degrees))
        )
        escape = EscapeDirection(
            TURN_LEFT if direction == "left" else TURN_RIGHT,
            steps,
            opening=str(payload.get("opening", "unknown")).strip() or "unknown",
            reason=str(payload.get("reason", "")).strip(),
        )
        if self.show_output:
            print(
                f"[ModelA escape scout] {escape.describe(turn_degrees=turn_degrees)}",
                flush=True,
            )
        return escape

    def _update_completion_status(
        self,
        tracker: SubtaskTracker,
        actions: Tuple[str, ...],
        image_bytes: bytes,
        distance: Optional[float],
    ) -> None:
        """Collect a status proposal and let the tracker's invariants filter it."""
        active = tracker.active_subtask
        if active is None:
            return
        prompt = (
            f"Task memory:\n{self.memory.context()}\n"
            f"Subtask tracker:\n{tracker.render()}\n"
            f"Recent actions: {list(actions)}\n"
            f"Active subtask index: {active.index}\n"
            f"Active subtask text: {active.text}\n"
            "Report the status of the active subtask only."
        )
        reply = self._llm(
            [prompt, image_bytes],
            system_prompt=COMPLETION_STATUS_PROMPT,
            max_tokens=320,
            temperature=0,
        ).strip()
        if not reply:
            raise ValueError("Completion-status updater returned an empty reply.")
        result = tracker.apply(reply, measured_distance_m=distance)
        if self.show_output:
            for rejection in result.rejections:
                print(f"[ModelA tracker rejected] {rejection}", flush=True)

    def _infer_directive(
        self,
        image_bytes: bytes,
        memory_context: str,
        actions: Tuple[str, ...],
        decision: StopDecision,
    ) -> str:
        """Turn the current observation and task memory into ModelB's next directive."""
        permission = (
            "permitted" if decision.allowed else f"NOT permitted ({decision.reason})"
        )
        prompt = (
            f"Task memory:\n{memory_context}\n"
            f"Recent actions: {list(actions)}\n"
            f"Stopping is currently {permission}.\n"
            "Analyze the current RGB observation and issue ModelB's next directive."
        )
        directive = self._llm(
            [prompt, image_bytes],
            system_prompt=DIRECTIVE_PROMPT,
            max_tokens=96,
            temperature=0,
        ).strip()
        if not directive:
            raise ValueError("Thinker returned an empty directive.")
        return directive
