"""ModelA: asynchronous long/short-horizon multimodal thinker."""

import threading
from typing import Any, Callable, Optional, Tuple

from agentflow.agents.engine.factory import create_llm_engine
from .Actor import Actor, DEFAULT_MODEL_PATH
from .memory import CompositeMemory


SUBTASK_DECOMPOSITION_PROMPT = """You are the task planner for a visual navigation robot.
Given the original task, decompose it into an ordered, minimal list of observable
navigation subtasks. For every subtask, include a `Completion status` initialized
to `NOT STARTED`. The final subtask must describe arriving close to (within about
2 meters of) the destination/landmark named in the instruction, e.g. "Approach and
stop within 2 meters of the rug" -- not merely bringing the destination into view.
Preserve the original task and use this exact structure:
Original task: ...
1. Subtask: ... | Completion status: NOT STARTED
2. Subtask: ... | Completion status: NOT STARTED
Do not provide action tokens or a next-action directive."""


COMPLETION_STATUS_PROMPT = """You maintain an existing visual-navigation subtask
tracker. Do not create, remove, reorder, split, or rewrite subtasks. Using the
current RGB observation and recent actions, update only each subtask's
`Completion status` to `NOT STARTED`, `IN PROGRESS`, `COMPLETE`, or `BLOCKED`
(with a short reason only when BLOCKED).

Strict rule for arrival/stop subtasks (e.g. "stop near the rug"): mark COMPLETE
ONLY when the robot is close to that location, roughly within 2 meters -- the
target should occupy a large portion of the frame with little floor space left
between the robot and it. Merely seeing the target appear in the frame, even
clearly, keeps the subtask IN PROGRESS. Judge distance conservatively: if you
are unsure whether the robot is within 2 meters, keep the subtask IN PROGRESS
rather than COMPLETE.

Return the same tracker format. Do not provide action tokens or a next-action
directive."""


DIRECTIVE_PROMPT = """You are the multimodal visual thinker for a navigation robot.
Inspect the current RGB image and combine it with the task memory and recent actions.
Report local progress, visible obstacles, and a precise next directive for ModelB.
ModelB can only move forward 0.25 m or turn left/right 15 degrees.

Stopping rule: the robot must be within about 2 meters of the final destination
before it may stop. Seeing the target object/area in the distance is NOT enough
to justify stopping -- judge proximity from how large the target appears in frame
and how much open floor remains between the robot and it. While the target is
still small/far, issue directives that keep approaching it (e.g. "move forward
toward the rug, it is still far ahead"). Only instruct ModelB to stop once the
robot is close enough that a couple more forward steps would reach the target, or
it is already adjacent to it. Never issue a stop directive on the same step the
target first becomes visible unless it is already that close.

Do not output an action token; output a concise directive."""


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
        self._subtask_tracker: Optional[str] = None
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
        """The fixed subtask list with its latest completion statuses."""
        with self._lock:
            return self._subtask_tracker

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
        """Update the tracker for a terminal frame without a new directive."""
        image_bytes = self.actor.rgb_to_bytes(rgb_image)
        actions = self._recent_actions()
        with self._lock:
            tracker = self._subtask_tracker
        tracker = self._updated_tracker(tracker, actions, image_bytes)
        update_status = getattr(self.memory, "update_subgoal_status", None)
        if callable(update_status):
            update_status(tracker)
        with self._lock:
            self._subtask_tracker = tracker
        return tracker

    def reset(self, goal: str) -> None:
        """Reset episode-scoped planner state without reloading weights."""
        with self._condition:
            self._pending = None
            self._completed_count = self._submitted_count
            self._condition.notify_all()
        with self._lock:
            self.goal = goal
            self._subtask_tracker = None
            self._directive = goal
            self._error = None

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
                tracker = self._updated_tracker(
                    tracker,
                    actions,
                    image_bytes,
                )
                if self.show_output:
                    label = (
                        "subtasks"
                        if was_uninitialized
                        else "completion status"
                    )
                    print(f"[ModelA {label}] {tracker}", flush=True)
                update_status = getattr(
                    self.memory,
                    "update_subgoal_status",
                    None,
                )
                if callable(update_status):
                    update_status(tracker)
                if tracker_updated_callback is not None:
                    tracker_updated_callback(tracker)
                memory_context = self.memory.context()
                directive = self._infer_directive(image_bytes, memory_context, actions)
                self._thought_count += 1
                with self._lock:
                    self._subtask_tracker, self._directive, self._error = tracker, directive, None
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

    def _updated_tracker(
        self,
        tracker: Optional[str],
        actions: Tuple[str, ...],
        image_bytes: bytes,
    ) -> str:
        if tracker is None:
            return self._decompose_original_task()
        return self._update_completion_status(
            tracker,
            actions,
            image_bytes,
        )

    def _decompose_original_task(self) -> str:
        """Create the tracker once, using only the task supplied at construction."""
        tracker = self._llm(
            f"Original task: {self.goal}",
            system_prompt=SUBTASK_DECOMPOSITION_PROMPT,
            max_tokens=240,
            temperature=0,
        ).strip()
        if not tracker:
            raise ValueError("Subtask planner returned an empty tracker.")
        return tracker

    def _update_completion_status(self, tracker: str, actions: Tuple[str, ...], image_bytes: bytes) -> str:
        """Update statuses without allowing the model to re-plan the task."""
        prompt = (
            f"Task memory:\n{self.memory.context()}\n"
            f"Existing subtask tracker:\n{tracker}\n"
            f"Recent actions: {list(actions)}\n"
            "Update only the Completion status fields."
        )
        updated_tracker = self._llm(
            [prompt, image_bytes],
            system_prompt=COMPLETION_STATUS_PROMPT,
            max_tokens=240,
            temperature=0,
        ).strip()
        if not updated_tracker:
            raise ValueError("Completion-status updater returned an empty tracker.")
        return updated_tracker

    def _infer_directive(self, image_bytes: bytes, memory_context: str, actions: Tuple[str, ...]) -> str:
        """Turn the current observation and task memory into ModelB's next directive."""
        prompt = (
            f"Task memory:\n{memory_context}\n"
            f"Recent actions: {list(actions)}\n"
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
