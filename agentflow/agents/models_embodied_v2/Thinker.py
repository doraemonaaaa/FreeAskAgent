"""ModelA: asynchronous long/short-horizon multimodal thinker."""

import threading
from collections import deque
from typing import Any, Deque, List, Optional, Tuple

from agentflow.agents.engine.factory import create_llm_engine
from .Actor import Actor, DEFAULT_MODEL_PATH
from .memory import TaskMemory
from .memory.rag import NavigationRAG
from .memory.short_term import ShortTermThinker


SUBTASK_DECOMPOSITION_PROMPT = """You are the task planner for a visual navigation robot.
Given the original task, decompose it into an ordered, minimal list of observable
navigation subtasks. For every subtask, include a `Completion status` initialized
to `NOT STARTED`. Preserve the original task and use this exact structure:
Original task: ...
1. Subtask: ... | Completion status: NOT STARTED
2. Subtask: ... | Completion status: NOT STARTED
Do not provide action tokens or a next-action directive."""


COMPLETION_STATUS_PROMPT = """You maintain an existing visual-navigation subtask
tracker. Do not create, remove, reorder, split, or rewrite subtasks. Using the
current RGB observation, recent actions, and retrieved knowledge, update only each
subtask's `Completion status` to `NOT STARTED`, `IN PROGRESS`, `COMPLETE`, or
`BLOCKED` (with a short reason only when BLOCKED). Return the same tracker format.
Do not provide action tokens or a next-action directive."""


class Thinker:
    """Keep ModelA planning asynchronously while ModelB continues acting."""

    def __init__(self, goal: str, actor: Actor, *, planner_model_path: str = DEFAULT_MODEL_PATH, bootstrap_instruction: Optional[str] = None, debug_performance: bool = False, use_cache: bool = False, show_output: bool = True, long_term_interval: int = 8, rag_documents: Optional[List[str]] = None, rag_path: Optional[str] = None):
        if long_term_interval < 1:
            raise ValueError("long_term_interval must be at least 1.")
        self.goal, self.actor, self.show_output = goal, actor, show_output
        # Retained for API compatibility; subtask decomposition happens exactly once.
        self.long_term_interval = long_term_interval
        llm = create_llm_engine(model_string=f"local-qwen3vl-{planner_model_path}", is_multimodal=True, use_cache=use_cache, debug_performance=debug_performance)
        self.rag = NavigationRAG(documents=rag_documents, path=rag_path)
        self._llm, self.short_term = llm, ShortTermThinker(llm)
        self.task_memory = TaskMemory(goal)
        self._subtask_tracker: Optional[str] = None
        self._directive, self._actions = bootstrap_instruction or goal, deque(maxlen=8)
        self._lock, self._condition = threading.Lock(), threading.Condition()
        self._pending: Optional[Tuple[int, bytes, Tuple[str, ...], str]] = None
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

    def submit_observation(self, rgb_image: Any, *, wait_for_completion: bool = False) -> str:
        """Submit an observation and optionally wait for its updated directive."""
        image_bytes = self.actor.rgb_to_bytes(rgb_image)
        with self._lock:
            directive, actions = self._directive, tuple(self._actions)
            self.task_memory.record_input(rgb_image)
        with self._condition:
            self._submitted_count += 1
            request_id = self._submitted_count
            self._pending = (request_id, image_bytes, actions, directive)
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
        with self._lock:
            self._actions.append(action)
            self.task_memory.record_event("Actor action", action)

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
                request_id, image_bytes, actions, prior = self._pending
                self._pending = None
            try:
                retrieved = self.rag.search(f"{self.goal} {prior}")
                with self._lock:
                    tracker = self._subtask_tracker
                if tracker is None:
                    tracker = self._decompose_original_task()
                    if self.show_output:
                        print(f"[ModelA subtasks] {tracker}", flush=True)
                else:
                    tracker = self._update_completion_status(tracker, actions, retrieved, image_bytes)
                    if self.show_output:
                        print(f"[ModelA completion status] {tracker}", flush=True)
                with self._lock:
                    self.task_memory.update_subgoal_status(tracker)
                    memory_context = self.task_memory.context()
                directive = self.short_term.analyze(image_bytes, memory_context, actions, retrieved)
                self._thought_count += 1
                with self._lock:
                    self._subtask_tracker, self._directive, self._error = tracker, directive, None
                    self.task_memory.record_event("Thinker directive", directive)
                with self._condition:
                    self._completed_count = request_id
                    self._condition.notify_all()
                if self.show_output:
                    print(f"[ModelA short-term] {directive}", flush=True)
            except Exception as exc:
                with self._lock:
                    self._error = exc
                with self._condition:
                    self._completed_count = request_id
                    self._condition.notify_all()
                if self.show_output:
                    print(f"[ModelA thinker error] {exc}", flush=True)

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

    def _update_completion_status(self, tracker: str, actions: Tuple[str, ...], retrieved: List[str], image_bytes: bytes) -> str:
        """Update statuses without allowing the model to re-plan the task."""
        prompt = (
            f"Task memory:\n{self.task_memory.context()}\n"
            f"Existing subtask tracker:\n{tracker}\n"
            f"Recent actions: {list(actions)}\n"
            f"Retrieved knowledge: {list(retrieved)}\n"
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
