"""ModelA: asynchronous long/short-horizon multimodal thinker."""

import threading
from collections import deque
from typing import Any, Deque, List, Optional, Tuple

from agentflow.agents.engine.factory import create_llm_engine
from .Actor import Actor, DEFAULT_MODEL_PATH
from .long_term import LongTermPlanner
from .rag import NavigationRAG
from .short_term import ShortTermThinker


class Thinker:
    """Keep ModelA planning asynchronously while ModelB continues acting."""

    def __init__(self, goal: str, actor: Actor, *, planner_model_path: str = DEFAULT_MODEL_PATH, bootstrap_instruction: Optional[str] = None, debug_performance: bool = False, use_cache: bool = False, show_output: bool = True, long_term_interval: int = 8, rag_documents: Optional[List[str]] = None, rag_path: Optional[str] = None):
        if long_term_interval < 1:
            raise ValueError("long_term_interval must be at least 1.")
        self.goal, self.actor, self.show_output = goal, actor, show_output
        self.long_term_interval = long_term_interval
        llm = create_llm_engine(model_string=f"local-qwen3vl-{planner_model_path}", is_multimodal=True, use_cache=use_cache, debug_performance=debug_performance)
        self.rag = NavigationRAG(documents=rag_documents, path=rag_path)
        self.long_term, self.short_term = LongTermPlanner(llm, goal), ShortTermThinker(llm)
        self._directive, self._actions = bootstrap_instruction or goal, deque(maxlen=8)
        self._lock, self._condition = threading.Lock(), threading.Condition()
        self._pending: Optional[Tuple[bytes, Tuple[str, ...], str]] = None
        self._closed, self._error, self._thought_count = False, None, 0
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    @property
    def directive(self) -> str:
        with self._lock:
            return self._directive

    def submit_observation(self, rgb_image: Any) -> str:
        image_bytes = self.actor.rgb_to_bytes(rgb_image)
        with self._lock:
            directive, actions = self._directive, tuple(self._actions)
        with self._condition:
            self._pending = (image_bytes, actions, directive)
            self._condition.notify()
        return directive

    def record_action(self, action: str) -> None:
        with self._lock:
            self._actions.append(action)

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
                image_bytes, actions, prior = self._pending
                self._pending = None
            try:
                retrieved = self.rag.search(f"{self.goal} {prior}")
                if self._thought_count % self.long_term_interval == 0:
                    plan = self.long_term.update(actions, retrieved, image_bytes)
                    if self.show_output:
                        print(f"[ModelA long-term] {plan}", flush=True)
                else:
                    plan = self.long_term.plan
                directive = self.short_term.analyze(image_bytes, plan, actions, retrieved)
                self._thought_count += 1
                with self._lock:
                    self._directive, self._error = directive, None
                if self.show_output:
                    print(f"[ModelA short-term] {directive}", flush=True)
            except Exception as exc:
                with self._lock:
                    self._error = exc
                if self.show_output:
                    print(f"[ModelA thinker error] {exc}", flush=True)
