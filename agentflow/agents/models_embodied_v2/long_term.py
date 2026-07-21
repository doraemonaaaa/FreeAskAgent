from typing import Sequence


LONG_TERM_PROMPT = """You are the long-horizon planner for visual navigation.
Decompose the goal into a short route plan and assess progress from recent control
history. Use retrieved navigation knowledge when relevant. Produce a concise plan
for the short-horizon visual thinker, not an action token. The robot may only
advance 0.1 m or turn 15 degrees per ModelB action."""


class LongTermPlanner:
    """Low-frequency task decomposition; it accepts optional multimodal evidence."""

    def __init__(self, llm, goal: str):
        self.llm = llm
        self.goal = goal
        self.plan = goal

    def update(self, actions: Sequence[str], retrieved: Sequence[str], image_bytes=None) -> str:
        prompt = (
            f"Goal: {self.goal}\n"
            f"Previous route plan: {self.plan}\n"
            f"Recent actions: {list(actions)}\n"
            f"Retrieved knowledge: {list(retrieved)}\n"
            "Update the route plan and identify the current subtask."
        )
        content = [prompt, image_bytes] if image_bytes is not None else prompt
        plan = self.llm(content, system_prompt=LONG_TERM_PROMPT, max_tokens=160, temperature=0).strip()
        if not plan:
            raise ValueError("Long-term planner returned an empty plan.")
        self.plan = plan
        return plan
