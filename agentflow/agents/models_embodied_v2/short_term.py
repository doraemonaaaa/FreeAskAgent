from typing import Sequence


SHORT_TERM_PROMPT = """You are the short-horizon multimodal visual thinker for a
navigation robot. Inspect the current RGB image and combine it with the route plan,
recent actions, and retrieved experience. Report local progress, visible obstacles,
and a precise next directive for ModelB. ModelB can only move forward 0.1 m or turn
left/right 15 degrees. Do not output an action token; output a concise directive."""


class ShortTermThinker:
    """High-frequency RGB observation analysis for the action model."""

    def __init__(self, llm):
        self.llm = llm

    def analyze(self, image_bytes: bytes, route_plan: str, actions: Sequence[str], retrieved: Sequence[str]) -> str:
        prompt = (
            f"Route plan: {route_plan}\n"
            f"Recent actions: {list(actions)}\n"
            f"Retrieved knowledge: {list(retrieved)}\n"
            "Analyze the current RGB observation and issue ModelB's next directive."
        )
        directive = self.llm(
            [prompt, image_bytes], system_prompt=SHORT_TERM_PROMPT, max_tokens=96, temperature=0
        ).strip()
        if not directive:
            raise ValueError("Short-term thinker returned an empty directive.")
        return directive
