"""ModelB: Qwen3-VL policy that emits one discrete navigation action token."""

import io
import re
from pathlib import Path
from typing import Any, List, Optional, Sequence, Union

from agentflow.agents.engine.factory import create_llm_engine

DEFAULT_MODEL_PATH = "models/Qwen3-VL-8B-Instruct"
FORWARD = "FORWARD"
TURN_LEFT = "TURN_LEFT"
TURN_RIGHT = "TURN_RIGHT"
STOP = "STOP"
ACTION_TOKENS = (FORWARD, TURN_LEFT, TURN_RIGHT, STOP)
# The action space offered whenever the stop gate has not cleared STOP.
MOVE_TOKENS = (FORWARD, TURN_LEFT, TURN_RIGHT)
ACTION_ALIASES = {
    "MOVE_FORWARD": FORWARD,
    "MOVE_FORWARD_0.1_METERS": FORWARD,
    "MOVE_FORWARD_0.25_METERS": FORWARD,
    "TURN_LEFT_15_DEGREES": TURN_LEFT,
    "TURN_RIGHT_15_DEGREES": TURN_RIGHT,
}
# STOP is absent from `{allowed}` whenever the gate has not cleared it, so the
# prompt only has to state the step sizes the model cannot infer from the image.
ACTOR_PROMPT_TEMPLATE = """Visual navigation action policy.
Reply with exactly one token from this list, nothing else: {allowed}
FORWARD moves 0.25 m; TURN_LEFT/TURN_RIGHT turn 15 degrees, so reaching a
destination takes many steps.
Only pick FORWARD when the floor directly ahead is clear for the next 0.25 m.
Seeing the target is not a path to it: if a wall or furniture stands between you
and it, turn toward drivable floor -- a doorway, a corridor, a gap -- and go
around.
{stop_rule}{obstacle_rule}"""

STOP_AVAILABLE_RULE = "Arrival is confirmed: pick STOP only if the image agrees you are there."

STOP_MASKED_RULE = "Arrival is not confirmed, so STOP is unavailable this step; make progress instead."

# FORWARD is masked out of `{allowed}` once the gate has evidence the last
# FORWARD moved nothing, so driving into the wall again is unrepresentable.
FORWARD_MASKED_RULE = (
    "\nThe way straight ahead is blocked and FORWARD is unavailable this step: "
    "turn toward the side showing open floor or a doorway."
)

# Kept for callers that still import the pre-masking prompt.
ACTOR_PROMPT = ACTOR_PROMPT_TEMPLATE.format(
    allowed=", ".join(ACTION_TOKENS),
    stop_rule=STOP_AVAILABLE_RULE,
    obstacle_rule="",
)


class Actor:
    """ModelB action policy with the fixed action token space."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, *, debug_performance: bool = True, use_cache: bool = False, show_output: bool = True):
        self.model_path = model_path
        self.show_output = show_output
        self.llm = create_llm_engine(model_string=f"local-qwen3vl-{model_path}", is_multimodal=True, use_cache=use_cache, debug_performance=debug_performance)

    def act(self, rgb_image: Any, instruction: str = "", *, context: str = "", allowed_actions: Sequence[str] = ACTION_TOKENS) -> str:
        """Select one action, restricted to `allowed_actions`.

        `context` carries the goal and subtask progress so the policy never acts
        on a bare directive, and masking STOP out of `allowed_actions` makes an
        early stop unrepresentable rather than merely discouraged.
        """
        allowed = self._validate_allowed(allowed_actions)
        image_bytes = self.rgb_to_bytes(rgb_image)
        system_prompt = ACTOR_PROMPT_TEMPLATE.format(
            allowed=", ".join(allowed),
            stop_rule=STOP_AVAILABLE_RULE if STOP in allowed else STOP_MASKED_RULE,
            obstacle_rule="" if FORWARD in allowed else FORWARD_MASKED_RULE,
        )
        prompt = "\n".join(
            part
            for part in (
                context,
                f"Navigation instruction: {instruction}",
                f"Select exactly one action from: {', '.join(allowed)}",
            )
            if part
        )
        response = self.llm([prompt, image_bytes], system_prompt=system_prompt, max_tokens=8, temperature=0)
        return self.parse_action(response, allowed=allowed, show_output=self.show_output)

    @staticmethod
    def _validate_allowed(allowed_actions: Sequence[str]) -> tuple:
        allowed = tuple(dict.fromkeys(allowed_actions))
        if not allowed:
            raise ValueError("allowed_actions must contain at least one action token.")
        unknown = [action for action in allowed if action not in ACTION_TOKENS]
        if unknown:
            raise ValueError(f"Unknown action tokens {unknown}; expected a subset of {ACTION_TOKENS}.")
        if allowed == (STOP,):
            raise ValueError("allowed_actions must contain at least one movement action.")
        return allowed

    def ask(self, message: str = "", image_paths: Optional[Union[str, List[str]]] = None, *, max_tokens: Optional[int] = None) -> str:
        if not image_paths:
            raise ValueError("Actor requires one RGB image to select an action.")
        paths = [image_paths] if isinstance(image_paths, str) else image_paths
        if len(paths) != 1:
            raise ValueError("Actor accepts exactly one RGB image per action.")
        return self.act(paths[0], message)

    @staticmethod
    def rgb_to_bytes(rgb_image: Any) -> bytes:
        if isinstance(rgb_image, bytes):
            return rgb_image
        if isinstance(rgb_image, (str, Path)):
            return Path(rgb_image).expanduser().read_bytes()
        try:
            from PIL import Image
            if hasattr(rgb_image, "shape"):
                image = Image.fromarray(rgb_image, mode="RGB")
            elif isinstance(rgb_image, Image.Image):
                image = rgb_image.convert("RGB")
            else:
                raise TypeError
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            return buffer.getvalue()
        except (TypeError, ValueError) as exc:
            raise TypeError("rgb_image must be encoded bytes, an image path, a PIL image, or an HWC RGB numpy array.") from exc

    @staticmethod
    def parse_action(response: str, *, allowed: Sequence[str] = ACTION_TOKENS, show_output: bool = False) -> str:
        """Parse one action token, substituting a legal one if the model ignores the mask."""
        accepted = (*ACTION_TOKENS, *ACTION_ALIASES)
        pattern = r"\b(?:" + "|".join(
            sorted(
                (re.escape(token) for token in accepted),
                key=len,
                reverse=True,
            )
        ) + r")\b"
        matches = re.findall(pattern, response.upper())
        if len(matches) != 1:
            raise ValueError(f"ModelB returned invalid action {response!r}; expected one of {ACTION_TOKENS}.")
        action = ACTION_ALIASES.get(matches[0], matches[0])
        if action in allowed:
            return action
        # A masked token still came back: substitute rather than abort the
        # rollout, but never silently -- this is the failure worth watching.
        fallback = FORWARD if FORWARD in allowed else allowed[0]
        if show_output:
            print(f"[ModelB masked action] {action} is not allowed this step; substituting {fallback}.", flush=True)
        return fallback
