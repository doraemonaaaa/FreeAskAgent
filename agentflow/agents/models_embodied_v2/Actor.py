"""ModelB: Qwen3-VL policy that emits one discrete navigation action token."""

import io
import re
from pathlib import Path
from typing import Any, List, Optional, Union

from agentflow.agents.engine.factory import create_llm_engine

DEFAULT_MODEL_PATH = "models/Qwen3-VL-8B-Instruct"
FORWARD = "FORWARD"
TURN_LEFT = "TURN_LEFT"
TURN_RIGHT = "TURN_RIGHT"
STOP = "STOP"
ACTION_TOKENS = (FORWARD, TURN_LEFT, TURN_RIGHT, STOP)
ACTOR_PROMPT = """Return exactly one token: FORWARD, TURN_LEFT, TURN_RIGHT, or STOP.
Given an RGB image and navigation instruction, output no explanation."""


class Actor:
    """ModelB action policy with the fixed action token space."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, *, debug_performance: bool = True, use_cache: bool = False):
        self.model_path = model_path
        self.llm = create_llm_engine(model_string=f"local-qwen3vl-{model_path}", is_multimodal=True, use_cache=use_cache, debug_performance=debug_performance)

    def act(self, rgb_image: Any, instruction: str = "") -> str:
        image_bytes = self.rgb_to_bytes(rgb_image)
        response = self.llm([f"Select the next action. Navigation instruction: {instruction}", image_bytes], system_prompt=ACTOR_PROMPT, max_tokens=8, temperature=0)
        return self.parse_action(response)

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
    def parse_action(response: str) -> str:
        matches = re.findall(r"\b(?:FORWARD|TURN_LEFT|TURN_RIGHT|STOP)\b", response)
        if len(matches) != 1:
            raise ValueError(f"ModelB returned invalid action {response!r}; expected one of {ACTION_TOKENS}.")
        return matches[0]
