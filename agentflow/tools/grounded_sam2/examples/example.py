"""Demo: grounded text prompts -> masks via GroundingDINO + SAM2."""
from __future__ import annotations

import json
import os
from pathlib import Path

from agentflow.agents.tools.grounded_sam2.tool import GroundedSAM2_Tool

IMAGE_PATH = os.environ.get("GROUND_SAM_IMAGE", "/home/pengyh/workspace/FreeAskAgent/input_img1.jpg")
OUTPUT_DIR = Path(os.environ.get("GROUND_SAM_OUTPUT", "tmp"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEXT_PROMPTS = os.environ.get("GROUND_SAM_PROMPTS", "car, tree, street").split(",")
DEBUG = os.environ.get("GROUND_SAM_DEBUG", "0") == "1"


def main() -> None:
    tool = GroundedSAM2_Tool(
        model_cfg="sam2.1_hiera_l",
        device="cuda",
        dino_config_path=os.environ.get("GROUNDING_DINO_CONFIG"),
        dino_checkpoint_path=os.environ.get("GROUNDING_DINO_CHECKPOINT"),
    )

    if not Path(IMAGE_PATH).exists():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    prompts = [p.strip() for p in TEXT_PROMPTS if p.strip()]
    result = tool.execute(
        image_path=IMAGE_PATH,
        text_prompts=prompts,
        box_threshold=float(os.environ.get("GROUND_SAM_BOX_THRESH", 0.3)),
        text_threshold=float(os.environ.get("GROUND_SAM_TEXT_THRESH", 0.25)),
        top_k=int(os.environ.get("GROUND_SAM_TOPK", 10)),
        output_dir=str(OUTPUT_DIR),
        return_masks=DEBUG,  # set GROUND_SAM_DEBUG=1 to inspect masks in JSON
    )

    print("Grounded SAM2 summary:")
    print(json.dumps({
        "num_masks": result.get("num_masks"),
        "visualization_path": result.get("visualization_path"),
    }, indent=2))

    # Optional debug: print first few mask stats to quickly see if the mask exists and its size
    if DEBUG and result.get("masks"):
        preview = []
        for m in result["masks"][:3]:
            preview.append({
                "phrase": m.get("prompt_phrase"),
                "score": m.get("score"),
                "bbox": m.get("bbox"),
                "area": m.get("area"),
            })
        print("Mask preview:")
        print(json.dumps(preview, indent=2))


if __name__ == "__main__":
    main()
