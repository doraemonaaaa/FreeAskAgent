# Grounded SAM2 Tool

Text-prompted segmentation that combines **GroundingDINO** (text-conditioned detection) with **SAM2** (mask refinement).

## Dependencies
- `pip install groundingdino-py sam-2` (or your preferred GroundingDINO + SAM2 builds)
- Provide GroundingDINO assets via args/env, or place them in `grounded_sam2/assets` (default lookup):
  - `GROUNDING_DINO_CONFIG` (default: `grounded_sam2/assets/GroundingDINO_SwinT_OGC.py`)
  - `GROUNDING_DINO_CHECKPOINT` (default: `grounded_sam2/assets/groundingdino_swint_ogc.pth`)

## Usage (Python)
```python
from agentflow.agentflow.tools.grounded_sam2.tool import GroundedSAM2_Tool

tool = GroundedSAM2_Tool(
    model_cfg="sam2.1_hiera_l",
    device="cuda",
    dino_config_path="/path/to/GroundingDINO_SwinT_OGC.py",
    dino_checkpoint_path="/path/to/groundingdino_swint_ogc.pth",
)

result = tool.execute(
    image_path="/path/to/image.jpg",
    text_prompts=["person", "car"],
    box_threshold=0.3,
    text_threshold=0.25,
    top_k=10,
    output_dir="tmp",
    return_masks=False,
)
print(result.get("visualization_path"))
```

## Inputs
- `image_path`: image file path
- `text_prompts`: list of phrases to ground
- `box_threshold`: GroundingDINO box threshold (default 0.3)
- `text_threshold`: GroundingDINO text threshold (default 0.25)
- `top_k`: cap number of grounded boxes before SAM2 refinement (default 10)
- `output_dir`: where to save visualization (default `tmp`)
- `return_masks`: include mask arrays in output (default False)

## Outputs
- `masks`: refined masks (with bbox/center/area)
- `boxes`, `phrases`, `logits`: raw GroundingDINO outputs (normalized xyxy boxes)
- `visualization_path`: saved overlay of grounded boxes
- `num_masks`, `image_path`, metadata

## Notes
- This tool mirrors the interface style of `SAM2_Perception_Tool`, but requires GroundingDINO assets.
- If dependencies are missing, it will raise a clear error prompting install/config paths.
