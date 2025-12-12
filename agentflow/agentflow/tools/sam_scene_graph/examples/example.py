"""Quick demo: run SAM2 segmentation and build a scene graph."""
from __future__ import annotations

import json
import os
from pathlib import Path

from agentflow.agentflow.tools.sam_perception.tool import SAM2_Perception_Tool
from agentflow.agentflow.tools.sam_scene_graph.tool import SAM2_SceneGraph_Tool

IMAGE_PATH = os.environ.get("SAM_DEMO_IMAGE", "/home/pengyh/workspace/FreeAskAgent/input_img1.jpg")
OUTPUT_DIR = Path(os.environ.get("SAM_DEMO_OUTPUT", "tmp"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    sam_tool = SAM2_Perception_Tool(model_cfg="sam2.1_hiera_l", device="cuda")
    graph_tool = SAM2_SceneGraph_Tool()

    if not Path(IMAGE_PATH).exists():
        raise FileNotFoundError(f"Demo image not found: {IMAGE_PATH}")

    sam_result = sam_tool.execute(
        image_path=IMAGE_PATH,
        mode="automatic",
        top_k=5,
        output_dir=str(OUTPUT_DIR),
    )

    graph = graph_tool.execute(
        sam_result=sam_result,
        min_area=1500,
        relation_threshold=0.04,
        save_path=str(OUTPUT_DIR / "demo_scene_graph.json"),
    )

    print("Scene graph summary:")
    print(json.dumps({
        "num_objects": len(graph.get("objects", [])),
        "num_relations": len(graph.get("relations", [])),
        "saved_to": graph.get("saved_to"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
