import os
import sys
import json

from agentflow.agentflow.tools.sam_perception.tool import SAM2_Perception_Tool

# 初始化SAM2工具（可选模型：sam2.1_hiera_tiny, sam2.1_hiera_small, sam2.1_hiera_base_plus, sam2.1_hiera_large）
tool = SAM2_Perception_Tool(
    model_cfg="sam2.1_hiera_l",
    device="cuda"
)

# 测试图片路径
image_path = "/home/pengyh/workspace/FreeAskAgent/input_img1.jpg"

if not os.path.exists(image_path):
    print(f"Error: Image not found at {image_path}")
    exit(1)

# 自动分割并保存可视化
result = tool.execute(
    image_path=image_path,
    mode="automatic",
    top_k=5,
    output_dir="tmp"
)

print("SAM2自动分割结果:")
print(json.dumps(result, indent=2, default=str))

if "visualization_path" in result:
    print("可视化图片路径:", result["visualization_path"])

if "error" in result:
    print("Error:", result["error"])
else:
    print("Scene analysis:", result["scene_analysis"]["analysis"])
    print("Suggested direction:", result["scene_analysis"]["suggested_direction"])
    print("Free space ratio:", result["scene_analysis"]["free_space_ratio"])
