# SAM 2 Perception Tool

SAM 2 (Segment Anything Model 2) 是 Meta 发布的下一代分割模型，相比 SAM 1 有显著改进。

## SAM 2 vs SAM 1 对比

| 特性 | SAM 1 | SAM 2 |
|------|-------|-------|
| **图像分割** | ✅ | ✅ |
| **视频分割** | ❌ | ✅ 支持对象追踪 |
| **模型大小** | vit_b/l/h | tiny/small/base_plus/large |
| **推理速度** | 较慢 | 更快 (6x faster) |
| **Mask-to-Mask 细化** | ❌ | ✅ |
| **内存效率** | 较低 | 更高 |

## 安装

```bash
# 方式 1: 从 PyPI 安装
pip install sam-2

# 方式 2: 从源码安装 (推荐)
pip install git+https://github.com/facebookresearch/sam2.git

# 安装依赖
pip install opencv-python numpy torch torchvision
```

## 下载模型检查点

SAM 2.1 检查点下载：

```bash
# 创建检查点目录
mkdir -p checkpoints && cd checkpoints

# 下载 SAM 2.1 检查点 (选择一个)
# tiny - 最快，适合实时处理 (~39MB)
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt

# small - 快速 (~46MB)
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt

# base_plus - 平衡 (~81MB)
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt

# large - 最佳质量 (~225MB)
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

## 模型选择指南

| 模型 | 大小 | 速度 | 质量 | 推荐场景 |
|------|------|------|------|----------|
| `sam2.1_hiera_tiny` | ~39MB | 最快 | 可接受 | 实时处理、边缘设备 |
| `sam2.1_hiera_small` | ~46MB | 快 | 良好 | 一般应用 |
| `sam2.1_hiera_base_plus` | ~81MB | 中等 | 很好 | 平衡质量与速度 |
| `sam2.1_hiera_large` | ~225MB | 较慢 | 最佳 | 高精度需求 |

## 基本用法

### 初始化

```python
from agentflow.agentflow.tools.sam_perception.tool_sam2 import SAM2_Perception_Tool

# 使用不同模型大小
tool = SAM2_Perception_Tool(model_cfg="sam2.1_hiera_large")  # 最佳质量
tool = SAM2_Perception_Tool(model_cfg="sam2.1_hiera_tiny")   # 最快速度

# 指定检查点路径
tool = SAM2_Perception_Tool(
    model_cfg="sam2.1_hiera_large",
    checkpoint_path="/path/to/sam2.1_hiera_large.pt",
    device="cuda"
)
```

### 自动模式 (场景理解)

```python
# 自动分割 - 检测图像中所有对象
result = tool.execute(
    image_path="path/to/image.jpg",
    mode="automatic",
    top_k=10,
    output_dir="./output"
)

# 获取导航分析
print(result["scene_analysis"]["analysis"])
print(f"建议方向: {result['scene_analysis']['suggested_direction']}")
print(f"可通行空间比例: {result['scene_analysis']['free_space_ratio']}")
```

### 点提示模式 (分割特定对象)

```python
# 使用点提示分割特定对象
result = tool.execute(
    image_path="path/to/image.jpg",
    mode="point",
    points=[[320, 240]],  # x, y 坐标
    point_labels=[1],      # 1 = 前景, 0 = 背景
)

# 多个点以获得更精确的分割
result = tool.execute(
    image_path="path/to/image.jpg",
    mode="point",
    points=[[320, 240], [350, 260]],  # 多个前景点
    point_labels=[1, 1],
)

# 前景 + 背景点用于细化
result = tool.execute(
    image_path="path/to/image.jpg",
    mode="point",
    points=[[320, 240], [100, 100]],
    point_labels=[1, 0],  # 第一个点是前景，第二个是背景
)
```

### 边界框模式

```python
# 在边界框内分割对象
result = tool.execute(
    image_path="path/to/image.jpg",
    mode="box",
    box=[100, 100, 400, 300],  # [x1, y1, x2, y2]
)
```

### 视频模式 (SAM 2 新功能!)

```python
# 在视频中追踪对象 - SAM 2 独有功能
result = tool.execute(
    video_path="path/to/video.mp4",
    mode="video",
    points=[[320, 240]],  # 初始帧上的点提示
    point_labels=[1],
    output_dir="./video_output",
    frame_indices=None,  # None = 自动采样，或指定 [0, 10, 20, ...]
)

# 使用边界框初始化视频追踪
result = tool.execute(
    video_path="path/to/video.mp4",
    mode="video",
    box=[100, 100, 400, 300],
    output_dir="./video_output",
)

# 视频结果包含每帧的追踪信息
for frame in result["tracked_frames"]:
    print(f"Frame {frame['frame_idx']}: {len(frame['objects'])} objects tracked")
```

## 输出格式

### 图像模式输出

```python
{
    "image_path": "path/to/image.jpg",
    "image_size": {"width": 640, "height": 480},
    "mode": "automatic",
    "model": "sam2.1_hiera_large",
    "masks": [
        {
            "id": 0,
            "bbox": [x, y, w, h],
            "bbox_xyxy": [x1, y1, x2, y2],
            "center": [cx, cy],
            "area": 12345,
            "predicted_iou": 0.95,
            "stability_score": 0.98,
        },
        # ... 更多 masks
    ],
    "num_masks": 10,
    "scene_analysis": {
        "obstacles": [...],
        "num_obstacles": 5,
        "obstacles_in_path": 2,
        "free_space_ratio": 0.65,
        "suggested_direction": "forward",
        "analysis": "Detected 5 objects. 2 objects blocking the center path. ..."
    },
    "visualization_path": "./output/image_sam2_vis.jpg"
}
```

### 视频模式输出

```python
{
    "video_path": "path/to/video.mp4",
    "video_info": {
        "total_frames": 300,
        "fps": 30.0,
        "width": 1920,
        "height": 1080,
    },
    "mode": "video",
    "model": "sam2.1_hiera_large",
    "tracked_frames": [
        {
            "frame_idx": 0,
            "objects": [
                {
                    "obj_id": 1,
                    "bbox": [x1, y1, x2, y2],
                    "center": [cx, cy],
                    "area": 12345,
                    "score": 0.95,
                },
            ],
        },
        # ... 更多帧
    ],
    "num_tracked_frames": 30,
}
```

## 场景分析 (导航应用)

`scene_analysis` 字段提供导航相关信息：

- **obstacles**: 检测到的对象列表，包含位置 (left/center/right) 和深度 (near/medium/far)
- **obstacles_in_path**: 直接路径上的障碍物数量 (底部中心区域)
- **free_space_ratio**: 路径上的可通行空间比例 (0.0 = 完全阻塞, 1.0 = 完全畅通)
- **suggested_direction**: 建议的导航方向 ("forward", "left", "right", "stop")
- **analysis**: 人类可读的分析文本

## 与 Agent Solver 集成

```python
from agentflow.agentflow.solver_fast import construct_fast_solver

# 在 solver 中启用 SAM 2 Perception Tool
solver = construct_fast_solver(
    llm_engine_name="gpt-4o",
    enabled_tools=["SAM2_Perception_Tool", "Base_Generator_Tool"],
    tool_engine=["Default", "gpt-4o"],
    output_types="direct",
    enable_multimodal=True,
)

# 使用感知能力解决问题
output = solver.solve(
    "分析这张图片中的障碍物，告诉我是否可以安全前进",
    image_paths=["path/to/image.jpg"]
)
```

## 向后兼容

`tool_sam2.py` 提供了一个别名 `SAM_Perception_Tool`，可以作为 SAM 1 的替代品：

```python
# 这两种导入方式都可以工作
from agentflow.agentflow.tools.sam_perception.tool_sam2 import SAM2_Perception_Tool
from agentflow.agentflow.tools.sam_perception.tool_sam2 import SAM_Perception_Tool  # 别名

# 原有代码可以继续使用
tool = SAM_Perception_Tool(model_cfg="sam2.1_hiera_large")
```

## 常见问题

### Q: 选择 SAM 1 还是 SAM 2？

**推荐使用 SAM 2**，除非：
- 你有特定于 SAM 1 的代码依赖
- 需要使用 SAM 1 特定的检查点

### Q: GPU 内存不足？

尝试使用更小的模型：
```python
tool = SAM2_Perception_Tool(model_cfg="sam2.1_hiera_tiny")
```

### Q: 如何提高分割质量？

1. 使用 `sam2.1_hiera_large` 模型
2. 提供更精确的点或框提示
3. 使用前景+背景点组合

### Q: 视频处理太慢？

1. 使用 `sam2.1_hiera_tiny` 或 `sam2.1_hiera_small`
2. 减少处理的帧数 (`frame_indices` 参数)
3. 确保使用 GPU

## 参考链接

- [SAM 2 GitHub](https://github.com/facebookresearch/sam2)
- [SAM 2 论文](https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/)
- [SAM 2 Demo](https://sam2.metademolab.com/)

## 本地 config 和 checkpoint 路径说明（自定义项目结构推荐）

> **2025年12月更新：本工具已支持直接在 sam_perception 子目录下放置 config 和 checkpoint，无需依赖 site-packages 路径，也无需软链。**

- **配置文件(config)**：
  - 路径：`agentflow/agentflow/tools/sam_perception/configs/sam2.1/sam2.1_hiera_large.yaml`
  - 你可以把所有模型的 yaml 配置都放在 `sam_perception/configs/sam2.1/` 目录下。
- **权重文件(checkpoint)**：
  - 路径：`agentflow/agentflow/tools/sam_perception/checkpoints/sam2.1_hiera_large.pt`
  - 你可以把所有模型的 pt 文件都放在 `sam_perception/checkpoints/` 目录下。

**代码会自动用相对路径查找，无论你在哪里运行都能找到。**

例如：
```python
# 只需这样初始化即可，无需指定绝对路径
from agentflow.agentflow.tools.sam_perception.tool import SAM2_Perception_Tool

tool = SAM2_Perception_Tool(model_cfg="sam2.1_hiera_large")
# 会自动加载：
#   configs/sam2.1/sam2.1_hiera_large.yaml
#   checkpoints/sam2.1_hiera_large.pt
```

> ⚠️ 如果缺少 config 或 checkpoint，会报错提示缺哪个文件。

**这样管理后，迁移/多环境/多人协作都很方便！**
