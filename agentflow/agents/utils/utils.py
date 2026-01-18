
import os
from typing import List, Sequence
from pyparsing import Any
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image

# image meta info
def get_image_info(image_input: Any) -> Dict[str, Any]:
    normalized = normalize_image_inputs(image_input)
    paths = normalized["paths"]
    descriptions = normalized["descriptions"]

    if not paths:
        return {}

    frames: List[Dict[str, Any]] = []
    for index, (path, desc) in enumerate(zip(paths, descriptions)):
        frame_info: Dict[str, Any] = {
            "index": index,
            "image_path": path,
            "filename": os.path.basename(path),
            "description": desc  # 新增：如果有就放进去
        }
        if os.path.isfile(path):
            try:
                with Image.open(path) as img:
                    width, height = img.size
                frame_info.update({
                    "width": width,
                    "height": height
                })
            except Exception as e:
                print(f"Error processing image file '{path}': {str(e)}")
                frame_info["error"] = str(e)
        else:
            frame_info["exists"] = False
        frames.append(frame_info)

    image_info: Dict[str, Any] = {
        "frame_count": len(frames),
        "frames": frames
    }
    if len(frames) == 1:
        image_info.update(frames[0])
    return image_info

# 将可能嵌套的路径输入转换为一个平坦的字符串list
def normalize_image_inputs(image_input: Any) -> Dict[str, Any]:
    """
    返回:
    {
        "paths": List[str],
        "descriptions": List[str | None]   # 与 paths 一一对应
    }
    """
    paths: List[str] = []
    descriptions: List[Optional[str]] = []

    if not image_input:
        return {"paths": paths, "descriptions": descriptions}

    # 处理单个字符串/路径
    if isinstance(image_input, (str, os.PathLike)):
        path = os.fspath(image_input)
        if path:
            paths.append(path)
            descriptions.append(None)
        return {"paths": paths, "descriptions": descriptions}

    # 处理列表（可能嵌套）
    if isinstance(image_input, Sequence):
        for item in image_input:
            if isinstance(item, dict) and "path" in item:
                # 支持 {"path": "xxx.jpg", "description": "xxx"} 格式
                path = os.fspath(item["path"])
                desc = item.get("description")  # 可选
                if path:
                    paths.append(path)
                    descriptions.append(desc)
            elif isinstance(item, (str, os.PathLike)):
                path = os.fspath(item)
                if path:
                    paths.append(path)
                    descriptions.append(None)
            else:
                # 递归处理嵌套
                sub = normalize_image_inputs(item)
                paths.extend(sub["paths"])
                descriptions.extend(sub["descriptions"])
        return {"paths": paths, "descriptions": descriptions}

    return {"paths": paths, "descriptions": descriptions}

def append_image_bytes(
    input_data: List[Any],
    image_input: Any,  # 改为接受 Any，支持新格式
    *,
    log_prefix: str = "Image"
) -> None:
    normalized = normalize_image_inputs(image_input)
    paths = normalized["paths"]
    descriptions = normalized["descriptions"]

    if not paths:
        return

    # 判断是否有任意单独描述
    has_individual_desc = any(d is not None for d in descriptions)

    # 如果没有单独描述，且多张图 → 加统一描述
    if not has_individual_desc and len(paths) > 1:
        filenames = ", ".join(os.path.basename(p) for p in paths)
        input_data.append(
            f"Consider the following {len(paths)} frames in chronological order: {filenames}."
        )

    for path, desc in zip(paths, descriptions):
        if not os.path.isfile(path):
            print(f"[{log_prefix}] Warning: image file not found '{path}' - skipping.")
            continue
        try:
            with open(path, "rb") as f:
                if desc is not None:          # 有单独描述 → 先加描述
                    input_data.append(desc)
                input_data.append(f.read())       # 再加图片字节
        except Exception as e:
            print(f"[{log_prefix}] Error reading image file '{path}': {e}")


def make_json_serializable(obj):
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, dict):
        return {make_json_serializable(key): make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(element) for element in obj]
    elif hasattr(obj, '__dict__'):
        return make_json_serializable(obj.__dict__)
    else:
        return str(obj)
    

def make_json_serializable_truncated(obj, max_length: int = 100000):
    if isinstance(obj, (int, float, bool, type(None))):
        if isinstance(obj, (int, float)) and len(str(obj)) > max_length:
            return str(obj)[:max_length - 3] + "..."
        return obj
    elif isinstance(obj, str):
        return obj if len(obj) <= max_length else obj[:max_length - 3] + "..."
    elif isinstance(obj, dict):
        return {make_json_serializable_truncated(key, max_length): make_json_serializable_truncated(value, max_length) 
                for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable_truncated(element, max_length) for element in obj]
    elif hasattr(obj, '__dict__'):
        return make_json_serializable_truncated(obj.__dict__, max_length)
    else:
        result = str(obj)
        return result if len(result) <= max_length else result[:max_length - 3] + "..."
    