
import os
from typing import List, Sequence
from pyparsing import Any
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image

def get_image_info(image_input: Any) -> Dict[str, Any]:
    image_paths = normalize_image_paths(image_input)
    if not image_paths:
        return {}

    frames: List[Dict[str, Any]] = []
    for index, path in enumerate(image_paths):
        frame_info: Dict[str, Any] = {
            "index": index,
            "image_path": path,
            "filename": os.path.basename(path)
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

def normalize_image_paths(image_input: Any) -> List[str]:
    paths: List[str] = []
    if not image_input:
        return paths
    if isinstance(image_input, (bytes, bytearray)):
        return paths
    if isinstance(image_input, (str, os.PathLike)):
        candidate = os.fspath(image_input)
        if candidate:
            paths.append(candidate)
        return paths
    if isinstance(image_input, Sequence):
        for item in image_input:
            paths.extend(normalize_image_paths(item))
        return paths
    return paths


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
    