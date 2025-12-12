"""Scene graph construction utility that consumes SAM2 segmentation outputs."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


from agentflow.agentflow.tools.base import BaseTool
from agentflow.agentflow.tools.sam_scene_graph.scene_graph_viz import plot_scene_graph_matplotlib

TOOL_NAME = "SAM2_SceneGraph_Tool"


@dataclass
class _ImageSize:
    width: int
    height: int

    @property
    def area(self) -> int:
        return max(1, self.width * self.height)


class SAM2_SceneGraph_Tool(BaseTool):
    """Generate a lightweight scene graph (objects + relations) from SAM2 masks."""

    def __init__(self) -> None:
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "Build a relational scene graph from SAM2_Perception_Tool outputs to power "
                "downstream reasoning, navigation, or visualization tasks."
            ),
            tool_version="0.1.0",
            input_types={
                "sam_result": "dict - Raw output from SAM2_Perception_Tool.execute (optional if sam_result_path provided)",
                "sam_result_path": "str - Path to a JSON file dumped from SAM2 results",
                "min_area": "int - Minimum pixel area to keep an object (default: 2000)",
                "relation_threshold": "float - Minimum normalized distance to emit spatial relations (default: 0.05)",
                "agent_position": "tuple[float,float] - Reference point used for agent-object relations",
                "include_agent_relations": "bool - Whether to insert agent->object edges",
                "save_path": "str - Optional path to store the generated scene graph JSON",
            },
            output_type="dict - Scene graph dictionary with objects, relations, and metadata",
            demo_commands=[
                {
                    "command": "graph = tool.execute(sam_result=result, save_path=\"tmp/scene_graph.json\")",
                    "description": "Convert an in-memory SAM result to a scene graph and persist it"
                },
                {
                    "command": "graph = tool.execute(sam_result_path=\"tmp/sam_output.json\")",
                    "description": "Load SAM results from disk and compute a graph"
                },
            ],
            user_metadata={
                "recommended_pipeline": "Run SAM2_Perception_Tool first, then pipe the result here for structure-aware reasoning"
            },
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def execute(
        self,
        sam_result: Optional[Dict[str, Any]] = None,
        sam_result_path: Optional[str] = None,
        min_area: int = 2000,
        relation_threshold: float = 0.05,
        agent_position: Tuple[float, float] = (0.0, 0.0),
        include_agent_relations: bool = True,
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a scene graph using the provided SAM result, and auto-visualize to tmp/scene_graph.png."""
        data = self._load_sam_result(sam_result, sam_result_path)
        if data is None:
            raise ValueError("Either sam_result or sam_result_path must be provided")

        image_size = self._resolve_image_size(data)
        if image_size is None:
            raise ValueError("SAM result is missing image_size information")

        masks = data.get("masks") or []
        if not isinstance(masks, list):
            raise ValueError("SAM result 'masks' field must be a list")

        analysis = (data.get("scene_analysis") or {}).get("obstacles") or []
        obstacles_by_id = {obs.get("id"): obs for obs in analysis if isinstance(obs, dict)}

        objects = self._build_objects(masks, image_size, min_area, obstacles_by_id)
        relations = self._build_relations(objects, relation_threshold)

        if include_agent_relations:
            relations.extend(self._agent_relations(objects, agent_position))

        graph = {
            "objects": objects,
            "relations": relations,
            "metadata": {
                "image_path": data.get("image_path"),
                "image_size": {"width": image_size.width, "height": image_size.height},
                "num_input_masks": len(masks),
                "num_objects": len(objects),
                "num_relations": len(relations),
                "filters": {"min_area": min_area, "relation_threshold": relation_threshold},
            },
        }

        if save_path:
            self._save_graph(graph, save_path)
            graph["saved_to"] = os.path.abspath(save_path)

        # --- Auto-visualize to tmp/scene_graph.png ---
        try:
            from types import SimpleNamespace
            class MemoryLike(SimpleNamespace):
                def get_scene_graph(self):
                    return graph
            os.makedirs("tmp", exist_ok=True)
            vis_path = os.path.abspath("tmp/scene_graph.png")
            plot_scene_graph_matplotlib(MemoryLike(), outfile=vis_path, agent_position=agent_position)
            graph["scene_graph_image"] = vis_path
        except Exception as e:
            graph["scene_graph_image_error"] = str(e)

        return graph

    # ------------------------------------------------------------------
    # Object extraction helpers
    # ------------------------------------------------------------------
    def _build_objects(
        self,
        masks: List[Dict[str, Any]],
        image_size: _ImageSize,
        min_area: int,
        obstacle_map: Dict[Any, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        objects: List[Dict[str, Any]] = []
        for raw in masks:
            area = int(raw.get("area", 0))
            if area < min_area:
                continue

            bbox = raw.get("bbox_xyxy") or self._bbox_from_wh(raw.get("bbox"))
            if not bbox:
                continue

            center_px = self._center_from_mask(raw, bbox)
            position = self._normalize_center(center_px, image_size)
            bbox_norm = self._normalize_bbox(bbox, image_size)

            obj_id = f"obj_{raw.get('id', len(objects))}"
            obstacle_info = obstacle_map.get(raw.get("id"), {})

            attributes = {
                "depth": obstacle_info.get("depth"),
                "position_bucket": obstacle_info.get("position"),
                "in_path": bool(obstacle_info.get("in_path")),
            }

            obj_payload = {
                "object_id": obj_id,
                "label": f"segment_{raw.get('id', len(objects))}",
                "category": obstacle_info.get("position", "object"),
                "bbox": bbox,
                "bbox_normalized": bbox_norm,
                "position": position,
                "area": area,
                "area_ratio": round(area / image_size.area, 5),
                "predicted_iou": raw.get("predicted_iou"),
                "stability_score": raw.get("stability_score"),
                "is_static": True,
                "attributes": attributes,
            }
            objects.append(obj_payload)

        return objects

    def _build_relations(
        self,
        objects: List[Dict[str, Any]],
        relation_threshold: float,
    ) -> List[Dict[str, Any]]:
        relations: List[Dict[str, Any]] = []
        count = len(objects)
        for i in range(count):
            subj = objects[i]
            for j in range(i + 1, count):
                obj = objects[j]
                dx = obj["position"]["x"] - subj["position"]["x"]
                dy = obj["position"]["y"] - subj["position"]["y"]

                if abs(dx) >= relation_threshold:
                    if dx > 0:
                        relations.append(self._relation(subj, obj, "left_of"))
                    else:
                        relations.append(self._relation(obj, subj, "left_of"))

                if abs(dy) >= relation_threshold:
                    if dy > 0:
                        relations.append(self._relation(obj, subj, "ahead_of"))
                    else:
                        relations.append(self._relation(subj, obj, "ahead_of"))

        return relations

    def _agent_relations(
        self,
        objects: List[Dict[str, Any]],
        agent_position: Tuple[float, float],
    ) -> List[Dict[str, Any]]:
        ax, ay = agent_position
        agent_relations: List[Dict[str, Any]] = []
        for obj in objects:
            predicate = "observes"
            if obj.get("attributes", {}).get("in_path"):
                predicate = "in_path_alert"

            agent_relations.append(
                {
                    "subject_id": "agent",
                    "predicate": predicate,
                    "object_id": obj["object_id"],
                    "distance": self._distance(ax, ay, obj["position"]["x"], obj["position"]["y"]),
                }
            )
        return agent_relations

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _relation(subject: Dict[str, Any], obj: Dict[str, Any], predicate: str) -> Dict[str, Any]:
        return {
            "subject_id": subject["object_id"],
            "predicate": predicate,
            "object_id": obj["object_id"],
        }

    @staticmethod
    def _distance(ax: float, ay: float, bx: float, by: float) -> float:
        dx = bx - ax
        dy = by - ay
        return round((dx * dx + dy * dy) ** 0.5, 4)

    @staticmethod
    def _center_from_mask(raw: Dict[str, Any], bbox: List[float]) -> Tuple[float, float]:
        center = raw.get("center")
        if isinstance(center, list) and len(center) >= 2:
            return float(center[0]), float(center[1])
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2, (y1 + y2) / 2

    @staticmethod
    def _bbox_from_wh(bbox_wh: Optional[List[float]]) -> Optional[List[float]]:
        if not bbox_wh or len(bbox_wh) < 4:
            return None
        x, y, w, h = bbox_wh[:4]
        return [x, y, x + w, y + h]

    @staticmethod
    def _normalize_center(center: Tuple[float, float], image_size: _ImageSize) -> Dict[str, float]:
        width = max(1, image_size.width)
        height = max(1, image_size.height)
        x_norm = center[0] / width
        y_norm = center[1] / height
        # center coords with origin at image center, +x right, +y forward (toward top)
        x_centered = round(x_norm - 0.5, 5)
        y_centered = round(0.5 - y_norm, 5)
        return {"x": x_centered, "y": y_centered}

    @staticmethod
    def _normalize_bbox(bbox: List[float], image_size: _ImageSize) -> List[float]:
        width = max(1, image_size.width)
        height = max(1, image_size.height)
        x1, y1, x2, y2 = bbox
        return [
            round(x1 / width, 5),
            round(y1 / height, 5),
            round(x2 / width, 5),
            round(y2 / height, 5),
        ]

    @staticmethod
    def _load_sam_result(
        sam_result: Optional[Dict[str, Any]],
        sam_result_path: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        if sam_result is not None:
            return sam_result
        if not sam_result_path:
            return None
        path = os.path.abspath(sam_result_path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"SAM result file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _resolve_image_size(data: Dict[str, Any]) -> Optional[_ImageSize]:
        size = data.get("image_size")
        if isinstance(size, dict):
            width = size.get("width")
            height = size.get("height")
            if isinstance(width, int) and isinstance(height, int):
                return _ImageSize(width=width, height=height)
        return None

    @staticmethod
    def _save_graph(graph: Dict[str, Any], save_path: str) -> None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(graph, f, ensure_ascii=False, indent=2)


# Backward compatibility alias
SceneGraph_From_SAM_Tool = SAM2_SceneGraph_Tool
