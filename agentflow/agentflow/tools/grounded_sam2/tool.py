"""Grounded SAM 2 tool: text-prompted segmentation via GroundingDINO + SAM2."""
from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from agentflow.agentflow.tools.base import BaseTool

TOOL_NAME = "GroundedSAM2_Tool"


class GroundedSAM2_Tool(BaseTool):
    """Text-prompted segmentation using GroundingDINO for detection and SAM2 for masks."""

    def __init__(
        self,
        model_cfg: str = "sam2.1_hiera_l",
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        dino_config_path: Optional[str] = None,
        dino_checkpoint_path: Optional[str] = None,
    ) -> None:
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "Grounded SAM 2: detect objects from text prompts via GroundingDINO, then refine masks with SAM2."
            ),
            tool_version="0.1.0",
            input_types={
                "image_path": "str - Path to the input image file",
                "text_prompts": "list[str] - Text prompts to ground (comma/space separated phrase list), set according to your needs",
                "box_threshold": "float - GroundingDINO box threshold (default: 0.3)",
                "text_threshold": "float - GroundingDINO text threshold (default: 0.25)",
                "top_k": "int - Limit number of grounded boxes before SAM2 refinement (default: 10)",
                "output_dir": "str - Directory to save visualization (default: tmp)",
                "return_masks": "bool - Whether to include mask arrays in output (default: False)",
            },
            output_type="dict - Dictionary with grounded boxes, masks, and optional visualization_path",
            demo_commands=[
                {
                    "command": 'tool.execute(image_path="/path/to/image.jpg", text_prompts=["person", "car"], output_dir="tmp")',
                    "description": "Ground by text and save visualization"
                }
            ],
            user_metadata={
                "dependencies": "pip install groundingdino-py sam-2",
                "notes": "Provide GroundingDINO config/checkpoint via args or env: GROUNDING_DINO_CONFIG / GROUNDING_DINO_CHECKPOINT",
            },
        )

        self.model_cfg = model_cfg
        self.checkpoint_path = checkpoint_path
        self.device = device

        self._tool_dir = os.path.dirname(os.path.abspath(__file__))
        self._sam2_dir = self._tool_dir  # keep naming parity with existing code
        self.config_file = os.path.join(self._sam2_dir, "..", "sam_perception", "configs", "sam2.1", f"{self.model_cfg}.yaml")
        self.default_checkpoint_path = os.path.join(self._sam2_dir, "..", "sam_perception", "checkpoints", f"{self.model_cfg}.pt")
        self.checkpoint_path = checkpoint_path or self.default_checkpoint_path

        assets_dir = os.path.join(self._tool_dir, "assets")
        default_dino_config = os.path.join(assets_dir, "GroundingDINO_SwinT_OGC.py")
        default_dino_ckpt = os.path.join(assets_dir, "groundingdino_swint_ogc.pth")

        self.dino_config_path = dino_config_path or os.environ.get("GROUNDING_DINO_CONFIG") or default_dino_config
        self.dino_checkpoint_path = dino_checkpoint_path or os.environ.get("GROUNDING_DINO_CHECKPOINT") or default_dino_ckpt

        self.sam2_model = None
        self.image_predictor = None
        self.grounding_model = None
        self._sam_initialized = False
        self._dino_initialized = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def execute(
        self,
        image_path: str,
        text_prompts: List[str],
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
        top_k: int = 10,
        output_dir: str = "tmp",
        return_masks: bool = False,
    ) -> Dict[str, Any]:
        if not image_path or not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not text_prompts:
            raise ValueError("text_prompts must be a non-empty list of strings")

        self._lazy_init_sam()
        self._lazy_init_dino()

        image_source, image_tensor = self._load_image_for_dino(image_path)
        boxes, logits, phrases = self._predict_boxes(text_prompts, image_tensor, box_threshold, text_threshold)

        # Keep top_k by confidence (logits are confidences)
        if logits:
            order = sorted(range(len(logits)), key=lambda i: logits[i], reverse=True)
            boxes = [boxes[i] for i in order]
            logits = [logits[i] for i in order]
            phrases = [phrases[i] for i in order]

        if len(boxes) > top_k:
            boxes = boxes[:top_k]
            logits = logits[:top_k]
            phrases = phrases[:top_k]

        # Refine masks with SAM2
        refined = self._refine_masks_with_sam(image_source, boxes, phrases, return_masks)

        # Compose output
        result = {
            "image_path": image_path,
            "model": self.model_cfg,
            "prompts": text_prompts,
            "boxes": boxes,
            "phrases": phrases,
            "logits": logits,
            "masks": refined["masks_public"],
            "num_masks": len(refined["masks_public"]),
        }

        # Visualization
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            vis_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_grounded_sam2_vis.jpg")
            self._visualize(image_source, refined["masks_viz"], phrases, vis_path)
            result["visualization_path"] = vis_path

        return result

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------
    def _lazy_init_sam(self):
        if self._sam_initialized:
            return
        try:
            import torch
            from hydra import initialize_config_dir, compose
            from hydra.core.global_hydra import GlobalHydra
            from hydra.utils import instantiate
            from omegaconf import OmegaConf
            from sam2.build_sam import _load_checkpoint
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            if self.device == "cuda" and not torch.cuda.is_available():
                print("Warning: CUDA not available, falling back to CPU")
                self.device = "cpu"

            if not os.path.exists(self.checkpoint_path):
                raise FileNotFoundError(f"SAM2 checkpoint not found: {self.checkpoint_path}")
            if not os.path.exists(self.config_file):
                raise FileNotFoundError(f"SAM2 config not found: {self.config_file}")

            GlobalHydra.instance().clear()
            cfg_dir = os.path.dirname(os.path.abspath(self.config_file))
            cfg_name = os.path.basename(self.config_file).replace(".yaml", "")
            with initialize_config_dir(config_dir=cfg_dir, version_base=None):
                cfg = compose(config_name=cfg_name)
            OmegaConf.resolve(cfg)
            model = instantiate(cfg.model, _recursive_=True)
            _load_checkpoint(model, self.checkpoint_path)
            model = model.to(self.device)
            model.eval()
            self.sam2_model = model
            self.image_predictor = SAM2ImagePredictor(model)
            self._sam_initialized = True
            print(f"SAM2 initialized: {self.model_cfg} on {self.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize SAM2: {e}")

    def _lazy_init_dino(self):
        if self._dino_initialized:
            return
        try:
            from groundingdino.util.inference import load_model
            if not self.dino_config_path or not self.dino_checkpoint_path:
                raise FileNotFoundError("GroundingDINO config/checkpoint not provided. Set dino_config_path/dino_checkpoint_path or env vars GROUNDING_DINO_CONFIG / GROUNDING_DINO_CHECKPOINT.")
            if not os.path.exists(self.dino_config_path):
                raise FileNotFoundError(f"GroundingDINO config not found: {self.dino_config_path}")
            if not os.path.exists(self.dino_checkpoint_path):
                raise FileNotFoundError(f"GroundingDINO checkpoint not found: {self.dino_checkpoint_path}")
            self.grounding_model = load_model(self.dino_config_path, self.dino_checkpoint_path, device=self.device)
            self._dino_initialized = True
            print("GroundingDINO initialized")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize GroundingDINO: {e}")

    # ------------------------------------------------------------------
    # Core steps
    # ------------------------------------------------------------------
    @staticmethod
    def _load_image_for_dino(image_path: str):
        from groundingdino.util.inference import load_image
        return load_image(image_path)

    def _predict_boxes(self, text_prompts: List[str], image_tensor, box_threshold: float, text_threshold: float):
        from groundingdino.util.inference import predict
        from torchvision.ops import box_convert

        caption = "; ".join(text_prompts)
        boxes, logits, phrases = predict(
            model=self.grounding_model,
            image=image_tensor,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        # GroundingDINO returns cxcywh, convert to xyxy
        if boxes.numel() > 0:
            boxes = box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")

        # boxes are normalized xyxy in 0-1 range; logits are confidences
        boxes_list = boxes.tolist() if hasattr(boxes, "tolist") else boxes
        logits_list = logits.tolist() if hasattr(logits, "tolist") else logits
        phrases_list = phrases if isinstance(phrases, list) else [str(p) for p in phrases]
        return boxes_list, logits_list, phrases_list

    def _refine_masks_with_sam(self, image_rgb: np.ndarray, boxes: List[List[float]], phrases: List[str], return_masks: bool):
        import torch
        masks_public: List[Dict[str, Any]] = []
        masks_viz: List[Dict[str, Any]] = []
        h, w = image_rgb.shape[:2]
        with torch.inference_mode():
            self.image_predictor.set_image(image_rgb)
            for idx, box in enumerate(boxes):
                if not box or len(box) < 4:
                    continue
                input_box = np.array(box, dtype=float)
                # GroundingDINO predict returns normalized xyxy; scale to pixel coords for SAM2
                if input_box.max() <= 1.1:
                    input_box[0] *= w
                    input_box[2] *= w
                    input_box[1] *= h
                    input_box[3] *= h
                # Clamp to image bounds to avoid out-of-frame boxes
                input_box[0::2] = np.clip(input_box[0::2], 0, w - 1)
                input_box[1::2] = np.clip(input_box[1::2], 0, h - 1)
                masks, scores, _ = self.image_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box,
                    multimask_output=True,
                )
                best_idx = int(np.argmax(scores)) if len(scores) else 0
                mask = masks[best_idx]
                score = float(scores[best_idx]) if len(scores) else None
                mask_info = self._mask_to_info(mask)
                if mask_info:
                    mask_info.update({
                        "id": idx,
                        "prompt_box": box,
                        "prompt_phrase": phrases[idx] if idx < len(phrases) else None,
                        "score": score,
                        "segmentation": mask,  # keep ndarray for viz
                    })
                    # For API output
                    public_info = dict(mask_info)
                    if not return_masks:
                        public_info.pop("segmentation", None)
                    else:
                        public_info["segmentation"] = mask.tolist()
                    masks_public.append(public_info)
                    masks_viz.append(mask_info)
        return {"masks_public": masks_public, "masks_viz": masks_viz}

    @staticmethod
    def _mask_to_info(mask: np.ndarray) -> Optional[Dict[str, Any]]:
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not np.any(rows) or not np.any(cols):
            return None
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
        area = int(np.sum(mask))
        return {
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "area": area,
            "center": [int((x1 + x2) / 2), int((y1 + y2) / 2)],
            "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
        }

    @staticmethod
    def _visualize(image: np.ndarray, masks: List[Dict[str, Any]], phrases: List[str], output_path: str):
        import cv2
        vis = image.copy()
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(len(masks), 3))
        for idx, m in enumerate(masks):
            if "bbox" not in m:
                continue
            x1, y1, x2, y2 = m["bbox"]
            color = colors[idx].tolist()
            # overlay mask if present
            seg = m.get("segmentation")
            if seg is not None:
                seg_bool = seg.astype(bool)
                colored = np.zeros_like(vis)
                colored[seg_bool] = color
                vis = cv2.addWeighted(vis, 1.0, colored, 0.6, 0)
                # draw mask contour to make it visible on bright regions
                contours, _ = cv2.findContours(seg_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, contours, -1, color, 2)
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = m.get("prompt_phrase") or (phrases[idx] if idx < len(phrases) else "")
            if label:
                cv2.putText(vis, label, (int(x1), max(10, int(y1) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, vis)
        print(f"Visualization saved to: {output_path}")

    # ------------------------------------------------------------------
    # Metadata helper
    # ------------------------------------------------------------------
    def get_metadata(self):
        meta = super().get_metadata()
        meta.update({
            "model_cfg": self.model_cfg,
            "device": self.device,
            "sam_checkpoint": self.checkpoint_path,
            "dino_config": self.dino_config_path,
            "dino_checkpoint": self.dino_checkpoint_path,
        })
        return meta


# Backward compatibility alias
Grounded_SAM2_Tool = GroundedSAM2_Tool
