"""
SAM 2 (Segment Anything Model 2) Perception Tool for Agent

This tool uses Meta's Segment Anything Model 2 to provide perception capabilities
for the agent, including:
- Automatic mask generation for all objects in an image
- Point-prompted segmentation
- Box-prompted segmentation
- Video object segmentation and tracking (NEW in SAM 2)
- Object detection and scene understanding

SAM 2 improvements over SAM 1:
- Real-time video segmentation with object tracking
- Faster inference speed
- Better segmentation quality
- Smaller model sizes available (tiny, small, base_plus, large)
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple
from agentflow.tools.base import BaseTool

# Tool name mapping - this defines the external name for this tool
TOOL_NAME = "SAM2_Perception_Tool"

LIMITATION = f"""
The {TOOL_NAME} has the following limitations:
1. Requires GPU for optimal performance (CPU mode is significantly slower)
2. May not detect very small objects or objects with low contrast
3. Does not provide semantic labels for segmented objects (only masks)
4. Memory intensive for high-resolution images and long videos
5. Video mode requires sufficient GPU memory for frame processing
"""

BEST_PRACTICE = f"""
For optimal results with the {TOOL_NAME}:
1. Use clear, well-lit images for better segmentation quality
2. For specific object segmentation, provide point or box prompts
3. Use automatic mode for scene understanding and object discovery
4. Combine with other tools (like image captioning) for semantic understanding
5. For navigation tasks, focus on obstacle detection and free space analysis
6. For video: use smaller models (tiny/small) for real-time processing
7. For accuracy: use larger models (base_plus/large) for better quality
"""


class SAM2_Perception_Tool(BaseTool):
    """
    SAM 2 Perception Tool for visual perception in agent systems.
    
    Supports multiple modes:
    - automatic: Generate masks for all objects in the image
    - point: Segment based on point prompts (x, y coordinates)
    - box: Segment based on bounding box prompts
    - video: Track objects across video frames (NEW in SAM 2)
    
    Model sizes available:
    - sam2.1_hiera_tiny: Fastest, smallest (good for real-time)
    - sam2.1_hiera_small: Fast, small
    - sam2.1_hiera_base_plus: Balanced speed/quality
    - sam2.1_hiera_large: Best quality, slower
    """
    
    require_llm_engine = False  # SAM is a vision model, not LLM

    def __init__(
        self,
        model_cfg: str = "sam2.1_hiera_l",
        checkpoint_path: str = None,
        device: str = "cuda"
    ):
        """
        Initialize SAM 2 Perception Tool.
        
        Args:
            model_cfg: SAM 2 model config name:
                - "sam2.1_hiera_tiny" - Fastest, for real-time
                - "sam2.1_hiera_small" - Fast
                - "sam2.1_hiera_base_plus" - Balanced
                - "sam2.1_hiera_large" - Best quality (default)
            checkpoint_path: Path to SAM 2 checkpoint file. If None, will auto-download
            device: Device to run model on - "cuda" or "cpu"
        """
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description="A perception tool using Segment Anything Model 2 (SAM 2) for image and video segmentation. Supports automatic mask generation, point-based segmentation, box-based segmentation, and video object tracking for navigation and scene understanding.",
            tool_version="2.1.0",
            input_types={
                "image_path": "str - Path to the input image file (for image mode)",
                "video_path": "str - Path to the input video file (for video mode)",
                "mode": "str - Segmentation mode: 'automatic', 'point', 'box', or 'video' (default: 'automatic')",
                "points": "list[list[int]] - List of [x, y] point coordinates for point mode (optional)",
                "point_labels": "list[int] - Labels for each point: 1 for foreground, 0 for background (optional)",
                "box": "list[int] - Bounding box [x1, y1, x2, y2] for box mode (optional)",
                "output_dir": "str - Directory to save visualization results (optional)",
                "return_masks": "bool - Whether to return mask arrays (default: False)",
                "top_k": "int - Number of top masks to return in automatic mode (default: 10), Choose parameter according to your needs or scene complexity.",
                "frame_indices": "list[int] - Specific frame indices to process in video mode (optional)",
            },
            output_type="dict - Dictionary containing segmentation results with mask info, bounding boxes, and optionally mask arrays",
            demo_commands=[
                {
                    "command": 'result = tool.execute(image_path="path/to/image.jpg", mode="automatic", top_k=10, output_dir="tmp")',
                    "description": "Automatically segment all objects in the image and save to tmp"
                },
                {
                    "command": 'result = tool.execute(image_path="path/to/image.jpg", mode="point", points=[[100, 200]], point_labels=[1])',
                    "description": "Segment object at the specified point (foreground)"
                },
                {
                    "command": 'result = tool.execute(image_path="path/to/image.jpg", mode="box", box=[50, 50, 200, 200])',
                    "description": "Segment object within the specified bounding box"
                },
                {
                    "command": 'result = tool.execute(video_path="path/to/video.mp4", mode="video", points=[[100, 200]], point_labels=[1])',
                    "description": "Track object across video frames starting from point prompt"
                },
            ],
            user_metadata={
                "limitation": LIMITATION,
                "best_practice": BEST_PRACTICE
            }
        )
        
        self.model_cfg = model_cfg
        # Always resolve config/checkpoint relative to this file
        self._sam2_dir = os.path.dirname(os.path.abspath(__file__))
        # Local filesystem config path (may or may not exist)
        self.config_file = os.path.join(self._sam2_dir, "configs", "sam2.1", f"{self.model_cfg}.yaml")
        # Hydra / sam2 package config name fallback is not used here.
        # Keep _hydra_config_name explicit and set to None to avoid deriving from a short map.
        self._hydra_config_name = None
        self.default_checkpoint_path = os.path.join(self._sam2_dir, "checkpoints", f"{self.model_cfg}.pt")
        self.checkpoint_path = checkpoint_path or self.default_checkpoint_path
        self.device = device
        self.sam2_model = None
        self.image_predictor = None
        self.video_predictor = None
        self.mask_generator = None
        self._is_initialized = False
        
        print(f"Initializing SAM 2 Perception Tool with model: {self.model_cfg}")
    
    def _lazy_init(self):
        """Lazy initialization of SAM 2 model to avoid loading at import time."""
        if self._is_initialized:
            return
        
        try:
            import torch
            
            # Determine device
            if self.device == "cuda" and not torch.cuda.is_available():
                print("Warning: CUDA not available, falling back to CPU")
                self.device = "cpu"
            
            # Try to import SAM 2
            try:
                from sam2.build_sam import build_sam2, build_sam2_video_predictor
                from sam2.sam2_image_predictor import SAM2ImagePredictor
                from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            except ImportError:
                raise ImportError(
                    "sam2 package not found. Please install with:\n"
                    "pip install sam-2\n"
                    "Or from source:\n"
                    "pip install git+https://github.com/facebookresearch/sam2.git"
                )

            print(f"Loading SAM 2 model: {self.model_cfg}")
            print(f"Local config path: {self.config_file}")
            print(f"Package config name (fallback): {self._hydra_config_name}")
            print(f"Checkpoint path: {self.checkpoint_path}")

            if not os.path.exists(self.checkpoint_path):
                raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoint_path}")

            # Use local config if it exists
            if os.path.exists(self.config_file):
                from hydra import initialize_config_dir, compose
                from hydra.core.global_hydra import GlobalHydra
                
                config_dir = os.path.dirname(self.config_file)
                config_name = os.path.basename(self.config_file).replace('.yaml', '')
                
                # Clear any existing Hydra instance
                GlobalHydra.instance().clear()
                
                # Initialize with the local config directory
                with initialize_config_dir(config_dir=config_dir, version_base=None):
                    cfg = compose(config_name=config_name)
                
                # Build model from config and checkpoint
                from hydra.utils import instantiate
                from omegaconf import OmegaConf
                
                OmegaConf.resolve(cfg)
                self.sam2_model = instantiate(cfg.model, _recursive_=True)
                
                # Load checkpoint
                from sam2.build_sam import _load_checkpoint
                _load_checkpoint(self.sam2_model, self.checkpoint_path)
                self.sam2_model = self.sam2_model.to(self.device)
                self.sam2_model.eval()
            else:
                raise FileNotFoundError(f"Config file not found: {self.config_file}")
            
            # Initialize image predictor
            self.image_predictor = SAM2ImagePredictor(self.sam2_model)
            
            # Initialize automatic mask generator
            self.mask_generator = SAM2AutomaticMaskGenerator(
                model=self.sam2_model,
                points_per_side=32,
                points_per_batch=64,
                pred_iou_thresh=0.7,
                stability_score_thresh=0.92,
                stability_score_offset=0.7,
                crop_n_layers=1,
                box_nms_thresh=0.7,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100,
                use_m2m=True,  # Use mask-to-mask refinement
            )
            
            self._is_initialized = True
            print(f"SAM 2 model loaded successfully on {self.device}")
            
        except Exception as e:
            import traceback
            print(f"Error initializing SAM 2: {e}")
            print(traceback.format_exc())
            raise
    
    def _init_video_predictor(self):
        """Initialize video predictor (separate from image predictor)."""
        if self.video_predictor is not None:
            return
        
        try:
            from sam2.build_sam import build_sam2_video_predictor

            if self.checkpoint_path:
                self.video_predictor = build_sam2_video_predictor(
                    config_file=self.config_file,
                    ckpt_path=self.checkpoint_path,
                    device=self.device,
                )
            else:
                self.video_predictor = build_sam2_video_predictor(
                    config_file=self.config_file,
                    ckpt_path=None,
                    device=self.device,
                )
            print("SAM 2 video predictor initialized")
        except Exception as e:
            print(f"Warning: Could not initialize video predictor: {e}")
            self.video_predictor = None
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load image from path and convert to RGB numpy array."""
        from PIL import Image
        
        image = Image.open(image_path)
        image = np.array(image.convert("RGB"))
        return image
    
    def _mask_to_info(self, mask: np.ndarray, score: float = None, area: int = None) -> Dict[str, Any]:
        """Convert mask to info dictionary with bounding box and statistics."""
        # Get bounding box from mask
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return None
        
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
                              
        mask_area = int(np.sum(mask)) if area is None else area
        
        return {
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "area": mask_area,
            "score": float(score) if score is not None else None,
            "center": [int((x1 + x2) / 2), int((y1 + y2) / 2)],
            "width": int(x2 - x1),
            "height": int(y2 - y1),
        }
    
    def _visualize_masks(self, image: np.ndarray, masks: List[Dict], output_path: str):
        """Save visualization of masks overlaid on image."""
        import cv2
        
        # Create a copy of the image
        vis_image = image.copy()
        
        # Generate random colors for each mask
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(len(masks), 3))
        
        for idx, mask_info in enumerate(masks):
            if "segmentation" in mask_info:
                mask = mask_info["segmentation"]
                if isinstance(mask, list):
                    mask = np.array(mask)
            else:
                continue
            
            color = colors[idx].tolist()
            
            # Create colored mask overlay
            colored_mask = np.zeros_like(vis_image)
            colored_mask[mask] = color
            
            # Blend with original image
            vis_image = cv2.addWeighted(vis_image, 1, colored_mask, 0.4, 0)
            
            # Draw bounding box
            if "bbox" in mask_info:
                bbox = mask_info["bbox"]
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    # Handle both [x, y, w, h] and [x1, y1, x2, y2] formats
                    if x2 < x1:  # Likely [x, y, w, h] format
                        x2 = x1 + x2
                        y2 = y1 + y2
                    cv2.rectangle(vis_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        # Convert RGB to BGR for saving
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, vis_image)
        print(f"Visualization saved to: {output_path}")
    
    def execute(
        self,
        image_path: str = None,
        video_path: str = None,
        mode: str = "automatic",
        points: List[List[int]] = None,
        point_labels: List[int] = None,
        box: List[int] = None,
        output_dir: str = None,
        return_masks: bool = False,
        top_k: int = 10,
        frame_indices: List[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute SAM 2 perception on the input image or video.
        
        Args:
            image_path: Path to input image (for image modes)
            video_path: Path to input video (for video mode)
            mode: Segmentation mode - "automatic", "point", "box", or "video"
            points: List of [x, y] coordinates for point/video mode
            point_labels: Labels for points (1=foreground, 0=background)
            box: Bounding box [x1, y1, x2, y2] for box mode
            output_dir: Directory to save visualization
            return_masks: Whether to return mask arrays
            top_k: Number of top masks to return in automatic mode
            frame_indices: Specific frame indices for video mode
            
        Returns:
            Dictionary with segmentation results
        """
        try:
            # Lazy initialization
            self._lazy_init()
            
            import torch

            # Enforce defaults if not provided by caller
            if output_dir is None:
                output_dir = "tmp"
            
            if mode == "video":
                return self._execute_video(
                    video_path=video_path,
                    points=points,
                    point_labels=point_labels,
                    box=box,
                    output_dir=output_dir,
                    return_masks=return_masks,
                    frame_indices=frame_indices,
                )
            
            # Image modes
            if image_path is None:
                raise ValueError("image_path must be provided for image modes")
            
            # Load image
            image = self._load_image(image_path)
            image_height, image_width = image.shape[:2]
            
            result = {
                "image_path": image_path,
                "image_size": {"width": image_width, "height": image_height},
                "mode": mode,
                "model": self.model_cfg,
                "masks": [],
                "num_masks": 0,
            }
            
            if mode == "automatic":
                # Automatic mask generation
                masks = self.mask_generator.generate(image)
                
                # Sort by area (largest first) and take top_k
                masks = sorted(masks, key=lambda x: x["area"], reverse=True)[:top_k]
                
                for idx, mask_data in enumerate(masks):
                    mask_info = {
                        "id": idx,
                        "bbox": mask_data["bbox"],  # [x, y, w, h] format from SAM
                        "area": mask_data["area"],
                        "predicted_iou": mask_data["predicted_iou"],
                        "stability_score": mask_data["stability_score"],
                        "point_coords": mask_data.get("point_coords", []),
                    }
                    
                    # Convert bbox from [x, y, w, h] to [x1, y1, x2, y2]
                    x, y, w, h = mask_data["bbox"]
                    mask_info["bbox_xyxy"] = [x, y, x + w, y + h]
                    mask_info["center"] = [x + w // 2, y + h // 2]
                    
                    if return_masks:
                        mask_info["segmentation"] = mask_data["segmentation"].tolist()
                    
                    result["masks"].append(mask_info)
                
                result["num_masks"] = len(result["masks"])
                
            elif mode == "point":
                if points is None:
                    raise ValueError("Points must be provided for point mode")
                
                # Set image for predictor
                with torch.inference_mode():
                    self.image_predictor.set_image(image)
                    
                    # Convert to numpy arrays
                    input_points = np.array(points)
                    input_labels = np.array(point_labels) if point_labels else np.ones(len(points))
                    
                    # Predict masks
                    masks, scores, logits = self.image_predictor.predict(
                        point_coords=input_points,
                        point_labels=input_labels,
                        multimask_output=True,
                    )
                
                # Process each predicted mask
                for idx, (mask, score) in enumerate(zip(masks, scores)):
                    mask_info = self._mask_to_info(mask, score)
                    if mask_info:
                        mask_info["id"] = idx
                        mask_info["prompt_points"] = points
                        mask_info["prompt_labels"] = point_labels if point_labels else [1] * len(points)
                        
                        if return_masks:
                            mask_info["segmentation"] = mask.tolist()
                        
                        result["masks"].append(mask_info)
                
                result["num_masks"] = len(result["masks"])
                result["prompt_points"] = points
                result["prompt_labels"] = point_labels if point_labels else [1] * len(points)
                
            elif mode == "box":
                if box is None:
                    raise ValueError("Box must be provided for box mode")
                
                # Set image for predictor
                with torch.inference_mode():
                    self.image_predictor.set_image(image)
                    
                    # Convert to numpy array
                    input_box = np.array(box)
                    
                    # Predict mask
                    masks, scores, logits = self.image_predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=input_box,
                        multimask_output=True,
                    )
                
                # Process each predicted mask
                for idx, (mask, score) in enumerate(zip(masks, scores)):
                    mask_info = self._mask_to_info(mask, score)
                    if mask_info:
                        mask_info["id"] = idx
                        mask_info["prompt_box"] = box
                        
                        if return_masks:
                            mask_info["segmentation"] = mask.tolist()
                        
                        result["masks"].append(mask_info)
                
                result["num_masks"] = len(result["masks"])
                result["prompt_box"] = box
                
            else:
                raise ValueError(f"Unknown mode: {mode}. Supported modes: automatic, point, box, video")
            
            # Save visualization if output_dir is specified
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                image_name = Path(image_path).stem
                vis_path = os.path.join(output_dir, f"{image_name}_sam2_vis.jpg")
                
                # Need to re-run to get segmentation masks for visualization
                if mode == "automatic":
                    masks_with_seg = self.mask_generator.generate(image)
                    masks_with_seg = sorted(masks_with_seg, key=lambda x: x["area"], reverse=True)[:top_k]
                    self._visualize_masks(image, masks_with_seg, vis_path)
                else:
                    # For point/box mode, visualize with masks
                    vis_masks = []
                    for mask_info in result["masks"]:
                        if "segmentation" in mask_info:
                            vis_masks.append({
                                "segmentation": np.array(mask_info["segmentation"]),
                                "bbox": mask_info["bbox"]
                            })
                    if vis_masks:
                        self._visualize_masks(image, vis_masks, vis_path)
                
                result["visualization_path"] = vis_path
            
            # Add scene analysis summary for navigation
            result["scene_analysis"] = self._analyze_scene_for_navigation(result["masks"], image_width, image_height)
            
            return result
            
        except Exception as e:
            import traceback
            return {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "image_path": image_path,
                "video_path": video_path,
                "mode": mode,
            }
    
    def _execute_video(
        self,
        video_path: str,
        points: List[List[int]] = None,
        point_labels: List[int] = None,
        box: List[int] = None,
        output_dir: str = None,
        return_masks: bool = False,
        frame_indices: List[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute SAM 2 video object tracking.
        
        This is a NEW feature in SAM 2 that allows tracking objects across video frames.
        """
        import torch
        import cv2
        
        if video_path is None:
            raise ValueError("video_path must be provided for video mode")
        
        if points is None and box is None:
            raise ValueError("Either points or box must be provided for video mode")
        
        # Initialize video predictor
        self._init_video_predictor()
        
        if self.video_predictor is None:
            return {
                "error": "Video predictor not available. Please ensure SAM 2 is properly installed.",
                "video_path": video_path,
                "mode": "video",
            }
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        result = {
            "video_path": video_path,
            "video_info": {
                "total_frames": total_frames,
                "fps": fps,
                "width": width,
                "height": height,
            },
            "mode": "video",
            "model": self.model_cfg,
            "tracked_frames": [],
        }
        
        # Determine which frames to process
        if frame_indices is None:
            # Sample frames (every 10th frame by default)
            frame_indices = list(range(0, total_frames, 10))
        
        # Extract frames
        frames = []
        frame_paths = []
        temp_dir = output_dir or "/tmp/sam2_video_frames"
        os.makedirs(temp_dir, exist_ok=True)
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                frame_path = os.path.join(temp_dir, f"frame_{frame_idx:06d}.jpg")
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
        
        cap.release()
        
        try:
            # Initialize video state
            with torch.inference_mode():
                inference_state = self.video_predictor.init_state(video_path=temp_dir)
                
                # Add prompts on first frame
                if points is not None:
                    input_points = np.array(points)
                    input_labels = np.array(point_labels) if point_labels else np.ones(len(points))
                    _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=0,
                        obj_id=1,
                        points=input_points,
                        labels=input_labels,
                    )
                elif box is not None:
                    input_box = np.array(box)
                    _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=0,
                        obj_id=1,
                        box=input_box,
                    )
                
                # Propagate through video
                video_segments = {}
                for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(inference_state):
                    video_segments[out_frame_idx] = {
                        "obj_ids": out_obj_ids,
                        "masks": (out_mask_logits > 0.0).cpu().numpy(),
                    }
                
                # Process results
                for frame_idx in sorted(video_segments.keys()):
                    segment = video_segments[frame_idx]
                    masks = segment["masks"]
                    
                    frame_result = {
                        "frame_idx": frame_idx,
                        "objects": [],
                    }
                    
                    for obj_idx, obj_id in enumerate(segment["obj_ids"]):
                        mask = masks[obj_idx][0]  # First channel
                        mask_info = self._mask_to_info(mask)
                        if mask_info:
                            mask_info["obj_id"] = int(obj_id)
                            if return_masks:
                                mask_info["segmentation"] = mask.tolist()
                            frame_result["objects"].append(mask_info)
                    
                    result["tracked_frames"].append(frame_result)
                
                # Reset state
                self.video_predictor.reset_state(inference_state)
            
            result["num_tracked_frames"] = len(result["tracked_frames"])
            
            # Save video visualization if output_dir specified
            if output_dir:
                result["frame_output_dir"] = temp_dir
            
        except Exception as e:
            import traceback
            result["error"] = str(e)
            result["traceback"] = traceback.format_exc()
        
        return result
    
    def _analyze_scene_for_navigation(
        self, 
        masks: List[Dict], 
        image_width: int, 
        image_height: int
    ) -> Dict[str, Any]:
        """
        Analyze segmentation results for navigation purposes.
        
        Returns analysis including:
        - Obstacle detection (objects in path)
        - Free space estimation
        - Suggested navigation direction
        """
        if not masks:
            return {
                "obstacles": [],
                "free_space_ratio": 1.0,
                "suggested_direction": "forward",
                "analysis": "No objects detected, path appears clear"
            }
        
        # Define regions of interest for navigation
        # Bottom center region is most important (immediate path)
        bottom_center_x1 = image_width // 3
        bottom_center_x2 = 2 * image_width // 3
        bottom_center_y1 = 2 * image_height // 3
        bottom_center_y2 = image_height
        
        obstacles = []
        total_obstacle_area = 0
        
        for mask in masks:
            bbox = mask.get("bbox_xyxy", mask.get("bbox", [0, 0, 0, 0]))
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                if x2 < 100:  # Likely [x, y, w, h] format
                    x1, y1, w, h = mask.get("bbox", [0, 0, 0, 0])
                    x2, y2 = x1 + w, y1 + h
                
                center = mask.get("center", [(x1+x2)//2, (y1+y2)//2])
                
                # Check if object is in the bottom center region (obstacle in path)
                in_path = (
                    x1 < bottom_center_x2 and x2 > bottom_center_x1 and
                    y1 < bottom_center_y2 and y2 > bottom_center_y1
                )
                
                # Determine position relative to center
                center_x = center[0] if isinstance(center, list) else center
                if center_x < image_width // 3:
                    position = "left"
                elif center_x > 2 * image_width // 3:
                    position = "right"
                else:
                    position = "center"
                
                # Determine depth (based on y position - lower y = further)
                center_y = center[1] if isinstance(center, list) else center
                if center_y < image_height // 3:
                    depth = "far"
                elif center_y < 2 * image_height // 3:
                    depth = "medium"
                else:
                    depth = "near"
                
                obstacle_info = {
                    "id": mask.get("id", 0),
                    "bbox": [x1, y1, x2, y2],
                    "center": center,
                    "position": position,
                    "depth": depth,
                    "in_path": in_path,
                    "area": mask.get("area", 0),
                }
                obstacles.append(obstacle_info)
                
                if in_path:
                    total_obstacle_area += mask.get("area", 0)
        
        # Calculate free space ratio in bottom center
        bottom_center_area = (bottom_center_x2 - bottom_center_x1) * (bottom_center_y2 - bottom_center_y1)
        free_space_ratio = max(0, 1 - total_obstacle_area / bottom_center_area)
        
        # Suggest navigation direction
        left_obstacles = sum(1 for o in obstacles if o["position"] == "left" and o["in_path"])
        right_obstacles = sum(1 for o in obstacles if o["position"] == "right" and o["in_path"])
        center_obstacles = sum(1 for o in obstacles if o["position"] == "center" and o["in_path"])
        
        if center_obstacles == 0 and free_space_ratio > 0.5:
            suggested_direction = "forward"
        elif left_obstacles < right_obstacles:
            suggested_direction = "left"
        elif right_obstacles < left_obstacles:
            suggested_direction = "right"
        else:
            suggested_direction = "stop"  # Too many obstacles
        
        # Generate analysis text
        near_obstacles = [o for o in obstacles if o["depth"] == "near"]
        analysis = f"Detected {len(obstacles)} objects. "
        if near_obstacles:
            analysis += f"{len(near_obstacles)} objects are near. "
        if center_obstacles > 0:
            analysis += f"{center_obstacles} objects blocking the center path. "
        analysis += f"Free space ratio: {free_space_ratio:.2f}. "
        analysis += f"Suggested direction: {suggested_direction}."
        
        return {
            "obstacles": obstacles,
            "num_obstacles": len(obstacles),
            "obstacles_in_path": sum(1 for o in obstacles if o["in_path"]),
            "free_space_ratio": round(free_space_ratio, 3),
            "suggested_direction": suggested_direction,
            "analysis": analysis,
        }
    
    def get_metadata(self):
        """Return tool metadata."""
        metadata = super().get_metadata()
        metadata["model_cfg"] = self.model_cfg
        metadata["device"] = self.device
        metadata["sam_version"] = "2.1"
        return metadata


# Backward compatibility alias
SAM_Perception_Tool = SAM2_Perception_Tool


if __name__ == "__main__":
    """
    Test the SAM 2 Perception Tool.
    
    Run:
        cd agentflow/tools/sam_perception
        python tool.py --image /path/to/image.jpg --mode automatic --output-dir ./sam2_output
    """
    import os
    import argparse
    
    parser = argparse.ArgumentParser(description="Run SAM2_Perception_Tool on an input image.")
    parser.add_argument("--image", type=str, help="Path to the input image file")
    parser.add_argument("--mode", type=str, default="automatic", choices=["automatic", "point", "box"], help="Segmentation mode")
    parser.add_argument("--output-dir", type=str, default="./sam2_output", help="Directory to save visualization results")
    parser.add_argument("--model-cfg", type=str, default="sam2.1_hiera_l", help="SAM2 model config name")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt file")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to run on")
    parser.add_argument("--points", type=str, default=None, help="Point prompts as 'x1,y1;x2,y2'")
    parser.add_argument("--box", type=str, default=None, help="Box prompt as 'x1,y1,x2,y2'")
    parser.add_argument("--top-k", type=int, default=5, help="Top K masks to keep in automatic mode")
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Script directory: {script_dir}")
    
    # Initialize tool with selectable model settings
    tool = SAM2_Perception_Tool(
        model_cfg=args.model_cfg,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    
    # Print metadata
    metadata = tool.get_metadata()
    print("Tool Metadata:")
    print(json.dumps(metadata, indent=2, default=str))
    
    # Validate and parse prompts
    parsed_points = None
    if args.points:
        try:
            pts = []
            for p in args.points.split(";"):
                if not p:
                    continue
                x_str, y_str = p.split(",")
                pts.append([int(float(x_str)), int(float(y_str))])
            parsed_points = pts if pts else None
        except Exception as e:
            print(f"Warning: Failed to parse --points '{args.points}': {e}")
            parsed_points = None
    
    parsed_box = None
    if args.box:
        try:
            parts = [int(float(t)) for t in args.box.split(",")]
            if len(parts) == 4:
                parsed_box = parts
            else:
                print("Warning: --box must have 4 comma-separated values 'x1,y1,x2,y2'.")
        except Exception as e:
            print(f"Warning: Failed to parse --box '{args.box}': {e}")
            parsed_box = None
    
    # Run on provided image if valid
    if args.image and os.path.exists(args.image):
        print(f"\nRunning on image: {args.image}")
        result = tool.execute(
            image_path=args.image,
            mode=args.mode,
            points=parsed_points,
            box=parsed_box,
            top_k=args.top_k,
            output_dir=args.output_dir,
        )
        
        print("\nResult Summary:")
        print(f"Number of masks: {result.get('num_masks', 0)}")
        vis_path = result.get("visualization_path")
        if vis_path:
            print(f"Visualization saved: {vis_path}")
        sa = result.get('scene_analysis', {})
        if sa:
            print(f"Scene analysis: {sa.get('analysis', 'N/A')}")
        if "error" in result:
            print(f"Error: {result['error']}")
    else:
        print("\nNo valid --image provided or file does not exist.")
        print("Example: python tool.py --image /path/to/image.jpg --mode automatic --output-dir ./sam2_output")
    
    print("\nDone!")
