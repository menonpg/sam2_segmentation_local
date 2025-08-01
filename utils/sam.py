import torch
import numpy as np
from PIL import Image
from typing import Any, List
import supervision as sv

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    print("SAM2 not found. Please install sam2 package.")

def load_sam_image_model(device: torch.device) -> Any:
    """
    Load SAM2 image model.
    
    Args:
        device: The device to load the model on
        
    Returns:
        SAM2 image predictor
    """
    if not SAM2_AVAILABLE:
        raise ImportError(
            "SAM2 is not installed. Please install it from: "
            "https://github.com/facebookresearch/segment-anything-2"
        )
    
    # SAM2 model configurations - using SAM2.1 configs to match SAM2.1 models
    model_configs = {
        "large": {
            "config": "sam2.1_hiera_l.yaml",
            "checkpoint": "segment-anything-2/checkpoints/sam2.1_hiera_large.pt"
        },
        "base_plus": {
            "config": "sam2.1_hiera_b+.yaml", 
            "checkpoint": "segment-anything-2/checkpoints/sam2.1_hiera_base_plus.pt"
        },
        "small": {
            "config": "sam2.1_hiera_s.yaml",
            "checkpoint": "segment-anything-2/checkpoints/sam2.1_hiera_small.pt"
        },
        "tiny": {
            "config": "sam2.1_hiera_t.yaml",
            "checkpoint": "segment-anything-2/checkpoints/sam2.1_hiera_tiny.pt"
        }
    }
    
    # Choose model size based on device capabilities
    if device.type == "cuda":
        # Use large model for CUDA
        model_size = "large"
    elif device.type == "mps":
        # Use base_plus for Apple Silicon
        model_size = "base_plus"
    else:
        # Use small model for CPU
        model_size = "small"
    
    try:
        config = model_configs[model_size]
        
        # Build SAM2 model
        sam2_model = build_sam2(
            config["config"],
            config["checkpoint"],
            device=device
        )
        
        # Create image predictor
        predictor = SAM2ImagePredictor(sam2_model)
        
        print(f"SAM2 {model_size} model loaded on {device}")
        return predictor
        
    except Exception as e:
        print(f"Failed to load SAM2 {model_size} model, trying smaller model...")
        
        # Fallback to smaller models
        fallback_order = ["base_plus", "small", "tiny"]
        if model_size in fallback_order:
            fallback_order = fallback_order[fallback_order.index(model_size) + 1:]
        
        for fallback_size in fallback_order:
            try:
                config = model_configs[fallback_size]
                sam2_model = build_sam2(
                    config["config"],
                    config["checkpoint"],
                    device=device
                )
                predictor = SAM2ImagePredictor(sam2_model)
                print(f"SAM2 {fallback_size} model loaded on {device} (fallback)")
                return predictor
            except Exception as fallback_e:
                print(f"Failed to load SAM2 {fallback_size}: {fallback_e}")
                continue
        
        raise RuntimeError(f"Failed to load any SAM2 model: {e}")

def load_sam_video_model(device: torch.device) -> Any:
    """
    Load SAM2 video model.
    
    Args:
        device: The device to load the model on
        
    Returns:
        SAM2 video predictor
    """
    if not SAM2_AVAILABLE:
        raise ImportError(
            "SAM2 is not installed. Please install it from: "
            "https://github.com/facebookresearch/segment-anything-2"
        )
    
    try:
        from sam2.build_sam import build_sam2_video_predictor
        
        # Choose model configuration based on device
        if device.type == "cuda":
            config = "sam2_hiera_l.yaml"
            checkpoint = "sam2_hiera_large.pt"
        elif device.type == "mps":
            config = "sam2_hiera_b+.yaml"
            checkpoint = "sam2_hiera_base_plus.pt"
        else:
            config = "sam2_hiera_s.yaml"
            checkpoint = "sam2_hiera_small.pt"
        
        predictor = build_sam2_video_predictor(
            config,
            checkpoint,
            device=device
        )
        
        print(f"SAM2 video model loaded on {device}")
        return predictor
        
    except Exception as e:
        raise RuntimeError(f"Failed to load SAM2 video model: {e}")

@torch.inference_mode()
def run_sam_inference(
    predictor: Any,
    image: Image.Image,
    detections: sv.Detections
) -> sv.Detections:
    """
    Run SAM2 inference to generate masks from bounding boxes.
    
    Args:
        predictor: SAM2 image predictor
        image: Input PIL image
        detections: Supervision detections with bounding boxes
        
    Returns:
        Updated detections with masks
    """
    if len(detections) == 0:
        return detections
    
    # Convert PIL image to numpy array
    image_array = np.array(image)
    
    # Set image for predictor
    predictor.set_image(image_array)
    
    # Prepare input boxes (convert from xyxy to xywh format if needed)
    input_boxes = detections.xyxy
    
    # Convert boxes to the format expected by SAM2
    if len(input_boxes) > 0:
        # SAM2 expects boxes in xyxy format, which is what we have
        input_boxes_tensor = torch.tensor(input_boxes, device=predictor.device)
        
        # Run prediction
        masks, scores, logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes_tensor,
            multimask_output=False,
        )
        
        # Process masks
        if masks is not None and len(masks) > 0:
            # masks shape: (num_boxes, num_masks_per_box, H, W)
            # We want the best mask for each box
            if len(masks.shape) == 4:
                # Take the first (and typically best) mask for each box
                final_masks = masks[:, 0, :, :]  # Shape: (num_boxes, H, W)
            else:
                final_masks = masks
            
            # Convert to boolean masks
            boolean_masks = final_masks > 0.5
            
            # Update detections with masks
            detections.mask = boolean_masks.cpu().numpy()
        
        # Update confidence scores if available
        if scores is not None and len(scores) > 0:
            if hasattr(detections, 'confidence') and detections.confidence is not None:
                # Combine existing confidence with SAM scores
                sam_scores = scores[:, 0] if len(scores.shape) > 1 else scores
                detections.confidence = detections.confidence * sam_scores.cpu().numpy()
            else:
                # Use SAM scores as confidence
                sam_scores = scores[:, 0] if len(scores.shape) > 1 else scores
                detections.confidence = sam_scores.cpu().numpy()
    
    return detections

def postprocess_masks(masks: np.ndarray, original_size: tuple) -> np.ndarray:
    """
    Post-process SAM2 masks.
    
    Args:
        masks: Raw masks from SAM2
        original_size: Original image size (width, height)
        
    Returns:
        Processed masks
    """
    if masks is None or len(masks) == 0:
        return masks
    
    # Ensure masks are boolean
    if masks.dtype != bool:
        masks = masks > 0.5
    
    return masks

def filter_masks_by_area(
    detections: sv.Detections,
    min_area: int = 100,
    max_area: int = None
) -> sv.Detections:
    """
    Filter detections by mask area.
    
    Args:
        detections: Input detections
        min_area: Minimum mask area in pixels
        max_area: Maximum mask area in pixels (None for no limit)
        
    Returns:
        Filtered detections
    """
    if detections.mask is None or len(detections.mask) == 0:
        return detections
    
    # Calculate mask areas
    mask_areas = [np.sum(mask) for mask in detections.mask]
    
    # Create filter
    valid_indices = []
    for i, area in enumerate(mask_areas):
        if area >= min_area:
            if max_area is None or area <= max_area:
                valid_indices.append(i)
    
    if len(valid_indices) == 0:
        # Return empty detections
        return sv.Detections.empty()
    
    # Filter detections
    filtered_detections = detections[valid_indices]
    
    return filtered_detections

def merge_overlapping_masks(
    detections: sv.Detections,
    iou_threshold: float = 0.5
) -> sv.Detections:
    """
    Merge overlapping masks based on IoU threshold.
    
    Args:
        detections: Input detections
        iou_threshold: IoU threshold for merging
        
    Returns:
        Merged detections
    """
    if detections.mask is None or len(detections.mask) <= 1:
        return detections
    
    # This is a simplified version - full implementation would
    # calculate IoU between all mask pairs and merge overlapping ones
    return detections

def get_mask_statistics(detections: sv.Detections) -> dict:
    """
    Get statistics about the generated masks.
    
    Args:
        detections: Input detections with masks
        
    Returns:
        Dictionary with mask statistics
    """
    if detections.mask is None or len(detections.mask) == 0:
        return {
            "num_masks": 0,
            "total_area": 0,
            "avg_area": 0,
            "min_area": 0,
            "max_area": 0
        }
    
    areas = [np.sum(mask) for mask in detections.mask]
    
    return {
        "num_masks": len(detections.mask),
        "total_area": sum(areas),
        "avg_area": np.mean(areas),
        "min_area": min(areas),
        "max_area": max(areas)
    }
