import torch
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Any
import supervision as sv

def run_sam_point_inference(
    predictor: Any,
    image: Image.Image,
    points: List[Tuple[int, int]],
    point_labels: List[int]
) -> sv.Detections:
    """
    Run SAM2 inference with point prompts.
    
    Args:
        predictor: SAM2 image predictor
        image: Input PIL image
        points: List of (x, y) coordinates
        point_labels: List of labels (1 for positive, 0 for negative)
        
    Returns:
        Supervision detections with masks
    """
    if len(points) == 0:
        return sv.Detections.empty()
    
    # Convert PIL image to numpy array
    image_array = np.array(image)
    
    # Set image for predictor
    predictor.set_image(image_array)
    
    # Convert points to numpy array
    input_points = np.array(points)
    input_labels = np.array(point_labels)
    
    # Run prediction
    masks, scores, logits = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True,
    )
    
    # Process masks
    if masks is not None and len(masks) > 0:
        # Take the best mask (highest score)
        best_mask_idx = np.argmax(scores)
        best_mask = masks[best_mask_idx]
        best_score = scores[best_mask_idx]
        
        # Create bounding box from mask
        if np.any(best_mask):
            y_indices, x_indices = np.where(best_mask)
            x_min, x_max = x_indices.min(), x_indices.max()
            y_min, y_max = y_indices.min(), y_indices.max()
            bbox = np.array([[x_min, y_min, x_max, y_max]])
            
            # Create detections
            detections = sv.Detections(
                xyxy=bbox,
                mask=best_mask[np.newaxis, :, :],
                confidence=np.array([best_score])
            )
            
            return detections
    
    return sv.Detections.empty()

def run_sam_box_inference(
    predictor: Any,
    image: Image.Image,
    boxes: List[Tuple[int, int, int, int]]
) -> sv.Detections:
    """
    Run SAM2 inference with bounding box prompts.
    
    Args:
        predictor: SAM2 image predictor
        image: Input PIL image
        boxes: List of (x_min, y_min, x_max, y_max) coordinates
        
    Returns:
        Supervision detections with masks
    """
    if len(boxes) == 0:
        return sv.Detections.empty()
    
    # Convert PIL image to numpy array
    image_array = np.array(image)
    
    # Set image for predictor
    predictor.set_image(image_array)
    
    all_masks = []
    all_scores = []
    all_boxes = []
    
    for box in boxes:
        # Convert box to numpy array
        input_box = np.array([[box[0], box[1], box[2], box[3]]])
        
        # Run prediction
        masks, scores, logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box,
            multimask_output=False,
        )
        
        if masks is not None and len(masks) > 0:
            mask = masks[0]  # Take the first (and only) mask
            score = scores[0] if scores is not None else 1.0
            
            all_masks.append(mask)
            all_scores.append(score)
            all_boxes.append(box)
    
    if all_masks:
        # Stack masks
        masks_array = np.stack(all_masks)
        
        # Create detections
        detections = sv.Detections(
            xyxy=np.array(all_boxes),
            mask=masks_array,
            confidence=np.array(all_scores)
        )
        
        return detections
    
    return sv.Detections.empty()

def run_sam_everything_inference(
    predictor: Any,
    image: Image.Image,
    points_per_side: int = 32,
    pred_iou_thresh: float = 0.88,
    stability_score_thresh: float = 0.95,
    crop_n_layers: int = 0,
    crop_n_points_downscale_factor: int = 1,
    min_mask_region_area: int = 0
) -> sv.Detections:
    """
    Run SAM2 automatic mask generation (segment everything).
    
    Args:
        predictor: SAM2 image predictor
        image: Input PIL image
        points_per_side: Number of points per side for grid
        pred_iou_thresh: IoU threshold for mask prediction
        stability_score_thresh: Stability score threshold
        crop_n_layers: Number of crop layers
        crop_n_points_downscale_factor: Downscale factor for crop points
        min_mask_region_area: Minimum area for mask regions
        
    Returns:
        Supervision detections with masks
    """
    try:
        # Try to use SAM2's automatic mask generator
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        
        # Create automatic mask generator from the predictor's model
        mask_generator = SAM2AutomaticMaskGenerator(
            model=predictor.model,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            crop_n_layers=crop_n_layers,
            crop_n_points_downscale_factor=crop_n_points_downscale_factor,
            min_mask_region_area=min_mask_region_area,
        )
        
        # Convert PIL image to numpy array
        image_array = np.array(image)
        
        # Generate masks
        masks_data = mask_generator.generate(image_array)
        
        if masks_data and len(masks_data) > 0:
            # Extract masks, boxes, and scores
            masks = []
            boxes = []
            scores = []
            
            for mask_data in masks_data:
                mask = mask_data['segmentation']
                bbox = mask_data['bbox']  # [x, y, w, h] format
                score = mask_data.get('predicted_iou', 0.9)
                
                # Convert bbox from [x, y, w, h] to [x_min, y_min, x_max, y_max]
                x, y, w, h = bbox
                box = [x, y, x + w, y + h]
                
                masks.append(mask)
                boxes.append(box)
                scores.append(score)
            
            if masks:
                # Stack masks
                masks_array = np.stack(masks)
                
                # Create detections
                detections = sv.Detections(
                    xyxy=np.array(boxes),
                    mask=masks_array,
                    confidence=np.array(scores)
                )
                
                return detections
        
    except ImportError:
        # Fallback to grid-based approach if automatic mask generator is not available
        print("SAM2AutomaticMaskGenerator not available, using fallback grid approach")
        return _run_sam_everything_fallback(predictor, image, points_per_side, pred_iou_thresh, stability_score_thresh, min_mask_region_area)
    
    except Exception as e:
        print(f"Error in automatic mask generation: {e}")
        # Try fallback approach
        return _run_sam_everything_fallback(predictor, image, points_per_side, pred_iou_thresh, stability_score_thresh, min_mask_region_area)

def _run_sam_everything_fallback(
    predictor: Any,
    image: Image.Image,
    points_per_side: int,
    pred_iou_thresh: float,
    stability_score_thresh: float,
    min_mask_region_area: int
) -> sv.Detections:
    """
    Fallback implementation for segment everything using grid sampling.
    """
    # Convert PIL image to numpy array
    image_array = np.array(image)
    
    # Set image for predictor
    predictor.set_image(image_array)
    
    # Generate points grid
    h, w = image_array.shape[:2]
    
    # Create grid of points - reduce density for better performance
    grid_size = min(points_per_side, 16)  # Limit grid size
    points = []
    for y in np.linspace(h//4, 3*h//4, grid_size, dtype=int):
        for x in np.linspace(w//4, 3*w//4, grid_size, dtype=int):
            points.append([x, y])
    
    if len(points) == 0:
        return sv.Detections.empty()
    
    all_masks = []
    all_scores = []
    all_boxes = []
    
    # Process points in smaller batches to avoid memory issues
    batch_size = 4
    for i in range(0, len(points), batch_size):
        batch_points = points[i:i+batch_size]
        
        try:
            for point in batch_points:
                # Run prediction on single point
                input_point = np.array([point])
                input_label = np.array([1])  # Positive point
                
                masks, scores, logits = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=True,
                )
                
                if masks is not None and len(masks) > 0:
                    # Take the best mask
                    best_idx = np.argmax(scores)
                    mask = masks[best_idx]
                    score = scores[best_idx]
                    
                    if score > pred_iou_thresh and np.sum(mask) >= min_mask_region_area:
                        # Calculate bounding box
                        if np.any(mask):
                            y_indices, x_indices = np.where(mask)
                            x_min, x_max = x_indices.min(), x_indices.max()
                            y_min, y_max = y_indices.min(), y_indices.max()
                            
                            # Check if this mask is too similar to existing masks
                            is_duplicate = False
                            for existing_mask in all_masks:
                                overlap = np.sum(mask & existing_mask) / np.sum(mask | existing_mask)
                                if overlap > 0.5:  # 50% overlap threshold
                                    is_duplicate = True
                                    break
                            
                            if not is_duplicate:
                                all_masks.append(mask)
                                all_scores.append(score)
                                all_boxes.append([x_min, y_min, x_max, y_max])
        
        except Exception as e:
            print(f"Error processing batch: {e}")
            continue
    
    if all_masks:
        # Stack masks
        masks_array = np.stack(all_masks)
        
        # Create detections
        detections = sv.Detections(
            xyxy=np.array(all_boxes),
            mask=masks_array,
            confidence=np.array(all_scores)
        )
        
        return detections
    
    return sv.Detections.empty()

def create_point_annotations(points: List[Tuple[int, int]], labels: List[int]) -> np.ndarray:
    """
    Create point annotations for visualization.
    
    Args:
        points: List of (x, y) coordinates
        labels: List of labels (1 for positive, 0 for negative)
        
    Returns:
        Annotated image array
    """
    # This is a placeholder - in the actual app, we'll overlay points on the image
    return np.array([])

def mask_to_rle(mask: np.ndarray) -> dict:
    """
    Convert mask to RLE format for efficient storage.
    
    Args:
        mask: Binary mask array
        
    Returns:
        RLE dictionary
    """
    # Flatten mask
    flat_mask = mask.flatten()
    
    # Find runs
    runs = []
    current_val = flat_mask[0]
    run_length = 1
    
    for i in range(1, len(flat_mask)):
        if flat_mask[i] == current_val:
            run_length += 1
        else:
            runs.extend([run_length, current_val])
            current_val = flat_mask[i]
            run_length = 1
    
    # Add final run
    runs.extend([run_length, current_val])
    
    return {
        'size': mask.shape,
        'counts': runs
    }

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
            "max_area": 0,
            "coverage_percent": 0
        }
    
    # Calculate areas
    areas = [np.sum(mask) for mask in detections.mask]
    total_pixels = detections.mask[0].size if len(detections.mask) > 0 else 1
    
    return {
        "num_masks": len(detections.mask),
        "total_area": sum(areas),
        "avg_area": np.mean(areas),
        "min_area": min(areas),
        "max_area": max(areas),
        "coverage_percent": (sum(areas) / total_pixels) * 100
    }
