#!/usr/bin/env python3
"""
SAM2 Interactive Segmentation App
A Streamlit application with intuitive point-clicking and box-drawing on images.
"""

import streamlit as st
import numpy as np
import torch
from PIL import Image
import supervision as sv
import json
import io
from streamlit_drawable_canvas import st_canvas

from utils.device import get_device, setup_torch_optimizations
from utils.sam import load_sam_image_model
from utils.sam_interactive import (
    run_sam_point_inference, 
    run_sam_box_inference, 
    run_sam_everything_inference,
    get_mask_statistics
)

# Page configuration
st.set_page_config(
    page_title="SAM2 Interactive Segmentation",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .mode-description {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .stats-box {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .instructions-box {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        border-left: 4px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'sam_model_loaded' not in st.session_state:
    st.session_state.sam_model_loaded = False
    st.session_state.sam_model = None
    st.session_state.device = None
    st.session_state.current_image = None
    st.session_state.current_masks = None
    st.session_state.point_mode = "positive"  # Track current point mode

# Colors and annotators for visualization
COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700', '#32CD32', '#8A2BE2']
COLOR_PALETTE = sv.ColorPalette.from_hex(COLORS)
MASK_ANNOTATOR = sv.MaskAnnotator(
    color=COLOR_PALETTE,
    color_lookup=sv.ColorLookup.INDEX,
    opacity=0.6
)
BOX_ANNOTATOR = sv.BoxAnnotator(
    color=COLOR_PALETTE, 
    color_lookup=sv.ColorLookup.INDEX,
    thickness=2
)

@st.cache_resource
def load_sam_model():
    """Load SAM2 model with caching."""
    device = get_device()
    setup_torch_optimizations(device)
    
    with st.spinner("Loading SAM2 model..."):
        sam_model = load_sam_image_model(device=device)
    
    return sam_model, device

def annotate_image_with_masks(image: np.ndarray, detections: sv.Detections) -> np.ndarray:
    """Annotate image with masks and boxes."""
    if detections is None or len(detections) == 0:
        return image
    
    annotated = image.copy()
    
    # Fix mask data types - ensure masks are boolean
    if detections.mask is not None:
        # Convert masks to boolean if they aren't already
        if detections.mask.dtype != bool:
            detections.mask = detections.mask.astype(bool)
        annotated = MASK_ANNOTATOR.annotate(annotated, detections)
    
    if detections.xyxy is not None:
        annotated = BOX_ANNOTATOR.annotate(annotated, detections)
    
    return annotated

def extract_points_from_canvas(canvas_result, original_image_size, canvas_size, current_mode="positive"):
    """Extract point coordinates from canvas drawing data and scale to original image."""
    if canvas_result.json_data is None:
        return [], []
    
    points = []
    labels = []
    
    # Calculate scaling factors
    scale_x = original_image_size[0] / canvas_size[0]  # width scaling
    scale_y = original_image_size[1] / canvas_size[1]  # height scaling
    
    objects = canvas_result.json_data.get("objects", [])
    for obj in objects:
        if obj.get("type") == "circle":
            # Extract center coordinates
            canvas_x = obj.get("left", 0) + obj.get("radius", 0)
            canvas_y = obj.get("top", 0) + obj.get("radius", 0)
            
            # Scale to original image coordinates
            orig_x = int(canvas_x * scale_x)
            orig_y = int(canvas_y * scale_y)
            points.append((orig_x, orig_y))
            
            # Use current mode to determine label for ALL circles  
            # When user is in "positive" mode, all circles are positive
            # When user is in "negative" mode, all circles are negative
            if current_mode == "positive":
                labels.append(0)  # Positive (SAM2 uses 0 = include)
            else:  # negative mode
                labels.append(1)  # Negative (SAM2 uses 1 = exclude)
    
    return points, labels

def extract_boxes_from_canvas(canvas_result, original_image_size, canvas_size):
    """Extract bounding box coordinates from canvas drawing data and scale to original image."""
    if canvas_result.json_data is None:
        return []
    
    boxes = []
    
    # Calculate scaling factors
    scale_x = original_image_size[0] / canvas_size[0]  # width scaling
    scale_y = original_image_size[1] / canvas_size[1]  # height scaling
    
    objects = canvas_result.json_data.get("objects", [])
    
    for obj in objects:
        if obj.get("type") == "rect":
            canvas_x1 = obj.get("left", 0)
            canvas_y1 = obj.get("top", 0)
            canvas_width = obj.get("width", 0)
            canvas_height = obj.get("height", 0)
            
            canvas_x2 = canvas_x1 + canvas_width
            canvas_y2 = canvas_y1 + canvas_height
            
            # Scale to original image coordinates
            orig_x1 = int(canvas_x1 * scale_x)
            orig_y1 = int(canvas_y1 * scale_y)
            orig_x2 = int(canvas_x2 * scale_x)
            orig_y2 = int(canvas_y2 * scale_y)
            
            boxes.append((orig_x1, orig_y1, orig_x2, orig_y2))
    
    return boxes

def main():
    # Header
    st.markdown('<h1 class="main-header">SAM2 Interactive Segmentation 🎯</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    **Interactive image segmentation using SAM2 (Segment Anything 2)**
    
    This app lets you segment objects by **directly clicking and drawing on images**:
    - **Point Mode**: Click points to indicate what to segment
    - **Box Mode**: Draw rectangles around objects  
    - **Auto Mode**: Automatically find all objects
    """)
    
    # Load SAM2 model
    if not st.session_state.sam_model_loaded:
        try:
            sam_model, device = load_sam_model()
            st.session_state.sam_model = sam_model
            st.session_state.device = device
            st.session_state.sam_model_loaded = True
            
            # Device info
            device_info = f"🖥️ **Device:** {device}"
            if device.type == "cuda":
                device_info += f" (GPU: {torch.cuda.get_device_name()})"
            elif device.type == "mps":
                device_info += " (Apple Silicon)"
            st.success(f"✅ SAM2 loaded successfully! {device_info}")
            
        except Exception as e:
            st.error(f"❌ Error loading SAM2: {str(e)}")
            st.stop()
    
    # Sidebar
    st.sidebar.title("🛠️ Configuration")
    
    # Mode selection
    mode = st.sidebar.selectbox(
        "Segmentation Mode",
        ["Point Clicking", "Bounding Boxes", "Auto Segment Everything"],
        help="Choose how you want to segment the image"
    )
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Input")
        
        # Image upload
        uploaded_file = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG"
        )
        
        if uploaded_file is not None:
            # Load and display image
            image = Image.open(uploaded_file).convert("RGB")
            st.session_state.current_image = image
            
            # Show image info
            st.write(f"**Image Size:** {image.size[0]} × {image.size[1]} pixels")
            
            # Mode-specific interactive canvas
            if mode == "Point Clicking":
                st.markdown('<div class="instructions-box">', unsafe_allow_html=True)
                st.markdown("**🎯 Point Clicking Mode**")
                st.markdown("""
                **Instructions:**
                1. **Green circles** = Include this area (positive points)
                2. **Red circles** = Exclude this area (negative points)
                3. **Click and drag** to place circles on the image
                4. Use the color picker to switch between green (include) and red (exclude)
                """)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Initialize point mode if not set
                if 'point_mode' not in st.session_state:
                    st.session_state.point_mode = "positive"
                
                # Color picker for positive/negative points
                col_pos, col_neg = st.columns(2)
                with col_pos:
                    if st.button("🟢 Positive Points (Include)", type="primary" if st.session_state.point_mode == "positive" else "secondary"):
                        st.session_state.point_mode = "positive"
                        st.rerun()
                with col_neg:
                    if st.button("🔴 Negative Points (Exclude)", type="primary" if st.session_state.point_mode == "negative" else "secondary"):
                        st.session_state.point_mode = "negative"
                        st.rerun()
                
                # Set canvas colors based on current mode
                if st.session_state.point_mode == "positive":
                    fill_color = "rgba(0, 255, 0, 0.3)"  # Green with transparency
                    stroke_color = "#00ff00"  # Green
                    mode_info = "🟢 **Current Mode:** Positive Points (Include areas)"
                else:
                    fill_color = "rgba(255, 0, 0, 0.3)"  # Red with transparency
                    stroke_color = "#ff0000"  # Red
                    mode_info = "🔴 **Current Mode:** Negative Points (Exclude areas)"
                
                st.markdown(mode_info)
                
                # Canvas for point clicking - scale to fit properly
                canvas_width = min(600, image.size[0])
                canvas_height = int(canvas_width * image.size[1] / image.size[0])
                if canvas_height > 400:
                    canvas_height = 400
                    canvas_width = int(canvas_height * image.size[0] / image.size[1])
                
                canvas_result = st_canvas(
                    fill_color=fill_color,
                    stroke_width=2,
                    stroke_color=stroke_color,
                    background_image=image,
                    update_streamlit=True,
                    height=canvas_height,
                    width=canvas_width,
                    drawing_mode="circle",
                    point_display_radius=10,
                    key="point_canvas",
                )
                
                # Process points and segment
                if st.button("🎯 Segment from Points", type="primary"):
                    if canvas_result.json_data:
                        points, labels = extract_points_from_canvas(
                            canvas_result, 
                            image.size, 
                            (canvas_width, canvas_height),
                            st.session_state.point_mode
                        )
                        if points:
                            with st.spinner("Generating segmentation masks..."):
                                detections = run_sam_point_inference(
                                    st.session_state.sam_model,
                                    image,
                                    points,
                                    labels
                                )
                                st.session_state.current_masks = detections
                        else:
                            st.warning("Please click some points on the image!")
                    else:
                        st.warning("Please click some points on the image!")
            
            elif mode == "Bounding Boxes":
                st.markdown('<div class="instructions-box">', unsafe_allow_html=True)
                st.markdown("**📦 Bounding Box Mode**")
                st.markdown("""
                **Instructions:**
                1. **Draw rectangles** around objects you want to segment
                2. **Click and drag** to create bounding boxes
                3. You can draw multiple boxes for different objects
                """)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Canvas for box drawing - scale to fit properly
                canvas_width = min(600, image.size[0])
                canvas_height = int(canvas_width * image.size[1] / image.size[0])
                if canvas_height > 400:
                    canvas_height = 400
                    canvas_width = int(canvas_height * image.size[0] / image.size[1])
                
                canvas_result = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.2)",  # Orange fill
                    stroke_width=3,
                    stroke_color="#ff6347",
                    background_image=image,
                    update_streamlit=True,
                    height=canvas_height,
                    width=canvas_width,
                    drawing_mode="rect",
                    key="box_canvas",
                )
                
                # Process boxes and segment
                if st.button("🎯 Segment from Boxes", type="primary"):
                    if canvas_result.json_data:
                        boxes = extract_boxes_from_canvas(
                            canvas_result, 
                            image.size, 
                            (canvas_width, canvas_height)
                        )
                        if boxes:
                            with st.spinner("Generating segmentation masks..."):
                                detections = run_sam_box_inference(
                                    st.session_state.sam_model,
                                    image,
                                    boxes
                                )
                                st.session_state.current_masks = detections
                        else:
                            st.warning("Please draw some bounding boxes on the image!")
                    else:
                        st.warning("Please draw some bounding boxes on the image!")
            
            else:  # Auto Segment Everything
                st.markdown('<div class="instructions-box">', unsafe_allow_html=True)
                st.markdown("**🔍 Auto Segment Everything Mode**")
                st.markdown("Automatically detect and segment all objects in the image.")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Just show the image (no interaction needed)
                st.image(image, caption="Input Image", use_column_width=True)
                
                # Parameters for auto segmentation
                with st.expander("⚙️ Advanced Parameters"):
                    points_per_side = st.slider("Points per side", 8, 64, 16, help="Higher values = more detailed but slower")
                    pred_iou_thresh = st.slider("Prediction IoU threshold", 0.5, 0.99, 0.88, help="Higher values = stricter filtering")
                    stability_score_thresh = st.slider("Stability score threshold", 0.5, 0.99, 0.95, help="Higher values = more stable masks")
                    min_mask_region_area = st.number_input("Minimum mask area (pixels)", 0, 10000, 100, help="Filter out very small regions")
                
                # Segment button
                if st.button("🔍 Auto Segment Everything", type="primary"):
                    with st.spinner("Automatically segmenting all objects... This may take a moment."):
                        detections = run_sam_everything_inference(
                            st.session_state.sam_model,
                            image,
                            points_per_side=points_per_side,
                            pred_iou_thresh=pred_iou_thresh,
                            stability_score_thresh=stability_score_thresh,
                            min_mask_region_area=min_mask_region_area
                        )
                        st.session_state.current_masks = detections
    
    with col2:
        st.subheader("📥 Results")
        
        if st.session_state.current_image is not None and st.session_state.current_masks is not None:
            detections = st.session_state.current_masks
            
            if len(detections) > 0:
                # Annotate image with masks
                image_array = np.array(st.session_state.current_image)
                annotated_image = annotate_image_with_masks(image_array, detections)
                
                st.image(annotated_image, caption="Segmented Image", use_column_width=True)
                
                # Show statistics
                stats = get_mask_statistics(detections)
                st.markdown('<div class="stats-box">', unsafe_allow_html=True)
                st.markdown("**📊 Segmentation Statistics:**")
                st.write(f"• **Number of masks:** {stats['num_masks']}")
                st.write(f"• **Total area:** {stats['total_area']:,} pixels")
                st.write(f"• **Coverage:** {stats['coverage_percent']:.1f}% of image")
                if stats['num_masks'] > 0:
                    st.write(f"• **Average mask size:** {stats['avg_area']:.0f} pixels")
                    st.write(f"• **Size range:** {stats['min_area']:.0f} - {stats['max_area']:.0f} pixels")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Download options
                st.subheader("💾 Download Results")
                
                col_dl1, col_dl2 = st.columns(2)
                
                with col_dl1:
                    # Download annotated image
                    annotated_pil = Image.fromarray(annotated_image)
                    img_buffer = io.BytesIO()
                    annotated_pil.save(img_buffer, format='PNG')
                    st.download_button(
                        label="📸 Download Segmented Image",
                        data=img_buffer.getvalue(),
                        file_name="segmented_image.png",
                        mime="image/png"
                    )
                
                with col_dl2:
                    # Download masks data
                    masks_data = {
                        "image_size": st.session_state.current_image.size,
                        "num_masks": len(detections),
                        "masks": [],
                        "boxes": detections.xyxy.tolist() if detections.xyxy is not None else [],
                        "scores": detections.confidence.tolist() if detections.confidence is not None else []
                    }
                    
                    # Convert masks to lists (simplified)
                    if detections.mask is not None:
                        for mask in detections.mask:
                            masks_data["masks"].append(mask.astype(bool).tolist())
                    
                    json_data = json.dumps(masks_data, indent=2)
                    st.download_button(
                        label="📊 Download Masks JSON",
                        data=json_data,
                        file_name="masks_data.json",
                        mime="application/json"
                    )
                
            else:
                st.info("No objects detected. Try adjusting the parameters or using different input points/boxes.")
        
        else:
            st.info("👆 Upload an image and interact with it to see results here")
    
    # Instructions
    st.markdown("---")
    st.subheader("📋 How to Use")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🎯 Point Clicking Mode**
        1. Upload an image
        2. Click **green circles** on areas to include
        3. Click **red circles** on areas to exclude
        4. Click "Segment from Points"
        """)
    
    with col2:
        st.markdown("""
        **📦 Bounding Box Mode**
        1. Upload an image
        2. **Click and drag** to draw rectangles
        3. Draw multiple boxes for different objects
        4. Click "Segment from Boxes"
        """)
    
    with col3:
        st.markdown("""
        **🔍 Auto Segment Mode**
        1. Upload an image
        2. Optionally adjust parameters
        3. Click "Auto Segment Everything"
        4. Wait for automatic processing
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("**SAM2 Interactive Segmentation** - Powered by Meta's Segment Anything 2")

if __name__ == "__main__":
    main()
