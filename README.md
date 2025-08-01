# SAM2 Interactive Segmentation App 🎯

A complete **Streamlit web application** for interactive image segmentation using **Meta's SAM2 (Segment Anything 2)** with **Florence2** integration. This app provides an intuitive interface for segmenting objects in images through direct interaction - click points, draw bounding boxes, or automatically segment everything.

## ✨ Features

### 🎯 **Point Clicking Mode**
- Click **green circles** on areas to include (positive points)
- Click **red circles** on areas to exclude (negative points) 
- Dynamic color switching with visual feedback
- Precise coordinate mapping from canvas to original image

### 📦 **Bounding Box Mode**
- Draw rectangles directly on images
- Support for multiple objects
- Real-time bounding box visualization

### 🔍 **Auto Segment Everything Mode**
- Automatic object detection and segmentation
- Adjustable parameters for fine-tuning
- Uses SAM2's built-in automatic mask generator

### 🖥️ **Multi-Platform Support**
- **Apple Silicon (M1-M3 Macs)**: Optimized MPS acceleration
- **NVIDIA GPUs**: CUDA support
- **CPU**: Fallback for any system

### 📊 **Advanced Features**
- Real-time segmentation statistics
- Colored mask overlays with transparency
- Export segmented images (PNG)
- Export mask data (JSON)
- Interactive canvas with proper scaling
- Session state management

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+** (tested with Python 3.13)
- **Git** for cloning repositories
- **4GB+ VRAM** recommended for GPU acceleration (optional)

### One-Command Setup

```bash
# Clone this repository or download the GIT folder contents
# cd into the directory containing setup.sh

chmod +x setup.sh
./setup.sh
```

The setup script will:
1. ✅ Create a Python virtual environment
2. ✅ Install exact dependency versions 
3. ✅ Clone and install SAM2
4. ✅ Download model checkpoints (~1.5GB)
5. ✅ Configure SAM2.1 compatibility
6. ✅ Test the installation

### Manual Setup (if needed)

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Clone and install SAM2
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e .

# 4. Download model checkpoints
cd checkpoints
./download_ckpts.sh
cd ../..

# 5. Create config symlinks
cd segment-anything-2/sam2
ln -sf configs/sam2.1/sam2.1_hiera_l.yaml sam2.1_hiera_l.yaml
ln -sf configs/sam2.1/sam2.1_hiera_b+.yaml sam2.1_hiera_b+.yaml
ln -sf configs/sam2.1/sam2.1_hiera_s.yaml sam2.1_hiera_s.yaml
ln -sf configs/sam2.1/sam2.1_hiera_t.yaml sam2.1_hiera_t.yaml
cd ../..
```

## 🎮 Usage

### Running the App

```bash
# Activate virtual environment (if not already active)
source venv/bin/activate

# Start the Streamlit app
streamlit run sam2_interactive_app.py
```

The app will open in your browser at `http://localhost:8502`

### How to Use

1. **Upload an Image**: Drag & drop or browse for JPG/PNG files
2. **Choose Mode**: Point Clicking, Bounding Boxes, or Auto Segment
3. **Interact**: 
   - **Point Mode**: Click green/red buttons then click on image
   - **Box Mode**: Draw rectangles around objects
   - **Auto Mode**: Adjust parameters and click "Auto Segment"
4. **View Results**: See colored masks, statistics, and download options

### Testing Installation

```bash
python3 test_installation.py
```

## 📋 System Requirements

### Minimum Requirements
- **OS**: macOS 10.15+, Ubuntu 18.04+, Windows 10+ (with WSL)
- **Python**: 3.8 or higher
- **RAM**: 8GB+ recommended
- **Storage**: 4GB free space (for models and dependencies)

### Recommended for Best Performance
- **Apple Silicon Macs**: M1/M2/M3 with 16GB+ unified memory
- **NVIDIA GPUs**: RTX 3060+ with 8GB+ VRAM
- **CPU**: 8+ cores for CPU-only inference

## 🔧 Troubleshooting

### Common Issues

**1. SAM2 Model Loading Fails**
```bash
# Check if checkpoints downloaded correctly
ls -la segment-anything-2/checkpoints/
# Should show .pt files (~1.5GB total)
```

**2. Canvas Not Showing Image**
- Ensure image is uploaded and in JPG/PNG format
- Try refreshing the browser
- Check browser console for errors

**3. Points/Boxes Not Working**
- Make sure you've selected the correct mode
- Click the green/red buttons to switch point types
- Verify coordinates are being detected (check terminal output)

**4. SSL/Network Issues (Corporate Networks)**
```bash
# Set environment variables
export PYTHONHTTPSVERIFY=0
export SSL_VERIFY=false
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.



**5. Memory Issues**
- Reduce image size before upload
- Lower "Points per side" in Auto Segment mode
- Use CPU mode if GPU memory is insufficient

### Version Compatibility

This setup uses **exact tested versions**:
- **Streamlit 1.28.0** (required for drawable-canvas compatibility)
- **PyTorch 2.0+** with automatic MPS/CUDA detection
- **Python 3.8-3.13** (tested with 3.13)

## 🏗️ Architecture

### Project Structure
```
GIT/
├── sam2_interactive_app.py      # Main Streamlit application
├── utils/                       # Utility modules
│   ├── __init__.py
│   ├── device.py               # Device detection and optimization
│   ├── sam.py                  # SAM2 model loading and inference
│   ├── sam_interactive.py      # Interactive segmentation functions
│   └── florence.py             # Florence2 integration (optional)
├── requirements.txt            # Exact dependency versions
├── setup.sh                   # Automated setup script
├── README.md                  # This file
└── test_installation.py       # Installation verification
```

### Key Components

**Device Detection (`utils/device.py`)**
- Automatic detection of Apple Silicon MPS, CUDA, or CPU
- Platform-specific optimizations
- Memory management for different hardware

**SAM2 Integration (`utils/sam.py`)**
- Model loading with fallback sizes (large → base_plus → small → tiny)
- Device-specific model selection
- Proper config and checkpoint handling

**Interactive Functions (`utils/sam_interactive.py`)**
- Point-based segmentation with positive/negative labels
- Bounding box segmentation for multiple objects
- Automatic mask generation with parameter tuning
- Coordinate transformation between canvas and image

**Streamlit App (`sam2_interactive_app.py`)**
- Modern UI with custom CSS styling
- Real-time canvas interaction
- Session state management
- Export functionality

## 🤝 Contributing

### Development Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd sam2-interactive-app

# Follow the setup instructions above
./setup.sh

# Make changes and test
python3 test_installation.py
streamlit run sam2_interactive_app.py
```

### Adding Features
- New segmentation modes: Add to `utils/sam_interactive.py`
- UI improvements: Modify `sam2_interactive_app.py`
- Model integrations: Create new utility modules

## 📄 License

This project integrates multiple components:
- **SAM2**: Apache 2.0 License (Meta)
- **Florence2**: MIT License (Microsoft)
- **Custom Code**: MIT License

## 🙏 Acknowledgments

- **Meta AI** for SAM2 (Segment Anything 2)
- **Microsoft** for Florence2
- **Streamlit** for the amazing web framework
- **Roboflow** for supervision library

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Run `python3 test_installation.py` to verify setup
3. Check Streamlit logs in the terminal
4. Create an issue with error details and system info

---

**🎯 Ready to segment anything? Run `./setup.sh` and start exploring!**
