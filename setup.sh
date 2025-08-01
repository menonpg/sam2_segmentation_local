#!/bin/bash

# SAM2 Interactive Segmentation App - Complete Setup Script
# This script sets up everything needed to run the SAM2 app from scratch
# Tested on macOS with Apple Silicon, but should work on Linux and Windows with WSL

set -e  # Exit on any error

echo "🚀 SAM2 Interactive Segmentation App Setup"
echo "==========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is available
print_status "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
print_success "Found Python $PYTHON_VERSION"

# Check if we're in the right directory (should contain sam2_interactive_app.py)
if [ ! -f "sam2_interactive_app.py" ]; then
    print_error "sam2_interactive_app.py not found. Please run this script from the GIT directory."
    exit 1
fi

# Create virtual environment
print_status "Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install exact versions from requirements.txt
print_status "Installing Python dependencies..."
pip install -r requirements.txt

# Clone SAM2 repository
print_status "Setting up SAM2..."
if [ ! -d "segment-anything-2" ]; then
    print_status "Cloning SAM2 repository..."
    git clone https://github.com/facebookresearch/segment-anything-2.git
    print_success "SAM2 repository cloned"
else
    print_warning "SAM2 repository already exists"
fi

# Install SAM2
print_status "Installing SAM2..."
cd segment-anything-2
pip install -e .
cd ..

# Download SAM2 model checkpoints
print_status "Downloading SAM2 model checkpoints..."
cd segment-anything-2/checkpoints
if [ ! -f "sam2.1_hiera_base_plus.pt" ]; then
    ./download_ckpts.sh
    print_success "Model checkpoints downloaded"
else
    print_warning "Model checkpoints already exist"
fi
cd ../..

# Create necessary symlinks for SAM2.1 configs
print_status "Setting up SAM2.1 configuration symlinks..."
cd segment-anything-2/sam2
ln -sf configs/sam2.1/sam2.1_hiera_l.yaml sam2.1_hiera_l.yaml 2>/dev/null || true
ln -sf configs/sam2.1/sam2.1_hiera_b+.yaml sam2.1_hiera_b+.yaml 2>/dev/null || true
ln -sf configs/sam2.1/sam2.1_hiera_s.yaml sam2.1_hiera_s.yaml 2>/dev/null || true
ln -sf configs/sam2.1/sam2.1_hiera_t.yaml sam2.1_hiera_t.yaml 2>/dev/null || true
cd ../..
print_success "Configuration symlinks created"

# Test the installation
print_status "Testing SAM2 installation..."
python3 -c "
try:
    from sam2.build_sam import build_sam2
    print('✅ SAM2 installed successfully')
except ImportError as e:
    print(f'❌ SAM2 installation failed: {e}')
    exit(1)
"

print_status "Testing device detection..."
python3 -c "
import torch
from utils.device import get_device
device = get_device()
print(f'✅ Device detected: {device}')
if device.type == 'mps':
    print('🍎 Apple Silicon (MPS) support enabled')
elif device.type == 'cuda':
    print('🖥️ CUDA GPU support enabled')
else:
    print('💻 Using CPU')
"

# Create a simple test script
print_status "Creating test script..."
cat > test_installation.py << 'EOF'
#!/usr/bin/env python3
"""
Quick test to verify SAM2 installation is working
"""
import torch
from utils.device import get_device, setup_torch_optimizations
from utils.sam import load_sam_image_model

def test_installation():
    print("🧪 Testing SAM2 installation...")
    
    # Test device detection
    device = get_device()
    print(f"✅ Device: {device}")
    
    # Test torch optimizations
    setup_torch_optimizations(device)
    print("✅ Torch optimizations applied")
    
    # Test SAM2 model loading
    try:
        sam_model = load_sam_image_model(device=device)
        print("✅ SAM2 model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ SAM2 model loading failed: {e}")
        return False

if __name__ == "__main__":
    success = test_installation()
    if success:
        print("\n🎉 Installation test passed! You're ready to run the app.")
        print("Run: streamlit run sam2_interactive_app.py")
    else:
        print("\n❌ Installation test failed. Check the error messages above.")
        exit(1)
EOF

print_success "Test script created"

echo ""
echo "🎉 Setup Complete!"
echo "==================="
echo ""
echo "Your SAM2 Interactive Segmentation App is ready to use!"
echo ""
echo "To run the app:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Start the Streamlit app: streamlit run sam2_interactive_app.py"
echo ""
echo "To test the installation:"
echo "python3 test_installation.py"
echo ""
echo "The app will be available at: http://localhost:8502"
echo ""
print_success "Setup completed successfully!"

# Deactivate virtual environment
deactivate 2>/dev/null || true
