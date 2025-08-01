import torch
import platform
import subprocess
import sys

def get_device():
    """
    Automatically detect the best available device for the current system.
    Supports CUDA, Apple Silicon (MPS), and CPU.
    """
    # Check for CUDA
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
        return device
    
    # Check for Apple Silicon (MPS)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Additional check for macOS version compatibility
        if platform.system() == "Darwin":
            try:
                # MPS requires macOS 12.3+ and certain Metal performance shaders
                device = torch.device("mps")
                print("Using Apple Silicon (MPS) device")
                return device
            except Exception as e:
                print(f"MPS device not available: {e}")
    
    # Fallback to CPU
    device = torch.device("cpu")
    print("Using CPU device")
    return device

def setup_torch_optimizations(device):
    """
    Setup torch optimizations based on the device type.
    """
    if device.type == "cuda":
        # Enable optimizations for CUDA
        try:
            # Enable mixed precision if supported
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Set memory allocation strategy
            if hasattr(torch.cuda, 'set_memory_fraction'):
                torch.cuda.set_memory_fraction(0.9)  # Use 90% of GPU memory
            
            print("CUDA optimizations enabled")
        except Exception as e:
            print(f"Could not enable CUDA optimizations: {e}")
    
    elif device.type == "mps":
        # MPS-specific optimizations
        try:
            # Enable Metal Performance Shaders optimizations
            torch.backends.mps.allow_tf32 = True
            print("MPS optimizations enabled")
        except Exception as e:
            print(f"Could not enable MPS optimizations: {e}")
    
    else:
        # CPU optimizations
        try:
            # Set number of threads for CPU inference
            torch.set_num_threads(min(8, torch.get_num_threads()))
            print(f"CPU optimizations enabled with {torch.get_num_threads()} threads")
        except Exception as e:
            print(f"Could not enable CPU optimizations: {e}")

def get_device_info():
    """
    Get detailed information about the current device.
    """
    device = get_device()
    info = {
        "device": str(device),
        "type": device.type,
        "platform": platform.system(),
        "architecture": platform.machine(),
    }
    
    if device.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name()
        info["gpu_memory"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        info["cuda_version"] = torch.version.cuda
    
    elif device.type == "mps":
        info["metal_support"] = torch.backends.mps.is_built()
        
    info["torch_version"] = torch.__version__
    
    return info

def check_memory_usage(device):
    """
    Check current memory usage for the device.
    """
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        return {
            "allocated": f"{allocated:.2f} GB",
            "cached": f"{cached:.2f} GB", 
            "total": f"{total:.2f} GB",
            "usage_percent": f"{(allocated/total)*100:.1f}%"
        }
    
    elif device.type == "mps":
        # MPS doesn't have direct memory query methods like CUDA
        return {
            "info": "MPS memory usage monitoring not directly available"
        }
    
    else:
        # CPU memory would need psutil or similar
        return {
            "info": "CPU memory monitoring requires additional dependencies"
        }
