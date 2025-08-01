import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
from typing import Tuple, Dict, Any

# Florence2 task constants
FLORENCE_DETAILED_CAPTION_TASK = "<DETAILED_CAPTION>"
FLORENCE_CAPTION_TO_PHRASE_GROUNDING_TASK = "<CAPTION_TO_PHRASE_GROUNDING>"
FLORENCE_OPEN_VOCABULARY_DETECTION_TASK = "<OPEN_VOCABULARY_DETECTION>"

def load_florence_model(device: torch.device) -> Tuple[Any, Any]:
    """
    Load Florence2 model and processor.
    
    Args:
        device: The device to load the model on
        
    Returns:
        Tuple of (model, processor)
    """
    # Try different models in order of preference
    model_candidates = [
        "microsoft/Florence-2-large",
        "microsoft/Florence-2-base",
        "microsoft/Florence-2-base-ft"
    ]
    
    # Configure SSL and download settings
    import ssl
    import os
    
    # Try to handle SSL issues in corporate networks
    try:
        # Disable SSL verification if needed (not recommended for production)
        ssl._create_default_https_context = ssl._create_unverified_context
        print("⚠️ SSL verification disabled due to certificate issues")
    except:
        pass
    
    # Set offline mode if models are cached
    offline_mode = os.environ.get('HF_HUB_OFFLINE', '0') == '1'
    
    # Check if we have cached files for Florence-2-large
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub/models--microsoft--Florence-2-large/snapshots/main")
    if os.path.exists(cache_dir) and os.path.exists(os.path.join(cache_dir, "config.json")):
        print("🔍 Found cached Florence-2-large files, using offline mode")
        offline_mode = True
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        # Force only this model since we have it cached
        model_candidates = ["microsoft/Florence-2-large"]
    
    # Try each model candidate
    for model_id in model_candidates:
        try:
            print(f"🔄 Trying to load {model_id}...")
            
            # Load processor
            processor = AutoProcessor.from_pretrained(
                model_id, 
                trust_remote_code=True,
                local_files_only=offline_mode,
                use_auth_token=False
            )
            
            # Load model with appropriate precision based on device
            if device.type == "cuda":
                # Use bfloat16 for CUDA if supported, otherwise float16
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        torch_dtype=torch.bfloat16,
                        trust_remote_code=True,
                        device_map=device,
                        local_files_only=offline_mode
                    )
                except Exception:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        torch_dtype=torch.float16,
                        trust_remote_code=True,
                        device_map=device,
                        local_files_only=offline_mode
                    )
            elif device.type == "mps":
                # MPS works best with float32 for compatibility
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    local_files_only=offline_mode
                )
                model = model.to(device)
            else:
                # CPU - use float32
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    local_files_only=offline_mode
                )
                model = model.to(device)
            
            model.eval()
            
            print(f"✅ Florence2 model ({model_id}) loaded on {device}")
            return model, processor
            
        except Exception as e:
            print(f"❌ Failed to load {model_id}: {e}")
            if model_id == model_candidates[-1]:  # Last model failed
                print(f"\n💡 All Florence2 models failed to load. Final error: {e}")
                print("\n🔧 Troubleshooting steps:")
                print("1. Try the Florence2 downloader:")
                print("   python download_florence.py")
                print("2. For SSL issues in corporate networks:")
                print("   export PYTHONHTTPSVERIFY=0")
                print("3. Try using mobile hotspot or different network")
                print("4. Manual download from HuggingFace website")
                raise
            else:
                print(f"⚠️ Trying next model...")
                continue

def prepare_florence_prompt(task: str, text: str = None) -> str:
    """
    Prepare the prompt for Florence2 based on the task.
    
    Args:
        task: The Florence2 task type
        text: Optional text input for the task
        
    Returns:
        Formatted prompt string
    """
    if task == FLORENCE_DETAILED_CAPTION_TASK:
        return task
    elif task == FLORENCE_CAPTION_TO_PHRASE_GROUNDING_TASK:
        return f"{task} {text}" if text else task
    elif task == FLORENCE_OPEN_VOCABULARY_DETECTION_TASK:
        return f"{task} {text}" if text else task
    else:
        return task

@torch.inference_mode()
def run_florence_inference(
    model: Any,
    processor: Any,
    device: torch.device,
    image: Image.Image,
    task: str,
    text: str = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Run Florence2 inference on an image.
    
    Args:
        model: Florence2 model
        processor: Florence2 processor
        device: Device to run inference on
        image: Input PIL image
        task: Florence2 task type
        text: Optional text input
        
    Returns:
        Tuple of (prompt_used, results_dict)
    """
    # Prepare prompt
    prompt = prepare_florence_prompt(task, text)
    
    # Process inputs
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    )
    
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Set up generation parameters based on device
    generation_kwargs = {
        "max_new_tokens": 1024,
        "early_stopping": False,
        "do_sample": False,
        "num_beams": 3,
    }
    
    # Add device-specific optimizations
    if device.type == "cuda":
        # Use autocast for mixed precision on CUDA
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                **generation_kwargs
            )
    elif device.type == "mps":
        # MPS doesn't support autocast in the same way
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            **generation_kwargs
        )
    else:
        # CPU inference
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            **generation_kwargs
        )
    
    # Decode the generated text
    generated_text = processor.batch_decode(
        generated_ids, 
        skip_special_tokens=False
    )[0]
    
    # Parse the response
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task,
        image_size=(image.width, image.height)
    )
    
    return prompt, parsed_answer

def validate_florence_result(result: Dict[str, Any], task: str) -> bool:
    """
    Validate Florence2 inference result.
    
    Args:
        result: Florence2 inference result
        task: The task that was performed
        
    Returns:
        True if result is valid, False otherwise
    """
    if not isinstance(result, dict):
        return False
    
    if task == FLORENCE_DETAILED_CAPTION_TASK:
        return FLORENCE_DETAILED_CAPTION_TASK in result
    elif task == FLORENCE_CAPTION_TO_PHRASE_GROUNDING_TASK:
        return FLORENCE_CAPTION_TO_PHRASE_GROUNDING_TASK in result
    elif task == FLORENCE_OPEN_VOCABULARY_DETECTION_TASK:
        return FLORENCE_OPEN_VOCABULARY_DETECTION_TASK in result
    
    return True

def format_florence_result(result: Dict[str, Any], task: str) -> str:
    """
    Format Florence2 result for display.
    
    Args:
        result: Florence2 inference result
        task: The task that was performed
        
    Returns:
        Formatted result string
    """
    if task == FLORENCE_DETAILED_CAPTION_TASK:
        return result.get(FLORENCE_DETAILED_CAPTION_TASK, "No caption generated")
    elif task == FLORENCE_CAPTION_TO_PHRASE_GROUNDING_TASK:
        grounding_result = result.get(FLORENCE_CAPTION_TO_PHRASE_GROUNDING_TASK, {})
        if isinstance(grounding_result, dict):
            bboxes = grounding_result.get('bboxes', [])
            return f"Found {len(bboxes)} grounded phrases"
        return "No phrases grounded"
    elif task == FLORENCE_OPEN_VOCABULARY_DETECTION_TASK:
        detection_result = result.get(FLORENCE_OPEN_VOCABULARY_DETECTION_TASK, {})
        if isinstance(detection_result, dict):
            bboxes = detection_result.get('bboxes', [])
            return f"Detected {len(bboxes)} objects"
        return "No objects detected"
    
    return str(result)
