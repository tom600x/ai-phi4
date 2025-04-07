import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import gc
import psutil
import platform

def load_phi3_model(model_path="/home/TomAdmin/phi-3-mini-128k-instruct", use_gpu=False, low_memory=True):
    """Load the Phi-3 model and tokenizer from the specified path."""
    print(f"Loading Phi-3 model from {model_path}")
    
    # Check if the model path exists
    if not os.path.exists(model_path):
        print(f"WARNING: Model path {model_path} does not exist!")
        model_path = input("Please enter the correct path to the phi3 model: ")
        if not os.path.exists(model_path):
            raise ValueError(f"Model path {model_path} still does not exist!")
    
    print(f"Using model path: {model_path}")
    
    # Print system information
    print("\nSystem Information:")
    print(f"OS: {platform.system()} {platform.version()}")
    print(f"Python: {platform.python_version()}")
    print(f"CPU: {platform.processor()}")
    
    # Memory information
    mem_info = psutil.virtual_memory()
    print(f"Total RAM: {mem_info.total / 1e9:.2f} GB")
    print(f"Available RAM: {mem_info.available / 1e9:.2f} GB")
    print(f"Used RAM: {mem_info.used / 1e9:.2f} GB ({mem_info.percent}%)")
    
    # Force CPU usage if requested or if GPU not available
    if not use_gpu or not torch.cuda.is_available():
        if use_gpu and not torch.cuda.is_available():
            print("GPU requested but CUDA is not available. Using CPU.")
        else:
            print("Using CPU as requested.")
        device_map = "cpu"
        # Set PyTorch to only use CPU
        torch.set_num_threads(psutil.cpu_count(logical=True))
        print(f"Using {torch.get_num_threads()} CPU threads")
    else:
        device_map = "auto"
        print(f"Using GPU - CUDA available with {torch.cuda.device_count()} device(s)")
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Call garbage collection before loading model
    gc.collect()
    
    try:
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        
        # Load model with optimizations for CPU
        print("Loading model (this may take a while on CPU)...")
        
        # Memory optimization options for CPU
        model_kwargs = {
            "trust_remote_code": True,
            "local_files_only": True,
            "device_map": device_map,
            "attn_implementation": "eager"  # Avoid flash attention issues
        }
        
        # Add CPU-specific optimizations
        if device_map == "cpu":
            model_kwargs["torch_dtype"] = torch.float32  # Use float32 on CPU for better compatibility
            # 8-bit quantization was causing "init_empty_weights" error, so we'll use a different approach
            if low_memory:
                try:
                    # First try with 8-bit if bitsandbytes is installed
                    import bitsandbytes
                    model_kwargs["load_in_8bit"] = True
                    print("Using 8-bit quantization for lower memory usage")
                except ImportError:
                    # Fall back to 4-bit if supported
                    try:
                        from transformers import BitsAndBytesConfig
                        model_kwargs["quantization_config"] = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float32
                        )
                        print("Using 4-bit quantization for lower memory usage")
                    except (ImportError, AttributeError):
                        # If all else fails, load with default settings
                        print("Quantization libraries not available. Loading model with standard settings.")
                        if "load_in_8bit" in model_kwargs:
                            del model_kwargs["load_in_8bit"]
        else:
            model_kwargs["torch_dtype"] = torch.float16
        
        # Try to import accelerate explicitly to avoid init_empty_weights error
        try:
            import accelerate
            from accelerate import init_empty_weights
            print(f"Using accelerate version: {accelerate.__version__}")
        except ImportError:
            print("Accelerate package not found. This might cause issues with model loading.")
        except AttributeError:
            print("Your version of accelerate doesn't have init_empty_weights. Using alternative loading method.")
        
        # Load the model with error handling for different transformers versions
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **model_kwargs
            )
        except (NameError, AttributeError) as e:
            if "init_empty_weights" in str(e):
                print("Encountered init_empty_weights error. Trying alternative loading method...")
                # Try loading with CPU first then moving if needed
                cpu_kwargs = model_kwargs.copy()
                cpu_kwargs["device_map"] = None  # Don't use device_map
                if "load_in_8bit" in cpu_kwargs:
                    del cpu_kwargs["load_in_8bit"]  # Remove quantization that might cause issues
                if "quantization_config" in cpu_kwargs:
                    del cpu_kwargs["quantization_config"]
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    **cpu_kwargs
                )
                
                # Now move to the right device if needed
                if device_map != "cpu" and torch.cuda.is_available():
                    model = model.to("cuda")
            else:
                raise e
        
        # Print model and tokenizer info
        print(f"Model type: {type(model).__name__}")
        print(f"Model device: {next(model.parameters()).device}")
        
        # Memory usage after loading
        mem_info = psutil.virtual_memory()
        print(f"RAM usage after model loading: {mem_info.used / 1e9:.2f} GB ({mem_info.percent}%)")
        
        print("Model and tokenizer loaded successfully!")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def generate_response(model, tokenizer, prompt, temperature=0.7):
    """Generate a response using the Phi-3 model."""
    try:
        print(f"Processing prompt: '{prompt}'")
        
        # Memory usage before inference
        mem_before = psutil.virtual_memory()
        print(f"RAM before inference: {mem_before.used / 1e9:.2f} GB ({mem_before.percent}%)")
        
        # Simple tokenization
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Move inputs to the model device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # CPU-friendly generation parameters
        generate_kwargs = {
            "temperature": temperature,
            "do_sample": True
        }
        
        # For CPU, add more efficiency options
        if device.type == "cpu":
            generate_kwargs["num_beams"] = 1  # Reduce beam search complexity
        
        # Generation
        print("Generating response (this may take longer on CPU)...")
        output = model.generate(**inputs, **generate_kwargs)
        
        # Decode the full output
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Memory usage after inference
        mem_after = psutil.virtual_memory()
        print(f"RAM after inference: {mem_after.used / 1e9:.2f} GB ({mem_after.percent}%)")
        
        print(f"Response generated! Length: {len(response)} chars")
        
        # Clean up to free memory
        del output
        gc.collect()
        
        return response
    
    except Exception as e:
        print(f"Error during generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error generating response: {str(e)}"

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run Phi-3 model with CPU optimizations")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available (default is CPU-only)")
    parser.add_argument("--model-path", type=str, default="/home/TomAdmin/phi-3-mini-128k-instruct", 
                        help="Path to the Phi-3 model directory")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for text generation (higher = more random)")
    parser.add_argument("--low-memory", action="store_true", 
                        help="Enable 8-bit quantization for lower memory usage")
    
    args = parser.parse_args()
    
    try:
        # Show intro message
        print("=" * 50)
        print("Phi-3 CPU Tester")
        print("=" * 50)
        print("This script is optimized for CPU usage.")
        print("Loading and inference will be slower than on GPU but more memory-efficient.")
        print("=" * 50)
        
        # Load the model and tokenizer with CPU optimizations
        model, tokenizer = load_phi3_model(
            model_path=args.model_path, 
            use_gpu=args.gpu,
            low_memory=args.low_memory
        )
        
        # Test with a single prompt
        print("\nTesting model with a simple prompt...")
        test_prompt = "Hello, how are you?"
        print(f"Test prompt: '{test_prompt}'")
        
        test_response = generate_response(model, tokenizer, test_prompt, 
                                          temperature=args.temperature)
        print(f"Response: {test_response}\n")
        
        # Interactive mode
        print("\nEntering interactive mode. Type 'exit' to quit.")
        while True:
            user_input = input("\nEnter your prompt: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            
            print("Generating response...")
            response = generate_response(model, tokenizer, user_input, 
                                        temperature=args.temperature)
            print(f"\nResponse:\n{response}")
            
            # Clear some memory between generations
            gc.collect()
    
    except Exception as e:
        print(f"Critical error: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting tips:")
        print("1. Make sure the model path is correct and contains all necessary files")
        print("2. If you're getting out-of-memory errors:")
        print("   - Try with --low-memory flag for 8-bit quantization")
        print("   - Consider using a smaller model variant")
        print("   - Close other applications to free up memory")
        print("3. Be patient, CPU inference is much slower than GPU")

if __name__ == "__main__":
    main()