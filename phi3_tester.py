import torch
import os
import numpy as np
from transformers import AutoTokenizer
import argparse
import gc
import psutil
import platform
from typing import Dict, List, Union, Optional
import onnxruntime as ort

def load_phi3_model(model_path="/home/TomAdmin/phi-3-mini-4k-instruct-onnx", use_gpu=False, low_memory=True):
    """Load the Phi-3 ONNX model and tokenizer from the specified path."""
    print(f"Loading Phi-3 ONNX model from {model_path}")
    
    # Check if the model path exists
    if not os.path.exists(model_path):
        print(f"WARNING: Model path {model_path} does not exist!")
        model_path = input("Please enter the correct path to the phi3 ONNX model: ")
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
    
    # Determine provider based on hardware and user preference
    provider_options = []
    
    if use_gpu and ('CUDAExecutionProvider' in ort.get_available_providers()):
        print("Using GPU with CUDA for inference.")
        provider_options = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        if use_gpu and 'CUDAExecutionProvider' not in ort.get_available_providers():
            print("GPU requested but CUDA provider not available. Using CPU.")
        else:
            print("Using CPU as requested.")
        provider_options = ['CPUExecutionProvider']
        
        # Set number of threads for CPU
        thread_count = psutil.cpu_count(logical=True)
        print(f"Using {thread_count} CPU threads")
    
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
        
        # Check if model file exists
        model_files = [f for f in os.listdir(model_path) if f.endswith('.onnx')]
        if not model_files:
            raise FileNotFoundError(f"No .onnx files found in {model_path}")
        
        print(f"Found ONNX model files: {model_files}")
        
        # Determine main model file - typically model.onnx or something similar
        main_model_file = "model.onnx"
        if "model.onnx" not in model_files:
            # Try to find a suitable model file
            if len(model_files) == 1:
                main_model_file = model_files[0]
            else:
                # Look for likely candidates
                for file in model_files:
                    if "decoder" in file.lower() or "model" in file.lower():
                        main_model_file = file
                        break
                else:
                    main_model_file = model_files[0]  # Use the first one if no better match
        
        model_path_full = os.path.join(model_path, main_model_file)
        print(f"Loading ONNX model from: {model_path_full}")
        
        # Set up session options
        sess_options = ort.SessionOptions()
        
        if low_memory:
            print("Enabling memory optimizations for lower memory usage")
            sess_options.enable_mem_pattern = True
            sess_options.enable_mem_reuse = True
            sess_options.optimize_for_inference = True
            
            if 'CPUExecutionProvider' in provider_options:
                # Memory optimizations for CPU
                sess_options.intra_op_num_threads = thread_count
                
        # Create ONNX runtime session
        session = ort.InferenceSession(
            model_path_full, 
            sess_options=sess_options,
            providers=provider_options
        )
        
        # Get model metadata
        model_metadata = session.get_modelmeta()
        print(f"Model producer: {model_metadata.producer_name}")
        print(f"Graph inputs: {session.get_inputs()}")
        
        # Memory usage after loading
        mem_info = psutil.virtual_memory()
        print(f"RAM usage after model loading: {mem_info.used / 1e9:.2f GB ({mem_info.percent}%)")
        
        print("Model and tokenizer loaded successfully!")
        return session, tokenizer
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def generate_response(session, tokenizer, prompt, temperature=0.7):
    """Generate a response using the Phi-3 ONNX model."""
    try:
        print(f"Processing prompt: '{prompt}'")
        
        # Memory usage before inference
        mem_before = psutil.virtual_memory()
        print(f"RAM before inference: {mem_before.used / 1e9:.2f GB ({mem_before.percent}%)")
        
        # Tokenize input
        input_tokens = tokenizer(prompt, return_tensors="np")
        input_ids = input_tokens["input_ids"]
        
        # Get the input names from the model
        input_names = [input.name for input in session.get_inputs()]
        output_names = [output.name for output in session.get_outputs()]
        
        print(f"Model input names: {input_names}")
        print(f"Model output names: {output_names}")
        
        # For greedy decoding
        max_new_tokens = 500
        eos_token_id = tokenizer.eos_token_id
        
        # Manual generation loop
        print("Generating response (this may take longer on CPU)...")
        all_tokens = input_ids.copy()
        
        try:
            # Try to prepare full inputs based on the expected input names
            ort_inputs = {}
            
            # Common input name patterns
            if "input_ids" in input_names:
                ort_inputs["input_ids"] = input_ids.astype(np.int64)
            
            # Add attention mask if needed
            if "attention_mask" in input_names:
                attention_mask = np.ones(input_ids.shape, dtype=np.int64)
                ort_inputs["attention_mask"] = attention_mask
                
            # Add any other required inputs with default values
            for name in input_names:
                if name not in ort_inputs:
                    if "position" in name or "index" in name:
                        ort_inputs[name] = np.array([0], dtype=np.int64)
                    elif "past" in name or "cache" in name:
                        # For past key/value caches, typically start with None
                        ort_inputs[name] = np.array([0], dtype=np.int64)
            
            # For one-shot inference without caching (simplified approach)
            if len(output_names) == 1 and ("logits" in output_names[0] or "next_token" in output_names[0]):
                # Simple output with just logits - do manual token generation
                for _ in range(max_new_tokens):
                    # Run inference to get next token logits
                    outputs = session.run(output_names, ort_inputs)
                    
                    # Get logits from the last token
                    if "logits" in output_names[0]:
                        logits = outputs[0][0, -1, :]
                    else:
                        logits = outputs[0][0, :]
                    
                    # Apply temperature
                    if temperature > 0:
                        logits = logits / temperature
                    
                    # Simple greedy sampling
                    next_token_id = np.argmax(logits)
                    
                    # Append to generated tokens
                    all_tokens = np.concatenate([all_tokens, np.array([[next_token_id]])], axis=1)
                    
                    # Update inputs for next iteration
                    ort_inputs["input_ids"] = np.array([[next_token_id]], dtype=np.int64)
                    if "attention_mask" in ort_inputs:
                        ort_inputs["attention_mask"] = np.ones((1, 1), dtype=np.int64)
                    
                    # Check if we've generated the EOS token
                    if next_token_id == eos_token_id:
                        break
            else:
                # More complex model - run full inference in one go
                # If we can't generate token by token, we'll try to run the whole model once
                outputs = session.run(output_names, ort_inputs)
                
                # Try to extract something useful from the outputs
                if any("token_ids" in name for name in output_names):
                    # Find output with token_ids
                    for i, name in enumerate(output_names):
                        if "token_ids" in name or "output_ids" in name:
                            all_tokens = outputs[i]
                            break
        except Exception as e:
            print(f"Error during ONNX inference: {str(e)}")
            print("Trying alternative approach with just input_ids...")
            
            # Simplified approach - just pass input_ids and try to get something back
            try:
                outputs = session.run(None, {"input_ids": input_ids.astype(np.int64)})
                if len(outputs) > 0 and hasattr(outputs[0], "shape") and len(outputs[0].shape) > 1:
                    all_tokens = outputs[0]
            except Exception as e2:
                print(f"Alternative approach also failed: {str(e2)}")
                return f"Error: Unable to generate response with the ONNX model. {str(e)}"
        
        # Decode the generated tokens
        try:
            # Convert from numpy to list for tokenizer
            if isinstance(all_tokens, np.ndarray):
                if len(all_tokens.shape) > 1:
                    token_ids = all_tokens[0].tolist()
                else:
                    token_ids = all_tokens.tolist()
            else:
                token_ids = all_tokens
                
            response = tokenizer.decode(token_ids, skip_special_tokens=True)
            
            # If the response still contains the prompt, extract only the new part
            if prompt in response:
                response = response[response.find(prompt) + len(prompt):]
        except Exception as e:
            print(f"Error during decoding: {str(e)}")
            response = f"Error decoding response: {str(e)}"
        
        # Memory usage after inference
        mem_after = psutil.virtual_memory()
        print(f"RAM after inference: {mem_after.used / 1e9:.2f GB ({mem_after.percent}%)")
        
        print(f"Response generated! Length: {len(response)} chars")
        
        # Clean up to free memory
        gc.collect()
        
        return response
    
    except Exception as e:
        print(f"Error during generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error generating response: {str(e)}"

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run Phi-3 ONNX model with CPU optimizations")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available (default is CPU-only)")
    parser.add_argument("--model-path", type=str, default="/home/TomAdmin/phi-3-mini-4k-instruct-onnx", 
                        help="Path to the Phi-3 ONNX model directory")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for text generation (higher = more random)")
    parser.add_argument("--low-memory", action="store_true", 
                        help="Enable memory optimizations for lower memory usage")
    
    args = parser.parse_args()
    
    try:
        # Show intro message
        print("=" * 50)
        print("Phi-3 ONNX Tester")
        print("=" * 50)
        print("This script is optimized for the ONNX version of Phi-3.")
        print("ONNX provides better performance on CPU compared to PyTorch.")
        print("=" * 50)
        
        # Load the model and tokenizer with optimizations
        session, tokenizer = load_phi3_model(
            model_path=args.model_path, 
            use_gpu=args.gpu,
            low_memory=args.low_memory
        )
        
        # Test with a single prompt
        print("\nTesting model with a simple prompt...")
        test_prompt = "Hello, how are you?"
        print(f"Test prompt: '{test_prompt}'")
        
        test_response = generate_response(session, tokenizer, test_prompt, 
                                         temperature=args.temperature)
        print(f"Response: {test_response}\n")
        
        # Interactive mode
        print("\nEntering interactive mode. Type 'exit' to quit.")
        while True:
            user_input = input("\nEnter your prompt: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            
            print("Generating response...")
            response = generate_response(session, tokenizer, user_input, 
                                      temperature=args.temperature)
            print(f"\nResponse:\n{response}")
            
            # Clear some memory between generations
            gc.collect()
    
    except Exception as e:
        print(f"Critical error: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting tips:")
        print("1. Make sure the model path is correct and contains the ONNX model files")
        print("2. Install onnxruntime with: pip install onnxruntime")
        print("3. For GPU support: pip install onnxruntime-gpu")
        print("4. If you're getting out-of-memory errors:")
        print("   - Try with --low-memory flag")
        print("   - Close other applications to free up memory")
        print("5. Make sure your ONNX model is compatible - it should be the Phi-3-mini-4k-instruct-onnx version")

if __name__ == "__main__":
    main()