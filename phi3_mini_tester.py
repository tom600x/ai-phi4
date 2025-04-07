import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import optimum.onnxruntime

def load_phi3_mini_model(model_path="/home/TomAdmin/phi-3-mini-4k-instruct-onnx", use_gpu=True):
    """Load the Phi-3-mini-4k-instruct ONNX model and tokenizer from the specified path."""
    print(f"Loading Phi-3-mini ONNX model from {model_path}")
    
    # Check if the model path exists
    if not os.path.exists(model_path):
        print(f"WARNING: Model path {model_path} does not exist!")
        model_path = input("Please enter the correct path to the phi-3-mini model: ")
        if not os.path.exists(model_path):
            raise ValueError(f"Model path {model_path} still does not exist!")
    
    print(f"Using model path: {model_path}")
    
    # Determine device based on user preference and availability
    if use_gpu and torch.cuda.is_available():
        device = "cuda"
        print(f"Using GPU - CUDA available with {torch.cuda.device_count()} device(s)")
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        if use_gpu and not torch.cuda.is_available():
            print("GPU requested but CUDA is not available. Falling back to CPU.")
        else:
            print("Using CPU as requested.")
        device = "cpu"
    
    try:
        # Load tokenizer and ONNX model
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        
        # For ONNX models, we use optimum.onnxruntime
        model = optimum.onnxruntime.ORTModelForCausalLM.from_pretrained(
            model_path,
            device=device,
            trust_remote_code=True,
            local_files_only=True
        )
        
        # Print model and tokenizer info for debugging
        print(f"Model type: {type(model).__name__}")
        print("Model and tokenizer loaded successfully!")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def generate_response(model, tokenizer, prompt, max_length=100, temperature=0.7):
    """Generate a response using the Phi-3-mini ONNX model."""
    try:
        print(f"Processing prompt: '{prompt}'")
        
        # For instruction-tuned models, we format the prompt for instruction following
        formatted_prompt = f"<|user|>\n{prompt}\n<|assistant|>"
        
        # Tokenize the input
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        
        # Generate the response
        output = model.generate(
            **inputs, 
            max_length=max_length,
            temperature=temperature,
            do_sample=True
        )
        
        # Decode the full output
        full_response = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract only the assistant's response (if needed)
        response = full_response.split("<|assistant|>")[-1].strip()
        
        print(f"Response generated! Length: {len(response)} chars")
        
        return response
    
    except Exception as e:
        print(f"Error during generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error generating response: {str(e)}"

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run Phi-3-mini ONNX model with GPU or CPU")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage even if GPU is available")
    parser.add_argument("--model-path", type=str, default="/home/TomAdmin/phi-3-mini-4k-instruct-onnx", 
                        help="Path to the Phi-3-mini-4k-instruct-onnx model directory")
    parser.add_argument("--max-length", type=int, default=200,
                        help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for text generation (higher = more random)")
    
    args = parser.parse_args()
    
    try:
        # Load the model and tokenizer with GPU/CPU preference
        model, tokenizer = load_phi3_mini_model(model_path=args.model_path, use_gpu=not args.cpu)
        
        # Interactive mode
        print("\nPhi-3-mini-4k-instruct ONNX model interactive mode. Type 'exit' to quit.")
        while True:
            user_input = input("\nEnter your prompt: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            
            print("Generating response...")
            response = generate_response(model, tokenizer, user_input, 
                                        max_length=args.max_length,
                                        temperature=args.temperature)
            print(f"\nResponse:\n{response}")
    
    except Exception as e:
        print(f"Critical error: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting tips:")
        print("1. Make sure the model path is correct and contains all necessary files")
        print("2. Ensure you have sufficient GPU memory if using CUDA")
        print("3. Try with --cpu flag if having GPU memory issues")
        print("4. Ensure you have the optimum and onnxruntime-gpu (or onnxruntime) packages installed")

if __name__ == "__main__":
    main()