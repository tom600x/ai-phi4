import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

def load_phi3_model(model_path="/home/TomAdmin/phi-3-mini-128k-instruct", use_gpu=True):
    """Load the Phi-3 model and tokenizer from the specified path."""
    print(f"Loading Phi-3 model from {model_path}")
    
    # Check if the model path exists
    if not os.path.exists(model_path):
        print(f"WARNING: Model path {model_path} does not exist!")
        model_path = input("Please enter the correct path to the phi3 model: ")
        if not os.path.exists(model_path):
            raise ValueError(f"Model path {model_path} still does not exist!")
    
    print(f"Using model path: {model_path}")
    
    # Determine device based on user preference and availability
    if use_gpu and torch.cuda.is_available():
        device_map = "auto"
        print(f"Using GPU - CUDA available with {torch.cuda.device_count()} device(s)")
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        if use_gpu and not torch.cuda.is_available():
            print("GPU requested but CUDA is not available. Falling back to CPU.")
        else:
            print("Using CPU as requested.")
        device_map = "cpu"
    
    try:
        # Load tokenizer and model directly from the local path
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device_map == "auto" else torch.float32,
            device_map=device_map,
            trust_remote_code=True,
            local_files_only=True
        )
        
        # Print model and tokenizer info for debugging
        print(f"Model type: {type(model).__name__}")
        print(f"Model device: {next(model.parameters()).device}")
        print("Model and tokenizer loaded successfully!")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def generate_response(model, tokenizer, prompt, max_length=100, temperature=0.7):
    """Generate a response using the Phi-3 model."""
    try:
        print(f"Processing prompt: '{prompt}'")
        
        # Simple tokenization
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Move inputs to the model device
        inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
        
        # Generation parameters
        output = model.generate(
            **inputs, 
            max_length=max_length,
            temperature=temperature,
            do_sample=True
        )
        
        # Decode the full output
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        
        print(f"Response generated! Length: {len(response)} chars")
        
        return response
    
    except Exception as e:
        print(f"Error during generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error generating response: {str(e)}"

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run Phi-3 model with GPU or CPU")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage even if GPU is available")
    parser.add_argument("--model-path", type=str, default="/home/TomAdmin/phi-3", 
                        help="Path to the Phi-3 model directory")
    parser.add_argument("--max-length", type=int, default=200,
                        help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for text generation (higher = more random)")
    
    args = parser.parse_args()
    
    try:
        # Load the model and tokenizer with GPU/CPU preference
        model, tokenizer = load_phi3_model(model_path=args.model_path, use_gpu=not args.cpu)
        
        # Test with a single prompt
        print("\nTesting model with a simple prompt...")
        test_prompt = "Hello, how are you?"
        print(f"Test prompt: '{test_prompt}'")
        
        test_response = generate_response(model, tokenizer, test_prompt, 
                                          max_length=args.max_length, 
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

if __name__ == "__main__":
    main()