import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_phi4_model(model_path="/home/TomAdmin/phi4"):
    """Load the Phi-4 model and tokenizer from the specified path."""
    print(f"Loading Phi-4 model from {model_path}")
    
    # Check if the model path exists
    if not os.path.exists(model_path):
        raise ValueError(f"Model path {model_path} does not exist!")
    
    # Load tokenizer and model directly from the local path
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True
    )
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=500, temperature=0.7):
    """Generate a response using the Phi-4 model."""
    # Format the input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,  # Use max_new_tokens instead of max_length
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.1
        )
    
    # Decode the response, avoiding the input prompt repetition
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Attempt to remove the original prompt from the response if it appears at the beginning
    if response.startswith(prompt):
        response = response[len(prompt):].strip()
        
    return response

def main():
    try:
        # Load the model and tokenizer
        model, tokenizer = load_phi4_model()
        
        # Test with a single prompt first to verify functionality
        print("\nTesting model with a simple prompt...")
        test_response = generate_response(model, tokenizer, "Hello, what can you do?", max_length=100)
        print(f"Response: {test_response}\n")
        
        # Interactive mode
        print("\nEntering interactive mode. Type 'exit' to quit.")
        while True:
            user_input = input("\nEnter your prompt: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            
            print("Generating response...")
            response = generate_response(model, tokenizer, user_input)
            print(f"\nResponse:\n{response}")
    
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure the model path is correct")
        print("2. Ensure you have sufficient GPU memory if using CUDA")
        print("3. Check that all required libraries are installed")

if __name__ == "__main__":
    main()