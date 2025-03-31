import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_phi4_model(model_path="/home/TomAdmin/phi-4"):
    """Load the Phi-4 model and tokenizer from the specified path."""
    print(f"Loading Phi-4 model from {model_path}")
    
    # Check if the model path exists
    if not os.path.exists(model_path):
        print(f"WARNING: Model path {model_path} does not exist!")
        model_path = input("Please enter the correct path to the phi4 model: ")
        if not os.path.exists(model_path):
            raise ValueError(f"Model path {model_path} still does not exist!")
    
    print(f"Using model path: {model_path}")
    
    try:
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
        
        print("Model and tokenizer loaded successfully!")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def generate_response(model, tokenizer, prompt, max_length=500, temperature=0.7):
    """Generate a response using the Phi-4 model."""
    try:
        print(f"Processing prompt: '{prompt}'")
        
        # Format the input with proper formatting for phi-4
        # First get the input token IDs
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        input_length = input_ids.shape[1]
        print(f"Input length: {input_length} tokens")
        
        # Generate response
        with torch.no_grad():
            generation_config = {
                "max_new_tokens": max_length,
                "temperature": temperature,
                "do_sample": True,
                "top_p": 0.95,
                "top_k": 50,
                "repetition_penalty": 1.1,
                "pad_token_id": tokenizer.eos_token_id
            }
            
            outputs = model.generate(
                input_ids,
                **generation_config
            )
            
            print(f"Total generated tokens: {outputs.shape[1]}")
            print(f"New tokens: {outputs.shape[1] - input_length}")
        
        # Extract only the newly generated tokens (skip the input)
        new_tokens = outputs[0, input_length:]
        
        # Decode only the new tokens
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            
        print(f"Response generated successfully! Length: {len(response)} chars")
        
        # Print the first 50 characters for debugging
        preview = response[:50] + "..." if len(response) > 50 else response
        print(f"Response preview: {preview}")
        
        return response
    
    except Exception as e:
        print(f"Error during generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error generating response: {str(e)}"

def main():
    try:
        # Load the model and tokenizer
        model, tokenizer = load_phi4_model()
        
        # Test with a single prompt first to verify functionality
        print("\nTesting model with a simple prompt...")
        test_prompt = "You are Phi-4, an AI assistant. Answer this question: What are three interesting facts about space?"
        print(f"Test prompt: '{test_prompt}'")
        
        test_response = generate_response(model, tokenizer, test_prompt, max_length=150)
        print(f"Response: {test_response}\n")
        
        # Interactive mode
        print("\nEntering interactive mode. Type 'exit' to quit.")
        while True:
            user_input = input("\nEnter your prompt: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            
            print("Generating response...")
            
            # Add a system prompt to help the model
            full_prompt = f"You are Phi-4, an AI assistant. Answer this question: {user_input}"
            
            response = generate_response(model, tokenizer, full_prompt)
            print(f"\nResponse:\n{response}")
    
    except Exception as e:
        print(f"Critical error: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting tips:")
        print("1. Make sure the model path is correct")
        print("2. Ensure you have sufficient GPU memory if using CUDA")
        print("3. Check that all required libraries are installed")
        print("4. Try installing the latest version of transformers: pip install -U transformers")
        print("5. If using CPU only, add 'device_map=\"cpu\"' to the model loading code")

if __name__ == "__main__":
    main()