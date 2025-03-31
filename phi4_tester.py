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
        
        # Ensure the tokenizer has proper pad token and EOS token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True
        )
        
        # Print model and tokenizer info for debugging
        print(f"Model type: {type(model).__name__}")
        print(f"Tokenizer type: {type(tokenizer).__name__}")
        print(f"Model device: {next(model.parameters()).device}")
        print(f"Vocab size: {len(tokenizer)}")
        print(f"EOS token: {tokenizer.eos_token}, ID: {tokenizer.eos_token_id}")
        print(f"PAD token: {tokenizer.pad_token}, ID: {tokenizer.pad_token_id}")
        
        print("Model and tokenizer loaded successfully!")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def generate_response(model, tokenizer, prompt, max_length=100, temperature=0.7):
    """Generate a response using the Phi-4 model."""
    try:
        print(f"Processing prompt: '{prompt}'")
        
        # Tokenize input with attention mask
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,  # Prevent overly long inputs
            return_attention_mask=True
        )
        
        # Move inputs to the same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        input_length = inputs["input_ids"].shape[1]
        print(f"Input length: {input_length} tokens")
        print(f"Input IDs: {inputs['input_ids'][0].tolist()[:10]}...")
        
        # Generate response - explicitly defining all parameters
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_length,  # Generate this many new tokens
                min_new_tokens=10,  # Force generation of at least 10 tokens
                temperature=temperature,
                do_sample=True,
                top_p=0.92,
                top_k=50,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
            
            total_tokens = outputs.shape[1]
            new_tokens = total_tokens - input_length
            print(f"Total generated tokens: {total_tokens}")
            print(f"New tokens: {new_tokens}")
            
            if new_tokens <= 0:
                print("WARNING: No new tokens generated!")
                print(f"Output shape: {outputs.shape}")
                print(f"First few output tokens: {outputs[0][:10].tolist()}")
        
        # Only extract the new tokens if we actually generated new ones
        if new_tokens > 0:
            # Extract only the newly generated tokens (skip the input)
            new_tokens_tensor = outputs[0, input_length:]
            response = tokenizer.decode(new_tokens_tensor, skip_special_tokens=True)
        else:
            # If no new tokens, try decoding the whole output
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Try to remove the input prompt if possible
            if prompt in response:
                response = response[len(prompt):].strip()
            else:
                response = "Model did not generate any response."
        
        print(f"Response generated successfully! Length: {len(response)} chars")
        print(f"Response preview: {response[:100] + '...' if len(response) > 100 else response}")
        
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
        test_prompt = "Write a short story about a robot learning to feel emotions."
        print(f"Test prompt: '{test_prompt}'")
        
        test_response = generate_response(model, tokenizer, test_prompt, max_length=200)
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
        print(f"Critical error: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting tips:")
        print("1. Make sure the model path is correct and contains all necessary files")
        print("2. Ensure you have sufficient GPU memory if using CUDA")
        print("3. Verify that all required libraries are installed: transformers, torch")
        print("4. Try updating libraries: pip install -U transformers torch")
        print("5. Try with device_map=\"cpu\" if having GPU memory issues")
        print("6. Check the model's specific requirements in its documentation")

if __name__ == "__main__":
    main()