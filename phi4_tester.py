import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_phi4_model(model_path="/home/TomAdmin/phi4"):
    """Load the Phi-4 model and tokenizer from the specified path."""
    print(f"Loading Phi-4 model from {model_path}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,  # Use float16 for efficiency
        device_map="auto"  # Automatically determine the best device
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
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.1
        )
    
    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    # Load the model and tokenizer
    model, tokenizer = load_phi4_model()
    
    # Test with different prompts
    test_prompts = [
        "Hello, how are you today?",
        "Explain the concept of quantum computing in simple terms.",
        "Write a short poem about artificial intelligence.",
        "What are the main differences between Python and JavaScript?"
    ]
    
    print("\n" + "="*50)
    print("PHI-4 MODEL TEST RESULTS")
    print("="*50)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nTest {i}: '{prompt}'\n")
        response = generate_response(model, tokenizer, prompt)
        print(f"Response:\n{response}\n")
        print("-"*50)
    
    # Interactive mode
    print("\nEntering interactive mode. Type 'exit' to quit.")
    while True:
        user_input = input("\nEnter your prompt: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        
        response = generate_response(model, tokenizer, user_input)
        print(f"\nResponse:\n{response}")

if __name__ == "__main__":
    main()