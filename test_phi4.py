#!/usr/bin/env python3
"""
Phi-4 Test Script
This script tests if Phi-4 model is properly installed and running.
"""

import time
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_phi4_model(prompt="Explain quantum computing in simple terms"):
    """Test if the Phi-4 model is working properly."""
    print("Starting Phi-4 test...")
    start_time = time.time()
    
    # Load the model and tokenizer
    print("Loading Phi-4 model and tokenizer...")
    try:
        model_name = "/home/TomAdmin/phi-4"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True
        )
        print(f"✅ Model '{model_name}' loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    # Test generation
    print("\nGenerating text with prompt:", prompt)
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Print model's response
        print("\n--- MODEL RESPONSE ---")
        print(response)
        print("--- END OF RESPONSE ---\n")
        
        # Calculate and print metrics
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"✅ Generation successful!")
        print(f"⏱️ Total time elapsed: {elapsed_time:.2f} seconds")
        print(f"📊 Using CPU")
        return True
    except Exception as e:
        print(f"❌ Error generating text: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("PHI-4 MODEL TEST")
    print("=" * 50)
    success = test_phi4_model()
    if success:
        print("\n✅ PHI-4 MODEL TEST PASSED - READY FOR USE!")
    else:
        print("\n❌ PHI-4 MODEL TEST FAILED")
    print("=" * 50)