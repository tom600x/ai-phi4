from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer only once at the start
print("Loading model and tokenizer, please wait...")
model = AutoModelForCausalLM.from_pretrained("/home/TomAdmin/output-model/phi-3-tuned")
tokenizer = AutoTokenizer.from_pretrained("/home/TomAdmin/output-model/phi-3-tuned")
print("Model loaded successfully!")

def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**inputs, max_length=100)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Interactive loop to get user questions
def main():
    print("\nPhi-3 Mini Question Answering System")
    print("Type 'exit' or 'quit' to end the program\n")
    
    while True:
        user_question = input("Enter your question: ")
        if user_question.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
            
        print("\nGenerating response...")
        response = generate_text(user_question)
        print(f"\nResponse: {response}\n")

if __name__ == "__main__":
    main()