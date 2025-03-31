from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("/home/TomAdmin/phi-4")
tokenizer = AutoTokenizer.from_pretrained("/home/TomAdmin/phi-4")

def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**inputs, max_length=100)
    return tokenizer.decode(output[0], skip_special_tokens=True)

print(generate_text("Hello, how are you?"))