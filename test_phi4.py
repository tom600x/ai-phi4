# This script demonstrates how to load a fine-tuned model using PEFT (Parameter-Efficient Fine-Tuning) with LoRA (Low-Rank Adaptation).

from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Load the base model
base_model = AutoModelForCausalLM.from_pretrained(
    "/home/TomAdmin/phi-4",  # Original model
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# 2. Load your fine-tuned LoRA adapter
adapter_path = "/home/TomAdmin/ai-phi-4/output/phi4-finetuned"  # Path to your saved model
model = PeftModel.from_pretrained(base_model, adapter_path)

# 3. Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("/home/TomAdmin/phi-4", trust_remote_code=True)

# 4. Generate text with your fine-tuned model
prompt = "Convert the following PL/SQL code to C# LINQ: \n\nSELECT * FROM users WHERE age > 30"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_length=512)
print(tokenizer.decode(output[0], skip_special_tokens=True))