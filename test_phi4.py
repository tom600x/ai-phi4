# This script demonstrates how to load a fully fine-tuned Phi-4 model.
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set paths for model loading
model_path = "/home/TomAdmin/ai-phi4/output/phi4-finetuned"  # Path to your fully fine-tuned model

# 1. Load the tokenizer from the fine-tuned model directory
print(f"Loading tokenizer from {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 2. Load the fine-tuned model
print(f"Loading fine-tuned model from {model_path}...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# 3. Format the prompt for conversation-style input
test_input = "Convert the following PL/SQL query to LINQ: CREATE OR REPLACE PROCEDURE call_rest_api(p_url IN VARCHAR2, p_response OUT CLOB) IS\n v_http_request UTL_HTTP.req;\n v_http_response UTL_HTTP.resp;\nBEGIN\n v_http_request := UTL_HTTP.begin_request(p_url, 'GET');\n v_http_response := UTL_HTTP.get_response(v_http_request);\n UTL_HTTP.read_text(v_http_response, p_response);\n UTL_HTTP.end_response(v_http_response);\nEXCEPTION\n WHEN others THEN\n DBMS_OUTPUT.PUT_LINE('Error: ' || SQLERRM);\nEND; Description: Call a REST API using PL/SQL and handle errors.."
formatted_input = f"<|user|>\n{test_input}\n<|assistant|>\n"

# 4. Generate text with your fine-tuned model
print("Generating response...")
inputs = tokenizer(formatted_input, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_length=512,
        temperature=0.1,
        top_p=0.9,
        repetition_penalty=1.2,
        do_sample=True
    )

# 5. Process and print the response
response = tokenizer.decode(output[0], skip_special_tokens=False)
# Extract just the assistant's response
assistant_response = response.split("<|assistant|>")[-1].strip()
print("\nModel response:")
print(assistant_response)