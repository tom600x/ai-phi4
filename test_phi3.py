from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import gc
import psutil
import threading
import queue
import time

# Configure environment for CPU optimization
os.environ["TOKENIZERS_PARALLELISM"] = "true"  # Enable parallel tokenization
os.environ["OMP_NUM_THREADS"] = "24"  # Use half of available vCPUs for better threading balance
os.environ["MKL_NUM_THREADS"] = "24"  # Intel MKL threading
os.environ["PYTORCH_CPU_ALLOC_CONF"] = "max_split_size_mb:128"  # Memory allocation optimization

# Print system information
def print_system_info():
    print("\n=== System Information ===")
    cpu_count = os.cpu_count()
    memory = psutil.virtual_memory()
    print(f"CPU: {cpu_count} cores")
    print(f"RAM: {memory.total / (1024**3):.2f} GB total, {memory.available / (1024**3):.2f} GB available")
    print("===========================\n")

# Memory management function
def free_memory():
    """Aggressively free CPU memory."""
    gc.collect()
    torch.cuda.empty_cache()  # Still call this even without GPU to be safe

print_system_info()

# Load the model and tokenizer with optimizations for CPU
print("Loading model and tokenizer, please wait...")
model = AutoModelForCausalLM.from_pretrained(
    "/home/TomAdmin/output-model/phi-3-tuned",
    low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16,  # bfloat16 can be faster on CPUs with AVX512 support
    device_map="cpu",
)

# CPU optimization - once model is loaded, optimize it for inference
model = model.eval()  # Set to evaluation mode

# Apply quantization for better CPU performance
try:
    # Int8 quantization can significantly speed up inference on CPU
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    print("Model successfully quantized to INT8 for faster CPU inference")
except Exception as e:
    print(f"Quantization failed: {e}, continuing with unquantized model")

tokenizer = AutoTokenizer.from_pretrained("/home/TomAdmin/output-model/phi-3-tuned")
print("Model loaded successfully!")

# Response cache to avoid recomputing identical prompts
response_cache = {}

# Create worker thread pool for concurrent processing
request_queue = queue.Queue()
result_queue = queue.Queue()
stop_event = threading.Event()

def generate_text(prompt, max_length=1000, temperature=0.7):
    # Check cache first
    cache_key = f"{prompt}_{max_length}_{temperature}"
    if cache_key in response_cache:
        return response_cache[cache_key]
    
    # Process text generation
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        output = model.generate(
            **inputs, 
            max_length=max_length,
            do_sample=(temperature > 0),
            temperature=temperature,
            use_cache=True  # Enable KV caching
        )
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Store in cache
    response_cache[cache_key] = response
    return response

# Worker function to process requests in parallel
def worker_thread():
    while not stop_event.is_set() or not request_queue.empty():
        try:
            prompt, max_length, temperature, task_id = request_queue.get(timeout=0.1)
            response = generate_text(prompt, max_length, temperature)
            result_queue.put((task_id, response))
            request_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            result_queue.put((task_id, f"Error: {str(e)}"))
            request_queue.task_done()

# Start worker thread pool with 2 threads (more can cause contention on CPU-only inference)
worker_threads = []
for _ in range(2):
    t = threading.Thread(target=worker_thread, daemon=True)
    worker_threads.append(t)
    t.start()

# Interactive loop to get user questions
def main():
    print("\nPhi-3 Mini Question Answering System (CPU-Optimized)")
    print("Type 'exit' or 'quit' to end the program")
    print("Type 'settings' to adjust generation parameters\n")
    
    task_counter = 0
    max_length = 100
    temperature = 0.7
    
    while True:
        user_question = input("Enter your question: ")
        if user_question.lower() in ['exit', 'quit']:
            print("Shutting down worker threads...")
            stop_event.set()
            for t in worker_threads:
                t.join(timeout=2.0)
            print("Goodbye!")
            break
            
        elif user_question.lower() == 'settings':
            try:
                max_length = int(input("Enter max response length (default 1000): ") or max_length)
                temperature = float(input("Enter temperature (0.0-1.0, default 0.7): ") or temperature)
                print(f"Settings updated: max_length={max_length}, temperature={temperature}")
            except ValueError:
                print("Invalid input. Using previous settings.")
            continue
            
        print("\nQueuing request...")
        task_id = task_counter
        task_counter += 1
        request_queue.put((user_question, max_length, temperature, task_id))
        
        # Wait for this specific result
        while True:
            try:
                result_id, response = result_queue.get(timeout=0.1)
                result_queue.task_done()
                
                if result_id == task_id:
                    print(f"\nResponse: {response}\n")
                    break
                else:
                    # Put back results for other requests
                    result_queue.put((result_id, response))
            except queue.Empty:
                print(".", end="", flush=True)
                time.sleep(0.5)

if __name__ == "__main__":
    main()