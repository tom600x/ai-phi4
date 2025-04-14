#!/usr/bin/env python
# Memory-efficient fine-tuning script for Phi-4
# Uses 8-bit quantization and LoRA to minimize GPU memory requirements
# Specifically optimized for phi4_fine_tuning_dataset.json

import json
import torch
import os
import gc
import psutil
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    TaskType
)
from datasets import Dataset
import argparse
import numpy as np

# Set environment variables to optimize memory usage
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Only use the first GPU
DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Memory optimization settings
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def print_system_info():
    """Print system information including GPU, CPU and memory."""
    print("\n=== System Information ===")
    cpu_count = os.cpu_count()
    memory = psutil.virtual_memory()
    print(f"CPU: {cpu_count} cores")
    print(f"RAM: {memory.total / (1024**3):.2f} GB total, {memory.available / (1024**3):.2f} GB available")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"GPU: {gpu_count} devices")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"  GPU #{i}: {gpu_name}, {gpu_mem:.2f} GB")
            print(f"  Current memory usage: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB allocated, {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB reserved")
    else:
        print("No GPU available, using CPU only")
    print("===========================\n")


def free_memory():
    """Aggressively free GPU and CPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                
        print(f"Memory after cleanup - " + 
              " ".join([f"GPU {i}: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB allocated" 
                       for i in range(torch.cuda.device_count())]))
    else:
        print("No CUDA available for memory cleanup")


def load_dataset(json_path, max_samples=None):
    """Load and process conversation dataset from JSON/JSONL file."""
    try:
        print(f"Loading dataset from {json_path}...")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            # First try as a regular JSON array or object
            try:
                data = json.load(f)
                # If it's not a list, make it a list
                if not isinstance(data, list):
                    data = [data]
            except json.JSONDecodeError:
                # If regular JSON fails, try as JSONL format (each line is a JSON object)
                f.seek(0)
                data = []
                for i, line in enumerate(f):
                    if max_samples is not None and i >= max_samples:
                        break
                    
                    if line.strip():
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
                
                if not data:
                    raise ValueError(f"Could not parse {json_path} as JSON or JSONL")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise
    
    # Extract and format conversations
    formatted_data = []
    for item in data:
        if "messages" in item:
            conversation = ""
            messages = item["messages"]
            
            for i, message in enumerate(messages):
                role = message.get("role", "")
                content = message.get("content", "")
                
                if role == "user":
                    conversation += f"<|user|>\n{content}\n"
                elif role == "assistant":
                    conversation += f"<|assistant|>\n{content}\n"
                elif role == "system":
                    conversation += f"<|system|>\n{content}\n"
            
            formatted_data.append({"text": conversation})
        # Handle legacy format with input/output fields
        elif "input" in item and "output" in item:
            conversation = f"<|user|>\n{item['input']}\n<|assistant|>\n{item['output']}\n"
            formatted_data.append({"text": conversation})
    
    print(f"Processed {len(formatted_data)} conversations from the dataset")
    
    # Create dataset and split into train/validation
    dataset = Dataset.from_list(formatted_data)
    train_val_split = dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)
    
    return {
        'train': train_val_split['train'],
        'validation': train_val_split['test']
    }


def tokenize_dataset(dataset, tokenizer, max_length=512):
    """Tokenize the dataset for training with a shorter sequence length to save memory."""
    def tokenize_function(examples):
        results = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Set labels equal to input_ids for causal language modeling
        results["labels"] = results["input_ids"].clone()
        
        # Set padding token ids to -100 so they don't contribute to the loss
        for i, label in enumerate(results["labels"]):
            mask = results["attention_mask"][i] == 0
            results["labels"][i][mask] = -100
            
        return results
    
    # Apply tokenization to each split using smaller batch size to save memory
    tokenized_dataset = {}
    for split in dataset:
        print(f"Tokenizing {split} split...")
        tokenized_dataset[split] = dataset[split].map(
            tokenize_function,
            batched=True,
            batch_size=8,  # Small batch size for tokenizing
            remove_columns=["text"],
            desc=f"Tokenizing {split} dataset"
        )
        free_memory()
    
    print("Dataset tokenized successfully.")
    return tokenized_dataset


def memory_efficient_fine_tune(
    model_path: str, 
    dataset_path: str, 
    output_dir: str, 
    num_epochs: int = 3,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
    learning_rate: float = 2e-4,
    max_length: int = 512,
    max_samples: int = None,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    save_strategy: str = "steps",
    save_steps: int = 50
):
    """
    Memory-efficient fine-tuning for Phi-4 using LoRA and 8-bit quantization.
    """
    print_system_info()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Ensure we have proper padding and EOS tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 8-bit quantization config for memory efficiency
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False
    )
    
    # Load model with quantization
    print(f"Loading model from {model_path} with 8-bit quantization...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map={'': 0},  # Place on GPU 0
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
    except Exception as e:
        print(f"Error loading model with quantization: {e}")
        print("Trying more aggressive memory saving options...")
        
        # More aggressive config with 4-bit
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
    
    # Prepare model for kbit training
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    
    # Define LoRA configuration
    print("Setting up LoRA adapters...")
    # For Phi-4, these are the typical attention layers that work well with LoRA
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False
    )
    
    # Get PEFT model
    model = get_peft_model(model, lora_config)
    free_memory()
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    # Load and process dataset
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_dataset(dataset_path, max_samples=max_samples)
    print(f"Dataset loaded: {len(dataset['train'])} training samples, {len(dataset['validation'])} validation samples")
    free_memory()
    
    # Tokenize the dataset
    print("Tokenizing dataset...")
    tokenized_dataset = tokenize_dataset(dataset, tokenizer, max_length=max_length)
    free_memory()
    
    # Prepare data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Training arguments
    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy=save_strategy,
        save_steps=save_steps if save_strategy == "steps" else None,
        save_total_limit=2,
        fp16=True,  # Use fp16 for faster training with quantized models
        report_to="none",  # Disable reporting to save memory
        gradient_checkpointing=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # Fine-tune the model
    print("Starting training...")
    try:
        trainer.train()
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("\nERROR: CUDA out of memory even with memory optimizations.")
            print("Try these extreme settings:")
            print("1. Set max_samples to 10 to first verify the pipeline works")
            print("2. Decrease max_length to 256")
            print("3. Use lora_r=4 (smaller adapter)")
        raise
    
    # Save the fine-tuned model (LoRA adapter)
    print(f"Saving LoRA adapter to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Training complete!")
    
    return model, tokenizer


def test_fine_tuned_model(model_path, base_model_path, test_input):
    """Test the fine-tuned model with a sample input."""
    print(f"Loading fine-tuned LoRA adapter from {model_path}...")
    
    # Load tokenizer from base model
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    
    # First load the base model with 8-bit quantization
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Then load the adapter on top of it
    from peft import PeftModel
    model = PeftModel.from_pretrained(base_model, model_path)
    
    # Format special SQL conversion input if needed
    if not test_input.startswith("Convert the following PL/SQL code to C# LINQ:"):
        formatted_input = f"Convert the following PL/SQL code to C# LINQ: \n\n```sql\n{test_input}\n```\n"
    else:
        formatted_input = test_input
    
    # Format with chat markers
    chat_formatted_input = f"<|user|>\n{formatted_input}\n<|assistant|>\n"
    
    # Generate a response
    inputs = tokenizer(chat_formatted_input, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    print("Generating response...")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=1024,  # Shorter to save memory
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.1,
            do_sample=True
        )
    
    # Decode the response
    response = tokenizer.decode(output[0], skip_special_tokens=False)
    
    # Extract just the assistant's response
    assistant_response = response.split("<|assistant|>")[-1].strip()
    
    return assistant_response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Memory-efficient fine-tuning for Phi-4 using LoRA and quantization")
    parser.add_argument("--model_path", type=str, default="microsoft/Phi-4", 
                        help="Path or name of the pre-trained model")
    parser.add_argument("--dataset_path", type=str, default="phi4_fine_tuning_dataset.json", 
                        help="Path to the input dataset file (JSON or JSONL format)")
    parser.add_argument("--output_dir", type=str, default="./phi4-lora-finetuned", 
                        help="Directory to save the fine-tuned LoRA adapter")
    parser.add_argument("--num_epochs", type=int, default=3, 
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, 
                        help="Training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, 
                        help="Number of updates steps to accumulate before backward pass")
    parser.add_argument("--learning_rate", type=float, default=2e-4, 
                        help="Learning rate for training")
    parser.add_argument("--max_length", type=int, default=512, 
                        help="Maximum sequence length")
    parser.add_argument("--max_samples", type=int, default=None, 
                        help="Maximum number of samples to use from the dataset")
    parser.add_argument("--lora_r", type=int, default=8, 
                        help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=16, 
                        help="LoRA alpha scaling factor")
    parser.add_argument("--lora_dropout", type=float, default=0.05, 
                        help="LoRA dropout probability")
    parser.add_argument("--save_strategy", type=str, default="steps", 
                        choices=["epoch", "steps", "no"],
                        help="When to save model checkpoints")
    parser.add_argument("--save_steps", type=int, default=50, 
                        help="Steps to save when using steps strategy")
    parser.add_argument("--test", action="store_true",
                        help="Test the fine-tuned model after training")
    parser.add_argument("--test_only", action="store_true",
                        help="Only test an existing fine-tuned model (no training)")
    parser.add_argument("--test_input", type=str, 
                        default="SELECT * FROM employees WHERE department_id = 10;",
                        help="Test input for the model")
    
    args = parser.parse_args()
    
    # Perform fine-tuning unless test_only is specified
    if not args.test_only:
        model, tokenizer = memory_efficient_fine_tune(
            model_path=args.model_path,
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            max_samples=args.max_samples,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            save_strategy=args.save_strategy,
            save_steps=args.save_steps
        )
    
    # Test the model if requested
    if args.test or args.test_only:
        print("\nTesting fine-tuned model...")
        response = test_fine_tuned_model(
            args.output_dir, 
            args.model_path if not args.test_only else args.model_path,
            args.test_input
        )
        print("\nModel response:")
        print(response)