#!/usr/bin/env python
# Full model fine-tuning script for Microsoft Phi-4
# This script performs complete model fine-tuning (not just LoRA) and saves the full trained model
# Adapted from existing fine-tuning scripts for optimal compatibility with phi4_fine_tuning_dataset.json

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
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import argparse
import numpy as np

# Set environment variable to help with memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

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

def tokenize_dataset(dataset, tokenizer, max_length=2048):
    """Tokenize the dataset for training."""
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
    
    # Apply tokenization to each split
    tokenized_dataset = {}
    for split in dataset:
        print(f"Tokenizing {split} split...")
        tokenized_dataset[split] = dataset[split].map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc=f"Tokenizing {split} dataset"
        )
        free_memory()
    
    print("Dataset tokenized successfully.")
    return tokenized_dataset

def full_fine_tune_phi4(
    model_path: str, 
    dataset_path: str, 
    output_dir: str, 
    num_epochs: int = 3,
    batch_size: int = 1,  # Reduced default batch size
    gradient_accumulation_steps: int = 16,  # Increased gradient accumulation
    learning_rate: float = 2e-5,
    max_length: int = 1024,  # Reduced sequence length
    max_samples: int = None,
    save_strategy: str = "epoch",
    eval_steps: int = 500
):
    """
    Perform full fine-tuning on the Phi-4 model with the entire model's parameters.
    """
    print_system_info()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine appropriate compute type based on available hardware
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8 else torch.float16
    
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Ensure we have proper padding and EOS tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading model from {model_path}...")
    
    # More memory-efficient model loading with low_cpu_mem_usage and offload options
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=compute_dtype,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            offload_folder="offload_folder"
        )
    except Exception as e:
        print(f"Error loading model with device_map='auto': {e}")
        print("Trying with more conservative device map settings...")
        
        # Try loading with a more explicit device map if auto fails
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=compute_dtype,
            device_map="balanced",
            max_memory={0: "18GB", "cpu": "32GB"},
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
    
    # Prepare model for training
    model.config.use_cache = False  # Important for training efficiency
    
    free_memory()
    
    # Load and tokenize dataset
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
        mlm=False  # We want causal language modeling, not masked
    )
    
    # Define training arguments with optimized settings for Phi-4
    # Note: Removed evaluation_strategy parameter which is causing the error
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=10,
        save_strategy=save_strategy,
        save_total_limit=2,
        fp16=torch.cuda.is_available() and compute_dtype == torch.float16,
        bf16=torch.cuda.is_available() and compute_dtype == torch.bfloat16,
        report_to="tensorboard",
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,  # To help with memory usage
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        # Use these if your transformers version supports them:
        # eval_steps=eval_steps,
        # load_best_model_at_end=True, 
        # metric_for_best_model="loss",
        # greater_is_better=False,
    )
    
    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Fine-tune the model
    print("Starting training...")
    try:
        trainer.train()
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("\nERROR: CUDA out of memory. Try reducing batch size or using gradient accumulation.")
            print("Recommended settings:")
            print("1. Reduce batch_size to 1 or 2")
            print("2. Increase gradient_accumulation_steps to 8 or 16")
            print("3. Reduce max_length if your sequences allow it")
        raise
    
    # Save the fine-tuned model
    print(f"Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Training complete!")
    
    return model, tokenizer

def test_fine_tuned_model(model_path, test_input):
    """Test the fine-tuned model with a sample input."""
    print(f"Loading fine-tuned model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    
    # Format special SQL conversion input if needed
    if not test_input.startswith("Convert the following PL/SQL code to C# LINQ:"):
        formatted_input = f"Convert the following PL/SQL code to C# LINQ: \n\n```sql\n{test_input}\n```\n"
    else:
        formatted_input = test_input
    
    # Format with chat markers
    chat_formatted_input = f"<|user|>\n{formatted_input}\n<|assistant|>\n"
    
    # Generate a response
    inputs = tokenizer(chat_formatted_input, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    print("Generating response...")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=2048,
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
    parser = argparse.ArgumentParser(description="Full fine-tune the Microsoft Phi-4 model (complete model, not just LoRA)")
    parser.add_argument("--model_path", type=str, default="microsoft/Phi-4", 
                        help="Path or name of the pre-trained model")
    parser.add_argument("--dataset_path", type=str, default="phi4_fine_tuning_dataset.json", 
                        help="Path to the input dataset file (JSON or JSONL format)")
    parser.add_argument("--output_dir", type=str, default="./phi4-full-finetuned", 
                        help="Directory to save the fine-tuned model")
    parser.add_argument("--num_epochs", type=int, default=3, 
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, 
                        help="Training batch size (default: 1 to avoid OOM errors)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, 
                        help="Number of updates steps to accumulate before backward pass")
    parser.add_argument("--learning_rate", type=float, default=2e-5, 
                        help="Learning rate for training")
    parser.add_argument("--max_length", type=int, default=1024, 
                        help="Maximum sequence length (default: 1024 to reduce memory usage)")
    parser.add_argument("--max_samples", type=int, default=None, 
                        help="Maximum number of samples to use from the dataset")
    parser.add_argument("--save_strategy", type=str, default="epoch", 
                        choices=["epoch", "steps", "no"],
                        help="When to save model checkpoints")
    parser.add_argument("--eval_steps", type=int, default=500, 
                        help="Number of steps between evaluations")
    parser.add_argument("--test", action="store_true",
                        help="Test the fine-tuned model after training")
    parser.add_argument("--test_only", action="store_true",
                        help="Only test an existing fine-tuned model (no training)")
    parser.add_argument("--test_input", type=str, 
                        default="SELECT * FROM employees WHERE department_id = 10;",
                        help="Test input for the model")
    parser.add_argument("--offload_to_cpu", action="store_true", default=True,
                        help="Offload model layers to CPU to reduce GPU memory usage")
    
    args = parser.parse_args()
    
    # Perform fine-tuning unless test_only is specified
    if not args.test_only:
        model, tokenizer = full_fine_tune_phi4(
            model_path=args.model_path,
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            max_samples=args.max_samples,
            save_strategy=args.save_strategy,
            eval_steps=args.eval_steps
        )
    
    # Test the model if requested
    if args.test or args.test_only:
        print("\nTesting fine-tuned model...")
        response = test_fine_tuned_model(args.output_dir, args.test_input)
        print("\nModel response:")
        print(response)