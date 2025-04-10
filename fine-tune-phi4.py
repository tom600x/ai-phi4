#!/usr/bin/env python
# Fine-tuning script optimized for Phi-4 using high-performance hardware with extreme memory optimization
# Usage: python fine-tune-phi4.py

import json
import os
import gc
import torch
import psutil
import argparse
import numpy as np
from typing import Dict, List, Union, Optional
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Configure environment for optimal performance and memory management
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable parallel tokenization to save memory
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# Set PYTORCH_CUDA_ALLOC_CONF to avoid fragmentation and enable expandable segments
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64,expandable_segments:True"

# Add these to ensure PyTorch releases memory properly
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

def free_memory():
    """Aggressively free GPU and CPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    try:
        # For NVIDIA GPUs, try to reset the GPU memory stats
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
    except:
        pass
    
    print(f"Memory after cleanup - GPU: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated")

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

def validate_jsonl_dataset(dataset_path: str) -> bool:
    """
    Validate that the dataset file exists and has the expected format for Phi-4 fine-tuning.
    
    Args:
        dataset_path: Path to the dataset file
    
    Returns:
        True if valid, False otherwise
    """
    try:
        if not os.path.exists(dataset_path):
            print(f"Error: Dataset file {dataset_path} does not exist.")
            return False
            
        # Check if the file has content
        if os.path.getsize(dataset_path) == 0:
            print(f"Error: Dataset file {dataset_path} is empty.")
            return False
            
        # For JSONL files - check line by line
        with open(dataset_path, 'r', encoding='utf-8') as f:
            line_count = 0
            for i, line in enumerate(f):
                line_count += 1
                try:
                    # Parse each line as JSON
                    data = json.loads(line)
                    
                    # Validate expected Phi-4 format with messages array containing role and content
                    if 'messages' not in data:
                        print(f"Error: Line {i+1} is missing 'messages' field.")
                        return False
                    
                    messages = data['messages']
                    if not isinstance(messages, list) or len(messages) < 2:
                        print(f"Error: Line {i+1} should have at least 2 messages in the conversation.")
                        return False
                    
                    # Check that we have the right structure in each message
                    for msg in messages:
                        if not all(key in msg for key in ['role', 'content']):
                            print(f"Error: Line {i+1} has message missing 'role' or 'content'.")
                            return False
                        
                        if msg['role'] not in ['user', 'assistant', 'system']:
                            print(f"Error: Line {i+1} has invalid role: {msg['role']}")
                            return False
                    
                    # Check if conversation alternates between user and assistant
                    for j in range(len(messages) - 1):
                        if messages[j]['role'] == messages[j+1]['role']:
                            print(f"Warning: Line {i+1} has consecutive messages with the same role.")
                
                except json.JSONDecodeError:
                    print(f"Error: Line {i+1} is not valid JSON.")
                    return False
                
            print(f"Successfully validated {line_count} conversation examples.")
            return True
            
    except Exception as e:
        print(f"Error validating dataset: {str(e)}")
        return False

def load_phi4_dataset(dataset_path: str, max_samples=None):
    """
    Load and prepare the dataset for Phi-4 fine-tuning with memory optimizations.
    
    Args:
        dataset_path: Path to the dataset file
        max_samples: Maximum number of samples to load (for memory constraints)
    
    Returns:
        Processed dataset ready for training
    """
    print(f"Loading dataset from {dataset_path}...")
    
    # Load the dataset as JSONL
    raw_dataset = []
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            example = json.loads(line)
            raw_dataset.append(example)
    
    # Process the raw dataset into the format needed for fine-tuning
    processed_data = []
    
    # Special tokens for Phi-4 conversations
    for example in raw_dataset:
        # Format conversation according to the Phi-4 format
        formatted_text = ""
        for msg in example["messages"]:
            role = msg["role"]
            content = msg["content"]
            
            if role == "user":
                formatted_text += f"<|user|>\n{content}\n"
            elif role == "assistant":
                formatted_text += f"<|assistant|>\n{content}\n"
            elif role == "system":
                formatted_text += f"<|system|>\n{content}\n"
        
        processed_data.append({"text": formatted_text})
    
    free_memory()
    
    # Create a Dataset object
    dataset = Dataset.from_list(processed_data)
    
    # Split the dataset into train and validation
    train_val_split = dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)
    
    result_dataset = {
        'train': train_val_split['train'],
        'validation': train_val_split['test']
    }
    
    print(f"Dataset loaded and processed: {len(result_dataset['train'])} training examples, {len(result_dataset['validation'])} validation examples")
    return result_dataset

def tokenize_phi4_dataset(dataset, tokenizer, max_length=512):
    """
    Tokenize the dataset using the provided tokenizer with memory optimizations.
    
    Args:
        dataset: The dataset to tokenize
        tokenizer: The tokenizer to use
        max_length: Maximum sequence length
    
    Returns:
        Tokenized dataset
    """
    print(f"Tokenizing dataset with max_length={max_length}...")
    
    def tokenize_function(examples):
        texts = examples["text"]
        
        # Process in smaller chunks to save memory
        all_tokenized = {"input_ids": [], "attention_mask": [], "labels": []}
        chunk_size = 1  # Process one example at a time to minimize memory usage
        
        for i in range(0, len(texts), chunk_size):
            end_idx = min(i + chunk_size, len(texts))
            chunk = texts[i:end_idx]
            
            # Tokenize this small chunk
            tokenized = tokenizer(
                chunk,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            
            # Copy the tensors to lists to avoid keeping references to CUDA tensors
            for key in tokenized:
                all_tokenized.setdefault(key, []).extend(tokenized[key].cpu().tolist())
            
            # Set up labels equal to input_ids for causal language modeling
            all_tokenized.setdefault("labels", []).extend(tokenized["input_ids"].cpu().tolist())
            
            # Force cleanup after each chunk
            del tokenized
            free_memory()
            
        return all_tokenized
    
    # Apply tokenization to each split, with limited parallelism
    tokenized_dataset = {}
    for split in dataset:
        print(f"Tokenizing {split} split...")
        tokenized_dataset[split] = dataset[split].map(
            tokenize_function,
            batched=True,
            batch_size=4,  # Small batch size to reduce memory usage
            num_proc=1,    # Don't use multiple processes to save memory
            remove_columns=["text"],
            desc=f"Tokenizing {split} dataset"
        )
        free_memory()
    
    print("Dataset tokenized successfully.")
    return tokenized_dataset

def fine_tune_phi4(
    model_path: str, 
    dataset_path: str, 
    output_dir: str, 
    use_lora: bool = True,
    epochs: int = 3, 
    batch_size: int = 1,     # Reduced to 1
    gradient_accumulation_steps: int = 16,  # Increased to 16
    learning_rate: float = 2e-5,
    lora_r: int = 8,         # Reduced to 8
    lora_alpha: int = 16,    # Reduced to 16
    lora_dropout: float = 0.1,
    max_length: int = 512,   # Reduced to 512
    use_8bit: bool = True,   
    use_4bit: bool = False,  # New option for 4-bit quantization
    offload_modules: bool = True,  # Default to offload
    max_samples: int = None  # Limit number of training samples
):
    """
    Fine-tune Microsoft Phi-4 model with extreme memory optimization.
    """
    print_system_info()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Validate dataset first
    if not validate_jsonl_dataset(dataset_path):
        print("Dataset validation failed. Please check your dataset format.")
        return
    
    free_memory()
    
    print(f"Loading model: {model_path}")

    # Determine compute dtype based on available hardware
    compute_dtype = torch.float16
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        print("Using bfloat16 precision")
        compute_dtype = torch.bfloat16

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Make sure the tokenizer has padding token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = "</s>"
    
    # Configure memory-efficient loading
    device_map = "auto"
    if offload_modules:
        # Offload more layers to CPU
        device_map = {
            "model.embed_tokens": 0,
            "lm_head": 0,
            "model.norm": 0
        }
        
        # Offload most transformer blocks to CPU
        num_layers = 32  # Phi-4 typically has 32 layers
        for i in range(num_layers):
            # Keep only a few layers on GPU
            if i < 4 or i >= num_layers-4:
                device_map[f"model.layers.{i}"] = 0
            else:
                device_map[f"model.layers.{i}"] = "cpu"
                
        print("Using aggressive CPU offloading for model layers")
    
    # Configure quantization
    quantization_config = None
    if use_4bit:
        print("Using 4-bit quantization (most memory efficient)")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif use_8bit:
        print("Using 8-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=True
        )

    # Load model with extreme memory optimization
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            torch_dtype=compute_dtype,
            trust_remote_code=True,
            device_map=device_map,
            low_cpu_mem_usage=True,
            max_memory={0: "18GiB", "cpu": "32GiB"}  # Reduced GPU memory limit
        )
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("Trying with even more aggressive memory settings...")
        
        # Try again with more aggressive settings
        if not use_4bit:
            print("Switching to 4-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,  # Force float16
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,  # Force float16
            trust_remote_code=True,
            device_map="auto",  # Let the library decide mapping
            low_cpu_mem_usage=True,
            max_memory={0: "16GiB", "cpu": "32GiB"}  # Further reduce GPU memory
        )
    
    free_memory()
    
    # Set up LoRA if requested
    if use_lora:
        print("Setting up LoRA for parameter-efficient fine-tuning")
        if use_8bit or use_4bit:
            model = prepare_model_for_kbit_training(model)
            
        # Configure LoRA with more memory-efficient settings
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj"],  # Reduced target modules
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    free_memory()
    
    # Load dataset with potentially limited samples
    dataset = load_phi4_dataset(dataset_path, max_samples=max_samples)
    free_memory()
    
    # Tokenize dataset with memory optimizations
    tokenized_dataset = tokenize_phi4_dataset(dataset, tokenizer, max_length=max_length)
    free_memory()
    
    # Prepare data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We want causal language modeling, not masked
    )
    
    # Determine effective batch size
    num_gpus = max(1, torch.cuda.device_count())
    effective_batch_size = batch_size * gradient_accumulation_steps * num_gpus
    print(f"Effective batch size: {effective_batch_size} (batch_size={batch_size} × gradient_accumulation={gradient_accumulation_steps} × num_gpus={num_gpus})")
    
    # Configure training arguments with extreme memory-efficient settings
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=1,  # Set to minimum
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=50,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=1,  # Keep only 1 checkpoint to save space
        evaluation_strategy="steps",
        eval_steps=500,
        fp16=not (use_8bit or use_4bit) and compute_dtype == torch.float16,
        bf16=not (use_8bit or use_4bit) and compute_dtype == torch.bfloat16,
        report_to="tensorboard",
        run_name=f"phi4_finetune_{os.path.basename(output_dir)}",
        # Memory-optimized settings
        deepspeed=None,
        dataloader_num_workers=0,  # No parallel data loading
        group_by_length=False,  # Disable grouping to save memory
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        torch_compile=False,
        optim="adamw_torch",  # Use standard optimizer
        # Additional memory optimizations
        greater_is_better=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        hub_model_id=None,
        hub_strategy="every_save",
        remove_unused_columns=True,
        no_cuda=False,
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
    
    free_memory()
    
    # Train model
    print("\n=== Starting Training ===")
    
    try:
        trainer.train()
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("\nERROR: CUDA out of memory. Try the following:")
            print("1. Reduce batch_size to 1")
            print("2. Reduce max_length to 256")
            print("3. Use --use_4bit option for 4-bit quantization")
            print("4. Use --max_samples to limit the number of training examples")
            print("5. Try running on a machine with more GPU memory")
        raise
    
    # Save model
    print(f"\n=== Saving Model to {output_dir} ===")
    if use_lora:
        # For LoRA, save adapter weights
        model.save_pretrained(output_dir)
    else:
        # For full fine-tuning, save full model
        trainer.save_model(output_dir)
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    print(f"Model fine-tuning complete! Model saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Microsoft Phi-4 model with extreme memory optimization")
    parser.add_argument("--model_path", type=str, default="microsoft/Phi-4", help="Path or name of the Phi-4 model")
    parser.add_argument("--dataset_path", type=str, default="phi4_fine_tuning_dataset.json", help="Path to the JSONL dataset file")
    parser.add_argument("--output_dir", type=str, default="output/phi4-finetuned", help="Directory to save the fine-tuned model")
    parser.add_argument("--use_lora", action="store_true", default=True, help="Use LoRA for memory-efficient fine-tuning")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size per device")
    parser.add_argument("--gradient_accumulation", type=int, default=16, help="Number of steps to accumulate gradients")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for training")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout probability")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--use_8bit", action="store_true", default=True, help="Use 8-bit quantization")
    parser.add_argument("--use_4bit", action="store_true", help="Use 4-bit quantization (overrides 8-bit)")
    parser.add_argument("--offload_modules", action="store_true", default=True, help="Offload most model modules to CPU")
    parser.add_argument("--max_samples", type=int, help="Maximum number of samples to use for training (for memory constraints)")
    args = parser.parse_args()
    
    fine_tune_phi4(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        use_lora=args.use_lora,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        max_length=args.max_length,
        use_8bit=args.use_8bit,
        use_4bit=args.use_4bit,
        offload_modules=args.offload_modules,
        max_samples=args.max_samples
    )