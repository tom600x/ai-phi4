#!/usr/bin/env python
# Fine-tuning script optimized for Phi-4 using high-performance hardware
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

# Configure environment for optimal performance
os.environ["TOKENIZERS_PARALLELISM"] = "true"  # Enable parallel tokenization
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Ensure consistent GPU ordering
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"  # Optimize CUDA memory allocation

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

def load_phi4_dataset(dataset_path: str):
    """
    Load and prepare the dataset for Phi-4 fine-tuning.
    
    Args:
        dataset_path: Path to the dataset file
    
    Returns:
        Processed dataset ready for training
    """
    print(f"Loading dataset from {dataset_path}...")
    
    # Load the dataset as JSONL
    raw_dataset = []
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
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

def tokenize_phi4_dataset(dataset, tokenizer, max_length=2048):
    """
    Tokenize the dataset using the provided tokenizer.
    
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
        
        # Tokenize and prepare for training
        tokenized = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Set up labels equal to input_ids for causal language modeling
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    # Apply tokenization to each split
    tokenized_dataset = {}
    for split in dataset:
        tokenized_dataset[split] = dataset[split].map(
            tokenize_function,
            batched=True,
            num_proc=os.cpu_count(),  # Use all available CPUs for tokenization
            remove_columns=["text"],
            desc=f"Tokenizing {split} dataset"
        )
    
    print("Dataset tokenized successfully.")
    return tokenized_dataset

def fine_tune_phi4(
    model_path: str, 
    dataset_path: str, 
    output_dir: str, 
    use_lora: bool = True,
    epochs: int = 3, 
    batch_size: int = 4, 
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-5,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    max_length: int = 2048,
    use_8bit: bool = False,
):
    """
    Fine-tune Microsoft Phi-4 model with a custom dataset.

    Args:
        model_path: Path or name of the Phi-4 model
        dataset_path: Path to the JSONL dataset file
        output_dir: Directory to save the fine-tuned model
        use_lora: Whether to use LoRA for memory-efficient fine-tuning
        epochs: Number of training epochs
        batch_size: Training batch size per device
        gradient_accumulation_steps: Number of steps to accumulate gradients
        learning_rate: Learning rate for training
        lora_r: LoRA attention dimension
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout probability
        max_length: Maximum sequence length
        use_8bit: Whether to use 8-bit quantization
    """
    print_system_info()
    
    # Validate dataset first
    if not validate_jsonl_dataset(dataset_path):
        print("Dataset validation failed. Please check your dataset format.")
        return
    
    # Clean up memory before loading model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"Loading model: {model_path}")

    # Determine compute dtype based on available hardware
    compute_dtype = torch.float16
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        print("Using bfloat16 precision")
        compute_dtype = torch.bfloat16

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Make sure the tokenizer has padding, unk, sep and mask tokens
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = "</s>"
    
    # Configure quantization if requested
    quantization_config = None
    if use_8bit:
        print("Using 8-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=True
        )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        torch_dtype=compute_dtype,
        trust_remote_code=True,
        device_map="auto"  # Automatically determine optimal device mapping
    )
    
    # Set up LoRA if requested
    if use_lora:
        print("Setting up LoRA for parameter-efficient fine-tuning")
        if use_8bit:
            model = prepare_model_for_kbit_training(model)
            
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Load dataset
    dataset = load_phi4_dataset(dataset_path)
    
    # Tokenize dataset
    tokenized_dataset = tokenize_phi4_dataset(dataset, tokenizer, max_length=max_length)
    
    # Prepare data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We want causal language modeling, not masked
    )
    
    # Determine effective batch size
    num_gpus = max(1, torch.cuda.device_count())
    effective_batch_size = batch_size * gradient_accumulation_steps * num_gpus
    print(f"Effective batch size: {effective_batch_size} (batch_size={batch_size} × gradient_accumulation={gradient_accumulation_steps} × num_gpus={num_gpus})")
    
    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
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
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3,
        evaluation_strategy="epoch",
        eval_steps=500,
        fp16=not use_8bit and compute_dtype == torch.float16,  # Use fp16 only if not using 8-bit and not using bfloat16
        bf16=not use_8bit and compute_dtype == torch.bfloat16,  # Use bf16 only if not using 8-bit and using bfloat16
        report_to="tensorboard",
        run_name=f"phi4_finetune_{os.path.basename(output_dir)}",
        # High performance settings
        deepspeed=None,  # Optional: Use DeepSpeed for even better performance (requires separate config)
        dataloader_num_workers=os.cpu_count() // 2,  # Use half of available CPUs for data loading
        group_by_length=True,  # Group sequences of similar lengths to minimize padding
        gradient_checkpointing=True,  # Enable gradient checkpointing to save memory
        ddp_find_unused_parameters=False,  # More efficient distributed training
        torch_compile=False,  # Can be enabled for PyTorch 2.0+ for more speed
        optim="adamw_torch_fused"  # Use fused optimizer for better performance
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
    
    # Free memory before training
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Train model
    print("\n=== Starting Training ===")
    trainer.train()
    
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
    parser = argparse.ArgumentParser(description="Fine-tune Microsoft Phi-4 model")
    parser.add_argument("--model_path", type=str, default="microsoft/Phi-4", help="Path or name of the Phi-4 model")
    parser.add_argument("--dataset_path", type=str, default="phi4_fine_tuning_dataset.json", help="Path to the JSONL dataset file")
    parser.add_argument("--output_dir", type=str, default="output/phi4-finetuned", help="Directory to save the fine-tuned model")
    parser.add_argument("--use_lora", action="store_true", default=True, help="Use LoRA for memory-efficient fine-tuning")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size per device")
    parser.add_argument("--gradient_accumulation", type=int, default=4, help="Number of steps to accumulate gradients")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for training")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout probability")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--use_8bit", action="store_true", help="Use 8-bit quantization")
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
        use_8bit=args.use_8bit
    )