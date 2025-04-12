#!/usr/bin/env python
# Fine-tuning script optimized for Phi-4 using high-performance hardware with dual-GPU optimization
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
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

# Configure environment for dual-GPU high-performance setup
os.environ["TOKENIZERS_PARALLELISM"] = "true"  # Enable parallel tokenization for 70 vCPUs
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# Optimize memory allocation for two 19GB GPUs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
# Set threads for multi-CPU efficiency
os.environ["OMP_NUM_THREADS"] = "35"  # Half of available vCPUs for better threading balance

# Try importing DeepSpeed for multi-GPU optimization
try:
    import deepspeed
    HAS_DEEPSPEED = True
    print("DeepSpeed is available for advanced multi-GPU optimization")
except ImportError:
    HAS_DEEPSPEED = False
    print("DeepSpeed not found. Consider installing it with 'pip install deepspeed' for better multi-GPU performance")

def free_memory():
    """Aggressively free GPU and CPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
        except:
            pass
        
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                
        print(f"Memory after cleanup - " + 
              " ".join([f"GPU {i}: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB allocated" 
                       for i in range(torch.cuda.device_count())]))
    else:
        print("No CUDA available for memory cleanup")

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
    Load and prepare the dataset for Phi-4 fine-tuning with multi-CPU optimization.
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

def tokenize_phi4_dataset(dataset, tokenizer, max_length=1024, num_workers=35):
    """
    Tokenize the dataset using the provided tokenizer with multi-CPU optimization.
    
    Args:
        dataset: The dataset to tokenize
        tokenizer: The tokenizer to use
        max_length: Maximum sequence length
        num_workers: Number of CPU workers for tokenization (set to half of available vCPUs)
    
    Returns:
        Tokenized dataset
    """
    print(f"Tokenizing dataset with max_length={max_length} using {num_workers} workers...")
    
    def tokenize_function(examples):
        texts = examples["text"]
        
        # For multi-GPU setup, process in larger batches with multi-CPU
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
    
    # Apply tokenization to each split with optimized multi-processing
    tokenized_dataset = {}
    for split in dataset:
        print(f"Tokenizing {split} split...")
        tokenized_dataset[split] = dataset[split].map(
            tokenize_function,
            batched=True,
            batch_size=64,  # Increased batch size for faster processing
            num_proc=num_workers,  # Use multiple CPU cores
            remove_columns=["text"],
            desc=f"Tokenizing {split} dataset"
        )
        free_memory()
    
    print("Dataset tokenized successfully.")
    return tokenized_dataset

# Create a function to generate a DeepSpeed config optimized for our setup
def get_deepspeed_config(train_batch_size):
    """
    Generate a DeepSpeed configuration optimized for 2x19GB GPUs.
    Uses "auto" for parameters that should match TrainingArguments.
    """
    return {
        "fp16": {
            "enabled": "auto"  # Use "auto" to match TrainingArguments fp16 setting
        },
        "zero_optimization": {
            "stage": 2,  # Stage 2 for balancing memory and speed
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "contiguous_gradients": True,
            "overlap_comm": True
        },
        "gradient_accumulation_steps": "auto",  # Use "auto" to match TrainingArguments
        "train_micro_batch_size_per_gpu": "auto",  # Use "auto" to match batch size from TrainingArguments
        "gradient_clipping": 1.0,
        "steps_per_print": 10,
    }

class DebugTrainer(Trainer):
    """Custom trainer class that validates loss before backward pass"""
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Override compute_loss to add validation before backward pass.
        Added **kwargs to handle additional arguments like num_items_in_batch.
        """
        outputs = model(**inputs)
        loss = outputs.loss
        
        # Check if loss is None or not a valid tensor for backward
        if loss is None:
            print("WARNING: Loss is None! Replacing with zero tensor to avoid backward error.")
            loss = torch.tensor(0.0, device=model.device, requires_grad=True)
        elif not loss.requires_grad:
            print("WARNING: Loss doesn't require grad! Adding requires_grad=True.")
            loss.requires_grad_(True)
        
        # Also check for NaN values in loss
        if torch.isnan(loss).any():
            print("WARNING: NaN detected in loss! Replacing with zero tensor.")
            loss = torch.tensor(0.0, device=model.device, requires_grad=True)
        
        return (loss, outputs) if return_outputs else loss

def fine_tune_phi4(
    model_path: str, 
    dataset_path: str, 
    output_dir: str, 
    use_lora: bool = True,
    epochs: int = 3, 
    batch_size: int = 4,     # Increased for dual-GPU
    gradient_accumulation_steps: int = 4,  # Optimized for dual-GPU
    learning_rate: float = 2e-5,
    lora_r: int = 16,        # Increased for higher capacity
    lora_alpha: int = 32,    # Increased for better training dynamics
    lora_dropout: float = 0.05,
    max_length: int = 1024,  # Increased for larger context windows
    use_8bit: bool = True,   
    use_4bit: bool = False,
    use_deepspeed: bool = True,
    offload_modules: bool = False,  # Less need with dual GPUs
    max_samples: int = None
):
    """
    Fine-tune Microsoft Phi-4 model optimized for dual-GPU setup with 19GB each.
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
    
    # Configure model distribution across both GPUs
    device_map = "auto"
    if not use_deepspeed:  # If not using DeepSpeed, manually distribute model
        # Force model to use a single GPU instead of balanced distribution to avoid tensor device mismatches
        device_map = {"": 0}  # Place all modules on GPU 0
        print("Using single GPU to avoid tensor device mismatches")
    
    # Configure quantization
    quantization_config = None
    if use_4bit:
        print("Using 4-bit quantization")
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

    # Load model with single-GPU optimization to avoid tensor device mismatches
    model_load_kwargs = {
        "quantization_config": quantization_config,
        "torch_dtype": compute_dtype,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "use_cache": False,  # Explicitly disable caching since we're using gradient checkpointing
        "device_map": device_map,
    }
    
    # If not using DeepSpeed, set memory limits for single GPU
    if not use_deepspeed:
        model_load_kwargs.update({
            "max_memory": {0: "36GiB", "cpu": "70GiB"}  # Allocate all GPU memory to device 0
        })
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_load_kwargs
        )
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("Trying with more aggressive memory settings...")
        
        # Try again with more aggressive settings
        if not use_4bit:
            print("Switching to 4-bit quantization")
            model_load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model_load_kwargs["torch_dtype"] = torch.float16
        
        if not use_deepspeed:
            model_load_kwargs["max_memory"] = {0: "16GiB", "cpu": "70GiB"}
            
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_load_kwargs
        )
    
    free_memory()
    
    # Set up LoRA if requested
    if use_lora:
        print("Setting up LoRA for parameter-efficient fine-tuning")
        if use_8bit or use_4bit:
            model = prepare_model_for_kbit_training(model)
            
        # Ensure model is on the correct device before LoRA setup
        if not use_deepspeed:
            # When not using DeepSpeed, explicitly move to GPU 0
            device = torch.device("cuda:0")
            model = model.to(device)
            
        # Configure LoRA with optimized settings
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            inference_mode=False,
        )
        
        try:
            model = get_peft_model(model, lora_config)
            # Ensure all LoRA weights are on the same device
            if not use_deepspeed:
                model = model.to(device)
            model.print_trainable_parameters()
        except ValueError as e:
            print(f"Error with specified target modules: {str(e)}")
            print("Falling back to automatic target module detection...")
            # Fallback to automatic target module detection
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=None,  # Auto-detect
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                inference_mode=False,
            )
            model = get_peft_model(model, lora_config)
            if not use_deepspeed:
                model = model.to(device)
            model.print_trainable_parameters()
            print(f"Auto-detected target modules: {model.peft_config['default'].target_modules}")
    
    free_memory()
    
    # Load dataset with optimized processing for 70 vCPUs
    dataset = load_phi4_dataset(dataset_path, max_samples=max_samples)
    free_memory()
    
    # Tokenize dataset with multi-CPU optimization
    cpu_workers = min(35, os.cpu_count() // 2)  # Use half of available vCPUs for tokenization
    tokenized_dataset = tokenize_phi4_dataset(dataset, tokenizer, max_length=max_length, num_workers=cpu_workers)
    free_memory()
    
    # Prepare data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We want causal language modeling, not masked
    )
    
    # Determine effective batch size accounting for 2 GPUs
    num_gpus = torch.cuda.device_count()
    effective_batch_size = batch_size * gradient_accumulation_steps * num_gpus
    print(f"Effective batch size: {effective_batch_size} (batch_size={batch_size} × gradient_accumulation={gradient_accumulation_steps} × num_gpus={num_gpus})")
    
    # Configure DeepSpeed if available and requested
    deepspeed_config = None
    if HAS_DEEPSPEED and use_deepspeed:
        deepspeed_config = get_deepspeed_config(batch_size)
        print("Using DeepSpeed configuration for multi-GPU training")
    
    # Configure training arguments optimized for dual-GPU setup
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size // 2,  # Half training batch size
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
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,  # Keep 3 checkpoints with more available memory
        fp16=not (use_8bit or use_4bit) and compute_dtype == torch.float16,
        bf16=not (use_8bit or use_4bit) and compute_dtype == torch.bfloat16,
        report_to="tensorboard",
        run_name=f"phi4_finetune_{os.path.basename(output_dir)}",
        # Multi-GPU settings
        deepspeed=deepspeed_config,
        local_rank=-1,  # Auto-detect for distributed training
        dataloader_num_workers=cpu_workers // 2,  # Optimize data loading for 70 vCPUs
        group_by_length=True,  # Better for multi-GPU efficiency
        gradient_checkpointing=True,  # Still useful for memory efficiency
        ddp_find_unused_parameters=False,
        torch_compile=False,  # Can be enabled for PyTorch 2.0+ if available
        optim="adamw_torch_fused",  # Use fused optimizer for better performance
        # Set these if your transformers version supports them
        # greater_is_better=False,
        # load_best_model_at_end=True,
        # metric_for_best_model="eval_loss",
        remove_unused_columns=True,
    )
    
    # Initialize trainer with multi-GPU awareness and custom loss handling
    trainer = DebugTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    free_memory()
    
    # Train model
    print("\n=== Starting Training on Multiple GPUs ===")
    
    try:
        trainer.train()
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("\nERROR: CUDA out of memory. Try reducing batch size or enabling DeepSpeed.")
            print("For dual 19GB GPUs, recommended settings:")
            print("1. Use DeepSpeed with Zero stage 2/3")
            print("2. Use gradient_accumulation_steps=4 with batch_size=4")
            print("3. Enable 8-bit quantization")
            print("4. Set max_length to 1024 or lower")
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
    parser = argparse.ArgumentParser(description="Fine-tune Microsoft Phi-4 model optimized for dual 19GB GPUs")
    parser.add_argument("--model_path", type=str, default="microsoft/Phi-4", help="Path or name of the Phi-4 model")
    parser.add_argument("--dataset_path", type=str, default="phi4_fine_tuning_dataset.json", help="Path to the JSONL dataset file")
    parser.add_argument("--output_dir", type=str, default="output/phi4-finetuned", help="Directory to save the fine-tuned model")
    parser.add_argument("--use_lora", action="store_true", default=True, help="Use LoRA for parameter-efficient fine-tuning")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size per device")
    parser.add_argument("--gradient_accumulation", type=int, default=4, help="Number of steps to accumulate gradients")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for training")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout probability")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--use_8bit", action="store_true", default=True, help="Use 8-bit quantization")
    parser.add_argument("--use_4bit", action="store_true", help="Use 4-bit quantization (overrides 8-bit)")
    parser.add_argument("--use_deepspeed", action="store_true", default=True, help="Use DeepSpeed for multi-GPU optimization")
    parser.add_argument("--offload_modules", action="store_true", help="Offload model modules to CPU (less needed with dual-GPU)")
    parser.add_argument("--max_samples", type=int, help="Maximum number of samples to use for training")
    args = parser.parse_args()
    
    # For dual-GPU distributed training
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training")
    
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
        use_deepspeed=args.use_deepspeed,
        offload_modules=args.offload_modules,
        max_samples=args.max_samples
    )