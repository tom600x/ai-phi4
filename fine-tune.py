#pip install transformers datasets torch

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, Dataset, concatenate_datasets
import json
import os

# Disable tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def validate_dataset(dataset_path):
    """
    Validates that the dataset at the given path exists and has the expected format.
    
    Args:
        dataset_path (str): Path to the dataset file
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        # Check if file exists
        if not os.path.exists(dataset_path):
            print(f"Error: Dataset file {dataset_path} does not exist.")
            return False
        
        # For JSON files
        if dataset_path.endswith('.json'):
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Check if it's a list of records or has a specific structure
            if isinstance(data, list):
                if not data:
                    print("Error: Dataset is empty.")
                    return False
                
                # Check for input/output fields (typical for instruction tuning datasets)
                if all(isinstance(item, dict) and 'input' in item and 'output' in item for item in data):
                    print("Found dataset with input/output pairs format.")
                    return True
                # Check for text field (typical for general language model datasets)
                elif all(isinstance(item, dict) and 'text' in item for item in data):
                    print("Found dataset with text field format.")
                    return True
                else:
                    print(f"Error: Dataset must contain records with either 'text' or 'input'/'output' fields.")
                    print(f"Found structure: {list(data[0].keys()) if data else 'empty list'}")
                    return False
            elif isinstance(data, dict):
                if 'train' not in data:
                    print(f"Error: Dataset in dict format should have a 'train' key.")
                    return False
                if not data['train']:
                    print(f"Error: Dataset's train split is empty.")
                    return False
                
                # Check structure of train records
                train_records = data['train']
                if isinstance(train_records, list) and train_records:
                    first_record = train_records[0]
                    if isinstance(first_record, dict) and ('input' in first_record and 'output' in first_record) or 'text' in first_record:
                        return True
                
                print(f"Error: Dataset's train records must contain either 'text' or 'input'/'output' fields.")
                return False
        
        return True
    except json.JSONDecodeError:
        print(f"Error: {dataset_path} is not a valid JSON file.")
        return False
    except Exception as e:
        print(f"Error validating dataset: {str(e)}")
        return False

def fine_tune_model(model_path, dataset_path, output_dir, epochs=3, batch_size=1, learning_rate=5e-5, use_local_model=True):
    """
    Fine-tune a Hugging Face model with a custom dataset - ultra memory-efficient version.

    Args:
        model_path (str): Path to the pre-trained model on disk or model name on HF Hub
        dataset_path (str): Path to the dataset file (e.g., JSON, CSV).
        output_dir (str): Directory to save the fine-tuned model.
        epochs (int): Number of training epochs.
        batch_size (int): Training batch size (reduced to minimum).
        learning_rate (float): Learning rate for training.
        use_local_model (bool): Whether to load the model from a local path (True) or HF Hub (False).
    """
    # Validate the dataset first
    if not validate_dataset(dataset_path):
        print("Dataset validation failed. Please check your dataset format.")
        return
    
    # Import necessary modules
    import gc
    import torch
    import os
    import psutil
    
    # Aggressive memory management - force garbage collection
    gc.collect()
    
    # Get system memory info for logging
    memory_info = psutil.virtual_memory()
    print(f"System memory: {memory_info.total / (1024**3):.1f}GB total, {memory_info.available / (1024**3):.1f}GB available")
    print(f"CPU count: {os.cpu_count()} cores")
    
    # Set memory management environment variables
    os.environ["PYTORCH_CPU_ALLOC_CONF"] = "max_split_size_mb:128"
    
    print(f"Loading model from {'local path' if use_local_model else 'Hugging Face'}: {model_path}...")
    
    # CPU-optimized loading with extreme memory efficiency
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32,  # Use standard precision for better compatibility
        device_map={"": "cpu"},  # Force CPU for all modules
    )
    
    # Free up memory after model loading
    gc.collect()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Ensure the tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load the dataset
    print(f"Loading dataset from {dataset_path}...")
    if dataset_path.endswith('.json'):
        try:
            # Try the standard method first
            dataset = load_dataset('json', data_files=dataset_path)
        except Exception as e:
            print(f"Standard JSON loading failed: {str(e)}. Trying manual load...")
            # Fallback to manual loading
            with open(dataset_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Check if it's a plain list or has a train/test structure
            if isinstance(json_data, list):
                # Convert the list to a Dataset object
                dataset = {'train': Dataset.from_dict({'input': [item['input'] for item in json_data], 
                                                     'output': [item['output'] for item in json_data]})}
            else:
                print(f"Error: Unsupported JSON structure: {type(json_data)}")
                return
    elif dataset_path.endswith('.csv'):
        dataset = load_dataset('csv', data_files=dataset_path)
    else:
        raise ValueError(f"Unsupported dataset format for {dataset_path}. Use .json or .csv")
    
    # Create validation split if it doesn't exist
    if 'validation' not in dataset:
        train_testvalid = dataset['train'].train_test_split(test_size=0.2, seed=42)
        test_valid = train_testvalid['test'].train_test_split(test_size=0.5, seed=42)
        dataset = {
            'train': train_testvalid['train'], 
            'test': test_valid['test'], 
            'validation': test_valid['train']
        }
    
    print("Dataset structure:", {k: len(v) for k, v in dataset.items()})
    
    # Process the dataset in smaller chunks to save memory
    print("Processing dataset in smaller chunks to conserve memory...")
    
    # Function to process dataset in chunks
    def process_dataset_in_chunks(dataset, chunk_size=100):
        for split in dataset:
            total_samples = len(dataset[split])
            chunk_datasets = []
            
            for i in range(0, total_samples, chunk_size):
                end_idx = min(i + chunk_size, total_samples)
                print(f"Processing {split} chunk {i}-{end_idx} of {total_samples}")
                chunk = dataset[split].select(range(i, end_idx))
                
                # Process this chunk
                if 'text' not in chunk[0] and 'input' in chunk[0] and 'output' in chunk[0]:
                    chunk = chunk.map(
                        lambda example: {'text': f"<|user|>\n{example['input']}\n<|assistant|>\n{example['output']}\n"},
                        remove_columns=['input', 'output']
                    )
                
                # Tokenize the chunk
                tokenized_chunk = chunk.map(
                    lambda example: {
                        **tokenizer(
                            example['text'],
                            truncation=True,
                            padding='max_length',
                            max_length=max_length,
                        ),
                        'labels': tokenizer(
                            example['text'],
                            truncation=True,
                            padding='max_length',
                            max_length=max_length,
                        )['input_ids']
                    },
                    desc=f"Tokenizing {split} chunk"
                )
                
                chunk_datasets.append(tokenized_chunk)
                
                # Force garbage collection after each chunk
                gc.collect()
            
            # Combine all chunks
            dataset[split] = concatenate_datasets(chunk_datasets)
            
        return dataset
    
    # Create a much smaller max_length to reduce memory requirements
    max_length = min(512, tokenizer.model_max_length)  # Strict limit to 512 tokens for memory efficiency
    
    # Process dataset in chunks
    tokenized_dataset = process_dataset_in_chunks(dataset, chunk_size=100)
    
    # Force garbage collection before training
    gc.collect()
    
    # Define training arguments optimized for extreme memory constraints
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,  # Minimum batch size
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        save_steps=2_000,
        save_total_limit=1,  # Keep only 1 checkpoint to save memory
        logging_dir=f'{output_dir}/logs',
        logging_steps=100,
        # Ultra memory-efficient settings
        gradient_accumulation_steps=16,  # Accumulate gradients over many steps
        fp16=False,
        bf16=False,
        optim="adamw_torch",
        ddp_find_unused_parameters=False,
        dataloader_num_workers=1,  # Minimal parallelism to save memory
        group_by_length=True,
        weight_decay=0.01,
        no_cuda=True,
        # Do not keep unused elements in memory
        remove_unused_columns=True,
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
    )

    # Fine-tune the model
    print("Starting training...")
    
    # More aggressive memory cleanup
    gc.collect()
    
    trainer.train()

    # Save the fine-tuned model
    print(f"Saving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Model fine-tuned and saved to {output_dir}")

# Example usage
if __name__ == "__main__":
    fine_tune_model(
        model_path="/home/TomAdmin/phi-3-mini-128k-instruct",  # Local model path
        dataset_path="/home/TomAdmin/ai-phi4/fine_tuning_dataset.json",
        output_dir="/home/TomAdmin/output-model/phi-3-tuned",
        epochs=3,
        batch_size=1,
        learning_rate=5e-5,
        use_local_model=True  # Set to True to use a local model path
    )