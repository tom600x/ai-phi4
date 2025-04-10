#pip install transformers datasets torch

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import json
import os

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

def fine_tune_model(model_path, dataset_path, output_dir, epochs=3, batch_size=8, learning_rate=5e-5, use_local_model=True):
    """
    Fine-tune a Hugging Face model with a custom dataset - optimized for high-core CPU machine with large RAM.

    Args:
        model_path (str): Path to the pre-trained model on disk or model name on HF Hub
        dataset_path (str): Path to the dataset file (e.g., JSON, CSV).
        output_dir (str): Directory to save the fine-tuned model.
        epochs (int): Number of training epochs.
        batch_size (int): Training batch size (increased for high-memory machine).
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
    
    # Clear memory before loading model
    gc.collect()
    
    # Get system memory info for logging
    memory_info = psutil.virtual_memory()
    print(f"System memory: {memory_info.total / (1024**3):.1f}GB total, {memory_info.available / (1024**3):.1f}GB available")
    print(f"CPU count: {os.cpu_count()} cores")
    
    # Force CPU usage for the model (no GPU)
    print(f"Loading model from {'local path' if use_local_model else 'Hugging Face'}: {model_path}...")
    
    # CPU-optimized loading for high-memory machine
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        device_map="cpu",  # Explicitly use CPU
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
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
    
    # Check the dataset format - if input/output pairs, convert to text
    first_example = dataset['train'][0]
    if 'text' not in first_example and 'input' in first_example and 'output' in first_example:
        print("Converting input/output format to text format...")
        
        # Function to combine input and output into a single text field
        def convert_to_text_format(examples):
            texts = []
            for i, o in zip(examples['input'], examples['output']):
                # Format specifically for PL/SQL to LINQ conversion
                texts.append(f"<|user|>\n{i}\n<|assistant|>\n{o}\n")
            return {'text': texts}
        
        # Apply the conversion
        for split in dataset:
            dataset[split] = dataset[split].map(
                convert_to_text_format,
                batched=True,
                remove_columns=['input', 'output'],
                desc=f"Converting {split} split"
            )
        
        print("Dataset converted to text format.")
    
    # Verify dataset has text field now and print sample
    if 'text' not in dataset['train'][0]:
        print(f"Error: Could not process dataset format. Found fields: {list(dataset['train'][0].keys())}")
        return
    
    print("Sample data point:", dataset['train'][0]['text'][:100] + "...")
    
    # Create a larger max_length to take advantage of abundant RAM
    max_length = min(2048, tokenizer.model_max_length)  # Allow longer sequences for high-memory machine
    
    # Tokenize the dataset with larger max_length and prepare labels for causal language modeling
    def tokenize_function(examples):
        # Tokenize the text
        tokenized = tokenizer(
            examples['text'], 
            truncation=True, 
            padding='max_length', 
            max_length=max_length,
            return_tensors=None  # Return as python lists
        )
        
        # Set up labels for causal language modeling (same as input_ids)
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    print("Tokenizing dataset...")
    tokenized_dataset = {}
    for split in dataset:
        tokenized_dataset[split] = dataset[split].map(
            tokenize_function, 
            batched=True, 
            desc=f"Tokenizing {split} split"
        )
    
    # Define training arguments optimized for high-core CPU with large memory
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,  # Increased to 8 for high-memory machine
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        save_steps=500,
        save_total_limit=3,  # Can keep more checkpoints with high memory
        logging_dir=f'{output_dir}/logs',
        logging_steps=50,
        # CPU optimization parameters
        gradient_accumulation_steps=4,  # Reduced for higher batch size
        fp16=False,  # Disable mixed precision for CPU
        bf16=False,  # Disable bfloat16 for CPU
        optim="adamw_torch",
        ddp_find_unused_parameters=False,
        dataloader_num_workers=min(16, os.cpu_count() // 4),  # Use 1/4 of available CPUs for data loading
        group_by_length=True,
        weight_decay=0.01,
        no_cuda=True,  # Force no CUDA
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
        batch_size=8,
        learning_rate=5e-5,
        use_local_model=True  # Set to True to use a local model path
    )