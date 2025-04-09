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

def fine_tune_model(model_path, dataset_path, output_dir, epochs=3, batch_size=1, learning_rate=5e-5, use_local_model=True):
    """
    Fine-tune a Hugging Face model with a custom dataset with aggressive memory optimizations.

    Args:
        model_path (str): Path to the pre-trained model on disk or model name on HF Hub
        dataset_path (str): Path to the dataset file (e.g., JSON, CSV).
        output_dir (str): Directory to save the fine-tuned model.
        epochs (int): Number of training epochs.
        batch_size (int): Training batch size (default reduced to 1).
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
    
    # Clear memory before loading model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Set up memory-saving environment variables
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    # Force CPU offloading for parts of the model if needed
    print(f"Loading model from {'local path' if use_local_model else 'Hugging Face'}: {model_path}...")
    
    # Try to load with 8-bit quantization if available
    try:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,  # Use half precision
            max_memory={0: "8GiB"}  # Limit GPU memory usage
        )
        print("Loaded model with 8-bit quantization.")
    except (ImportError, ValueError) as e:
        print(f"8-bit quantization not available: {str(e)}. Falling back to standard loading with memory optimization.")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,  # Use half precision
        )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
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
    
    # Create a smaller max_length to reduce memory requirements
    max_length = min(512, tokenizer.model_max_length)  # Limit to 512 tokens max
    
    # Tokenize the dataset with the smaller max_length
    def tokenize_function(examples):
        return tokenizer(
            examples['text'], 
            truncation=True, 
            padding='max_length', 
            max_length=max_length
        )
    
    print("Tokenizing dataset...")
    tokenized_dataset = {}
    for split in dataset:
        tokenized_dataset[split] = dataset[split].map(
            tokenize_function, 
            batched=True, 
            desc=f"Tokenizing {split} split"
        )
    
    # Define training arguments with aggressive memory optimization
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,  # Set to 1
        per_device_eval_batch_size=batch_size,   # Set to 1
        learning_rate=learning_rate,
        save_steps=10_000,
        save_total_limit=1,  # Keep only 1 checkpoint
        logging_dir=f'{output_dir}/logs',
        logging_steps=100,
        # Memory optimization parameters
        gradient_accumulation_steps=16,  # Accumulate gradients over more steps
        fp16=True,                       # Use mixed precision training
        optim="adamw_torch",
        ddp_find_unused_parameters=False,
        dataloader_num_workers=0,        # No parallel loading
        group_by_length=True,
        weight_decay=0.01,
        no_cuda=not torch.cuda.is_available(),  # Use CPU if no GPU
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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
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