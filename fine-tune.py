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
                if not data or 'text' not in data[0]:
                    print(f"Error: Dataset must contain records with a 'text' field.")
                    print(f"Found structure: {list(data[0].keys()) if data else 'empty list'}")
                    return False
            elif isinstance(data, dict):
                if 'train' not in data:
                    print(f"Error: Dataset in dict format should have a 'train' key.")
                    return False
                if not data['train'] or 'text' not in data['train'][0]:
                    print(f"Error: Dataset's train split must contain records with a 'text' field.")
                    return False
        
        return True
    except json.JSONDecodeError:
        print(f"Error: {dataset_path} is not a valid JSON file.")
        return False
    except Exception as e:
        print(f"Error validating dataset: {str(e)}")
        return False

def fine_tune_model(model_name, dataset_path, output_dir, epochs=3, batch_size=8, learning_rate=5e-5):
    """
    Fine-tune a Hugging Face model with a custom dataset.

    Args:
        model_name (str): Name of the pre-trained model (e.g., "phi-4", "phi-3-mini-128k-instruct").
        dataset_path (str): Path to the dataset file (e.g., JSON, CSV).
        output_dir (str): Directory to save the fine-tuned model.
        epochs (int): Number of training epochs.
        batch_size (int): Training batch size.
        learning_rate (float): Learning rate for training.
    """
    # Validate the dataset first
    if not validate_dataset(dataset_path):
        print("Dataset validation failed. Please check your dataset format.")
        return
        
    # Load the pre-trained model and tokenizer
    print(f"Loading model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load the dataset
    print(f"Loading dataset from {dataset_path}...")
    if dataset_path.endswith('.json'):
        dataset = load_dataset('json', data_files=dataset_path)
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
    
    # Verify dataset has text field and print sample
    if 'text' not in dataset['train'][0]:
        print(f"Warning: Dataset doesn't contain 'text' field. Found fields: {list(dataset['train'][0].keys())}")
        print("Please ensure your dataset has the required format. First record sample:")
        print(dataset['train'][0])
        return
    
    print("Sample data point:", dataset['train'][0]['text'][:100] + "...")
    
    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(
            examples['text'], 
            truncation=True, 
            padding='max_length', 
            max_length=tokenizer.model_max_length
        )
    
    print("Tokenizing dataset...")
    tokenized_dataset = {}
    for split in dataset:
        tokenized_dataset[split] = dataset[split].map(
            tokenize_function, 
            batched=True, 
            desc=f"Tokenizing {split} split"
        )
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir=f'{output_dir}/logs',
        logging_steps=500,
        evaluation_strategy="steps",
        eval_steps=500,
        load_best_model_at_end=True,
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
    trainer.train()

    # Save the fine-tuned model
    print(f"Saving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Model fine-tuned and saved to {output_dir}")

# Example usage
if __name__ == "__main__":
    fine_tune_model(
        model_name="phi-3-mini-128k-instruct",  # Replace with "phi-4" or other model as needed
        dataset_path="fine_tuning_dataset.json",  # Update with the correct path
        output_dir="output/phi-3-tuned",
        epochs=3,
        batch_size=8,
        learning_rate=5e-5
    )