# Full fine-tune the Phi-4 model using a conversation dataset in JSONL format.
# This script performs full parameter fine-tuning on Phi-4 using conversation data.

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import prepare_model_for_kbit_training
from datasets import Dataset
import os

def load_dataset(json_path):
    """Load and process conversation dataset from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) if line.strip() else {} for line in f.readlines()]
        if not data or not isinstance(data[0], dict):
            # If the file isn't in JSONL format, try loading as a regular JSON array
            f.seek(0)
            data = json.load(f)
    
    # Extract and format conversations
    formatted_data = []
    for item in data:
        if "messages" in item:
            # Format the conversation as a single text for the model
            conversation = ""
            messages = item["messages"]
            for message in messages:
                role = message.get("role", "")
                content = message.get("content", "")
                conversation += f"{role}: {content}\n\n"
            
            formatted_data.append({"text": conversation})
    
    return Dataset.from_dict({"text": [item["text"] for item in formatted_data]})

def tokenize_dataset(dataset, tokenizer):
    """Tokenize the dataset for training."""
    def tokenize_function(examples):
        results = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=1024,  # Adjust based on your model and data
            return_tensors="pt"
        )
        
        # Set labels equal to input_ids for causal language modeling
        results["labels"] = results["input_ids"].clone()
        return results
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )
    
    return tokenized_dataset

def full_fine_tune_phi4(input_json_path, output_model_dir, num_epochs=3, batch_size=4, learning_rate=2e-5):
    """Perform full fine-tuning on the Phi-4 model."""
    # Load the dataset
    print(f"Loading dataset from {input_json_path}...")
    dataset = load_dataset(input_json_path)
    print(f"Dataset loaded with {len(dataset)} examples")
    
    # Split dataset into training and evaluation sets (90/10 split)
    dataset = dataset.train_test_split(test_size=0.1)
    
    # Load the pre-trained Phi-4 model and tokenizer
    model_name = "microsoft/Phi-4"  # Correct model identifier for Phi-4
    
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading model from {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    
    # Prepare model for training
    model.config.use_cache = False  # Important for training efficiency
    
    # Tokenize the dataset
    print("Tokenizing dataset...")
    tokenized_train_dataset = tokenize_dataset(dataset["train"], tokenizer)
    tokenized_eval_dataset = tokenize_dataset(dataset["test"], tokenizer)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_model_dir, exist_ok=True)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_model_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir=os.path.join(output_model_dir, 'logs'),
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="tensorboard",
        gradient_accumulation_steps=2,  # To help with memory usage
        gradient_checkpointing=True,    # To help with memory usage
        optim="adamw_torch",
    )
    
    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        tokenizer=tokenizer,
    )
    
    # Fine-tune the model
    print("Starting training...")
    trainer.train()
    
    # Save the fine-tuned model
    print(f"Saving model to {output_model_dir}...")
    trainer.save_model(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)
    print("Training complete!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Full fine-tune the Phi-4 model.")
    parser.add_argument("--input_json", type=str, default="phi4_fine_tuning_dataset.json", 
                        help="Path to the input JSON dataset.")
    parser.add_argument("--output_model_dir", type=str, default="./phi4-finetuned", 
                        help="Directory to save the fine-tuned model.")
    parser.add_argument("--num_epochs", type=int, default=3, 
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, 
                        help="Training batch size.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, 
                        help="Learning rate for training.")
    
    args = parser.parse_args()
    
    full_fine_tune_phi4(
        input_json_path=args.input_json,
        output_model_dir=args.output_model_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )