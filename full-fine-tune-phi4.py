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
    try:
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
                for line in f:
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
            
            formatted_data.append({"text": conversation})
        # Handle legacy format with input/output fields
        elif "input" in item and "output" in item:
            conversation = f"<|user|>\n{item['input']}\n<|assistant|>\n{item['output']}\n"
            formatted_data.append({"text": conversation})
    
    print(f"Processed {len(formatted_data)} conversations from the dataset")
    return Dataset.from_dict({"text": [item["text"] for item in formatted_data]})

def tokenize_dataset(dataset, tokenizer):
    """Tokenize the dataset for training."""
    def tokenize_function(examples):
        results = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=2048,  # Increased max length for longer conversations
            return_tensors="pt"
        )
        
        # Set labels equal to input_ids for causal language modeling
        results["labels"] = results["input_ids"].clone()
        
        # Set padding token ids to -100 so they don't contribute to the loss
        for i, label in enumerate(results["labels"]):
            mask = results["attention_mask"][i] == 0
            results["labels"][i][mask] = -100
            
        return results
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )
    
    return tokenized_dataset

def full_fine_tune_phi4(input_json_path, output_model_dir, model_path=None, num_epochs=3, batch_size=4, learning_rate=2e-5):
    """Perform full fine-tuning on the Phi-4 model."""
    # Load the dataset
    print(f"Loading dataset from {input_json_path}...")
    dataset = load_dataset(input_json_path)
    print(f"Dataset loaded with {len(dataset)} examples")
    
    # Split dataset into training and evaluation sets (90/10 split)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    
    # Load the pre-trained Phi-4 model and tokenizer
    if model_path is None:
        # Default paths for different operating systems
        if os.name == 'nt':  # Windows
            model_path = "microsoft/Phi-4"  # Use HuggingFace model on Windows
        else:
            model_path = "/home/TomAdmin/phi-4"  # Linux path
    
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Ensure we have proper padding and EOS tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
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
    
    # Define training arguments with optimized settings for Phi-4
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
    #    evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="tensorboard",
        gradient_accumulation_steps=4,  # Increased for better stability
        gradient_checkpointing=True,    # To help with memory usage
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
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
    
    # Check if test_input is already properly formatted
    if not test_input.startswith("Convert the following PL/SQL code to C# LINQ:"):
        # Format as a SQL conversion query similar to the dataset format
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
    import argparse
    
    parser = argparse.ArgumentParser(description="Full fine-tune the Phi-4 model.")
    parser.add_argument("--input_json", type=str, default="phi4_fine_tuning_dataset.json", 
                        help="Path to the input JSON dataset.")
    parser.add_argument("--output_model_dir", type=str, default="./phi4-finetuned", 
                        help="Directory to save the fine-tuned model.")
    parser.add_argument("--model_path", type=str, default=None, 
                        help="Path to the pre-trained Phi-4 model.")
    parser.add_argument("--num_epochs", type=int, default=3, 
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, 
                        help="Training batch size.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, 
                        help="Learning rate for training.")
    parser.add_argument("--test", action="store_true",
                        help="Test the fine-tuned model after training.")
    parser.add_argument("--test_input", type=str, 
                        default="Convert the following PL/SQL code to C# LINQ: \n\nSELECT * FROM employees WHERE department_id = 10;",
                        help="Test input for the fine-tuned model.")
    parser.add_argument("--test_only", action="store_true",
                        help="Only test the model without fine-tuning.")
    
    args = parser.parse_args()
    
    if not args.test_only:
        full_fine_tune_phi4(
            input_json_path=args.input_json,
            output_model_dir=args.output_model_dir,
            model_path=args.model_path,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
    
    if args.test or args.test_only:
        print("\nTesting fine-tuned model...")
        response = test_fine_tuned_model(args.output_model_dir, args.test_input)
        print("\nModel response:")
        print(response)