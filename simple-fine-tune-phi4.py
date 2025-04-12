# Fine-tune the Phi-4 model using a simple dataset in JSON format.
# This script demonstrates how to load a fine-tuned model using PEFT (Parameter-Efficient Fine-Tuning) with LoRA (Low-Rank Adaptation).
# This script demonstrates how to fine-tune the Phi-4 model using a simple dataset in JSON format.

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset

def load_dataset(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return Dataset.from_dict({"input": [item["input"] for item in data], "output": [item["output"] for item in data]})

def tokenize_dataset(dataset, tokenizer):
    def tokenize_function(examples):
        return tokenizer(examples["input"], padding="max_length", truncation=True, max_length=512)
    return dataset.map(tokenize_function, batched=True)

def fine_tune_phi4(input_json_path, output_model_dir, num_epochs=3, batch_size=8, learning_rate=5e-5):
    # Load the dataset
    dataset = load_dataset(input_json_path)
    
    # Load the pre-trained Phi-4 model and tokenizer
    model_name = "phi-4"  # Replace with the actual model name or path
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Tokenize the dataset
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_model_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir='./logs',
        logging_steps=500,
        evaluation_strategy="steps",
        eval_steps=500,
        warmup_steps=500,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
    )
    
    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )
    
    # Fine-tune the model
    trainer.train()
    
    # Save the fine-tuned model
    trainer.save_model(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune the Phi-4 model.")
    parser.add_argument("--input_json", type=str, required=True, help="Path to the input JSON dataset.")
    parser.add_argument("--output_model_dir", type=str, required=True, help="Directory to save the fine-tuned model.")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for training.")
    
    args = parser.parse_args()
    
    fine_tune_phi4(
        input_json_path=args.input_json,
        output_model_dir=args.output_model_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )

    # python simple-fine-tune-phi4.py --input_json /path/to/your/dataset.json --output_model_dir /path/to/save/model --num_epochs 3 --batch_size 8 --learning_rate 5e-5
