from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

def fine_tune_model(model_name, dataset_path, output_dir, epochs=3, batch_size=8, learning_rate=5e-5):
    """
    Fine-tune a Hugging Face model with a custom dataset.

    Args:
        model_name (str): Name of the pre-trained model (e.g., "t5-small", "phi-3-mini-128k-instruct").
        dataset_path (str): Path to the dataset file (e.g., JSON).
        output_dir (str): Directory to save the fine-tuned model.
        epochs (int): Number of training epochs.
        batch_size (int): Training batch size.
        learning_rate (float): Learning rate for training.
    """
    # Load the pre-trained model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load the dataset
    dataset = load_dataset('json', data_files=dataset_path)

    # Tokenize the dataset
    def preprocess_function(examples):
        inputs = [f"Convert PL/SQL to LINQ: {x}" for x in examples['input']]
        targets = examples['output']
        model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
        labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True)

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
        eval_dataset=tokenized_dataset['validation'] if 'validation' in tokenized_dataset else None,
    )

    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Model fine-tuned and saved to {output_dir}")

# Example usage
if __name__ == "__main__":
    fine_tune_model(
        model_name="t5-small",  # Replace with "phi-3-mini-128k-instruct" or another model
        dataset_path="fine_tuning_dataset.json",
        output_dir="output/plsql-to-linq-finetuned",
        epochs=3,
        batch_size=8,
        learning_rate=5e-5
    )