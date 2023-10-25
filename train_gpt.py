import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset
import json

# ------------------------------------------------------------------
# 1. Load and Prepare Data
# ------------------------------------------------------------------

def load_and_prepare_dataset(file_path, tokenizer):
    """Load conversations, format them, and tokenize for GPT-2 fine-tuning."""
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Format the dataset into a single text column
    formatted_texts = []
    for item in data:
        # Create a single string for each conversation pair
        text = f"User: {item['question']}\nBot: {item['answer']}{tokenizer.eos_token}"
        formatted_texts.append(text)

    # Create a Hugging Face Dataset object
    dataset = Dataset.from_dict({'text': formatted_texts})

    def tokenize_function(examples):
        # Tokenize the texts
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    return tokenized_dataset

# ------------------------------------------------------------------
# 2. Main Fine-Tuning Execution
# ------------------------------------------------------------------
if __name__ == '__main__':
    # --- Tokenizer and Model ---
    model_name = 'distilgpt2'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    # Set a padding token
    tokenizer.pad_token = tokenizer.eos_token
    
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # --- Dataset ---
    dataset = load_and_prepare_dataset('data/conversations.json', tokenizer)

    # --- Training ---
    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=3,              # total number of training epochs
        per_device_train_batch_size=2,   # batch size per device during training
        warmup_steps=10,                 # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=10,
        save_total_limit=2,              # limit the total amount of checkpoints
        report_to="none"                 # disable external reporting
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    print("Starting fine-tuning of DistilGPT-2...")
    trainer.train()
    print("Fine-tuning finished.")

    # --- Save the final model ---
    final_model_path = './fine_tuned_gpt2'
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"Model saved to {final_model_path}")

    # --- Inference Example ---
    print("\n--- Inference Example ---")
    # Load the fine-tuned model
    fine_tuned_model = GPT2LMHeadModel.from_pretrained(final_model_path)
    fine_tuned_tokenizer = GPT2Tokenizer.from_pretrained(final_model_path)

    # Format the prompt
    prompt = f"User: How are you doing?\nBot:"
    
    # Generate a response
    inputs = fine_tuned_tokenizer(prompt, return_tensors='pt')
    outputs = fine_tuned_model.generate(inputs.input_ids, max_length=50, pad_token_id=fine_tuned_tokenizer.eos_token_id)
    response = fine_tuned_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(response)
