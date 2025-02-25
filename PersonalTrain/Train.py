# Install required packages (if not already installed)
!pip install -q transformers datasets peft bitsandbytes huggingface_hub

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import bitsandbytes as bnb
from huggingface_hub import login

# Authenticate with Hugging Face
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")  # Use environment variable for security
login(token=HUGGINGFACE_TOKEN)

# Load tokenizer and base model
MODEL_NAME = "tiiuae/falcon-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Load dataset
dataset = load_dataset("text", data_files={"train": "/kaggle/input/common-diseases/medical_text_data.txt"})["train"]

# Tokenization function with labels
def tokenize_function(examples):
    tokenized = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
    tokenized["labels"] = tokenized["input_ids"].copy()  # Assign labels
    return tokenized

# Tokenize dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Load model with LoRA and 4-bit quantization
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
config = LoraConfig(r=8, lora_alpha=32, lora_dropout=0.1, bias="none")
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=quantization_config, device_map="auto")
model = get_peft_model(base_model, config)

# Training arguments
training_args = TrainingArguments(
    output_dir="./falcon_lora_finetuned",
    run_name="falcon_lora_training",
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,  # Improve memory efficiency
    num_train_epochs=3,
    save_strategy="epoch",
    fp16=True,  # Enable mixed precision training
    report_to=[],  # Disable wandb
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,
)

# Train model
trainer.train()

# Save model
model.save_pretrained("./falcon_lora_finetuned")
tokenizer.save_pretrained("./falcon_lora_finetuned")

print("Training complete! Model saved successfully.")
