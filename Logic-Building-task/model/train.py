"""
Model Training Script
Fine-tunes MarianMT model on English-Hindi word pairs from Hugging Face dataset.
"""

import os
import json
import torch
from transformers import (
    MarianMTModel, 
    MarianTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
from tqdm import tqdm


# Model configuration
MODEL_NAME = "Helsinki-NLP/opus-mt-en-hi"
OUTPUT_DIR = "model/saved_model"
DATA_PATH = "data/word_pairs.json"


def load_training_data(filepath: str):
    """Load word pairs from JSON file."""
    if not os.path.exists(filepath):
        print(f"Error: Data file not found at {filepath}")
        print("Please run 'python data/download_data.py' first to download the dataset.")
        return None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def prepare_dataset(word_pairs: list, tokenizer):
    """
    Prepare dataset for training.
    
    Args:
        word_pairs: List of dicts with 'english' and 'hindi' keys
        tokenizer: MarianTokenizer instance
    
    Returns:
        Hugging Face Dataset object
    """
    english_words = [pair['english'] for pair in word_pairs]
    hindi_words = [pair['hindi'] for pair in word_pairs]
    
    # Create dataset
    dataset = Dataset.from_dict({
        'english': english_words,
        'hindi': hindi_words
    })
    
    def tokenize_function(examples):
        # Tokenize inputs (English words)
        model_inputs = tokenizer(
            examples['english'],
            max_length=32,
            truncation=True,
            padding='max_length'
        )
        
        # Tokenize targets (Hindi words)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples['hindi'],
                max_length=32,
                truncation=True,
                padding='max_length'
            )
        
        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['english', 'hindi']
    )
    
    return tokenized_dataset


def train_model():
    """
    Main training function.
    Fine-tunes MarianMT model on English-Hindi word pairs.
    """
    print("=" * 70)
    print("English-Hindi Translation Model Training")
    print("=" * 70)
    
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    # Load training data
    print("\nLoading training data...")
    word_pairs = load_training_data(DATA_PATH)
    if word_pairs is None:
        return
    
    print(f"Loaded {len(word_pairs)} word pairs for training.")
    
    # Load pre-trained model and tokenizer
    print(f"\nLoading pre-trained model: {MODEL_NAME}")
    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
    model = MarianMTModel.from_pretrained(MODEL_NAME)
    
    # Move model to device
    model = model.to(device)
    
    # Prepare dataset
    print("\nPreparing dataset for training...")
    train_dataset = prepare_dataset(word_pairs, tokenizer)
    
    # Split into train and validation
    split_dataset = train_dataset.train_test_split(test_size=0.1)
    train_data = split_dataset['train']
    val_data = split_dataset['test']
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    # Training arguments - optimized for word-level fine-tuning
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=8,  # More epochs for better generalization
        per_device_train_batch_size=8,  # Smaller batch for small dataset
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=20,
        eval_strategy="epoch",  # Evaluate at end of each epoch
        save_strategy="epoch",  # Save at end of each epoch
        save_total_limit=2,
        load_best_model_at_end=True,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        report_to="none",  # Disable wandb/tensorboard
        learning_rate=3e-5,  # Lower learning rate for better fine-tuning
        metric_for_best_model="eval_loss",
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    
    # Start training
    print("\n" + "=" * 70)
    print("Starting model training...")
    print("=" * 70 + "\n")
    
    # Check for existing checkpoints to resume from
    import glob
    checkpoints = glob.glob(f"{OUTPUT_DIR}/checkpoint-*")
    if checkpoints:
        # Find the latest checkpoint
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
        print(f"Found checkpoint: {latest_checkpoint}")
        print("Resuming training from checkpoint...")
        trainer.train(resume_from_checkpoint=latest_checkpoint)
    else:
        print("Starting fresh training...")
        trainer.train()
    
    # Save final model
    print("\nSaving trained model...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("\n" + "=" * 70)
    print(f"Training complete! Model saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    train_model()
