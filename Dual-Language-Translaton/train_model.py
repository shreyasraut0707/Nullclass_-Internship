"""
Train Model - MAXIMUM ACCURACY VERSION
Uses more data, more epochs, and optimized parameters
"""

import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("   HIGH ACCURACY TRAINING - MAXIMUM PERFORMANCE")
print("=" * 60)
print()

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device.upper()}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

from transformers import (
    MarianMTModel, 
    MarianTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset, Dataset
import random


def load_data(src, tgt, num_samples=5000):
    """Load more training data."""
    print(f"Loading {num_samples} samples for {src}-{tgt}...")
    
    pairs = []
    try:
        dataset = load_dataset("opus100", f"{src}-{tgt}", split="train", trust_remote_code=True)
        
        # Get more diverse samples
        indices = list(range(min(num_samples * 2, len(dataset))))
        random.shuffle(indices)
        
        for idx in indices[:num_samples]:
            item = dataset[idx]
            src_text = item['translation'][src]
            tgt_text = item['translation'][tgt]
            # Filter quality - longer sentences for better learning
            if 15 < len(src_text) < 200 and 10 < len(tgt_text) < 300:
                pairs.append((src_text, tgt_text))
                if len(pairs) >= num_samples:
                    break
    except Exception as e:
        print(f"Error: {e}")
    
    print(f"Loaded {len(pairs)} pairs")
    return pairs


def create_dataset(pairs, tokenizer, max_len=150):
    """Create dataset with better tokenization."""
    sources = [p[0] for p in pairs]
    targets = [p[1] for p in pairs]
    
    inputs = tokenizer(sources, max_length=max_len, truncation=True, padding='max_length')
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_len, truncation=True, padding='max_length')
    
    inputs["labels"] = labels["input_ids"]
    return Dataset.from_dict(inputs)


def train(model_name, src, tgt, output_dir, lang_name, num_samples=5000):
    """Train with maximum accuracy settings."""
    
    print(f"\n{'='*60}")
    print(f"Training: English → {lang_name} (HIGH ACCURACY)")
    print(f"{'='*60}")
    
    # Load more data
    pairs = load_data(src, tgt, num_samples)
    
    # Better split: 90% train, 10% validation
    split = int(len(pairs) * 0.9)
    train_pairs = pairs[:split]
    val_pairs = pairs[split:]
    
    print(f"Train: {len(train_pairs)}, Validation: {len(val_pairs)}")
    
    # Load model
    print("Loading model...")
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    model.to(device)
    
    train_ds = create_dataset(train_pairs, tokenizer)
    val_ds = create_dataset(val_pairs, tokenizer)
    
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
    
    # OPTIMIZED TRAINING PARAMETERS FOR HIGH ACCURACY
    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,              # More epochs
        per_device_train_batch_size=16,   # Larger batch
        per_device_eval_batch_size=16,
        learning_rate=5e-5,               # Higher LR
        weight_decay=0.01,
        warmup_ratio=0.1,
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_steps=25,
        fp16=(device == "cuda"),
        report_to="none",
        overwrite_output_dir=True,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    
    print(f"\nTraining for 10 epochs with {len(train_pairs)} samples...")
    print("-" * 40)
    trainer.train()
    print("-" * 40)
    
    print(f"Saving to: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Test
    print("\nTesting:")
    test_model(output_dir, val_pairs[:3])
    
    return True


def test_model(path, pairs):
    """Quick test."""
    tokenizer = MarianTokenizer.from_pretrained(path)
    model = MarianMTModel.from_pretrained(path)
    model.to(device)
    model.eval()
    
    for en, exp in pairs:
        inputs = tokenizer(en, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            out = model.generate(**inputs, max_length=150, num_beams=5)
        
        result = tokenizer.decode(out[0], skip_special_tokens=True)
        print(f"  IN:  {en[:60]}...")
        print(f"  OUT: {result[:60]}...")
        print()


def main():
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Train with 5000 samples each
    train("Helsinki-NLP/opus-mt-en-fr", "en", "fr", 
          os.path.join(models_dir, "en-fr"), "French", 5000)
    
    train("Helsinki-NLP/opus-mt-en-hi", "en", "hi",
          os.path.join(models_dir, "en-hi"), "Hindi", 5000)
    
    print("\n" + "=" * 60)
    print("         TRAINING COMPLETE!")
    print("=" * 60)
    print("\nRun: python evaluate_model.py")


if __name__ == "__main__":
    main()
