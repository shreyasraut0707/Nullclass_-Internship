"""
Evaluate models on UNSEEN test data from Hugging Face
"""

import os
import warnings
warnings.filterwarnings('ignore')

import torch
from transformers import MarianMTModel, MarianTokenizer
from datasets import load_dataset

try:
    from sacrebleu.metrics import BLEU
    HAS_BLEU = True
except:
    HAS_BLEU = False


def evaluate_model(model_path, src_lang, tgt_lang, lang_name, num_test=200):
    """Evaluate model on unseen test data."""
    print(f"\n{'='*60}")
    print(f"Evaluating: English → {lang_name}")
    print(f"{'='*60}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    tokenizer = MarianTokenizer.from_pretrained(model_path)
    model = MarianMTModel.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    # Load UNSEEN test data from different split
    print("Loading TEST data from Hugging Face (unseen data)...")
    try:
        dataset = load_dataset("opus100", f"{src_lang}-{tgt_lang}", split="test", trust_remote_code=True)
        test_pairs = []
        for item in dataset.select(range(min(num_test, len(dataset)))):
            src = item['translation'][src_lang]
            tgt = item['translation'][tgt_lang]
            if len(src) > 10:
                test_pairs.append((src, tgt))
    except:
        print("Using validation split...")
        dataset = load_dataset("opus100", f"{src_lang}-{tgt_lang}", split="validation", trust_remote_code=True)
        test_pairs = []
        for item in dataset.select(range(min(num_test, len(dataset)))):
            src = item['translation'][src_lang]
            tgt = item['translation'][tgt_lang]
            if len(src) > 10:
                test_pairs.append((src, tgt))
    
    print(f"Testing on {len(test_pairs)} UNSEEN samples")
    
    predictions = []
    references = []
    
    for i, (en, expected) in enumerate(test_pairs):
        inputs = tokenizer(en, return_tensors="pt", padding=True, max_length=128, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=128, num_beams=5)
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(result)
        references.append(expected)
        
        if i < 3:
            print(f"\nExample {i+1}:")
            print(f"  EN: {en[:70]}...")
            print(f"  Predicted: {result[:70]}...")
            print(f"  Expected:  {expected[:70]}...")
    
    # Calculate BLEU
    if HAS_BLEU:
        bleu = BLEU()
        score = bleu.corpus_score(predictions, [references])
        print(f"\n{'='*60}")
        print(f"BLEU SCORE for {lang_name}: {score.score:.2f}")
        print(f"{'='*60}")
        return score.score
    else:
        print("sacrebleu not installed, run: pip install sacrebleu")
        return 0


def main():
    print("=" * 60)
    print("   MODEL ACCURACY ON UNSEEN TEST DATA")
    print("=" * 60)
    
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    
    # Evaluate French
    fr_score = evaluate_model(
        os.path.join(models_dir, "en-fr"),
        "en", "fr", "French", 200
    )
    
    # Evaluate Hindi  
    hi_score = evaluate_model(
        os.path.join(models_dir, "en-hi"),
        "en", "hi", "Hindi", 200
    )
    
    print("\n" + "=" * 60)
    print("            FINAL ACCURACY SUMMARY")
    print("=" * 60)
    print(f"\n{'Model':<25} {'BLEU Score':<15}")
    print("-" * 40)
    print(f"{'English → French':<25} {fr_score:.2f}")
    print(f"{'English → Hindi':<25} {hi_score:.2f}")
    print("-" * 40)
    print(f"{'AVERAGE':<25} {(fr_score + hi_score)/2:.2f}")
    print()
    print("NOTE: These scores are on UNSEEN test data,")
    print("not the training data. This shows real generalization!")
    print()
    print("BLEU Score Guide:")
    print("  50+ : Excellent (professional quality)")
    print("  40-50: Very Good")
    print("  30-40: Good (understandable)")
    print("  20-30: Acceptable")
    print("  <20 : Needs improvement")


if __name__ == "__main__":
    main()
