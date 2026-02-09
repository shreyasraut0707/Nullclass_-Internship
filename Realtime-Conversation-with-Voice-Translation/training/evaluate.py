"""
Model evaluation utilities.
"""

import os
import sys
import torch
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.seq2seq import Encoder, Decoder, Seq2Seq
from data.data_loader import load_data, TranslationDataset, collate_fn
from training.config import *


def translate_sentence(model, sentence, src_vocab, tgt_vocab, device, max_len=50):
    """
    Translate a single sentence.
    
    Args:
        model: Trained Seq2Seq model
        sentence: Input sentence as string
        src_vocab: Source vocabulary
        tgt_vocab: Target vocabulary
        device: Device to run on
        max_len: Maximum output length
    
    Returns:
        Translated sentence as string
    """
    model.eval()
    
    # Tokenize and convert to indices
    tokens = sentence.lower().split()
    indices = [src_vocab.word2idx.get(token, src_vocab.UNK_token) for token in tokens]
    indices.append(src_vocab.EOS_token)
    
    # Convert to tensor
    src_tensor = torch.LongTensor(indices).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Encode
        encoder_outputs, hidden, cell = model.encoder(src_tensor)
        
        # Prepare decoder initial state
        hidden = model._combine_bidirectional(hidden, model.decoder.num_layers)
        cell = model._combine_bidirectional(cell, model.decoder.num_layers)
        
        # Start with SOS token
        input_token = torch.LongTensor([tgt_vocab.SOS_token]).to(device)
        
        output_indices = []
        
        for _ in range(max_len):
            output, hidden, cell = model.decoder(input_token, hidden, cell, encoder_outputs)
            
            # Get the most probable token
            top_token = output.argmax(1).item()
            
            if top_token == tgt_vocab.EOS_token:
                break
            
            output_indices.append(top_token)
            input_token = torch.LongTensor([top_token]).to(device)
        
        # Convert indices to words
        output_tokens = [tgt_vocab.idx2word.get(idx, '<UNK>') for idx in output_indices]
        
        return ' '.join(output_tokens)


def calculate_bleu(model, test_pairs, src_vocab, tgt_vocab, device, num_samples=100):
    """Calculate BLEU score on test set."""
    try:
        from sacrebleu.metrics import BLEU
        bleu = BLEU()
    except:
        print("sacrebleu not available, skipping BLEU calculation")
        return None
    
    model.eval()
    
    predictions = []
    references = []
    
    # Sample random pairs
    sample_pairs = random.sample(test_pairs, min(num_samples, len(test_pairs)))
    
    for src_sentence, tgt_sentence in tqdm(sample_pairs, desc="Calculating BLEU"):
        translation = translate_sentence(model, src_sentence, src_vocab, tgt_vocab, device)
        
        predictions.append(translation)
        references.append([tgt_sentence])
    
    score = bleu.corpus_score(predictions, references)
    return score.score


def load_trained_model(model_path, device):
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    src_vocab = checkpoint['src_vocab']
    tgt_vocab = checkpoint['tgt_vocab']
    config = checkpoint['model_config']
    
    # Create model
    encoder = Encoder(
        len(src_vocab),
        config['embedding_dim'],
        config['encoder_hidden_dim'],
        config['num_layers'],
        config['dropout']
    )
    decoder = Decoder(
        len(tgt_vocab),
        config['embedding_dim'],
        config['encoder_hidden_dim'],
        config['decoder_hidden_dim'],
        config['num_layers'],
        config['dropout']
    )
    
    model = Seq2Seq(encoder, decoder, device).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, src_vocab, tgt_vocab


def evaluate_model(direction='en-es', model_type='best'):
    """
    Evaluate a trained model.
    
    Args:
        direction: 'en-es' or 'es-en'
        model_type: 'best' or 'final'
    """
    device = DEVICE
    print(f"Evaluating {direction} model on {device}")
    
    # Load model
    if model_type == 'best':
        model_path = os.path.join(CHECKPOINT_DIR, direction.replace('-', '_'), 'best_model.pth')
    else:
        model_path = os.path.join(MODEL_DIR, direction.replace('-', '_'), 'final_model.pth')
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
    
    model, src_vocab, tgt_vocab = load_trained_model(model_path, device)
    print(f"Model loaded from {model_path}")
    
    # Load test data
    data_file = os.path.join(DATA_DIR, 'processed', f'{direction.replace("-", "_")}_data.pkl')
    data = load_data(data_file)
    test_pairs = data['test']
    
    print(f"\nTest set size: {len(test_pairs)}")
    
    # Calculate BLEU score
    print("\nCalculating BLEU score...")
    bleu_score = calculate_bleu(model, test_pairs, src_vocab, tgt_vocab, device)
    if bleu_score:
        print(f"BLEU Score: {bleu_score:.2f}")
    
    # Test with some examples
    print("\nSample translations:")
    num_examples = 10
    sample_pairs = random.sample(test_pairs, min(num_examples, len(test_pairs)))
    
    for i, (src, tgt) in enumerate(sample_pairs, 1):
        translation = translate_sentence(model, src, src_vocab, tgt_vocab, device)
        print(f"\n{i}.")
        print(f"  Source: {src}")
        print(f"  Target: {tgt}")
        print(f"  Predicted: {translation}")


def interactive_translate(direction='en-es', model_type='best'):
    """Interactive translation mode."""
    device = DEVICE
    
    # Load model
    if model_type == 'best':
        model_path = os.path.join(CHECKPOINT_DIR, direction.replace('-', '_'), 'best_model.pth')
    else:
        model_path = os.path.join(MODEL_DIR, direction.replace('-', '_'), 'final_model.pth')
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
    
    model, src_vocab, tgt_vocab = load_trained_model(model_path, device)
    print(f"Model loaded. Ready for translation!")
    print(f"Direction: {direction}")
    print("Type 'quit' to exit\n")
    
    while True:
        sentence = input("Enter sentence to translate: ").strip()
        
        if sentence.lower() in ['quit', 'exit', 'q']:
            break
        
        if not sentence:
            continue
        
        translation = translate_sentence(model, sentence, src_vocab, tgt_vocab, device)
        print(f"Translation: {translation}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate translation model')
    parser.add_argument('--direction', type=str, default='en-es',
                       choices=['en-es', 'es-en'],
                       help='Translation direction')
    parser.add_argument('--model-type', type=str, default='best',
                       choices=['best', 'final'],
                       help='Which model to evaluate')
    parser.add_argument('--interactive', action='store_true',
                       help='Enter interactive translation mode')
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_translate(args.direction, args.model_type)
    else:
        evaluate_model(args.direction, args.model_type)
