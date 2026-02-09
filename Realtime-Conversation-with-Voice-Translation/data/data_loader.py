"""
Data loading and preprocessing utilities for English-Spanish translation.
This module handles downloading data from Hugging Face and preparing it for training.
"""

import os
import pickle
from datasets import load_dataset
from collections import Counter
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class Vocabulary:
    """Build and manage vocabulary for source and target languages."""
    
    def __init__(self, max_size=None):
        self.word2idx = {}
        self.idx2word = {}
        self.word_count = Counter()
        self.max_size = max_size
        
        # Special tokens
        self.PAD_token = 0
        self.SOS_token = 1
        self.EOS_token = 2
        self.UNK_token = 3
        
        self.word2idx['<PAD>'] = self.PAD_token
        self.word2idx['<SOS>'] = self.SOS_token
        self.word2idx['<EOS>'] = self.EOS_token
        self.word2idx['<UNK>'] = self.UNK_token
        
        self.idx2word[self.PAD_token] = '<PAD>'
        self.idx2word[self.SOS_token] = '<SOS>'
        self.idx2word[self.EOS_token] = '<EOS>'
        self.idx2word[self.UNK_token] = '<UNK>'
        
        self.n_words = 4
    
    def add_sentence(self, sentence):
        """Add all words in a sentence to vocabulary."""
        for word in sentence.split():
            self.add_word(word)
    
    def add_word(self, word):
        """Add a word to vocabulary."""
        self.word_count[word] += 1
        if word not in self.word2idx:
            self.word2idx[word] = self.n_words
            self.idx2word[self.n_words] = word
            self.n_words += 1
    
    def trim_to_size(self, max_size):
        """Trim vocabulary to most frequent words."""
        if max_size is None or self.n_words <= max_size:
            return
        
        # Keep special tokens
        most_common = self.word_count.most_common(max_size - 4)
        
        # Reset vocabulary
        self.word2idx = {}
        self.idx2word = {}
        
        self.word2idx['<PAD>'] = self.PAD_token
        self.word2idx['<SOS>'] = self.SOS_token
        self.word2idx['<EOS>'] = self.EOS_token
        self.word2idx['<UNK>'] = self.UNK_token
        
        self.idx2word[self.PAD_token] = '<PAD>'
        self.idx2word[self.SOS_token] = '<SOS>'
        self.idx2word[self.EOS_token] = '<EOS>'
        self.idx2word[self.UNK_token] = '<UNK>'
        
        self.n_words = 4
        
        for word, _ in most_common:
            self.word2idx[word] = self.n_words
            self.idx2word[self.n_words] = word
            self.n_words += 1
    
    def __len__(self):
        return self.n_words


class TranslationDataset(Dataset):
    """PyTorch Dataset for translation pairs."""
    
    def __init__(self, pairs, src_vocab, tgt_vocab):
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        src_sentence, tgt_sentence = self.pairs[idx]
        
        # Convert sentences to indices
        src_indices = self.sentence_to_indices(src_sentence, self.src_vocab)
        tgt_indices = self.sentence_to_indices(tgt_sentence, self.tgt_vocab)
        
        return torch.LongTensor(src_indices), torch.LongTensor(tgt_indices)
    
    def sentence_to_indices(self, sentence, vocab):
        """Convert sentence to list of word indices."""
        indices = [vocab.word2idx.get(word, vocab.UNK_token) 
                   for word in sentence.split()]
        indices.append(vocab.EOS_token)
        return indices


def normalize_sentence(sentence):
    """Basic sentence normalization."""
    sentence = sentence.lower().strip()
    # Remove extra spaces
    sentence = ' '.join(sentence.split())
    return sentence


def load_translation_data(direction='en-es', max_length=50, min_length=3, max_samples=None, max_vocab_size=15000):
    """
    Load and preprocess translation data from Hugging Face.
    
    Args:
        direction: 'en-es' for English to Spanish, 'es-en' for Spanish to English
        max_length: Maximum sentence length to include
        min_length: Minimum sentence length to include
        max_samples: Maximum number of samples to use (None for all)
        max_vocab_size: Maximum vocabulary size (limits to most frequent words)
    
    Returns:
        train_pairs, val_pairs, test_pairs, src_vocab, tgt_vocab
    """
    
    print("Loading dataset from Hugging Face...")
    print(f"Translation direction: {direction}")
    
    # Load the opus_books dataset (English-Spanish)
    try:
        dataset = load_dataset("opus_books", "en-es")
    except Exception as e:
        print(f"opus_books loading failed: {e}")
        # Fallback to another dataset if opus_books is not available
        print("Trying alternative dataset (Helsinki-NLP/tatoeba)...")
        try:
            dataset = load_dataset("Helsinki-NLP/tatoeba", lang1="en", lang2="es")
        except:
            print("Trying opus100 dataset...")
            dataset = load_dataset("Helsinki-NLP/opus-100", "en-es")
    
    print(f"Dataset loaded: {dataset}")
    
    # Process the data
    train_data = dataset['train']
    
    pairs = []
    print("Processing translation pairs...")
    
    for item in tqdm(train_data, desc="Loading data"):
        # Get English and Spanish text from dataset
        en_text = normalize_sentence(item['translation']['en'])
        es_text = normalize_sentence(item['translation']['es'])
        
        # Set source and target based on direction
        if direction == 'en-es':
            src_text = en_text
            tgt_text = es_text
        else:  # es-en
            src_text = es_text
            tgt_text = en_text
        
        # Filter by length
        src_words = len(src_text.split())
        tgt_words = len(tgt_text.split())
        
        if (min_length <= src_words <= max_length and 
            min_length <= tgt_words <= max_length):
            pairs.append((src_text, tgt_text))
        
        # Limit samples if specified
        if max_samples and len(pairs) >= max_samples:
            break
    
    print(f"Collected {len(pairs)} translation pairs")
    
    # Build vocabularies
    print("Building vocabularies...")
    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()
    
    for src, tgt in tqdm(pairs, desc="Building vocab"):
        src_vocab.add_sentence(src)
        tgt_vocab.add_sentence(tgt)
    
    # Trim vocabularies to max size for faster training
    if max_vocab_size:
        print(f"Trimming vocabularies to {max_vocab_size} most frequent words...")
        src_vocab.trim_to_size(max_vocab_size)
        tgt_vocab.trim_to_size(max_vocab_size)
    
    print(f"Source vocabulary size: {len(src_vocab)}")
    print(f"Target vocabulary size: {len(tgt_vocab)}")
    
    # Split data into train, validation, test (80%, 10%, 10%)
    total = len(pairs)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    
    train_pairs = pairs[:train_size]
    val_pairs = pairs[train_size:train_size + val_size]
    test_pairs = pairs[train_size + val_size:]
    
    print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}")
    
    return train_pairs, val_pairs, test_pairs, src_vocab, tgt_vocab


def save_data(data, filepath):
    """Save processed data to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to {filepath}")


def load_data(filepath):
    """Load processed data from disk."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    print(f"Data loaded from {filepath}")
    return data


def collate_fn(batch):
    """
    Custom collate function to handle variable length sequences.
    Pads sequences to the longest in the batch.
    """
    src_batch, tgt_batch = zip(*batch)
    
    # Get max lengths
    src_max_len = max(len(s) for s in src_batch)
    tgt_max_len = max(len(t) for t in tgt_batch)
    
    # Pad sequences
    src_padded = []
    tgt_padded = []
    
    for src, tgt in zip(src_batch, tgt_batch):
        src_pad = torch.cat([src, torch.LongTensor([0] * (src_max_len - len(src)))])
        tgt_pad = torch.cat([tgt, torch.LongTensor([0] * (tgt_max_len - len(tgt)))])
        src_padded.append(src_pad)
        tgt_padded.append(tgt_pad)
    
    return torch.stack(src_padded), torch.stack(tgt_padded)


if __name__ == "__main__":
    # Test data loading
    print("Testing data loading...")
    train_pairs, val_pairs, test_pairs, src_vocab, tgt_vocab = load_translation_data(
        direction='en-es',
        max_samples=10000
    )
    
    print("\nSample pairs:")
    for i in range(5):
        print(f"EN: {train_pairs[i][0]}")
        print(f"ES: {train_pairs[i][1]}")
        print()
    
    # Save the processed data
    save_data({
        'train': train_pairs,
        'val': val_pairs,
        'test': test_pairs,
        'src_vocab': src_vocab,
        'tgt_vocab': tgt_vocab
    }, 'data/processed/en_es_data.pkl')
