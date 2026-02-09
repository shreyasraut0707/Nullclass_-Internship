"""
Training script for translation model with checkpoint support.
This script supports resuming training from checkpoints.
Includes learning rate scheduling for better convergence.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import time
import math

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.seq2seq import Encoder, Decoder, Seq2Seq, init_weights, count_parameters
from data.data_loader import (
    load_translation_data, save_data, load_data,
    TranslationDataset, collate_fn
)
from training.config import *


def train_epoch(model, dataloader, optimizer, criterion, clip, device, scaler=None, teacher_forcing_ratio=0.5):
    """Train for one epoch."""
    model.train()
    epoch_loss = 0
    
    for i, (src, tgt) in enumerate(tqdm(dataloader, desc="Training")):
        src, tgt = src.to(device), tgt.to(device)
        
        optimizer.zero_grad()
        
        # Mixed Precision Training
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                output = model(src, tgt, teacher_forcing_ratio)
                
                output_dim = output.shape[-1]
                output = output[:, 1:].reshape(-1, output_dim)
                tgt = tgt[:, 1:].reshape(-1)
                
                loss = criterion(output, tgt)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard Training (Fall back if no scaler)
            output = model(src, tgt, teacher_forcing_ratio)
            
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            tgt = tgt[:, 1:].reshape(-1)
            
            loss = criterion(output, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        
        epoch_loss += loss.item()
        
        if (i + 1) % PRINT_EVERY == 0:
            avg_loss = epoch_loss / (i + 1)
            print(f"  Batch {i+1}/{len(dataloader)}, Loss: {avg_loss:.4f}")
    
    return epoch_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model."""
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for src, tgt in tqdm(dataloader, desc="Evaluating"):
            src, tgt = src.to(device), tgt.to(device)
            
            output = model(src, tgt, 0)  # No teacher forcing during evaluation
            
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            tgt = tgt[:, 1:].reshape(-1)
            
            loss = criterion(output, tgt)
            epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)


def save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss, best_val_loss,
                   src_vocab, tgt_vocab, direction, checkpoint_dir):
    """Save training checkpoint with all state information."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'best_val_loss': best_val_loss,
        'src_vocab': src_vocab,
        'tgt_vocab': tgt_vocab,
        'model_config': {
            'embedding_dim': EMBEDDING_DIM,
            'encoder_hidden_dim': ENCODER_HIDDEN_DIM,
            'decoder_hidden_dim': DECODER_HIDDEN_DIM,
            'num_layers': NUM_LAYERS,
            'dropout': DROPOUT
        }
    }
    
    # Save latest checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")
    
    # Save best model
    if val_loss <= best_val_loss:
        best_path = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f"Best model updated: {best_path} (Val Loss: {val_loss:.4f})")
    
    # Clean up old checkpoints (keep only last KEEP_CHECKPOINTS)
    cleanup_old_checkpoints(checkpoint_dir, epoch)


def cleanup_old_checkpoints(checkpoint_dir, current_epoch):
    """Remove old checkpoint files, keeping only the most recent ones."""
    checkpoint_files = []
    for f in os.listdir(checkpoint_dir):
        if f.startswith('checkpoint_epoch_') and f.endswith('.pth'):
            epoch_num = int(f.split('_')[-1].split('.')[0])
            checkpoint_files.append((epoch_num, f))
    
    # Sort by epoch number
    checkpoint_files.sort(reverse=True)
    
    # Remove old checkpoints
    for epoch_num, filename in checkpoint_files[KEEP_CHECKPOINTS:]:
        filepath = os.path.join(checkpoint_dir, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"Removed old checkpoint: {filename}")


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    """Load checkpoint and restore training state."""
    # Use weights_only=False to load custom classes like Vocabulary
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state if available
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['best_val_loss']
    
    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"Resuming from epoch {start_epoch}")
    print(f"Best validation loss so far: {best_val_loss:.4f}")
    
    return start_epoch, best_val_loss, checkpoint['src_vocab'], checkpoint['tgt_vocab']


def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in the directory."""
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoint_files = []
    for f in os.listdir(checkpoint_dir):
        if f.startswith('checkpoint_epoch_') and f.endswith('.pth'):
            epoch_num = int(f.split('_')[-1].split('.')[0])
            checkpoint_files.append((epoch_num, f))
    
    if not checkpoint_files:
        return None
    
    checkpoint_files.sort(reverse=True)
    latest_file = checkpoint_files[0][1]
    return os.path.join(checkpoint_dir, latest_file)


def epoch_time(start_time, end_time):
    """Calculate time taken for an epoch."""
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def get_teacher_forcing_ratio(epoch, num_epochs):
    """Gradually decrease teacher forcing ratio during training."""
    # Start with high TF (0.7) and decrease to 0.3 by end
    start_tf = TEACHER_FORCING_RATIO
    end_tf = 0.3
    return start_tf - (start_tf - end_tf) * (epoch / num_epochs)


def train(direction='en-es', resume=True):
    """
    Main training function.
    
    Args:
        direction: 'en-es' for English to Spanish, 'es-en' for Spanish to English
        resume: Whether to resume from checkpoint if available
    """
    print("=" * 60)
    print(f"Training {direction.upper()} translation model")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Max Samples: {MAX_SAMPLES}")
    print(f"Epochs: {NUM_EPOCHS}")
    print("=" * 60)
    
    # Set up checkpoint directory
    checkpoint_dir = os.path.join(CHECKPOINT_DIR, direction.replace('-', '_'))
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Check for existing checkpoint
    checkpoint_path = None
    if resume:
        checkpoint_path = find_latest_checkpoint(checkpoint_dir)
        if checkpoint_path:
            print(f"Found checkpoint: {checkpoint_path}")
        else:
            print("No checkpoint found, starting from scratch")
    
    # Load or prepare data
    data_file = os.path.join(DATA_DIR, 'processed', f'{direction.replace("-", "_")}_data_v2.pkl')
    
    if os.path.exists(data_file) and checkpoint_path:
        print(f"Loading processed data from {data_file}")
        data = load_data(data_file)
        train_pairs = data['train']
        val_pairs = data['val']
        test_pairs = data['test']
        src_vocab = data['src_vocab']
        tgt_vocab = data['tgt_vocab']
    else:
        print("Loading fresh data from Hugging Face...")
        train_pairs, val_pairs, test_pairs, src_vocab, tgt_vocab = load_translation_data(
            direction=direction,
            max_length=MAX_LENGTH,
            min_length=MIN_LENGTH,
            max_samples=MAX_SAMPLES,
            max_vocab_size=MAX_VOCAB_SIZE
        )
        
        # Save processed data
        save_data({
            'train': train_pairs,
            'val': val_pairs,
            'test': test_pairs,
            'src_vocab': src_vocab,
            'tgt_vocab': tgt_vocab
        }, data_file)
    
    print(f"\nData Statistics:")
    print(f"  Training pairs: {len(train_pairs)}")
    print(f"  Validation pairs: {len(val_pairs)}")
    print(f"  Test pairs: {len(test_pairs)}")
    print(f"  Source vocabulary: {len(src_vocab)} words")
    print(f"  Target vocabulary: {len(tgt_vocab)} words")
    
    # Create datasets
    train_dataset = TranslationDataset(train_pairs, src_vocab, tgt_vocab)
    val_dataset = TranslationDataset(val_pairs, src_vocab, tgt_vocab)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=0  # Set to 0 to avoid multiprocessing issues on Windows
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=0
    )
    
    print(f"\nTraining batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Create model
    encoder = Encoder(
        len(src_vocab),
        EMBEDDING_DIM,
        ENCODER_HIDDEN_DIM,
        NUM_LAYERS,
        DROPOUT
    )
    decoder = Decoder(
        len(tgt_vocab),
        EMBEDDING_DIM,
        ENCODER_HIDDEN_DIM,
        DECODER_HIDDEN_DIM,
        NUM_LAYERS,
        DROPOUT
    )
    
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)
    
    # Initialize weights
    init_weights(model)
    print(f"\nModel has {count_parameters(model):,} trainable parameters")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    # Learning rate scheduler - reduce LR when validation loss plateaus
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None
    
    # Load checkpoint if resuming
    start_epoch = 0
    best_val_loss = float('inf')
    
    if checkpoint_path and resume:
        start_epoch, best_val_loss, loaded_src_vocab, loaded_tgt_vocab = load_checkpoint(
            checkpoint_path, model, optimizer, scheduler, DEVICE
        )
        src_vocab = loaded_src_vocab
        tgt_vocab = loaded_tgt_vocab
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training from epoch {start_epoch + 1} to {NUM_EPOCHS}")
    print(f"{'='*60}\n")
    
    no_improve_count = 0
    early_stop_patience = 10  # More patience for longer training
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        start_time = time.time()
        
        # Get teacher forcing ratio for this epoch
        tf_ratio = get_teacher_forcing_ratio(epoch, NUM_EPOCHS)
        
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS} (Teacher Forcing: {tf_ratio:.2f})")
        print("-" * 40)
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, GRAD_CLIP, DEVICE, scaler, tf_ratio)
        val_loss = evaluate(model, val_loader, criterion, DEVICE)
        
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        print(f"\nEpoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s | LR: {current_lr:.6f}")
        print(f"  Train Loss: {train_loss:.4f} | Train PPL: {math.exp(min(train_loss, 10)):.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val PPL: {math.exp(min(val_loss, 10)):.4f}")
        print(f"  Best Val Loss: {best_val_loss:.4f}")
        
        # Save checkpoint after every epoch
        save_checkpoint(
            model, optimizer, scheduler, epoch, train_loss, val_loss, best_val_loss,
            src_vocab, tgt_vocab, direction, checkpoint_dir
        )
        
        # Early stopping check
        if no_improve_count >= early_stop_patience:
            print(f"\nEarly stopping triggered after {early_stop_patience} epochs without improvement")
            break
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation perplexity: {math.exp(min(best_val_loss, 10)):.4f}")
    print("=" * 60)
    
    # Save final model
    final_path = os.path.join(MODEL_DIR, direction.replace('-', '_'), 'final_model.pth')
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'src_vocab': src_vocab,
        'tgt_vocab': tgt_vocab,
        'model_config': {
            'embedding_dim': EMBEDDING_DIM,
            'encoder_hidden_dim': ENCODER_HIDDEN_DIM,
            'decoder_hidden_dim': DECODER_HIDDEN_DIM,
            'num_layers': NUM_LAYERS,
            'dropout': DROPOUT
        }
    }, final_path)
    print(f"Final model saved to {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train translation model')
    parser.add_argument('--direction', type=str, default='en-es',
                       choices=['en-es', 'es-en'],
                       help='Translation direction (en-es or es-en)')
    parser.add_argument('--no-resume', action='store_true',
                       help='Start training from scratch, ignore checkpoints')
    
    args = parser.parse_args()
    
    train(direction=args.direction, resume=not args.no_resume)
