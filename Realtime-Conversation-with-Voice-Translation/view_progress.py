"""
View training progress and checkpoint information.
"""

import os
import torch
import glob
import sys
# Add parent directory to path to find modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data.data_loader import Vocabulary  # Needed for unpickling


def view_checkpoint_info(checkpoint_path):
    """Display information about a checkpoint."""
    # weights_only=False is required to unpickle custom classes like Vocabulary
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    print(f"\nCheckpoint: {os.path.basename(checkpoint_path)}")
    print("-" * 60)
    print(f"  Epoch: {checkpoint['epoch'] + 1}")
    print(f"  Training Loss: {checkpoint['train_loss']:.4f}")
    print(f"  Validation Loss: {checkpoint['val_loss']:.4f}")
    print(f"  Best Validation Loss: {checkpoint['best_val_loss']:.4f}")
    print(f"  Source Vocabulary Size: {len(checkpoint['src_vocab'])}")
    print(f"  Target Vocabulary Size: {len(checkpoint['tgt_vocab'])}")


def view_training_progress(direction='en-es'):
    """View all checkpoints and training progress."""
    checkpoint_dir = os.path.join('checkpoints', direction.replace('-', '_'))
    
    if not os.path.exists(checkpoint_dir):
        print(f"No checkpoints found for {direction}")
        print(f"Training has not started yet.")
        return
    
    # Find all checkpoint files
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pth'))
    
    if not checkpoint_files:
        print(f"No checkpoints found in {checkpoint_dir}")
        return
    
    # Sort by epoch number
    checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    print(f"\nTraining Progress for {direction.upper()}")
    print("=" * 60)
    print(f"Checkpoint Directory: {checkpoint_dir}")
    print(f"Total Checkpoints: {len(checkpoint_files)}")
    
    # Show each checkpoint
    for checkpoint_file in checkpoint_files:
        view_checkpoint_info(checkpoint_file)
    
    # Check for best model
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        print("\n" + "=" * 60)
        view_checkpoint_info(best_model_path)
    
    # Show latest checkpoint
    if checkpoint_files:
        print("\n" + "=" * 60)
        print("Latest Checkpoint (will be used to resume training):")
        view_checkpoint_info(checkpoint_files[-1])


def view_all_progress():
    """View progress for all models."""
    print("\n" + "=" * 60)
    print("TRAINING PROGRESS SUMMARY")
    print("=" * 60)
    
    for direction in ['en-es', 'es-en']:
        view_training_progress(direction)
        print("\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        direction = sys.argv[1]
        view_training_progress(direction)
    else:
        view_all_progress()
