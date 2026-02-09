"""
Training configuration parameters.
Optimized for better model performance and lower loss.
"""

import torch

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model hyperparameters - Balanced for GTX 1650 GPU
EMBEDDING_DIM = 256          # Good size for word representations
ENCODER_HIDDEN_DIM = 256     # Smaller for faster training
DECODER_HIDDEN_DIM = 256     # Match encoder
NUM_LAYERS = 2               # 2 layers is good balance
DROPOUT = 0.3                # Prevent overfitting

# Training parameters - Optimized for faster training (7-8 hours total)
BATCH_SIZE = 128             # Larger batch for faster epochs
LEARNING_RATE = 0.001        # Standard learning rate
NUM_EPOCHS = 50              # Reasonable number of epochs
TEACHER_FORCING_RATIO = 0.5  # Standard TF ratio
GRAD_CLIP = 1.0              # Prevent exploding gradients

# Data parameters - Balanced for speed and quality
MAX_LENGTH = 25              # Shorter sentences for faster training
MIN_LENGTH = 3               # Filter very short sentences
MAX_SAMPLES = 50000          # Good amount of data
MAX_VOCAB_SIZE = 15000       # Limit vocab for smaller model

# Paths
DATA_DIR = 'data'
CHECKPOINT_DIR = 'checkpoints'
MODEL_DIR = 'models/translation'

# Checkpoint settings
SAVE_EVERY = 1  # Save checkpoint every N epochs
KEEP_CHECKPOINTS = 5  # Keep only the last N checkpoints

# Evaluation
BLEU_N_GRAMS = 4

# Training display
PRINT_EVERY = 100  # Print loss every N batches
