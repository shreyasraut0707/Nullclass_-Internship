"""Model utility functions."""

import torch
import os


def save_model(model, filepath, **kwargs):
    """Save model to file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        **kwargs
    }, filepath)


def load_model(model, filepath, device='cpu'):
    """Load model from file."""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device():
    """Get available device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
