# -*- coding: utf-8 -*-
"""
Training Visualization Module
Generates plots for model training metrics
Includes loss curves, accuracy plots, and sample predictions
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


class TrainingVisualizer:
    """
    Generates visualizations for model training and evaluation
    Saves plots to the outputs directory
    """
    
    def __init__(self, output_dir=None):
        if output_dir is None:
            output_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "outputs"
            )
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        self.colors = {
            'train': '#e94560',
            'val': '#0f3460',
            'accent': '#00d26a'
        }
    
    def plot_training_history(self, history, save=True):
        """
        Plot training and validation loss/accuracy curves
        
        Args:
            history: Keras training history object
            save: Whether to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Model Training History', fontsize=14, fontweight='bold')
        
        epochs = range(1, len(history.history['loss']) + 1)
        
        # Loss plot
        ax1 = axes[0]
        ax1.plot(epochs, history.history['loss'], 
                color=self.colors['train'], linewidth=2, 
                marker='o', markersize=4, label='Training Loss')
        ax1.plot(epochs, history.history['val_loss'], 
                color=self.colors['val'], linewidth=2,
                marker='s', markersize=4, label='Validation Loss')
        ax1.set_xlabel('Epoch', fontsize=11)
        ax1.set_ylabel('Loss', fontsize=11)
        ax1.set_title('Training and Validation Loss', fontsize=12)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2 = axes[1]
        ax2.plot(epochs, history.history['accuracy'], 
                color=self.colors['train'], linewidth=2,
                marker='o', markersize=4, label='Training Accuracy')
        ax2.plot(epochs, history.history['val_accuracy'], 
                color=self.colors['val'], linewidth=2,
                marker='s', markersize=4, label='Validation Accuracy')
        ax2.set_xlabel('Epoch', fontsize=11)
        ax2.set_ylabel('Accuracy', fontsize=11)
        ax2.set_title('Training and Validation Accuracy', fontsize=12)
        ax2.legend(loc='lower right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'training_history.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Saved training history plot to {filepath}")
        
        plt.show()
        return fig
    
    def plot_loss_comparison(self, history, save=True):
        """
        Create detailed loss comparison plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(1, len(history.history['loss']) + 1)
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        
        # Plot with filled area
        ax.fill_between(epochs, train_loss, alpha=0.3, color=self.colors['train'])
        ax.fill_between(epochs, val_loss, alpha=0.3, color=self.colors['val'])
        ax.plot(epochs, train_loss, color=self.colors['train'], 
                linewidth=2.5, label='Training Loss')
        ax.plot(epochs, val_loss, color=self.colors['val'], 
                linewidth=2.5, label='Validation Loss')
        
        # Mark best epoch
        best_epoch = np.argmin(val_loss) + 1
        best_val_loss = min(val_loss)
        ax.axvline(x=best_epoch, color=self.colors['accent'], 
                   linestyle='--', linewidth=2, alpha=0.7)
        ax.scatter([best_epoch], [best_val_loss], color=self.colors['accent'], 
                   s=100, zorder=5, label=f'Best Model (Epoch {best_epoch})')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss (Categorical Crossentropy)', fontsize=12)
        ax.set_title('Model Training Progress - Loss Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add annotation
        ax.annotate(f'Best Val Loss: {best_val_loss:.4f}',
                   xy=(best_epoch, best_val_loss),
                   xytext=(best_epoch + 2, best_val_loss + 0.1),
                   fontsize=10,
                   arrowprops=dict(arrowstyle='->', color='gray'))
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'loss_comparison.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Saved loss comparison plot to {filepath}")
        
        plt.show()
        return fig
    
    def plot_sample_predictions(self, samples, save=True):
        """
        Plot sample translation predictions
        
        Args:
            samples: List of tuples (english, predicted_hindi, actual_hindi)
        """
        fig, ax = plt.subplots(figsize=(12, len(samples) * 0.8 + 2))
        
        ax.axis('off')
        ax.set_title('Sample Translation Predictions', fontsize=14, fontweight='bold', pad=20)
        
        # Create table
        cell_text = []
        for eng, pred, actual in samples:
            cell_text.append([eng, pred, actual if actual else "N/A"])
        
        table = ax.table(
            cellText=cell_text,
            colLabels=['English Input', 'Predicted Hindi', 'Actual Hindi'],
            cellLoc='center',
            loc='center',
            colWidths=[0.35, 0.35, 0.3]
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)
        
        # Style header
        for j in range(3):
            table[(0, j)].set_facecolor(self.colors['val'])
            table[(0, j)].set_text_props(color='white', fontweight='bold')
        
        # Alternate row colors
        for i in range(1, len(samples) + 1):
            for j in range(3):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'sample_predictions.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Saved sample predictions to {filepath}")
        
        plt.show()
        return fig
    
    def plot_vocabulary_distribution(self, eng_vocab, hin_vocab, save=True):
        """
        Plot vocabulary size comparison
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Bar chart comparison
        ax1 = axes[0]
        languages = ['English', 'Hindi']
        sizes = [len(eng_vocab), len(hin_vocab)]
        bars = ax1.bar(languages, sizes, color=[self.colors['train'], self.colors['val']], 
                       width=0.6, edgecolor='white', linewidth=2)
        
        ax1.set_ylabel('Vocabulary Size', fontsize=12)
        ax1.set_title('Vocabulary Size Comparison', fontsize=12, fontweight='bold')
        
        # Add value labels on bars
        for bar, size in zip(bars, sizes):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                    f'{size:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Pie chart
        ax2 = axes[1]
        ax2.pie(sizes, labels=languages, autopct='%1.1f%%',
               colors=[self.colors['train'], self.colors['val']],
               explode=(0.02, 0.02), shadow=True,
               textprops={'fontsize': 11})
        ax2.set_title('Vocabulary Distribution', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'vocabulary_distribution.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Saved vocabulary distribution to {filepath}")
        
        plt.show()
        return fig
    
    def plot_model_architecture_summary(self, model_info, save=True):
        """
        Create a visual summary of model architecture
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.95, 'Model Architecture Summary', 
               fontsize=16, fontweight='bold', ha='center', transform=ax.transAxes)
        
        # Create info text
        info_text = f"""
        ══════════════════════════════════════════════════════
        
        MODEL TYPE: Sequence-to-Sequence with Attention
        
        ══════════════════════════════════════════════════════
        
        ENCODER:
        • Input: English sentences (max length: {model_info.get('max_encoder_len', 25)})
        • Embedding: {model_info.get('embedding_dim', 256)} dimensions
        • Bidirectional LSTM: {model_info.get('lstm_units', 256)} units
        
        DECODER:
        • Input: Hindi sentences (max length: {model_info.get('max_decoder_len', 25)})
        • Embedding: {model_info.get('embedding_dim', 256)} dimensions
        • LSTM: {model_info.get('lstm_units', 256) * 2} units
        • Attention Mechanism: Bahdanau Attention
        
        VOCABULARY:
        • English: {model_info.get('eng_vocab_size', 'N/A'):,} words
        • Hindi: {model_info.get('hin_vocab_size', 'N/A'):,} words
        
        TRAINING:
        • Dataset: IIT Bombay English-Hindi Corpus
        • Source: Hugging Face (cfilt/iitb-english-hindi)
        • Samples: {model_info.get('num_samples', 'N/A'):,}
        • Optimizer: Adam (lr=0.001)
        • Loss: Categorical Crossentropy
        
        ══════════════════════════════════════════════════════
        """
        
        ax.text(0.5, 0.5, info_text, fontsize=11, ha='center', va='center',
               transform=ax.transAxes, family='monospace',
               bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'))
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'model_architecture.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Saved model architecture summary to {filepath}")
        
        plt.show()
        return fig
    
    def create_training_report(self, history, model_info, samples=None):
        """
        Generate complete training report with all visualizations
        """
        print("\n" + "="*60)
        print("GENERATING TRAINING REPORT")
        print("="*60 + "\n")
        
        # Plot training history
        print("1. Plotting training history...")
        self.plot_training_history(history)
        
        # Plot loss comparison
        print("2. Plotting loss comparison...")
        self.plot_loss_comparison(history)
        
        # Plot model architecture
        print("3. Creating model architecture summary...")
        self.plot_model_architecture_summary(model_info)
        
        # Plot vocabulary distribution
        if 'eng_vocab' in model_info and 'hin_vocab' in model_info:
            print("4. Plotting vocabulary distribution...")
            self.plot_vocabulary_distribution(
                model_info['eng_vocab'], 
                model_info['hin_vocab']
            )
        
        # Plot sample predictions
        if samples:
            print("5. Plotting sample predictions...")
            self.plot_sample_predictions(samples)
        
        print("\n" + "="*60)
        print(f"Training report saved to: {self.output_dir}")
        print("="*60 + "\n")


def create_sample_visualizations():
    """Create sample visualizations for demonstration"""
    visualizer = TrainingVisualizer()
    
    # Create mock history for demonstration
    class MockHistory:
        def __init__(self):
            np.random.seed(42)
            epochs = 25
            
            # Simulated training metrics
            train_loss = 3.5 * np.exp(-0.15 * np.arange(epochs)) + 0.3 + np.random.normal(0, 0.05, epochs)
            val_loss = 3.8 * np.exp(-0.12 * np.arange(epochs)) + 0.4 + np.random.normal(0, 0.08, epochs)
            train_acc = 1 - 0.85 * np.exp(-0.18 * np.arange(epochs)) + np.random.normal(0, 0.02, epochs)
            val_acc = 1 - 0.9 * np.exp(-0.15 * np.arange(epochs)) + np.random.normal(0, 0.03, epochs)
            
            self.history = {
                'loss': train_loss.tolist(),
                'val_loss': val_loss.tolist(),
                'accuracy': np.clip(train_acc, 0, 1).tolist(),
                'val_accuracy': np.clip(val_acc, 0, 1).tolist()
            }
    
    mock_history = MockHistory()
    
    # Model info
    model_info = {
        'max_encoder_len': 25,
        'max_decoder_len': 25,
        'embedding_dim': 256,
        'lstm_units': 256,
        'eng_vocab_size': 15000,
        'hin_vocab_size': 18000,
        'num_samples': 30000,
        'eng_vocab': ['word'] * 15000,
        'hin_vocab': ['शब्द'] * 18000
    }
    
    # Sample predictions
    samples = [
        ("hello", "नमस्ते", "नमस्ते"),
        ("how are you", "आप कैसे हैं", "आप कैसे हैं"),
        ("thank you very much", "बहुत धन्यवाद", "आपका बहुत धन्यवाद"),
        ("good morning", "सुप्रभात", "शुभ प्रभात"),
        ("i am fine", "मैं ठीक हूं", "मैं ठीक हूँ"),
    ]
    
    visualizer.create_training_report(mock_history, model_info, samples)


if __name__ == "__main__":
    create_sample_visualizations()
