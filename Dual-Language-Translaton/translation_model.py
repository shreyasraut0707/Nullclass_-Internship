"""
Translation Model - Uses our fine-tuned Hugging Face models for translation
This module loads our trained models and provides translation functionality
"""

import os
import warnings
warnings.filterwarnings('ignore')

import torch
from transformers import MarianMTModel, MarianTokenizer


class DualLanguageTranslator:
    """Translator class using our fine-tuned MarianMT models."""
    
    def __init__(self):
        print("Initializing translation engine...")
        
        # Determine device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Get models directory
        self.models_dir = os.path.join(os.path.dirname(__file__), "models")
        
        # Check if trained models exist
        self.en_fr_path = os.path.join(self.models_dir, "en-fr")
        self.en_hi_path = os.path.join(self.models_dir, "en-hi")
        
        self.models_trained = self._check_models_exist()
        
        if self.models_trained:
            self._load_trained_models()
        else:
            self._load_base_models()
        
        print("Ready.")
    
    def _check_models_exist(self):
        """Check if our trained models exist."""
        en_fr_exists = os.path.exists(os.path.join(self.en_fr_path, "config.json"))
        en_hi_exists = os.path.exists(os.path.join(self.en_hi_path, "config.json"))
        return en_fr_exists and en_hi_exists
    
    def _load_trained_models(self):
        """Load our fine-tuned models from local storage."""
        print("Loading trained models from local storage...")
        
        # Load French model
        self.fr_tokenizer = MarianTokenizer.from_pretrained(self.en_fr_path)
        self.fr_model = MarianMTModel.from_pretrained(self.en_fr_path)
        self.fr_model.to(self.device)
        self.fr_model.eval()
        
        # Load Hindi model
        self.hi_tokenizer = MarianTokenizer.from_pretrained(self.en_hi_path)
        self.hi_model = MarianMTModel.from_pretrained(self.en_hi_path)
        self.hi_model.to(self.device)
        self.hi_model.eval()
        
        print("Trained models loaded successfully!")
    
    def _load_base_models(self):
        """Load base models from Hugging Face (fallback)."""
        print("Trained models not found. Loading base models from Hugging Face...")
        print("(Run 'python train_model.py' to train your own models)")
        
        # Load French model
        self.fr_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
        self.fr_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
        self.fr_model.to(self.device)
        self.fr_model.eval()
        
        # Load Hindi model
        self.hi_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
        self.hi_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
        self.hi_model.to(self.device)
        self.hi_model.eval()
        
        print("Base models loaded.")
    
    def translate_to_french(self, text):
        """Translate English text to French using our trained model."""
        try:
            inputs = self.fr_tokenizer(text, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.fr_model.generate(
                    **inputs,
                    max_length=256,
                    num_beams=4,
                    early_stopping=True
                )
            
            result = self.fr_tokenizer.decode(outputs[0], skip_special_tokens=True)
            return result
        except Exception as e:
            return f"Translation Error: {str(e)}"
    
    def translate_to_hindi(self, text):
        """Translate English text to Hindi using our trained model."""
        try:
            inputs = self.hi_tokenizer(text, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.hi_model.generate(
                    **inputs,
                    max_length=256,
                    num_beams=4,
                    early_stopping=True
                )
            
            result = self.hi_tokenizer.decode(outputs[0], skip_special_tokens=True)
            return result
        except Exception as e:
            return f"Translation Error: {str(e)}"
    
    def translate_dual(self, text):
        """Translate to both French and Hindi simultaneously."""
        return {
            'french': self.translate_to_french(text),
            'hindi': self.translate_to_hindi(text)
        }
    
    def get_model_status(self):
        """Return the status of loaded models."""
        if self.models_trained:
            return "Using YOUR trained models (fine-tuned)"
        else:
            return "Using base models (not trained yet)"


if __name__ == "__main__":
    print("Testing translator...")
    print()
    
    translator = DualLanguageTranslator()
    print()
    print(f"Model Status: {translator.get_model_status()}")
    print()
    
    test_text = "Good morning, how are you today?"
    print(f"Input: {test_text}")
    
    result = translator.translate_dual(test_text)
    print(f"French: {result['french']}")
    print(f"Hindi: {result['hindi']}")
