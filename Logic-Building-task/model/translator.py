"""
Translator Module
Handles loading the trained model and performing English to Hindi translation.
"""

import os
import torch
from transformers import MarianMTModel, MarianTokenizer


# Path to saved model or fallback to pre-trained
SAVED_MODEL_PATH = "model/saved_model"
PRETRAINED_MODEL = "Helsinki-NLP/opus-mt-en-hi"


class EnglishHindiTranslator:
    """
    Translator class that loads the trained MarianMT model
    and provides word translation functionality.
    """
    
    def __init__(self):
        """Initialize the translator by loading the model."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        self.model_source = None
        
    def load_model(self):
        """
        Load the translation model.
        First tries to load the fine-tuned model, falls back to pre-trained.
        """
        try:
            # Try to load saved/fine-tuned model first
            if os.path.exists(SAVED_MODEL_PATH) and os.listdir(SAVED_MODEL_PATH):
                print(f"Loading fine-tuned model from: {SAVED_MODEL_PATH}")
                self.tokenizer = MarianTokenizer.from_pretrained(SAVED_MODEL_PATH)
                self.model = MarianMTModel.from_pretrained(SAVED_MODEL_PATH)
                self.model_source = "Fine-tuned Model"
            else:
                # Fall back to pre-trained model
                print(f"Loading pre-trained model: {PRETRAINED_MODEL}")
                self.tokenizer = MarianTokenizer.from_pretrained(PRETRAINED_MODEL)
                self.model = MarianMTModel.from_pretrained(PRETRAINED_MODEL)
                self.model_source = "Pre-trained Model (Helsinki-NLP/opus-mt-en-hi)"
            
            # Move to device
            self.model = self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            self.is_loaded = True
            print(f"Model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self.is_loaded = False
            return False
    
    def translate(self, english_word: str) -> str:
        """
        Translate an English word to Hindi.
        
        Args:
            english_word: The English word to translate
            
        Returns:
            Hindi translation or error message
        """
        if not self.is_loaded:
            if not self.load_model():
                return "Error: Model not loaded"
        
        try:
            # Clean input
            english_word = english_word.strip().lower()
            
            # Dictionary lookup for common words (override ML for accuracy)
            common_translations = {
                "hello": "नमस्ते",
                "hi": "नमस्ते",
                "bye": "अलविदा",
                "goodbye": "अलविदा",
                "thanks": "धन्यवाद",
                "thank": "धन्यवाद",
                "please": "कृपया",
                "sorry": "क्षमा करें",
                "welcome": "स्वागत है",
                "yes": "हाँ",
                "no": "नहीं",
                "okay": "ठीक है",
                "ok": "ठीक है",
            }
            
            if english_word in common_translations:
                return common_translations[english_word]
            
            # Tokenize input
            inputs = self.tokenizer(
                english_word,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=32
            ).to(self.device)
            
            # Generate translation
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=32,
                    num_beams=4,
                    early_stopping=True
                )
            
            # Decode output
            hindi_translation = self.tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True
            )
            
            return hindi_translation
            
        except Exception as e:
            return f"Translation error: {str(e)}"
    
    def get_model_status(self) -> str:
        """
        Get the current model status.
        
        Returns:
            String describing model status
        """
        if self.is_loaded:
            return f"✓ {self.model_source} ({self.device.upper()})"
        else:
            return "✗ Model not loaded"


# Global translator instance
_translator = None


def get_translator() -> EnglishHindiTranslator:
    """
    Get the global translator instance.
    Creates a new instance if one doesn't exist.
    
    Returns:
        EnglishHindiTranslator instance
    """
    global _translator
    if _translator is None:
        _translator = EnglishHindiTranslator()
    return _translator


def translate_word(english_word: str) -> str:
    """
    Convenience function to translate a word.
    
    Args:
        english_word: English word to translate
        
    Returns:
        Hindi translation
    """
    translator = get_translator()
    return translator.translate(english_word)
