"""
Translator module using pre-trained MarianMT models for high-quality translation.
Uses Helsinki-NLP models from Hugging Face.
"""

from transformers import MarianMTModel, MarianTokenizer
import torch


class PretrainedTranslator:
    """Handle translation using pre-trained MarianMT models."""
    
    def __init__(self, device=None):
        """
        Initialize translator with pre-trained models.
        
        Args:
            device: Device to run on ('cpu' or 'cuda')
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load English to Spanish model
        print("Loading English to Spanish model (Helsinki-NLP/opus-mt-en-es)...")
        self.en_es_model_name = "Helsinki-NLP/opus-mt-en-es"
        self.en_es_tokenizer = MarianTokenizer.from_pretrained(self.en_es_model_name)
        self.en_es_model = MarianMTModel.from_pretrained(self.en_es_model_name).to(self.device)
        self.en_es_model.eval()
        
        # Load Spanish to English model
        print("Loading Spanish to English model (Helsinki-NLP/opus-mt-es-en)...")
        self.es_en_model_name = "Helsinki-NLP/opus-mt-es-en"
        self.es_en_tokenizer = MarianTokenizer.from_pretrained(self.es_en_model_name)
        self.es_en_model = MarianMTModel.from_pretrained(self.es_en_model_name).to(self.device)
        self.es_en_model.eval()
        
        print("Models loaded successfully!")
    
    def translate(self, text, direction='en-es', max_len=128):
        """
        Translate text.
        
        Args:
            text: Input text to translate
            direction: 'en-es' for English to Spanish, 'es-en' for Spanish to English
            max_len: Maximum output length
        
        Returns:
            Translated text
        """
        if not text or not text.strip():
            return ""
        
        text = text.strip()
        
        if direction == 'en-es':
            return self._translate(text, self.en_es_tokenizer, self.en_es_model, max_len)
        else:  # es-en
            return self._translate(text, self.es_en_tokenizer, self.es_en_model, max_len)
    
    def _translate(self, text, tokenizer, model, max_len):
        """Internal method to translate using a specific model."""
        with torch.no_grad():
            # Tokenize input
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate translation
            translated = model.generate(**inputs, max_length=max_len)
            
            # Decode output
            translation = tokenizer.decode(translated[0], skip_special_tokens=True)
            
            return translation


# For backward compatibility, also provide the Translator alias
Translator = PretrainedTranslator


if __name__ == "__main__":
    # Test the translator
    translator = PretrainedTranslator()
    
    test_sentences = [
        ("Hello, how are you?", "en-es"),
        ("I love you", "en-es"),
        ("What is your name?", "en-es"),
        ("The book is on the table", "en-es"),
        ("Good morning, my friend", "en-es"),
        ("Hola, ¿cómo estás?", "es-en"),
        ("Te quiero mucho", "es-en"),
        ("¿Cuál es tu nombre?", "es-en"),
    ]
    
    print("\n" + "="*60)
    print("Translation Tests")
    print("="*60)
    
    for text, direction in test_sentences:
        translation = translator.translate(text, direction)
        arrow = "→" if direction == "en-es" else "→"
        lang = "EN→ES" if direction == "en-es" else "ES→EN"
        print(f"[{lang}] {text}")
        print(f"        {translation}")
        print()
