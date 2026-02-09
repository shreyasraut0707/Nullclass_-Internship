"""
Translator module that uses trained models for translation.
"""

import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.seq2seq import Encoder, Decoder, Seq2Seq


class Translator:
    """Handle translation using trained models."""
    
    def __init__(self, en_to_es_model_path, es_to_en_model_path, device='cpu'):
        """
        Initialize translator with both models.
        
        Args:
            en_to_es_model_path: Path to English to Spanish model
            es_to_en_model_path: Path to Spanish to English model
            device: Device to run on ('cpu' or 'cuda')
        """
        self.device = torch.device(device)
        
        # Load English to Spanish model
        print("Loading English to Spanish model...")
        self.en_es_model, self.en_vocab, self.es_vocab = self._load_model(en_to_es_model_path)
        
        # Load Spanish to English model
        print("Loading Spanish to English model...")
        self.es_en_model, self.es_vocab_reverse, self.en_vocab_reverse = self._load_model(es_to_en_model_path)
        
        print("Models loaded successfully!")
    
    def _load_model(self, model_path):
        """Load a trained model from checkpoint."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
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
        
        model = Seq2Seq(encoder, decoder, self.device).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model, src_vocab, tgt_vocab
    
    def translate(self, text, direction='en-es', max_len=50):
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
        
        text = text.lower().strip()
        
        if direction == 'en-es':
            return self._translate_sentence(
                text,
                self.en_es_model,
                self.en_vocab,
                self.es_vocab,
                max_len
            )
        else:  # es-en
            return self._translate_sentence(
                text,
                self.es_en_model,
                self.es_vocab_reverse,
                self.en_vocab_reverse,
                max_len
            )
    
    def _translate_sentence(self, sentence, model, src_vocab, tgt_vocab, max_len):
        """Internal method to translate a sentence."""
        # Tokenize and convert to indices
        tokens = sentence.split()
        indices = [src_vocab.word2idx.get(token, src_vocab.UNK_token) for token in tokens]
        indices.append(src_vocab.EOS_token)
        
        # Convert to tensor
        src_tensor = torch.LongTensor(indices).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Encode
            encoder_outputs, hidden, cell = model.encoder(src_tensor)
            
            # Prepare decoder initial state
            hidden = model._combine_bidirectional(hidden, model.decoder.num_layers)
            cell = model._combine_bidirectional(cell, model.decoder.num_layers)
            
            # Start with SOS token
            input_token = torch.LongTensor([tgt_vocab.SOS_token]).to(self.device)
            
            output_indices = []
            
            for _ in range(max_len):
                output, hidden, cell = model.decoder(input_token, hidden, cell, encoder_outputs)
                
                # Get the most probable token
                top_token = output.argmax(1).item()
                
                if top_token == tgt_vocab.EOS_token:
                    break
                
                output_indices.append(top_token)
                input_token = torch.LongTensor([top_token]).to(self.device)
            
            # Convert indices to words
            output_tokens = [tgt_vocab.idx2word.get(idx, '<UNK>') for idx in output_indices]
            
            # Capitalize first letter
            translation = ' '.join(output_tokens)
            if translation:
                translation = translation[0].upper() + translation[1:]
            
            return translation
    
    def english_to_spanish(self, text):
        """Shortcut for English to Spanish translation."""
        return self.translate(text, direction='en-es')
    
    def spanish_to_english(self, text):
        """Shortcut for Spanish to English translation."""
        return self.translate(text, direction='es-en')


if __name__ == "__main__":
    # Test translator (requires trained models)
    en_es_path = 'checkpoints/en_es/best_model.pth'
    es_en_path = 'checkpoints/es_en/best_model.pth'
    
    if os.path.exists(en_es_path) and os.path.exists(es_en_path):
        translator = Translator(en_es_path, es_en_path)
        
        # Test English to Spanish
        en_text = "hello how are you"
        es_translation = translator.english_to_spanish(en_text)
        print(f"EN: {en_text}")
        print(f"ES: {es_translation}")
        
        # Test Spanish to English
        es_text = "hola como estas"
        en_translation = translator.spanish_to_english(es_text)
        print(f"ES: {es_text}")
        print(f"EN: {en_translation}")
    else:
        print("Models not found. Please train the models first.")
