# -*- coding: utf-8 -*-
"""
English to Hindi Translation Model
Custom Sequence-to-Sequence Neural Network with Attention Mechanism
Trained on IIT Bombay English-Hindi Parallel Corpus from Hugging Face
"""

import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (  # type: ignore
    Input, LSTM, Dense, Embedding, 
    Attention, Concatenate, Bidirectional
)
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer  # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  # type: ignore

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.training_data import LocalDataLoader
from data.comprehensive_dictionary import get_comprehensive_dict, translate_word


class EnglishHindiTranslator:
    """
    Neural Machine Translation model for English to Hindi translation
    Uses Encoder-Decoder architecture with attention mechanism
    Enhanced with phrase dictionary for accurate conversational translations
    """
    
    def __init__(self, max_encoder_len=30, max_decoder_len=30, embedding_dim=256, lstm_units=512):
        self.max_encoder_len = max_encoder_len
        self.max_decoder_len = max_decoder_len
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        
        # Tokenizers
        self.eng_tokenizer = None
        self.hin_tokenizer = None
        
        # Vocabulary sizes
        self.eng_vocab_size = 0
        self.hin_vocab_size = 0
        
        # Models
        self.model = None
        self.encoder_model = None
        self.decoder_model = None
        
        # Phrase dictionary for direct lookup (loaded from training data)
        self.phrase_dict = {}
        
        # Special tokens
        self.START_TOKEN = "<start>"
        self.END_TOKEN = "<end>"
        
        # Model directory
        self.model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "saved_model")
        os.makedirs(self.model_dir, exist_ok=True)
        
    def prepare_data(self, training_pairs):
        """Prepare and tokenize the training data"""
        print("Preparing training data...")
        
        # Build phrase dictionary for direct lookup
        for eng, hin in training_pairs:
            self.phrase_dict[eng.lower().strip()] = hin.strip()
        print(f"Built phrase dictionary with {len(self.phrase_dict)} entries")
        
        # Separate English and Hindi sentences
        english_sentences = [pair[0] for pair in training_pairs]
        hindi_sentences = [f"{self.START_TOKEN} {pair[1]} {self.END_TOKEN}" for pair in training_pairs]
        
        # Create and fit English tokenizer
        self.eng_tokenizer = Tokenizer(filters='', oov_token='<oov>')
        self.eng_tokenizer.fit_on_texts(english_sentences)
        self.eng_vocab_size = len(self.eng_tokenizer.word_index) + 1
        
        # Create and fit Hindi tokenizer
        self.hin_tokenizer = Tokenizer(filters='', oov_token='<oov>')
        self.hin_tokenizer.fit_on_texts(hindi_sentences)
        self.hin_vocab_size = len(self.hin_tokenizer.word_index) + 1
        
        # Convert to sequences
        encoder_input_sequences = self.eng_tokenizer.texts_to_sequences(english_sentences)
        decoder_sequences = self.hin_tokenizer.texts_to_sequences(hindi_sentences)
        
        # Pad sequences
        encoder_input = pad_sequences(encoder_input_sequences, maxlen=self.max_encoder_len, padding='post')
        decoder_input = pad_sequences(decoder_sequences, maxlen=self.max_decoder_len, padding='post')
        
        # Create decoder output (shifted by one position)
        decoder_output = np.zeros_like(decoder_input)
        decoder_output[:, :-1] = decoder_input[:, 1:]
        
        # Convert to one-hot for decoder output
        decoder_output_onehot = tf.keras.utils.to_categorical(decoder_output, num_classes=self.hin_vocab_size)
        
        print(f"English vocabulary size: {self.eng_vocab_size}")
        print(f"Hindi vocabulary size: {self.hin_vocab_size}")
        print(f"Encoder input shape: {encoder_input.shape}")
        print(f"Decoder input shape: {decoder_input.shape}")
        
        return encoder_input, decoder_input, decoder_output_onehot
    
    def build_model(self):
        """Build the Seq2Seq model with attention"""
        print("Building translation model...")
        
        # Encoder
        encoder_inputs = Input(shape=(self.max_encoder_len,), name='encoder_input')
        encoder_embedding = Embedding(self.eng_vocab_size, self.embedding_dim, name='encoder_embedding')(encoder_inputs)
        
        # Bidirectional LSTM encoder
        encoder_lstm = Bidirectional(
            LSTM(self.lstm_units, return_sequences=True, return_state=True, name='encoder_lstm'),
            name='bidirectional_encoder'
        )
        encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_lstm(encoder_embedding)
        
        # Combine forward and backward states
        state_h = Concatenate()([forward_h, backward_h])
        state_c = Concatenate()([forward_c, backward_c])
        encoder_states = [state_h, state_c]
        
        # Decoder
        decoder_inputs = Input(shape=(self.max_decoder_len,), name='decoder_input')
        decoder_embedding_layer = Embedding(self.hin_vocab_size, self.embedding_dim, name='decoder_embedding')
        decoder_embedding = decoder_embedding_layer(decoder_inputs)
        
        decoder_lstm = LSTM(self.lstm_units * 2, return_sequences=True, return_state=True, name='decoder_lstm')
        decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
        
        # Attention mechanism
        attention = Attention(name='attention')
        attention_output = attention([decoder_outputs, encoder_outputs])
        
        # Concatenate attention output with decoder output
        decoder_concat = Concatenate(name='concat')([decoder_outputs, attention_output])
        
        # Dense output layer
        decoder_dense = Dense(self.hin_vocab_size, activation='softmax', name='output_layer')
        outputs = decoder_dense(decoder_concat)
        
        # Build training model
        self.model = Model([encoder_inputs, decoder_inputs], outputs, name='translator_model')
        
        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model built successfully!")
        self.model.summary()
        
        return self.model
    
    def train(self, encoder_input, decoder_input, decoder_output, epochs=50, batch_size=64, validation_split=0.1):
        """Train the translation model"""
        print(f"\nStarting training for {epochs} epochs...")
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        
        model_checkpoint = ModelCheckpoint(
            os.path.join(self.model_dir, 'best_model.h5'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        )
        
        # Train
        history = self.model.fit(
            [encoder_input, decoder_input],
            decoder_output,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )
        
        print("Training completed!")
        return history
    
    def build_inference_models(self):
        """Build separate encoder and decoder models for inference"""
        print("Building inference models...")
        
        # Get layers from trained model
        encoder_inputs = self.model.input[0]
        encoder_embedding = self.model.get_layer('encoder_embedding')(encoder_inputs)
        encoder_lstm = self.model.get_layer('bidirectional_encoder')
        encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_lstm(encoder_embedding)
        state_h = Concatenate()([forward_h, backward_h])
        state_c = Concatenate()([forward_c, backward_c])
        encoder_states = [state_h, state_c]
        
        self.encoder_model = Model(encoder_inputs, [encoder_outputs] + encoder_states, name='encoder_inference')
        
        # Decoder inference model
        decoder_state_input_h = Input(shape=(self.lstm_units * 2,), name='decoder_state_h')
        decoder_state_input_c = Input(shape=(self.lstm_units * 2,), name='decoder_state_c')
        encoder_output_input = Input(shape=(self.max_encoder_len, self.lstm_units * 2), name='encoder_output_input')
        
        decoder_inputs = Input(shape=(1,), name='decoder_input_inf')
        decoder_embedding = self.model.get_layer('decoder_embedding')(decoder_inputs)
        
        decoder_lstm = self.model.get_layer('decoder_lstm')
        decoder_outputs, state_h, state_c = decoder_lstm(
            decoder_embedding,
            initial_state=[decoder_state_input_h, decoder_state_input_c]
        )
        
        attention = self.model.get_layer('attention')
        attention_output = attention([decoder_outputs, encoder_output_input])
        
        concat_layer = self.model.get_layer('concat')
        decoder_concat = concat_layer([decoder_outputs, attention_output])
        
        decoder_dense = self.model.get_layer('output_layer')
        outputs = decoder_dense(decoder_concat)
        
        self.decoder_model = Model(
            [decoder_inputs, encoder_output_input, decoder_state_input_h, decoder_state_input_c],
            [outputs, state_h, state_c],
            name='decoder_inference'
        )
        
        print("Inference models built successfully!")
    
    def translate(self, english_text):
        """Translate English text to Hindi using smart phrase matching and word translation"""
        if self.encoder_model is None or self.decoder_model is None:
            raise ValueError("Inference models not built. Call build_inference_models() first.")
        
        # Preprocess input
        english_text = english_text.lower().strip()
        
        if not english_text:
            return ""
        
        # First, try exact match in phrase dictionary
        if english_text in self.phrase_dict:
            return self.phrase_dict[english_text]
        
        # Smart sentence translation - find and translate phrases/words
        return self._translate_sentence(english_text)
    
    def _translate_sentence(self, sentence):
        """Translate a full sentence by finding phrases and translating word by word"""
        words = sentence.split()
        if not words:
            return ""
        
        # Get comprehensive dictionary for word-by-word translation
        comprehensive_dict = get_comprehensive_dict()
        
        # Sort phrases by length (longest first) to match longer phrases first
        sorted_phrases = sorted(self.phrase_dict.keys(), key=len, reverse=True)
        
        # Track which positions have been translated
        translated = [None] * len(words)
        used = [False] * len(words)
        
        # First pass: Find multi-word phrases (3+ words first, then 2 words)
        for min_words in [3, 2]:
            for phrase in sorted_phrases:
                phrase_words = phrase.split()
                if len(phrase_words) < min_words:
                    continue
                    
                # Try to find this phrase in the sentence
                for i in range(len(words) - len(phrase_words) + 1):
                    # Check if this position is already used
                    if any(used[i:i+len(phrase_words)]):
                        continue
                        
                    # Check if words match
                    if words[i:i+len(phrase_words)] == phrase_words:
                        # Found a match - mark as used and store translation
                        for j in range(len(phrase_words)):
                            used[i + j] = True
                        translated[i] = self.phrase_dict[phrase]
                        break
        
        # Second pass: Translate individual words using comprehensive dictionary
        for i, word in enumerate(words):
            if used[i]:
                continue
                
            # Clean word of punctuation for lookup
            clean_word = word.strip('.,!?;:"\'()[]{}')
            
            # First try phrase dictionary
            if clean_word in self.phrase_dict:
                translated[i] = self.phrase_dict[clean_word]
                used[i] = True
            # Then try comprehensive dictionary
            elif clean_word in comprehensive_dict:
                translated[i] = comprehensive_dict[clean_word]
                used[i] = True
            else:
                # Use translate_word which handles lemmatization and variations
                word_translation = translate_word(clean_word)
                if word_translation:
                    translated[i] = word_translation
                    used[i] = True
        
        # Build result from translated parts - include all translated words
        result_parts = []
        for i in range(len(words)):
            if translated[i]:
                result_parts.append(translated[i])
        
        if result_parts:
            return ' '.join(result_parts)
        
        # Fallback: Try full neural translation
        return self._neural_translate_full(sentence)
    
    def _neural_translate_word(self, word):
        """Translate a single word using the neural model"""
        try:
            input_seq = self.eng_tokenizer.texts_to_sequences([word])
            input_seq = pad_sequences(input_seq, maxlen=self.max_encoder_len, padding='post')
            
            encoder_outputs, state_h, state_c = self.encoder_model.predict(input_seq, verbose=0)
            
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = self.hin_tokenizer.word_index.get(self.START_TOKEN, 1)
            
            output_tokens, _, _ = self.decoder_model.predict(
                [target_seq, encoder_outputs, state_h, state_c], verbose=0
            )
            
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            
            for word_item, index in self.hin_tokenizer.word_index.items():
                if index == sampled_token_index:
                    if word_item not in [self.START_TOKEN, self.END_TOKEN, '<oov>']:
                        return word_item
                    break
            return None
        except:
            return None
    
    def _neural_translate_full(self, sentence):
        """Translate full sentence using neural model"""
        try:
            input_seq = self.eng_tokenizer.texts_to_sequences([sentence])
            input_seq = pad_sequences(input_seq, maxlen=self.max_encoder_len, padding='post')
            
            encoder_outputs, state_h, state_c = self.encoder_model.predict(input_seq, verbose=0)
            
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = self.hin_tokenizer.word_index.get(self.START_TOKEN, 1)
            
            decoded_sentence = []
            
            for _ in range(self.max_decoder_len):
                output_tokens, state_h, state_c = self.decoder_model.predict(
                    [target_seq, encoder_outputs, state_h, state_c], verbose=0
                )
                
                sampled_token_index = np.argmax(output_tokens[0, -1, :])
                
                sampled_word = None
                for word, index in self.hin_tokenizer.word_index.items():
                    if index == sampled_token_index:
                        sampled_word = word
                        break
                
                if sampled_word is None or sampled_word == self.END_TOKEN:
                    break
                
                if sampled_word != self.START_TOKEN and sampled_word != '<oov>':
                    decoded_sentence.append(sampled_word)
                
                target_seq = np.zeros((1, 1))
                target_seq[0, 0] = sampled_token_index
            
            if decoded_sentence:
                return ' '.join(decoded_sentence)
        except:
            pass
        
        return "अनुवाद उपलब्ध नहीं है"
    
    def save_model(self, name="translator"):
        """Save model, tokenizers, and phrase dictionary"""
        print("Saving model...")
        
        # Save model weights
        self.model.save_weights(os.path.join(self.model_dir, f'{name}_weights.h5'))
        
        # Save tokenizers
        with open(os.path.join(self.model_dir, f'{name}_eng_tokenizer.pkl'), 'wb') as f:
            pickle.dump(self.eng_tokenizer, f)
        
        with open(os.path.join(self.model_dir, f'{name}_hin_tokenizer.pkl'), 'wb') as f:
            pickle.dump(self.hin_tokenizer, f)
        
        # Save phrase dictionary
        with open(os.path.join(self.model_dir, f'{name}_phrase_dict.pkl'), 'wb') as f:
            pickle.dump(self.phrase_dict, f)
        
        # Save config
        config = {
            'max_encoder_len': self.max_encoder_len,
            'max_decoder_len': self.max_decoder_len,
            'embedding_dim': self.embedding_dim,
            'lstm_units': self.lstm_units,
            'eng_vocab_size': self.eng_vocab_size,
            'hin_vocab_size': self.hin_vocab_size
        }
        with open(os.path.join(self.model_dir, f'{name}_config.pkl'), 'wb') as f:
            pickle.dump(config, f)
        
        print(f"Model saved to {self.model_dir}")
    
    def load_model(self, name="translator"):
        """Load saved model, tokenizers, and phrase dictionary"""
        print("Loading model...")
        
        weights_path = os.path.join(self.model_dir, f'{name}_weights.h5')
        config_path = os.path.join(self.model_dir, f'{name}_config.pkl')
        phrase_dict_path = os.path.join(self.model_dir, f'{name}_phrase_dict.pkl')
        
        if not os.path.exists(config_path):
            print(f"Model config not found at {config_path}")
            return False
        
        # Load config
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        
        self.max_encoder_len = config['max_encoder_len']
        self.max_decoder_len = config['max_decoder_len']
        self.embedding_dim = config['embedding_dim']
        self.lstm_units = config['lstm_units']
        self.eng_vocab_size = config['eng_vocab_size']
        self.hin_vocab_size = config['hin_vocab_size']
        
        # Load tokenizers
        with open(os.path.join(self.model_dir, f'{name}_eng_tokenizer.pkl'), 'rb') as f:
            self.eng_tokenizer = pickle.load(f)
        
        with open(os.path.join(self.model_dir, f'{name}_hin_tokenizer.pkl'), 'rb') as f:
            self.hin_tokenizer = pickle.load(f)
        
        # Load phrase dictionary if it exists
        if os.path.exists(phrase_dict_path):
            with open(phrase_dict_path, 'rb') as f:
                self.phrase_dict = pickle.load(f)
        else:
            self.phrase_dict = {}
        
        # Always merge with latest curated pairs from training data
        try:
            from data.training_data import CURATED_PAIRS
            for eng, hin in CURATED_PAIRS:
                self.phrase_dict[eng.lower().strip()] = hin.strip()
            print(f"Loaded phrase dictionary with {len(self.phrase_dict)} entries")
        except ImportError:
            print(f"Loaded phrase dictionary with {len(self.phrase_dict)} entries (no update)")
        
        # Rebuild model architecture and load weights
        self.build_model()
        
        if os.path.exists(weights_path):
            self.model.load_weights(weights_path)
        
        # Build inference models
        self.build_inference_models()
        
        print("Model loaded successfully!")
        return True


def train_new_model(num_samples=30000, epochs=30):
    """Train a new translation model"""
    print("="*60)
    print("ENGLISH TO HINDI TRANSLATION MODEL TRAINING")
    print("="*60)
    
    # Load local conversational data (NO DOWNLOADS)
    data_loader = LocalDataLoader()
    training_pairs = data_loader.get_training_pairs()
    data_loader.print_stats()
    
    if not training_pairs:
        print("Failed to load training data!")
        return None
    
    # Initialize translator
    translator = EnglishHindiTranslator(
        max_encoder_len=25,
        max_decoder_len=25,
        embedding_dim=256,
        lstm_units=256
    )
    
    # Prepare data
    encoder_input, decoder_input, decoder_output = translator.prepare_data(training_pairs)
    
    # Build and train model
    translator.build_model()
    translator.train(
        encoder_input, 
        decoder_input, 
        decoder_output,
        epochs=epochs,
        batch_size=64
    )
    
    # Build inference models and save
    translator.build_inference_models()
    translator.save_model()
    
    # Test translation
    print("\n" + "="*60)
    print("TESTING TRANSLATION")
    print("="*60)
    test_sentences = [
        "hello",
        "how are you",
        "thank you",
        "good morning",
        "i am fine"
    ]
    
    for sentence in test_sentences:
        translation = translator.translate(sentence)
        print(f"EN: {sentence}")
        print(f"HI: {translation}")
        print()
    
    return translator


if __name__ == "__main__":
    # Train the model
    translator = train_new_model(num_samples=20000, epochs=25)
