"""
Real-time conversation system integrating speech recognition, translation, and TTS.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.speech_recognition_module import SpeechRecognizer
from src.text_to_speech import TextToSpeech
from src.translator import Translator


class ConversationSystem:
    """Manage real-time bilingual conversation."""
    
    def __init__(self, translator, tts_engine='pyttsx3'):
        """
        Initialize conversation system.
        
        Args:
            translator: Translator instance with loaded models
            tts_engine: TTS engine to use ('pyttsx3' or 'gtts')
        """
        self.translator = translator
        
        # Initialize speech recognizers
        self.en_recognizer = SpeechRecognizer(language='en-US')
        self.es_recognizer = SpeechRecognizer(language='es-ES')
        
        # Initialize TTS engines
        self.en_tts = TextToSpeech(engine=tts_engine, language='en')
        self.es_tts = TextToSpeech(engine=tts_engine, language='es')
        
        print("Conversation system initialized!")
    
    def english_to_spanish_conversation(self):
        """Handle English speaker's turn."""
        print("\n" + "="*60)
        print("English Speaker's Turn")
        print("="*60)
        
        # Listen to English speech
        print("\nEnglish speaker, please speak now...")
        english_text = self.en_recognizer.listen_from_microphone()
        
        if not english_text:
            print("No speech detected. Please try again.")
            return False
        
        print(f"\nRecognized (English): {english_text}")
        
        # Translate to Spanish
        print("Translating to Spanish...")
        spanish_translation = self.translator.english_to_spanish(english_text)
        print(f"Translation (Spanish): {spanish_translation}")
        
        # Speak in Spanish
        print("Speaking translation in Spanish...")
        self.es_tts.speak(spanish_translation)
        
        return True
    
    def spanish_to_english_conversation(self):
        """Handle Spanish speaker's turn."""
        print("\n" + "="*60)
        print("Spanish Speaker's Turn")
        print("="*60)
        
        # Listen to Spanish speech
        print("\nSpanish speaker, please speak now...")
        spanish_text = self.es_recognizer.listen_from_microphone()
        
        if not spanish_text:
            print("No se detecto voz. Por favor intente de nuevo.")
            return False
        
        print(f"\nRecognized (Spanish): {spanish_text}")
        
        # Translate to English
        print("Translating to English...")
        english_translation = self.translator.spanish_to_english(spanish_text)
        print(f"Translation (English): {english_translation}")
        
        # Speak in English
        print("Speaking translation in English...")
        self.en_tts.speak(english_translation)
        
        return True
    
    def start_conversation(self):
        """Start the conversation loop."""
        print("\n" + "="*60)
        print("REAL-TIME BILINGUAL CONVERSATION SYSTEM")
        print("English <-> Spanish")
        print("="*60)
        print("\nInstructions:")
        print("1. Each turn, select which language you'll speak")
        print("2. Speak clearly into the microphone")
        print("3. The system will translate and speak in the other language")
        print("4. Type 'quit' to exit")
        print("\n")
        
        while True:
            print("\n" + "-"*60)
            choice = input("\nWho wants to speak?\n1. English speaker\n2. Spanish speaker\n3. Quit\nEnter choice (1/2/3): ").strip()
            
            if choice == '1':
                self.english_to_spanish_conversation()
            elif choice == '2':
                self.spanish_to_english_conversation()
            elif choice == '3':
                print("\nThank you for using the conversation system!")
                print("Gracias por usar el sistema de conversacion!")
                break
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
    
    def demo_mode(self):
        """Run a demo conversation with text input instead of voice."""
        print("\n" + "="*60)
        print("DEMO MODE - Text Input")
        print("="*60)
        
        while True:
            print("\n" + "-"*60)
            choice = input("\nChoose language to translate FROM:\n1. English to Spanish\n2. Spanish to English\n3. Quit\nEnter choice (1/2/3): ").strip()
            
            if choice == '1':
                text = input("\nEnter English text: ").strip()
                if text:
                    translation = self.translator.english_to_spanish(text)
                    print(f"\nEnglish: {text}")
                    print(f"Spanish: {translation}")
                    print("\nSpeaking in Spanish...")
                    self.es_tts.speak(translation)
            
            elif choice == '2':
                text = input("\nEnter Spanish text: ").strip()
                if text:
                    translation = self.translator.spanish_to_english(text)
                    print(f"\nSpanish: {text}")
                    print(f"English: {translation}")
                    print("\nSpeaking in English...")
                    self.en_tts.speak(translation)
            
            elif choice == '3':
                print("\nExiting demo mode...")
                break
            
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    print("Testing Conversation System...")
    
    # Check if models exist
    en_es_path = 'checkpoints/en_es/best_model.pth'
    es_en_path = 'checkpoints/es_en/best_model.pth'
    
    if not os.path.exists(en_es_path) or not os.path.exists(es_en_path):
        print("Error: Trained models not found!")
        print("Please train the models first using:")
        print("  python training/train_translator.py --direction en-es")
        print("  python training/train_translator.py --direction es-en")
        sys.exit(1)
    
    # Initialize translator
    translator = Translator(en_es_path, es_en_path)
    
    # Create conversation system
    conversation = ConversationSystem(translator, tts_engine='pyttsx3')
    
    # Start conversation
    conversation.start_conversation()
