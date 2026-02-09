"""
Real-time Voice Conversation System
English <-> Spanish Translation with Voice Input/Output

This script demonstrates:
1. Voice input using speech recognition
2. Translation using our trained ML model + pre-trained model
3. Voice output using text-to-speech

For the internship project evaluation.
"""

import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.speech_recognition_module import SpeechRecognizer
    from src.text_to_speech import TextToSpeech
    from src.translator_pretrained import PretrainedTranslator
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure all dependencies are installed:")
    print("  pip install SpeechRecognition pyttsx3 gTTS pyaudio transformers sentencepiece")
    sys.exit(1)


class VoiceConversation:
    """Real-time voice conversation with translation."""
    
    def __init__(self):
        """Initialize the voice conversation system."""
        print("="*60)
        print("INITIALIZING VOICE CONVERSATION SYSTEM")
        print("="*60)
        
        # Initialize translator
        print("\n[1/4] Loading translation models...")
        self.translator = PretrainedTranslator()
        
        # Initialize speech recognizers
        print("\n[2/4] Initializing English speech recognizer...")
        try:
            self.en_recognizer = SpeechRecognizer(language='en-US')
        except Exception as e:
            print(f"Warning: Could not initialize English recognizer: {e}")
            self.en_recognizer = None
        
        print("\n[3/4] Initializing Spanish speech recognizer...")
        try:
            self.es_recognizer = SpeechRecognizer(language='es-ES')
        except Exception as e:
            print(f"Warning: Could not initialize Spanish recognizer: {e}")
            self.es_recognizer = None
        
        # Initialize TTS engines
        print("\n[4/4] Initializing text-to-speech engines...")
        try:
            self.en_tts = TextToSpeech(engine='pyttsx3', language='en')
            self.es_tts = TextToSpeech(engine='pyttsx3', language='es')
        except Exception as e:
            print(f"Warning: TTS initialization issue: {e}")
            # Try with gtts as fallback
            try:
                self.en_tts = TextToSpeech(engine='gtts', language='en')
                self.es_tts = TextToSpeech(engine='gtts', language='es')
            except:
                self.en_tts = None
                self.es_tts = None
        
        print("\n" + "="*60)
        print("SYSTEM READY!")
        print("="*60)
    
    def translate_english_to_spanish(self, text):
        """Translate English text to Spanish."""
        return self.translator.translate(text, 'en-es')
    
    def translate_spanish_to_english(self, text):
        """Translate Spanish text to English."""
        return self.translator.translate(text, 'es-en')
    
    def voice_english_to_spanish(self):
        """
        Voice mode: English speaker talks, system translates to Spanish
        and speaks the translation aloud.
        """
        print("\n" + "-"*60)
        print("🎤 ENGLISH TO SPANISH - Voice Mode")
        print("-"*60)
        
        if not self.en_recognizer:
            print("Error: Speech recognition not available")
            return self._fallback_english_to_spanish()
        
        print("\n🗣️  English speaker, please speak now...")
        english_text = self.en_recognizer.listen_from_microphone()
        
        if not english_text:
            print("❌ No speech detected. Please try again.")
            return False
        
        print(f"\n📝 Recognized (English): {english_text}")
        
        # Translate to Spanish
        spanish_translation = self.translate_english_to_spanish(english_text)
        print(f"🔄 Translation (Spanish): {spanish_translation}")
        
        # Speak in Spanish
        if self.es_tts:
            print("🔊 Speaking in Spanish...")
            self.es_tts.speak(spanish_translation)
        
        return True
    
    def voice_spanish_to_english(self):
        """
        Voice mode: Spanish speaker talks, system translates to English
        and speaks the translation aloud.
        """
        print("\n" + "-"*60)
        print("🎤 SPANISH TO ENGLISH - Voice Mode")
        print("-"*60)
        
        if not self.es_recognizer:
            print("Error: Speech recognition not available")
            return self._fallback_spanish_to_english()
        
        print("\n🗣️  Hablante español, por favor hable ahora...")
        spanish_text = self.es_recognizer.listen_from_microphone()
        
        if not spanish_text:
            print("❌ No se detectó voz. Por favor intente de nuevo.")
            return False
        
        print(f"\n📝 Recognized (Spanish): {spanish_text}")
        
        # Translate to English
        english_translation = self.translate_spanish_to_english(spanish_text)
        print(f"🔄 Translation (English): {english_translation}")
        
        # Speak in English
        if self.en_tts:
            print("🔊 Speaking in English...")
            self.en_tts.speak(english_translation)
        
        return True
    
    def _fallback_english_to_spanish(self):
        """Text fallback when voice is not available."""
        text = input("\n📝 Enter English text: ").strip()
        if text:
            translation = self.translate_english_to_spanish(text)
            print(f"\n🔄 Spanish: {translation}")
            if self.es_tts:
                print("🔊 Speaking in Spanish...")
                self.es_tts.speak(translation)
            return True
        return False
    
    def _fallback_spanish_to_english(self):
        """Text fallback when voice is not available."""
        text = input("\n📝 Ingrese texto en español: ").strip()
        if text:
            translation = self.translate_spanish_to_english(text)
            print(f"\n🔄 English: {translation}")
            if self.en_tts:
                print("🔊 Speaking in English...")
                self.en_tts.speak(translation)
            return True
        return False
    
    def demo_mode_text(self):
        """Demo mode with text input and voice output."""
        print("\n" + "="*60)
        print("📝 DEMO MODE - Text Input with Voice Output")
        print("="*60)
        
        while True:
            print("\n" + "-"*40)
            print("\nOptions:")
            print("1. English → Spanish (type English, hear Spanish)")
            print("2. Spanish → English (type Spanish, hear English)")
            print("3. Quit")
            
            choice = input("\nEnter choice (1/2/3): ").strip()
            
            if choice == '1':
                text = input("\n📝 Enter English text: ").strip()
                if text:
                    translation = self.translate_english_to_spanish(text)
                    print(f"\n🇬🇧 English: {text}")
                    print(f"🇪🇸 Spanish: {translation}")
                    if self.es_tts:
                        print("\n🔊 Speaking in Spanish...")
                        self.es_tts.speak(translation)
            
            elif choice == '2':
                text = input("\n📝 Ingrese texto en español: ").strip()
                if text:
                    translation = self.translate_spanish_to_english(text)
                    print(f"\n🇪🇸 Spanish: {text}")
                    print(f"🇬🇧 English: {translation}")
                    if self.en_tts:
                        print("\n🔊 Speaking in English...")
                        self.en_tts.speak(translation)
            
            elif choice == '3':
                print("\n👋 Goodbye! / ¡Adiós!")
                break
            
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
    
    def voice_conversation_mode(self):
        """Full voice conversation mode."""
        print("\n" + "="*60)
        print("🎤 VOICE CONVERSATION MODE")
        print("="*60)
        print("\nInstructions:")
        print("• Select who wants to speak")
        print("• Speak clearly into the microphone")
        print("• The system will translate and speak the result")
        
        while True:
            print("\n" + "-"*40)
            print("\nWho wants to speak?")
            print("1. 🇬🇧 English speaker → Spanish translation")
            print("2. 🇪🇸 Spanish speaker → English translation")
            print("3. 🚪 Quit")
            
            choice = input("\nEnter choice (1/2/3): ").strip()
            
            if choice == '1':
                self.voice_english_to_spanish()
            elif choice == '2':
                self.voice_spanish_to_english()
            elif choice == '3':
                print("\n👋 Thank you for using the conversation system!")
                print("   ¡Gracias por usar el sistema de conversación!")
                break
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
    
    def start(self):
        """Start the conversation system."""
        print("\n" + "="*60)
        print("🌐 REAL-TIME VOICE TRANSLATION SYSTEM")
        print("    English ↔ Spanish")
        print("="*60)
        
        print("\nSelect mode:")
        print("1. 🎤 Voice Conversation (speak into microphone)")
        print("2. 📝 Demo Mode (text input with voice output)")
        print("3. 🚪 Exit")
        
        choice = input("\nEnter choice (1/2/3): ").strip()
        
        if choice == '1':
            self.voice_conversation_mode()
        elif choice == '2':
            self.demo_mode_text()
        elif choice == '3':
            print("\n👋 Goodbye!")
        else:
            print("Invalid choice. Starting demo mode...")
            self.demo_mode_text()


def main():
    """Main entry point."""
    print("\n" + "="*60)
    print("REAL-TIME CONVERSATION WITH VOICE TRANSLATION")
    print("Internship Project - Machine Learning Based Translation")
    print("="*60)
    
    try:
        # Create and start the conversation system
        conversation = VoiceConversation()
        conversation.start()
    
    except KeyboardInterrupt:
        print("\n\n👋 Program interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
