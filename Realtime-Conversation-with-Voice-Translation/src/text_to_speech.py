"""
Text-to-speech module for converting text to voice.
Supports both English and Spanish.
"""

import pyttsx3
from gtts import gTTS
import os
import tempfile
import platform


class TextToSpeech:
    """Handle text-to-speech conversion."""
    
    def __init__(self, engine='pyttsx3', language='en'):
        """
        Initialize TTS engine.
        
        Args:
            engine: 'pyttsx3' for offline or 'gtts' for online
            language: 'en' for English, 'es' for Spanish
        """
        self.engine_type = engine
        self.language = language
        
        if engine == 'pyttsx3':
            self.engine = pyttsx3.init()
            self._setup_pyttsx3()
        else:
            self.engine = None
    
    def _setup_pyttsx3(self):
        """Configure pyttsx3 engine."""
        voices = self.engine.getProperty('voices')
        
        # Try to set appropriate voice based on language
        if self.language == 'es':
            # Look for Spanish voice
            for voice in voices:
                if 'spanish' in voice.name.lower() or 'es' in voice.id.lower():
                    self.engine.setProperty('voice', voice.id)
                    break
        else:
            # Use first English voice
            if voices:
                self.engine.setProperty('voice', voices[0].id)
        
        # Set speech rate
        rate = self.engine.getProperty('rate')
        self.engine.setProperty('rate', rate - 20)  # Slow down a bit
        
        # Set volume
        self.engine.setProperty('volume', 1.0)
    
    def speak(self, text):
        """
        Convert text to speech and play it.
        
        Args:
            text: Text to speak
        """
        if not text:
            print("No text to speak")
            return
        
        try:
            if self.engine_type == 'pyttsx3':
                self._speak_pyttsx3(text)
            else:
                self._speak_gtts(text)
        except Exception as e:
            print(f"Error speaking: {e}")
            # Fallback to alternative engine
            try:
                if self.engine_type == 'pyttsx3':
                    self._speak_gtts(text)
                else:
                    self._speak_pyttsx3(text)
            except:
                print("All TTS engines failed")
    
    def _speak_pyttsx3(self, text):
        """Speak using pyttsx3 (offline)."""
        self.engine.say(text)
        self.engine.runAndWait()
    
    def _speak_gtts(self, text):
        """Speak using gTTS (online)."""
        # Map language codes
        lang_code = 'es' if self.language == 'es' else 'en'
        
        # Create TTS
        tts = gTTS(text=text, lang=lang_code, slow=False)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            temp_file = fp.name
            tts.save(temp_file)
        
        # Play the file
        self._play_audio(temp_file)
        
        # Clean up
        try:
            os.remove(temp_file)
        except:
            pass
    
    def _play_audio(self, audio_file):
        """Play audio file based on OS."""
        system = platform.system()
        
        try:
            if system == 'Windows':
                os.system(f'start {audio_file}')
            elif system == 'Darwin':  # macOS
                os.system(f'afplay {audio_file}')
            else:  # Linux
                os.system(f'mpg123 {audio_file}')
        except Exception as e:
            print(f"Error playing audio: {e}")
    
    def save_to_file(self, text, filename):
        """
        Save speech to audio file.
        
        Args:
            text: Text to convert
            filename: Output file path
        """
        try:
            if self.engine_type == 'pyttsx3':
                self.engine.save_to_file(text, filename)
                self.engine.runAndWait()
            else:
                lang_code = 'es' if self.language == 'es' else 'en'
                tts = gTTS(text=text, lang=lang_code)
                tts.save(filename)
            print(f"Audio saved to {filename}")
        except Exception as e:
            print(f"Error saving audio: {e}")
    
    def set_language(self, language):
        """Change the TTS language."""
        self.language = language
        if self.engine_type == 'pyttsx3' and self.engine:
            self._setup_pyttsx3()
        print(f"Language set to {language}")
    
    def set_rate(self, rate):
        """Set speech rate (pyttsx3 only)."""
        if self.engine_type == 'pyttsx3' and self.engine:
            self.engine.setProperty('rate', rate)


if __name__ == "__main__":
    print("Testing Text-to-Speech")
    print("=" * 50)
    
    # Test English TTS
    print("\nTesting English TTS (pyttsx3)")
    en_tts = TextToSpeech(engine='pyttsx3', language='en')
    en_tts.speak("Hello! This is a test of the English text to speech system.")
    
    # Test Spanish TTS
    print("\nTesting Spanish TTS (pyttsx3)")
    es_tts = TextToSpeech(engine='pyttsx3', language='es')
    es_tts.speak("Hola! Esta es una prueba del sistema de texto a voz en español.")
    
    # Test gTTS if internet is available
    print("\nTesting English TTS (gTTS)")
    try:
        en_tts_google = TextToSpeech(engine='gtts', language='en')
        en_tts_google.speak("This is Google text to speech.")
    except:
        print("gTTS test failed (internet may not be available)")
    
    print("\nTesting Spanish TTS (gTTS)")
    try:
        es_tts_google = TextToSpeech(engine='gtts', language='es')
        es_tts_google.speak("Esto es Google texto a voz.")
    except:
        print("gTTS test failed (internet may not be available)")
