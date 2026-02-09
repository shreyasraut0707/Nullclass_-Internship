"""
Speech recognition module for converting voice to text.
Supports both English and Spanish.
"""

import speech_recognition as sr
import sounddevice as sd
import soundfile as sf
import numpy as np
import os
import tempfile


class SpeechRecognizer:
    """Handle speech-to-text conversion."""
    
    def __init__(self, language='en-US'):
        """
        Initialize speech recognizer.
        
        Args:
            language: Language code ('en-US' for English, 'es-ES' for Spanish)
        """
        self.recognizer = sr.Recognizer()
        self.language = language
        self.sample_rate = 16000
        
        # Adjust for ambient noise
        print("Adjusting for ambient noise... Please wait.")
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Ready!")
    
    def listen_from_microphone(self, timeout=5, phrase_time_limit=10):
        """
        Listen to microphone and convert speech to text.
        
        Args:
            timeout: Seconds to wait for speech to start
            phrase_time_limit: Maximum seconds for the phrase
        
        Returns:
            Recognized text or None if failed
        """
        try:
            with sr.Microphone() as source:
                print("Listening...")
                audio = self.recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit
                )
                print("Processing speech...")
                
                # Try Google Speech Recognition
                try:
                    text = self.recognizer.recognize_google(audio, language=self.language)
                    return text
                except sr.UnknownValueError:
                    print("Could not understand audio")
                    return None
                except sr.RequestError as e:
                    print(f"Could not request results; {e}")
                    return None
        
        except sr.WaitTimeoutError:
            print("No speech detected")
            return None
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def recognize_from_file(self, audio_file):
        """
        Recognize speech from an audio file.
        
        Args:
            audio_file: Path to audio file
        
        Returns:
            Recognized text or None if failed
        """
        try:
            with sr.AudioFile(audio_file) as source:
                audio = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio, language=self.language)
                return text
        except Exception as e:
            print(f"Error recognizing from file: {e}")
            return None
    
    def set_language(self, language):
        """Change the recognition language."""
        self.language = language
        print(f"Language set to {language}")


def record_audio(duration=5, sample_rate=16000):
    """
    Record audio from microphone.
    
    Args:
        duration: Recording duration in seconds
        sample_rate: Sample rate for recording
    
    Returns:
        Audio data as numpy array
    """
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='float32'
    )
    sd.wait()
    print("Recording finished")
    return audio


def save_audio(audio, filename, sample_rate=16000):
    """Save audio to file."""
    sf.write(filename, audio, sample_rate)
    print(f"Audio saved to {filename}")


def test_microphone():
    """Test if microphone is working."""
    try:
        with sr.Microphone() as source:
            print("Microphone test successful!")
            return True
    except Exception as e:
        print(f"Microphone error: {e}")
        return False


if __name__ == "__main__":
    print("Testing Speech Recognition")
    print("=" * 50)
    
    # Test microphone
    if not test_microphone():
        print("Please check your microphone connection")
        exit(1)
    
    # Test English recognition
    print("\nTesting English Recognition")
    en_recognizer = SpeechRecognizer(language='en-US')
    print("Say something in English...")
    en_text = en_recognizer.listen_from_microphone()
    if en_text:
        print(f"You said: {en_text}")
    
    # Test Spanish recognition
    print("\nTesting Spanish Recognition")
    es_recognizer = SpeechRecognizer(language='es-ES')
    print("Say something in Spanish...")
    es_text = es_recognizer.listen_from_microphone()
    if es_text:
        print(f"Dijiste: {es_text}")
