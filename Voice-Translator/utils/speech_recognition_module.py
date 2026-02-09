# -*- coding: utf-8 -*-
"""
Speech Recognition Module for Voice Translator
Captures audio from microphone and converts English speech to text
Uses Google Speech Recognition API via SpeechRecognition library
"""

import speech_recognition as sr
import time


class SpeechRecognizer:
    """
    Handles audio capture and speech-to-text conversion
    Supports English language recognition with error handling
    Optimized for real-time continuous listening
    """
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self.is_listening = False
        self._stop_flag = False  # Flag to immediately stop listening
        self._is_initialized = False  # Track if mic is calibrated
        
        # Recognition settings - COMPLETE SENTENCES with FAST response
        self.energy_threshold = 300  # Will be calibrated
        self.pause_threshold = 1.0   # 1 second pause = end of sentence (captures full sentence)
        self.phrase_time_limit = 10  # Max 10 seconds per sentence
        self.non_speaking_duration = 0.5  # Silence detection
        self.dynamic_energy = True   # Auto-adjust to environment
        
        # Language setting (English only)
        self.language = "en-US"
        
        # Error messages
        self.ERROR_NOT_UNDERSTOOD = "could_not_understand"
        self.ERROR_SERVICE = "service_error"
        self.ERROR_NO_AUDIO = "no_audio"
        
    def initialize_microphone(self):
        """Initialize the microphone and adjust for ambient noise - ONCE at startup"""
        try:
            self.microphone = sr.Microphone()
            
            # Calibrate for ambient noise ONCE
            with self.microphone as source:
                print("Calibrating microphone... Please be quiet for 1 second.")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                
                # Store the calibrated threshold
                calibrated = self.recognizer.energy_threshold
                print(f"Calibrated energy threshold: {calibrated}")
                
                # Ensure reasonable threshold (not too high, not too low)
                if calibrated > 400:
                    self.recognizer.energy_threshold = 400
                elif calibrated < 50:
                    self.recognizer.energy_threshold = 100
                
                # Apply settings for real-time listening
                self.recognizer.pause_threshold = self.pause_threshold
                self.recognizer.non_speaking_duration = self.non_speaking_duration
                self.recognizer.dynamic_energy_threshold = True
                self.recognizer.dynamic_energy_adjustment_damping = 0.15
                self.recognizer.dynamic_energy_ratio = 1.5
            
            self._is_initialized = True
            print(f"Microphone ready! Threshold: {self.recognizer.energy_threshold}")
            return True
            
        except Exception as e:
            print(f"Error initializing microphone: {e}")
            return False
    
    def list_microphones(self):
        """List all available microphones"""
        mic_list = sr.Microphone.list_microphone_names()
        print("\nAvailable Microphones:")
        for idx, name in enumerate(mic_list):
            print(f"  {idx}: {name}")
        return mic_list
    
    def listen_for_speech(self, timeout=5, phrase_time_limit=None):
        """
        Listen for speech IMMEDIATELY and return recognized text.
        Real-time with complete sentence capture.
        """
        # Check if stop was requested
        if self._stop_flag:
            return False, self.ERROR_NO_AUDIO, None
            
        if phrase_time_limit is None:
            phrase_time_limit = self.phrase_time_limit
        
        try:
            # Use stored microphone for instant response
            with sr.Microphone() as source:
                self.is_listening = True
                
                # Apply settings (no recalibration - instant start)
                self.recognizer.pause_threshold = self.pause_threshold
                self.recognizer.non_speaking_duration = self.non_speaking_duration
                
                # Check stop flag
                if self._stop_flag:
                    self.is_listening = False
                    return False, self.ERROR_NO_AUDIO, None
                
                # Listen IMMEDIATELY - no delay
                try:
                    audio = self.recognizer.listen(
                        source,
                        timeout=timeout,
                        phrase_time_limit=phrase_time_limit
                    )
                except sr.WaitTimeoutError:
                    self.is_listening = False
                    return False, self.ERROR_NO_AUDIO, None
                
                # Check stop flag after audio
                if self._stop_flag:
                    self.is_listening = False
                    return False, self.ERROR_NO_AUDIO, None
                
                self.is_listening = False
                
                # Recognize with Google
                try:
                    text = self.recognizer.recognize_google(audio, language=self.language)
                    
                    if text:
                        text = text.strip()
                        print(f"Recognized: {text}")
                        return True, text.lower(), 0.9
                    else:
                        return False, self.ERROR_NOT_UNDERSTOOD, None
                        
                except sr.UnknownValueError:
                    return False, self.ERROR_NOT_UNDERSTOOD, None
                    
                except sr.RequestError as e:
                    return False, f"{self.ERROR_SERVICE}: {e}", None
                
        except Exception as e:
            self.is_listening = False
            print(f"Speech recognition error: {e}")
            return False, f"Error: {str(e)}", None
    
    def _is_likely_english(self, text):
        """
        Basic check to verify text is likely English
        Based on common English words and character patterns
        """
        if not text:
            return False
        
        # Check for non-ASCII characters (could indicate non-English)
        try:
            text.encode('ascii')
        except UnicodeEncodeError:
            # Contains non-ASCII characters
            # Allow common punctuation but flag other scripts
            cleaned = ''.join(c for c in text if ord(c) < 128 or c in '.,!?\'"-')
            if len(cleaned) < len(text) * 0.7:  # More than 30% non-ASCII
                return False
        
        # Common English words to check presence
        common_words = {
            'the', 'is', 'are', 'a', 'an', 'in', 'to', 'of', 'and', 'it',
            'i', 'you', 'he', 'she', 'we', 'they', 'this', 'that', 'what',
            'how', 'where', 'when', 'why', 'who', 'can', 'will', 'have',
            'has', 'do', 'does', 'am', 'my', 'your', 'be', 'with', 'for'
        }
        
        words = text.lower().split()
        if not words:
            return True  # Single word, let it pass
        
        # Check if at least some common English words present
        common_count = sum(1 for w in words if w in common_words)
        
        # For short phrases, be more lenient
        if len(words) <= 3:
            return True
        
        # For longer phrases, expect some common words
        return common_count > 0 or len(words) <= 5
    
    def stop_listening(self):
        """Stop the current listening operation immediately and clear all pending data"""
        self._stop_flag = True
        self.is_listening = False
        # Create fresh recognizer to clear any buffered audio
        self.recognizer = sr.Recognizer()
        # Reapply settings
        self.recognizer.pause_threshold = self.pause_threshold
        self.recognizer.non_speaking_duration = self.non_speaking_duration
        self.recognizer.dynamic_energy_threshold = True
    
    def reset(self):
        """Reset the recognizer completely for fresh listening session"""
        self._stop_flag = False
        self.is_listening = False
        # Create completely fresh recognizer - no old audio data
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = self.pause_threshold
        self.recognizer.non_speaking_duration = self.non_speaking_duration
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.energy_threshold = 300  # Default threshold
    
    def get_status(self):
        """Get current status of the recognizer"""
        return {
            'is_listening': self.is_listening,
            'microphone_ready': self.microphone is not None,
            'language': self.language
        }


class ContinuousSpeechRecognizer(SpeechRecognizer):
    """
    Extended speech recognizer with continuous listening capability
    Useful for real-time translation applications
    """
    
    def __init__(self):
        super().__init__()
        self.callback = None
        self.stop_flag = False
        
    def set_callback(self, callback_function):
        """Set callback function to receive recognized text"""
        self.callback = callback_function
    
    def start_continuous_listening(self, callback=None):
        """
        Start continuous speech recognition
        Calls callback with (success, text, confidence) for each phrase
        """
        if callback:
            self.callback = callback
            
        if not self.callback:
            print("No callback function set!")
            return
        
        self.stop_flag = False
        
        while not self.stop_flag:
            success, text, confidence = self.listen_for_speech(timeout=3)
            
            if self.callback:
                self.callback(success, text, confidence)
            
            time.sleep(0.1)  # Small delay between phrases
    
    def stop_continuous_listening(self):
        """Stop continuous listening"""
        self.stop_flag = True
        self.stop_listening()


def test_speech_recognition():
    """Test the speech recognition module"""
    print("="*50)
    print("SPEECH RECOGNITION TEST")
    print("="*50)
    
    recognizer = SpeechRecognizer()
    
    # List available microphones
    recognizer.list_microphones()
    
    # Initialize microphone
    if not recognizer.initialize_microphone():
        print("Failed to initialize microphone!")
        return
    
    print("\nReady for speech recognition test.")
    print("You will have 3 attempts to test the recognition.\n")
    
    for i in range(3):
        print(f"\n--- Test {i+1}/3 ---")
        input("Press Enter when ready to speak...")
        
        success, text, confidence = recognizer.listen_for_speech()
        
        if success:
            print(f"\nRecognized: {text}")
            if confidence:
                print(f"Confidence: {confidence:.2%}")
        else:
            print(f"\nFailed: {text}")
            if text == recognizer.ERROR_NOT_UNDERSTOOD:
                print("Sorry, I couldn't understand. Please speak more clearly.")
            elif text == recognizer.ERROR_NO_AUDIO:
                print("No speech detected. Please try again.")


if __name__ == "__main__":
    test_speech_recognition()
