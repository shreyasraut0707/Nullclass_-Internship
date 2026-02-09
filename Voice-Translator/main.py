# -*- coding: utf-8 -*-
"""
Voice Translator - Main Entry Point
English to Hindi Voice Translation System

Author: Shreyas
Version: 1.0

Usage:
    python main.py          - Run the GUI application
    python main.py --test   - Test the trained model
"""

import sys
import os
import argparse

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def run_gui():
    """Launch the GUI application"""
    print("Starting Voice Translator...")
    try:
        from gui.app import main
        main()
    except Exception as e:
        print(f"\nApplication Error: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")  # Keep window open to see error


def test_model():
    """Test the trained model with sample inputs"""
    print("=" * 50)
    print("TESTING TRANSLATION MODEL")
    print("=" * 50)
    
    from model.translator import EnglishHindiTranslator
    
    translator = EnglishHindiTranslator()
    
    if not translator.load_model():
        print("\nError: Model not found!")
        print("Ensure saved_model/ directory contains the trained model files.")
        return
    
    # Test sentences
    test_sentences = [
        "hello",
        "how are you",
        "good morning",
        "thank you",
        "what is your name",
        "i am fine",
        "nice to meet you",
        "goodbye",
        "please help me",
        "i love india"
    ]
    
    print("\nTranslation Results:")
    print("-" * 50)
    
    for sentence in test_sentences:
        translation = translator.translate(sentence)
        print(f"English: {sentence}")
        print(f"Hindi:   {translation}")
        print("-" * 50)
    
    print("\nTest completed successfully!")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Voice Translator - English to Hindi Translation System'
    )
    
    parser.add_argument('--test', action='store_true',
                       help='Test the trained model with sample sentences')
    
    args = parser.parse_args()
    
    if args.test:
        test_model()
    else:
        run_gui()


if __name__ == "__main__":
    main()
