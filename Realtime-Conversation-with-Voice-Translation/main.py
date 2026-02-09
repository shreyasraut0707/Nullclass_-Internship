"""
Main application entry point for Real-Time Voice Translation System.
"""

import os
import sys
import argparse

from src.conversation import ConversationSystem
from src.translator import Translator


def check_models():
    """Check if trained models exist."""
    en_es_path = 'checkpoints/en_es/best_model.pth'
    es_en_path = 'checkpoints/es_en/best_model.pth'
    
    en_es_exists = os.path.exists(en_es_path)
    es_en_exists = os.path.exists(es_en_path)
    
    if not en_es_exists or not es_en_exists:
        print("\n" + "="*60)
        print("ERROR: Trained models not found!")
        print("="*60)
        print("\nYou need to train the translation models first.")
        print("\nTo train the models, run:")
        print("  1. python training/train_translator.py --direction en-es")
        print("  2. python training/train_translator.py --direction es-en")
        print("\nNote: Training may take several hours depending on your hardware.")
        print("The training process will save checkpoints automatically,")
        print("so you can stop and resume anytime.")
        print("="*60)
        return False
    
    return True


def main():
    """Main application."""
    parser = argparse.ArgumentParser(
        description='Real-Time Voice Translation System (English <-> Spanish)'
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run in demo mode with text input instead of voice'
    )
    parser.add_argument(
        '--tts-engine',
        type=str,
        default='pyttsx3',
        choices=['pyttsx3', 'gtts'],
        help='Text-to-speech engine to use'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*60)
    print(" REAL-TIME VOICE TRANSLATION SYSTEM")
    print(" English <-> Spanish")
    print("="*60)
    
    # Check if models are trained
    if not check_models():
        sys.exit(1)
    
    # Load models
    print("\nLoading translation models...")
    en_es_path = 'checkpoints/en_es/best_model.pth'
    es_en_path = 'checkpoints/es_en/best_model.pth'
    
    try:
        translator = Translator(en_es_path, es_en_path)
    except Exception as e:
        print(f"\nError loading models: {e}")
        sys.exit(1)
    
    # Initialize conversation system
    print(f"\nInitializing conversation system (TTS: {args.tts_engine})...")
    try:
        conversation = ConversationSystem(translator, tts_engine=args.tts_engine)
    except Exception as e:
        print(f"\nError initializing conversation system: {e}")
        sys.exit(1)
    
    # Start conversation
    if args.demo:
        conversation.demo_mode()
    else:
        conversation.start_conversation()


if __name__ == "__main__":
    main()
