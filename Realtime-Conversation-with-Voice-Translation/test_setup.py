"""
Test project setup and verify all components are ready.
Run this before training to catch any issues early.
"""

import sys
import os

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing package imports...")
    print("-" * 60)
    
    required_packages = {
        'torch': 'PyTorch',
        'datasets': 'Hugging Face Datasets',
        'transformers': 'Hugging Face Transformers',
        'speech_recognition': 'SpeechRecognition',
        'pyttsx3': 'pyttsx3 (TTS)',
        'gtts': 'gTTS',
        'numpy': 'NumPy',
        'tqdm': 'tqdm',
    }
    
    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - NOT FOUND")
            missing.append(package)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    print("\nAll packages installed successfully!")
    return True


def test_project_structure():
    """Verify project directory structure."""
    print("\nTesting project structure...")
    print("-" * 60)
    
    required_dirs = [
        'data',
        'models',
        'src',
        'training',
        'checkpoints'
    ]
    
    required_files = [
        'main.py',
        'requirements.txt',
        'README.md',
        'data/data_loader.py',
        'models/seq2seq.py',
        'training/train_translator.py',
        'training/config.py',
        'src/speech_recognition_module.py',
        'src/text_to_speech.py',
        'src/translator.py',
        'src/conversation.py'
    ]
    
    all_good = True
    
    # Check directories
    for dir_name in required_dirs:
        if os.path.isdir(dir_name):
            print(f"  ✓ {dir_name}/")
        else:
            print(f"  ✗ {dir_name}/ - NOT FOUND")
            all_good = False
    
    # Check files
    for file_path in required_files:
        if os.path.isfile(file_path):
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} - NOT FOUND")
            all_good = False
    
    if all_good:
        print("\nProject structure looks good!")
    else:
        print("\nSome files or directories are missing!")
    
    return all_good


def test_torch_device():
    """Check if CUDA is available."""
    print("\nTesting PyTorch device...")
    print("-" * 60)
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  Training will be FAST (1-3 hours per model)")
        else:
            print(f"  ℹ CUDA not available")
            print(f"  Using CPU for training")
            print(f"  Training will be SLOW (4-8 hours per model)")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_microphone():
    """Test microphone availability."""
    print("\nTesting microphone...")
    print("-" * 60)
    
    try:
        import speech_recognition as sr
        
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("  ✓ Microphone detected")
            print("  Microphone is working correctly")
        return True
    except Exception as e:
        print(f"  ✗ Microphone error: {e}")
        print("  Voice recognition may not work")
        return False


def test_tts():
    """Test text-to-speech."""
    print("\nTesting text-to-speech...")
    print("-" * 60)
    
    try:
        import pyttsx3
        engine = pyttsx3.init()
        print("  ✓ pyttsx3 working")
        return True
    except Exception as e:
        print(f"  ℹ pyttsx3 error: {e}")
        print("  Trying gTTS as fallback...")
        
        try:
            from gtts import gTTS
            print("  ✓ gTTS available (requires internet)")
            return True
        except:
            print("  ✗ No TTS engine available")
            return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("PROJECT SETUP VERIFICATION")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Package Imports", test_imports()))
    results.append(("Project Structure", test_project_structure()))
    results.append(("PyTorch Device", test_torch_device()))
    results.append(("Microphone", test_microphone()))
    results.append(("Text-to-Speech", test_tts()))
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status} - {test_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("\nYour project is ready for training!")
        print("\nNext steps:")
        print("  1. python training/train_translator.py --direction en-es")
        print("  2. python training/train_translator.py --direction es-en")
        print("  3. python main.py")
    else:
        print("✗ SOME TESTS FAILED")
        print("\nPlease fix the issues above before training.")
        print("Refer to QUICKSTART.md for setup instructions.")
    print("=" * 60)


if __name__ == "__main__":
    main()
