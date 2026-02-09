"""
Test Module - Tests for the translation system
"""

from validator import InputValidator


def test_validator():
    """Test the input validator."""
    print("Testing Input Validator")
    print("-" * 40)
    
    validator = InputValidator()
    
    test_cases = [
        ("Hello", False),
        ("Good morning", True),
        ("Hi!", False),
        ("", False),
        ("Hello World!", True),
        ("Technology", True),
        ("Test", False),
        ("Hello Sakshi", True),
    ]
    
    passed = 0
    failed = 0
    
    for input_text, expected in test_cases:
        is_valid, _ = validator.validate_input(input_text)
        letter_count = validator.count_letters(input_text)
        
        if is_valid == expected:
            status = "PASS"
            passed += 1
        else:
            status = "FAIL"
            failed += 1
        
        print(f"{status} | '{input_text}' ({letter_count} letters)")
    
    print("-" * 40)
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


def test_translation():
    """Test the translation engine."""
    print()
    print("Testing Translation Engine")
    print("-" * 40)
    
    from translation_model import DualLanguageTranslator
    
    translator = DualLanguageTranslator()
    
    test_texts = [
        "Good morning, how are you today?",
        "Hello Sakshi, what are you doing?",
    ]
    
    for text in test_texts:
        print(f"Input: {text}")
        french = translator.translate_to_french(text)
        hindi = translator.translate_to_hindi(text)
        print(f"French: {french}")
        print(f"Hindi: {hindi}")
        print()


if __name__ == "__main__":
    print()
    print("Dual Language Translator - Tests")
    print("=" * 40)
    print()
    
    validator_ok = test_validator()
    
    print()
    response = input("Test translations? [y/N]: ")
    if response.lower() == 'y':
        test_translation()
    
    print()
    if validator_ok:
        print("All tests passed!")
    print("Done.")
