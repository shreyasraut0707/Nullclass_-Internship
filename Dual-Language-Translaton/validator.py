"""
Input Validator - Validates text input for minimum letter requirement
"""

import re


class InputValidator:
    """Validates input text for minimum 10 letters."""
    
    @staticmethod
    def count_letters(text):
        """Count only alphabetic letters (excludes spaces, punctuation, numbers)."""
        letters_only = re.sub(r'[^a-zA-Z]', '', text)
        return len(letters_only)
    
    @staticmethod
    def validate_input(text):
        """Check if text has at least 10 letters."""
        if not text or text.strip() == "":
            return False, "Please enter some text to translate."
        
        letter_count = InputValidator.count_letters(text)
        
        if letter_count < 10:
            return False, (
                f"Upload Again!\n\n"
                f"Your input has only {letter_count} letter(s).\n"
                f"Minimum 10 letters required.\n\n"
                f"(Spaces and punctuation don't count)"
            )
        
        return True, f"Valid input ({letter_count} letters)"


if __name__ == "__main__":
    validator = InputValidator()
    
    test_inputs = ["Hello", "Good morning", "Hi!", "Technology"]
    
    print("Testing validator:")
    for text in test_inputs:
        is_valid, _ = validator.validate_input(text)
        letters = validator.count_letters(text)
        status = "PASS" if is_valid else "FAIL"
        print(f"'{text}' -> {letters} letters -> {status}")
