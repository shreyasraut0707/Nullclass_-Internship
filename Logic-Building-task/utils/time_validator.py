"""
Time Validator Module
Handles vowel detection and time-based translation validation logic.
"""

from datetime import datetime


VOWELS = ['A', 'E', 'I', 'O', 'U', 'a', 'e', 'i', 'o', 'u']


def is_vowel_word(word: str) -> bool:
    """
    Check if a word starts with a vowel.
    
    Args:
        word: The English word to check
        
    Returns:
        True if the word starts with a vowel (A, E, I, O, U), False otherwise
    """
    if not word or len(word) == 0:
        return False
    return word[0] in VOWELS


def is_within_allowed_time() -> bool:
    """
    Check if current time is between 9 PM and 10 PM.
    
    Returns:
        True if current hour is 21 (9 PM), False otherwise
    """
    current_hour = datetime.now().hour
    return current_hour == 21  # 9 PM = 21:00


def is_translation_allowed(word: str) -> tuple:
    """
    Determine if translation is allowed for the given word.
    
    Rules:
    - Words starting with consonants: Always allowed
    - Words starting with vowels: Only allowed between 9 PM and 10 PM
    
    Args:
        word: The English word to validate
        
    Returns:
        Tuple of (is_allowed: bool, error_message: str or None)
    """
    if not word or len(word.strip()) == 0:
        return False, "Please enter a valid word."
    
    word = word.strip()
    
    if is_vowel_word(word):
        if is_within_allowed_time():
            return True, None
        else:
            return False, "This word starts with a vowel. Please provide another word."
    
    # Word starts with consonant - always allowed
    return True, None


def get_current_time_status() -> str:
    """
    Get a human-readable string of current time and vowel translation status.
    
    Returns:
        Status string showing current time and whether vowel words are allowed
    """
    now = datetime.now()
    time_str = now.strftime("%I:%M %p")
    
    if is_within_allowed_time():
        status = "✓ Vowel words allowed (9 PM - 10 PM)"
    else:
        status = "✗ Vowel words not allowed (outside 9-10 PM)"
    
    return f"Current Time: {time_str} | {status}"
