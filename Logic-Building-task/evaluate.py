"""
Model Evaluation Script
Calculates accuracy of the fine-tuned English-Hindi translation model.
"""

import json
import os
from model.translator import get_translator

def evaluate_model():
    """Evaluate the fine-tuned model on test word pairs."""
    
    print("=" * 60)
    print("Model Accuracy Evaluation")
    print("=" * 60)
    
    # Test on verified dictionary words (these we know are correct)
    test_words = {
        # Common nouns
        "book": "किताब", "water": "पानी", "food": "खाना", "house": "घर",
        "sun": "सूर्य", "moon": "चंद्रमा", "tree": "पेड़", "flower": "फूल",
        "bird": "पक्षी", "dog": "कुत्ता", "cat": "बिल्ली", "man": "आदमी",
        "woman": "औरत", "child": "बच्चा", "boy": "लड़का", "girl": "लड़की",
        "mother": "माँ", "father": "पिता", "brother": "भाई", "sister": "बहन",
        "friend": "दोस्त", "teacher": "शिक्षक", "student": "छात्र", "school": "स्कूल",
        "road": "सड़क", "car": "कार", "bus": "बस", "train": "ट्रेन",
        "phone": "फोन", "computer": "कंप्यूटर", "table": "मेज", "chair": "कुर्सी",
        "door": "दरवाजा", "window": "खिड़की", "bed": "बिस्तर", "room": "कमरा",
        "money": "पैसा", "time": "समय", "day": "दिन", "night": "रात",
        "morning": "सुबह", "evening": "शाम", "year": "साल", "month": "महीना",
        "name": "नाम", "work": "काम", "love": "प्यार", "life": "जीवन",
        "world": "दुनिया", "country": "देश", "city": "शहर", "village": "गाँव",
        "river": "नदी", "mountain": "पहाड़", "rain": "बारिश", "fire": "आग",
        # Common adjectives
        "good": "अच्छा", "bad": "बुरा", "big": "बड़ा", "small": "छोटा",
        "new": "नया", "old": "पुराना", "hot": "गर्म", "cold": "ठंडा",
        "happy": "खुश", "beautiful": "सुंदर", "fast": "तेज़", "slow": "धीमा",
        "strong": "मजबूत", "weak": "कमजोर", "rich": "अमीर", "poor": "गरीब",
        "easy": "आसान", "hard": "कठिन", "clean": "साफ", "long": "लंबा",
        # Colors
        "red": "लाल", "blue": "नीला", "green": "हरा", "yellow": "पीला",
        "white": "सफेद", "black": "काला", "orange": "नारंगी", "pink": "गुलाबी",
        # Numbers
        "one": "एक", "two": "दो", "three": "तीन", "four": "चार",
        "five": "पाँच", "six": "छह", "seven": "सात", "eight": "आठ",
        "nine": "नौ", "ten": "दस", "hundred": "सौ", "thousand": "हज़ार",
        # Food
        "rice": "चावल", "bread": "रोटी", "milk": "दूध", "fruit": "फल",
        "apple": "सेब", "banana": "केला", "mango": "आम", "potato": "आलू",
        "onion": "प्याज", "tomato": "टमाटर", "sugar": "चीनी", "salt": "नमक",
        "tea": "चाय", "coffee": "कॉफी", "egg": "अंडा", "fish": "मछली",
        # Animals
        "horse": "घोड़ा", "cow": "गाय", "goat": "बकरी", "lion": "शेर",
        "tiger": "बाघ", "elephant": "हाथी", "monkey": "बंदर", "snake": "साँप",
        # Body parts
        "head": "सिर", "eye": "आँख", "ear": "कान", "nose": "नाक",
        "mouth": "मुँह", "hand": "हाथ", "foot": "पैर", "heart": "दिल",
    }
    
    print(f"\nTesting on {len(test_words)} verified dictionary words...")
    print("-" * 60)
    
    # Load translator
    translator = get_translator()
    translator.load_model()
    
    correct = 0
    partial_correct = 0
    total = len(test_words)
    
    results = []
    
    for english, expected_hindi in test_words.items():
        # Get translation
        predicted_hindi = translator.translate(english)
        
        # Check if correct
        if predicted_hindi.strip() == expected_hindi.strip():
            correct += 1
            status = "✓ EXACT"
        elif expected_hindi in predicted_hindi or predicted_hindi in expected_hindi:
            partial_correct += 1
            status = "~ PARTIAL"
        else:
            status = "✗ WRONG"
        
        results.append({
            'english': english,
            'expected': expected_hindi,
            'predicted': predicted_hindi,
            'status': status
        })
    
    # Calculate accuracy
    exact_accuracy = (correct / total) * 100
    partial_accuracy = ((correct + partial_correct) / total) * 100
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"\nTotal Test Samples: {total}")
    print(f"Exact Matches: {correct}")
    print(f"Partial Matches: {partial_correct}")
    print(f"Wrong: {total - correct - partial_correct}")
    
    print("\n" + "-" * 60)
    print(f"EXACT ACCURACY: {exact_accuracy:.2f}%")
    print(f"PARTIAL ACCURACY: {partial_accuracy:.2f}%")
    print("-" * 60)
    
    # Show sample translations
    print("\n📋 Sample Translations:")
    print("-" * 60)
    for i, r in enumerate(results[:10]):
        print(f"{i+1}. {r['english']} → Expected: {r['expected']} | Got: {r['predicted']} | {r['status']}")
    
    return exact_accuracy, partial_accuracy

if __name__ == "__main__":
    evaluate_model()
