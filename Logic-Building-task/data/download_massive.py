"""
Download massive dataset from Hugging Face for 10,000+ word pairs
"""

import os
import json
import re
from datasets import load_dataset
from tqdm import tqdm

def is_valid_hindi(text):
    hindi_pattern = re.compile(r'[\u0900-\u097F]')
    return bool(hindi_pattern.search(text))

def is_valid_english(text):
    return text.isalpha() and text.isascii() and len(text) > 1

def clean_word(word):
    if not word:
        return None
    word = re.sub(r'[^\w\s]', '', word).strip()
    if ' ' in word or len(word) < 2:
        return None
    return word.lower()

def download_massive_data():
    print("=" * 70)
    print("Downloading MASSIVE English-Hindi Dataset (10,000+ words)")
    print("=" * 70)
    
    word_pairs = {}
    
    # Load full IITB corpus
    print("\n[1/2] Loading IITB English-Hindi Corpus (500K samples)...")
    try:
        dataset = load_dataset("cfilt/iitb-english-hindi", split="train")
        
        for sample in tqdm(dataset.select(range(min(500000, len(dataset)))), desc="Processing"):
            en_text = sample['translation']['en'].strip()
            hi_text = sample['translation']['hi'].strip()
            
            en_words = en_text.split()
            hi_words = hi_text.split()
            
            # Extract aligned word pairs from short sentences
            if len(en_words) <= 3 and len(hi_words) <= 3 and len(en_words) == len(hi_words):
                for ew, hw in zip(en_words, hi_words):
                    en_word = clean_word(ew)
                    hi_word = hw.strip()
                    if en_word and is_valid_english(en_word) and is_valid_hindi(hi_word):
                        if en_word not in word_pairs:
                            word_pairs[en_word] = hi_word
                            
            if len(word_pairs) >= 8000:
                break
                
    except Exception as e:
        print(f"Error: {e}")
    
    print(f"   Words from IITB: {len(word_pairs)}")
    
    # Load Aarif dataset
    print("\n[2/2] Loading Aarif1430 dataset...")
    try:
        aarif = load_dataset("Aarif1430/english-to-hindi", split="train")
        
        for sample in tqdm(aarif, desc="Processing"):
            en_text = sample.get('English', sample.get('english', '')).strip()
            hi_text = sample.get('Hindi', sample.get('hindi', '')).strip()
            
            en_words = en_text.split()
            hi_words = hi_text.split()
            
            if len(en_words) <= 2 and len(hi_words) >= 1:
                for ew in en_words:
                    en_word = clean_word(ew)
                    if en_word and is_valid_english(en_word) and en_word not in word_pairs:
                        if is_valid_hindi(hi_words[0]):
                            word_pairs[en_word] = hi_words[0]
                            
    except Exception as e:
        print(f"Error: {e}")
    
    print(f"   Total words: {len(word_pairs)}")
    
    # Add proper translations (not transliterations)
    print("\n[Bonus] Adding proper Hindi translations...")
    proper_translations = {
        "hello": "नमस्ते", "hi": "नमस्ते", "bye": "अलविदा", "goodbye": "अलविदा",
        "thanks": "धन्यवाद", "thank": "धन्यवाद", "please": "कृपया", "sorry": "क्षमा",
        "welcome": "स्वागत", "yes": "हाँ", "no": "नहीं", "okay": "ठीक",
        "morning": "सुबह", "evening": "शाम", "night": "रात", "day": "दिन",
        "love": "प्रेम", "hate": "घृणा", "happy": "प्रसन्न", "sad": "दुखी",
        "friend": "मित्र", "enemy": "शत्रु", "family": "परिवार", "home": "घर",
        "school": "विद्यालय", "college": "महाविद्यालय", "university": "विश्वविद्यालय",
        "book": "पुस्तक", "pen": "कलम", "paper": "कागज", "story": "कथा",
        "water": "जल", "fire": "अग्नि", "earth": "पृथ्वी", "sky": "आकाश",
        "sun": "सूर्य", "moon": "चंद्र", "star": "तारा", "rain": "वर्षा",
        "king": "राजा", "queen": "रानी", "prince": "राजकुमार", "princess": "राजकुमारी",
        "god": "ईश्वर", "goddess": "देवी", "temple": "मंदिर", "prayer": "प्रार्थना",
        "mother": "माता", "father": "पिता", "brother": "भ्राता", "sister": "भगिनी",
        "son": "पुत्र", "daughter": "पुत्री", "husband": "पति", "wife": "पत्नी",
        "teacher": "गुरु", "student": "शिष्य", "doctor": "चिकित्सक", "nurse": "परिचारिका",
        "soldier": "सैनिक", "police": "पुलिस", "judge": "न्यायाधीश", "lawyer": "अधिवक्ता",
        "farmer": "कृषक", "worker": "कर्मचारी", "servant": "सेवक", "master": "स्वामी",
        "rich": "धनवान", "poor": "निर्धन", "strong": "बलवान", "weak": "दुर्बल",
        "old": "वृद्ध", "young": "युवा", "new": "नवीन", "ancient": "प्राचीन",
        "big": "विशाल", "small": "लघु", "long": "दीर्घ", "short": "लघु",
        "good": "उत्तम", "bad": "दुष्ट", "beautiful": "सुंदर", "ugly": "कुरूप",
        "hot": "उष्ण", "cold": "शीत", "warm": "गर्म", "cool": "ठंडा",
        "fast": "द्रुत", "slow": "मंद", "easy": "सरल", "difficult": "कठिन",
        "true": "सत्य", "false": "असत्य", "right": "सही", "wrong": "गलत",
        "life": "जीवन", "death": "मृत्यु", "birth": "जन्म", "age": "आयु",
        "health": "स्वास्थ्य", "disease": "रोग", "medicine": "औषधि", "hospital": "चिकित्सालय",
        "food": "भोजन", "hunger": "भूख", "thirst": "प्यास", "sleep": "निद्रा",
        "work": "कार्य", "rest": "विश्राम", "game": "खेल", "sport": "क्रीड़ा",
        "war": "युद्ध", "peace": "शांति", "victory": "विजय", "defeat": "पराजय",
        "country": "राष्ट्र", "nation": "देश", "state": "राज्य", "city": "नगर",
        "village": "ग्राम", "town": "कस्बा", "road": "मार्ग", "path": "पथ",
        "river": "नदी", "ocean": "सागर", "mountain": "पर्वत", "forest": "वन",
        "tree": "वृक्ष", "flower": "पुष्प", "fruit": "फल", "leaf": "पत्ता",
        "bird": "पक्षी", "animal": "पशु", "fish": "मत्स्य", "snake": "सर्प",
        "lion": "सिंह", "tiger": "व्याघ्र", "elephant": "गज", "horse": "अश्व",
        "cow": "गौ", "dog": "श्वान", "cat": "बिल्ली", "mouse": "मूषक",
        "eye": "नेत्र", "ear": "कर्ण", "nose": "नासिका", "mouth": "मुख",
        "hand": "हस्त", "foot": "पाद", "head": "शिर", "heart": "हृदय",
        "mind": "मन", "soul": "आत्मा", "body": "शरीर", "blood": "रक्त",
        "time": "काल", "place": "स्थान", "thing": "वस्तु", "matter": "पदार्थ",
        "power": "शक्ति", "energy": "ऊर्जा", "light": "प्रकाश", "darkness": "अंधकार",
        "sound": "ध्वनि", "silence": "मौन", "music": "संगीत", "dance": "नृत्य",
        "art": "कला", "science": "विज्ञान", "knowledge": "ज्ञान", "wisdom": "बुद्धि",
        "truth": "सत्य", "justice": "न्याय", "freedom": "स्वतंत्रता", "equality": "समानता",
        "love": "प्रेम", "hate": "द्वेष", "anger": "क्रोध", "fear": "भय",
        "joy": "आनंद", "sorrow": "दुख", "hope": "आशा", "despair": "निराशा",
    }
    
    for en, hi in proper_translations.items():
        word_pairs[en] = hi
    
    print(f"   Final total: {len(word_pairs)}")
    
    # Save
    pairs_list = [{"english": en, "hindi": hi} for en, hi in word_pairs.items()]
    
    os.makedirs("data", exist_ok=True)
    with open("data/word_pairs.json", 'w', encoding='utf-8') as f:
        json.dump(pairs_list, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'=' * 70}")
    print(f"DATASET CREATED: {len(pairs_list)} word pairs")
    print(f"{'=' * 70}")
    
    return pairs_list

if __name__ == "__main__":
    download_massive_data()
