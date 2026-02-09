"""
Training data loader with Hugging Face datasets for English-Hindi translation.
Uses the cfilt/iitb-english-hindi dataset - one of the best parallel corpora available.
"""

import numpy as np
import os
import pickle

# Curated conversational pairs for high-quality everyday translations
CURATED_PAIRS = [
    # Greetings
    ("hello", "नमस्ते"),
    ("hi", "नमस्ते"),
    ("good morning", "शुभ प्रभात"),
    ("good afternoon", "शुभ दोपहर"),
    ("good evening", "शुभ संध्या"),
    ("good night", "शुभ रात्रि"),
    ("goodbye", "अलविदा"),
    ("bye", "अलविदा"),
    ("see you later", "फिर मिलेंगे"),
    ("welcome", "स्वागत है"),
    ("nice to meet you", "आपसे मिलकर खुशी हुई"),
    
    # Basic questions and responses
    ("how are you", "आप कैसे हैं"),
    ("how are you doing", "आप कैसे हैं"),
    ("i am fine", "मैं ठीक हूं"),
    ("i am good", "मैं अच्छा हूं"),
    ("i am doing well", "मैं अच्छा कर रहा हूं"),
    ("what is your name", "आपका नाम क्या है"),
    ("my name is", "मेरा नाम है"),
    ("where are you from", "आप कहां से हैं"),
    ("i am from india", "मैं भारत से हूं"),
    ("what do you do", "आप क्या करते हैं"),
    ("what are you doing", "आप क्या कर रहे हैं"),
    ("what are you doing right now", "आप अभी क्या कर रहे हैं"),
    ("how old are you", "आपकी उम्र क्या है"),
    ("where do you live", "आप कहां रहते हैं"),
    ("what time is it", "कितने बजे हैं"),
    ("what is this", "यह क्या है"),
    ("who are you", "आप कौन हैं"),
    ("why", "क्यों"),
    ("when", "कब"),
    ("where", "कहां"),
    ("how", "कैसे"),
    ("what", "क्या"),
    
    # Common phrases
    ("thank you", "धन्यवाद"),
    ("thank you very much", "बहुत धन्यवाद"),
    ("thanks", "धन्यवाद"),
    ("please", "कृपया"),
    ("sorry", "माफ़ कीजिए"),
    ("excuse me", "क्षमा कीजिए"),
    ("yes", "हां"),
    ("no", "नहीं"),
    ("maybe", "शायद"),
    ("okay", "ठीक है"),
    ("i understand", "मैं समझ गया"),
    ("i do not understand", "मुझे समझ नहीं आया"),
    ("please repeat", "कृपया दोहराएं"),
    ("speak slowly", "धीरे बोलिए"),
    ("can you help me", "क्या आप मेरी मदद कर सकते हैं"),
    ("i need help", "मुझे मदद चाहिए"),
    
    # Actions and activities
    ("i am eating", "मैं खा रहा हूं"),
    ("i am drinking", "मैं पी रहा हूं"),
    ("i am sleeping", "मैं सो रहा हूं"),
    ("i am working", "मैं काम कर रहा हूं"),
    ("i am studying", "मैं पढ़ रहा हूं"),
    ("i am reading", "मैं पढ़ रहा हूं"),
    ("i am writing", "मैं लिख रहा हूं"),
    ("i am watching", "मैं देख रहा हूं"),
    ("i am listening", "मैं सुन रहा हूं"),
    ("i am playing", "मैं खेल रहा हूं"),
    ("i am walking", "मैं चल रहा हूं"),
    ("i am running", "मैं दौड़ रहा हूं"),
    ("i am cooking", "मैं खाना बना रहा हूं"),
    ("i am talking", "मैं बात कर रहा हूं"),
    ("i am thinking", "मैं सोच रहा हूं"),
    ("i am waiting", "मैं इंतज़ार कर रहा हूं"),
    
    # Questions with what/where/when/who/how
    ("what do you want", "आप क्या चाहते हैं"),
    ("what do you need", "आपको क्या चाहिए"),
    ("what do you like", "आपको क्या पसंद है"),
    ("what did you say", "आपने क्या कहा"),
    ("what happened", "क्या हुआ"),
    ("what is happening", "क्या हो रहा है"),
    ("where is it", "यह कहां है"),
    ("where are you going", "आप कहां जा रहे हैं"),
    ("where is the bathroom", "बाथरूम कहां है"),
    ("where is the station", "स्टेशन कहां है"),
    ("when will you come", "आप कब आएंगे"),
    ("when did you arrive", "आप कब आए"),
    ("who is there", "वहां कौन है"),
    ("how much does it cost", "इसकी कीमत क्या है"),
    ("how do you know", "आप कैसे जानते हैं"),
    ("how did you do that", "आपने यह कैसे किया"),
    
    # Common sentences
    ("i love you", "मैं तुमसे प्यार करता हूं"),
    ("i miss you", "मुझे तुम्हारी याद आती है"),
    ("i am happy", "मैं खुश हूं"),
    ("i am sad", "मैं दुखी हूं"),
    ("i am tired", "मैं थका हुआ हूं"),
    ("i am hungry", "मुझे भूख लगी है"),
    ("i am thirsty", "मुझे प्यास लगी है"),
    ("i am cold", "मुझे ठंड लग रही है"),
    ("i am hot", "मुझे गर्मी लग रही है"),
    ("i am sick", "मैं बीमार हूं"),
    ("i am busy", "मैं व्यस्त हूं"),
    ("i am free", "मैं फ्री हूं"),
    ("i am learning hindi", "मैं हिंदी सीख रहा हूं"),
    ("i speak english", "मैं अंग्रेज़ी बोलता हूं"),
    ("do you speak hindi", "क्या आप हिंदी बोलते हैं"),
    ("i do not speak hindi", "मैं हिंदी नहीं बोलता"),
    
    # Time related
    ("today", "आज"),
    ("tomorrow", "कल"),
    ("yesterday", "कल"),
    ("now", "अब"),
    ("later", "बाद में"),
    ("morning", "सुबह"),
    ("afternoon", "दोपहर"),
    ("evening", "शाम"),
    ("night", "रात"),
    ("this week", "इस हफ्ते"),
    ("next week", "अगले हफ्ते"),
    ("this month", "इस महीने"),
    ("this year", "इस साल"),
    
    # Numbers
    ("one", "एक"),
    ("two", "दो"),
    ("three", "तीन"),
    ("four", "चार"),
    ("five", "पांच"),
    ("six", "छह"),
    ("seven", "सात"),
    ("eight", "आठ"),
    ("nine", "नौ"),
    ("ten", "दस"),
    
    # Days
    ("monday", "सोमवार"),
    ("tuesday", "मंगलवार"),
    ("wednesday", "बुधवार"),
    ("thursday", "गुरुवार"),
    ("friday", "शुक्रवार"),
    ("saturday", "शनिवार"),
    ("sunday", "रविवार"),
    
    # Common verbs
    ("eat", "खाना"),
    ("drink", "पीना"),
    ("sleep", "सोना"),
    ("wake up", "जागना"),
    ("walk", "चलना"),
    ("run", "दौड़ना"),
    ("sit", "बैठना"),
    ("stand", "खड़े होना"),
    ("come", "आना"),
    ("go", "जाना"),
    ("see", "देखना"),
    ("hear", "सुनना"),
    ("speak", "बोलना"),
    ("read", "पढ़ना"),
    ("write", "लिखना"),
    ("learn", "सीखना"),
    ("teach", "सिखाना"),
    ("work", "काम करना"),
    ("play", "खेलना"),
    ("help", "मदद करना"),
    ("give", "देना"),
    ("take", "लेना"),
    ("buy", "खरीदना"),
    ("sell", "बेचना"),
    ("open", "खोलना"),
    ("close", "बंद करना"),
    ("start", "शुरू करना"),
    ("stop", "रुकना"),
    ("wait", "इंतज़ार करना"),
    ("call", "फ़ोन करना"),
    
    # Family
    ("mother", "माँ"),
    ("father", "पिताजी"),
    ("brother", "भाई"),
    ("sister", "बहन"),
    ("son", "बेटा"),
    ("daughter", "बेटी"),
    ("husband", "पति"),
    ("wife", "पत्नी"),
    ("friend", "दोस्त"),
    ("family", "परिवार"),
    
    # Places
    ("home", "घर"),
    ("office", "कार्यालय"),
    ("school", "स्कूल"),
    ("hospital", "अस्पताल"),
    ("market", "बाज़ार"),
    ("restaurant", "रेस्तरां"),
    ("hotel", "होटल"),
    ("airport", "हवाई अड्डा"),
    ("station", "स्टेशन"),
    ("bank", "बैंक"),
    
    # Food
    ("water", "पानी"),
    ("food", "खाना"),
    ("rice", "चावल"),
    ("bread", "रोटी"),
    ("milk", "दूध"),
    ("tea", "चाय"),
    ("coffee", "कॉफ़ी"),
    ("fruit", "फल"),
    ("vegetable", "सब्ज़ी"),
    
    # Adjectives
    ("good", "अच्छा"),
    ("bad", "बुरा"),
    ("big", "बड़ा"),
    ("small", "छोटा"),
    ("hot", "गर्म"),
    ("cold", "ठंडा"),
    ("new", "नया"),
    ("old", "पुराना"),
    ("fast", "तेज़"),
    ("slow", "धीमा"),
    ("beautiful", "सुंदर"),
    ("easy", "आसान"),
    ("difficult", "मुश्किल"),
    ("right", "सही"),
    ("wrong", "गलत"),
    
    # More conversational phrases
    ("lets go", "चलो"),
    ("come here", "यहां आओ"),
    ("go there", "वहां जाओ"),
    ("sit down", "बैठ जाओ"),
    ("stand up", "खड़े हो जाओ"),
    ("be quiet", "चुप रहो"),
    ("listen to me", "मेरी बात सुनो"),
    ("look at this", "इसे देखो"),
    ("tell me", "मुझे बताओ"),
    ("show me", "मुझे दिखाओ"),
    ("give me", "मुझे दो"),
    ("wait a moment", "एक पल रुको"),
    ("just a minute", "बस एक मिनट"),
    ("no problem", "कोई बात नहीं"),
    ("never mind", "कोई बात नहीं"),
    ("i think so", "मुझे ऐसा लगता है"),
    ("i dont think so", "मुझे ऐसा नहीं लगता"),
    ("of course", "बिल्कुल"),
    ("really", "सच में"),
    ("are you sure", "क्या आप निश्चित हैं"),
    ("i am sure", "मुझे यकीन है"),
    ("take care", "अपना ख्याल रखना"),
    ("have a nice day", "आपका दिन शुभ हो"),
    ("congratulations", "बधाई हो"),
    ("happy birthday", "जन्मदिन मुबारक"),
    ("best wishes", "शुभकामनाएं"),
    
    # Meeting and planning
    ("can we meet", "क्या हम मिल सकते हैं"),
    ("can we meet today", "क्या हम आज मिल सकते हैं"),
    ("can we meet tomorrow", "क्या हम कल मिल सकते हैं"),
    ("lets meet", "चलो मिलते हैं"),
    ("meet me", "मुझसे मिलो"),
    ("where should we meet", "हम कहां मिलें"),
    ("when should we meet", "हम कब मिलें"),
    ("i want to meet you", "मैं आपसे मिलना चाहता हूं"),
    ("see you soon", "जल्द मिलते हैं"),
    ("nice meeting you", "आपसे मिलकर अच्छा लगा"),
    
    # Current activities
    ("what you are doing", "आप क्या कर रहे हैं"),
    ("what are you doing now", "आप अभी क्या कर रहे हैं"),
    ("what you doing", "आप क्या कर रहे हैं"),
    ("what is going on", "क्या हो रहा है"),
    ("whats up", "क्या चल रहा है"),
    ("whats happening", "क्या हो रहा है"),
    ("how is it going", "कैसा चल रहा है"),
    ("how is everything", "सब कैसा है"),
    ("how is life", "ज़िंदगी कैसी है"),
    ("how is work", "काम कैसा है"),
    
    # Requests and questions
    ("can you help", "क्या आप मदद कर सकते हैं"),
    ("will you help me", "क्या आप मेरी मदद करेंगे"),
    ("please help", "कृपया मदद करें"),
    ("i need your help", "मुझे आपकी मदद चाहिए"),
    ("can i ask you something", "क्या मैं आपसे कुछ पूछ सकता हूं"),
    ("do you have time", "क्या आपके पास समय है"),
    ("are you free", "क्या आप फ्री हैं"),
    ("are you busy", "क्या आप व्यस्त हैं"),
    
    # Common responses
    ("i understand", "मैं समझ गया"),
    ("i got it", "मुझे समझ आ गया"),
    ("i see", "मैं समझा"),
    ("i know", "मुझे पता है"),
    ("i dont know", "मुझे नहीं पता"),
    ("let me think", "मुझे सोचने दो"),
    ("let me check", "मुझे देखने दो"),
    ("one moment please", "एक पल कृपया"),
    ("wait please", "कृपया रुकिए"),
    
    # Emotions and states
    ("i am very happy", "मैं बहुत खुश हूं"),
    ("i am very sad", "मैं बहुत दुखी हूं"),
    ("i am excited", "मैं उत्साहित हूं"),
    ("i am worried", "मैं चिंतित हूं"),
    ("i am confused", "मैं भ्रमित हूं"),
    ("i am angry", "मैं नाराज़ हूं"),
    ("i am sorry to hear that", "यह सुनकर दुख हुआ"),
    ("thats great", "यह बहुत अच्छा है"),
    ("thats wonderful", "यह अद्भुत है"),
    ("thats amazing", "यह अद्भुत है"),
    ("thats terrible", "यह भयानक है"),
    ("thats good news", "यह अच्छी खबर है"),
    ("thats bad news", "यह बुरी खबर है"),
    
    # Sports and entertainment
    ("cricket", "क्रिकेट"),
    ("match", "मैच"),
    ("game", "खेल"),
    ("player", "खिलाड़ी"),
    ("team", "टीम"),
    ("win", "जीत"),
    ("lose", "हार"),
    ("score", "स्कोर"),
    ("cricket match", "क्रिकेट मैच"),
    ("india cricket match", "भारत क्रिकेट मैच"),
    ("tomorrow there is", "कल है"),
    ("there is", "है"),
    ("there are", "हैं"),
    ("there is a match", "एक मैच है"),
    ("there is a cricket match", "एक क्रिकेट मैच है"),
    ("tomorrow there is a cricket match", "कल एक क्रिकेट मैच है"),
    ("tomorrow there is a india cricket match", "कल भारत का क्रिकेट मैच है"),
    ("india", "भारत"),
    ("pakistan", "पाकिस्तान"),
    ("australia", "ऑस्ट्रेलिया"),
    ("england", "इंग्लैंड"),
    ("football", "फुटबॉल"),
    ("hockey", "हॉकी"),
    ("movie", "फ़िल्म"),
    ("film", "फ़िल्म"),
    ("song", "गाना"),
    ("music", "संगीत"),
    ("dance", "नृत्य"),
    ("tv", "टीवी"),
    ("television", "टेलीविज़न"),
    ("watch", "देखना"),
    ("watching", "देख रहा हूं"),
    ("i am watching", "मैं देख रहा हूं"),
    ("lets watch", "चलो देखते हैं"),
    
    # Countries and places
    ("america", "अमेरिका"),
    ("china", "चीन"),
    ("japan", "जापान"),
    ("russia", "रूस"),
    ("germany", "जर्मनी"),
    ("france", "फ्रांस"),
    ("italy", "इटली"),
    ("spain", "स्पेन"),
    ("canada", "कनाडा"),
    ("brazil", "ब्राज़ील"),
    ("delhi", "दिल्ली"),
    ("mumbai", "मुंबई"),
    ("kolkata", "कोलकाता"),
    ("chennai", "चेन्नई"),
    ("bangalore", "बैंगलोर"),
    
    # Common nouns
    ("book", "किताब"),
    ("phone", "फ़ोन"),
    ("computer", "कंप्यूटर"),
    ("laptop", "लैपटॉप"),
    ("car", "कार"),
    ("bus", "बस"),
    ("train", "ट्रेन"),
    ("plane", "हवाई जहाज़"),
    ("money", "पैसा"),
    ("time", "समय"),
    ("place", "जगह"),
    ("thing", "चीज़"),
    ("person", "व्यक्ति"),
    ("people", "लोग"),
    ("man", "आदमी"),
    ("woman", "औरत"),
    ("boy", "लड़का"),
    ("girl", "लड़की"),
    ("child", "बच्चा"),
    ("children", "बच्चे"),
    ("baby", "बच्चा"),
    ("name", "नाम"),
    ("country", "देश"),
    ("city", "शहर"),
    ("village", "गांव"),
    ("road", "सड़क"),
    ("street", "गली"),
    ("house", "मकान"),
    ("room", "कमरा"),
    ("door", "दरवाज़ा"),
    ("window", "खिड़की"),
    ("table", "मेज़"),
    ("chair", "कुर्सी"),
    ("bed", "बिस्तर"),
    
    # Weather
    ("weather", "मौसम"),
    ("rain", "बारिश"),
    ("sun", "सूरज"),
    ("cloud", "बादल"),
    ("wind", "हवा"),
    ("hot weather", "गर्म मौसम"),
    ("cold weather", "ठंडा मौसम"),
    ("its raining", "बारिश हो रही है"),
    ("its sunny", "धूप है"),
    
    # More useful phrases
    ("i like", "मुझे पसंद है"),
    ("i dont like", "मुझे पसंद नहीं है"),
    ("i want", "मैं चाहता हूं"),
    ("i need", "मुझे चाहिए"),
    ("i have", "मेरे पास है"),
    ("i dont have", "मेरे पास नहीं है"),
    ("do you have", "क्या आपके पास है"),
    ("can i", "क्या मैं"),
    ("may i", "क्या मैं"),
    ("would you", "क्या आप"),
    ("could you", "क्या आप"),
    ("should i", "क्या मुझे"),
    ("must i", "क्या मुझे"),
    ("have to", "करना होगा"),
    ("need to", "करना है"),
    ("want to", "चाहता हूं"),
    ("going to", "जा रहा हूं"),
    ("about to", "होने वाला है"),
]


class HuggingFaceDataLoader:
    """Data loader that uses Hugging Face datasets for training."""
    
    def __init__(self, max_samples=30000, max_sentence_length=20, use_local_fallback=True):
        """
        Initialize the data loader.
        
        Args:
            max_samples: Maximum number of samples to load from dataset
            max_sentence_length: Maximum sentence length in words
            use_local_fallback: If True, use curated pairs if HF download fails
        """
        self.max_samples = max_samples
        self.max_sentence_length = max_sentence_length
        self.use_local_fallback = use_local_fallback
        self.cache_dir = os.path.join(os.path.dirname(__file__), '..', 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def load_data(self):
        """Load training data from Hugging Face or local cache."""
        cache_file = os.path.join(self.cache_dir, f'hf_data_{self.max_samples}.pkl')
        
        # Try to load from cache first
        if os.path.exists(cache_file):
            print("Loading cached training data...")
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                print(f"Loaded {len(data)} pairs from cache")
                return data
            except Exception as e:
                print(f"Cache load failed: {e}")
        
        # Try to download from Hugging Face
        try:
            print("Downloading English-Hindi dataset from Hugging Face...")
            print("This may take a few minutes for first-time download...")
            
            from datasets import load_dataset
            
            # Load the CFILT IITB English-Hindi parallel corpus
            # This is one of the best quality English-Hindi datasets
            dataset = load_dataset("cfilt/iitb-english-hindi", split="train", trust_remote_code=True)
            
            pairs = []
            for i, item in enumerate(dataset):
                if i >= self.max_samples:
                    break
                    
                # Get English and Hindi text
                en_text = item.get('translation', {}).get('en', '') or item.get('en', '')
                hi_text = item.get('translation', {}).get('hi', '') or item.get('hi', '')
                
                if not en_text or not hi_text:
                    continue
                
                # Clean and filter
                en_text = en_text.strip().lower()
                hi_text = hi_text.strip()
                
                # Filter by length (keep shorter sentences for better training)
                en_words = len(en_text.split())
                if en_words > self.max_sentence_length or en_words < 1:
                    continue
                    
                # Skip sentences with too many numbers or special chars
                if any(c in en_text for c in ['@', '#', '$', '%', '&', '*', '<', '>', '|']):
                    continue
                    
                pairs.append((en_text, hi_text))
                
                if len(pairs) % 5000 == 0:
                    print(f"Processed {len(pairs)} pairs...")
            
            # Add curated pairs for guaranteed quality
            for en, hi in CURATED_PAIRS:
                pairs.append((en.lower(), hi))
            
            # Remove duplicates while preserving order
            seen = set()
            unique_pairs = []
            for pair in pairs:
                if pair[0] not in seen:
                    seen.add(pair[0])
                    unique_pairs.append(pair)
            
            pairs = unique_pairs
            print(f"Loaded {len(pairs)} unique training pairs")
            
            # Cache for future use
            with open(cache_file, 'wb') as f:
                pickle.dump(pairs, f)
            print("Training data cached for future use")
            
            return pairs
            
        except ImportError:
            print("datasets library not installed. Installing...")
            import subprocess
            subprocess.check_call(['pip', 'install', 'datasets'])
            return self.load_data()  # Retry after installation
            
        except Exception as e:
            print(f"Failed to download from Hugging Face: {e}")
            
            if self.use_local_fallback:
                print("Using local curated pairs as fallback...")
                return [(en.lower(), hi) for en, hi in CURATED_PAIRS]
            else:
                raise


class LocalDataLoader:
    """Fallback data loader using curated local pairs."""
    
    def __init__(self):
        pass
        
    def load_data(self):
        """Return curated training pairs."""
        return [(en.lower(), hi) for en, hi in CURATED_PAIRS]


# Default data loader
def get_data_loader(use_huggingface=True, max_samples=30000):
    """Get the appropriate data loader."""
    if use_huggingface:
        return HuggingFaceDataLoader(max_samples=max_samples)
    else:
        return LocalDataLoader()


# For backwards compatibility
DataLoader = HuggingFaceDataLoader
