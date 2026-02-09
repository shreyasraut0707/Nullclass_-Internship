# 🎤 Voice Translator - English to Hindi

A real-time voice translation application that converts spoken English into Hindi text using a custom-trained neural machine translation model.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange)
![Status](https://img.shields.io/badge/Status-Complete-green)

## 📋 Overview

This project implements a complete voice translation pipeline:

1. **Speech Recognition** - Captures English audio from microphone in real-time
2. **Neural Translation** - Translates English text to Hindi using a custom Seq2Seq model
3. **GUI Display** - Shows results in a modern, dark-themed interface

The system operates during specific hours (9:00 PM - 10:00 PM) as per project requirements, with a testing mode available outside these hours.

## ✨ Features

- ✅ Real-time speech-to-text conversion
- ✅ Custom-trained neural machine translation model (Seq2Seq with Attention)
- ✅ 19,400+ phrase dictionary for accurate translations
- ✅ Modern dark-themed graphical user interface
- ✅ Time-restricted operation with testing mode
- ✅ Translation history tracking
- ✅ Audio clarity detection with repeat prompts
- ✅ Manual text translation option

## 🛠️ Requirements

- Python 3.8 or higher
- Windows 10/11 (for optimal microphone support)
- Working microphone
- Internet connection (for Google Speech Recognition API)

## 📦 Installation

1. **Clone or download this repository:**

```bash
git clone https://github.com/yourusername/voice-translator.git
cd voice-translator
```

2. **Install the required dependencies:**

```bash
pip install -r requirements.txt
```

3. **Ensure the trained model files are present in `saved_model/` directory.**

## 🚀 Usage

### Running the Application

```bash
python main.py
```

### How to Use

1. Launch the application using the command above
2. Wait for the model to load (status will show "✓ Loaded")
3. Click **"🎤 Start Listening"** button
4. Speak clearly in English
5. Pause for ~1 second when done speaking
6. View the Hindi translation in the right panel
7. Click **"⏹️ Stop Listening"** when finished

### Manual Translation

You can also type English text directly in the left panel and click **"Translate Text"** to get the Hindi translation.

## 📁 Project Structure

```
Voice Translator/
│
├── main.py                          # Application entry point
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
├── .gitignore                       # Git ignore rules
│
├── gui/
│   ├── __init__.py
│   └── app.py                       # Tkinter GUI application
│
├── model/
│   ├── __init__.py
│   └── translator.py                # Neural translation model
│
├── data/
│   ├── __init__.py
│   ├── training_data.py             # Training data loader
│   └── comprehensive_dictionary.py  # Word/phrase dictionary
│
├── utils/
│   ├── __init__.py
│   ├── speech_recognition_module.py # Real-time speech capture
│   ├── time_restriction.py          # Time-based access control
│   └── visualization.py             # Training visualizations
│
├── saved_model/                     # Trained model files
│   ├── best_model.h5                # Best model checkpoint
│   ├── translator_weights.h5        # Model weights
│   ├── translator_config.pkl        # Model configuration
│   ├── translator_eng_tokenizer.pkl # English tokenizer
│   ├── translator_hin_tokenizer.pkl # Hindi tokenizer
│   └── translator_phrase_dict.pkl   # Phrase dictionary
│
└── outputs/                         # Training visualizations
    ├── training_history.png
    ├── loss_comparison.png
    ├── sample_predictions.png
    └── model_architecture.png
```

## 🧠 Model Architecture

The translation system uses a hybrid approach for optimal accuracy:

### 1. Phrase Dictionary (Primary)

- **19,423** pre-mapped English-Hindi phrase pairs
- Instant lookup for common conversational phrases
- High accuracy for everyday expressions

### 2. Seq2Seq Neural Network (Fallback)

| Component           | Specification                              |
| ------------------- | ------------------------------------------ |
| Encoder             | Bidirectional LSTM (512 units)             |
| Decoder             | LSTM with Attention mechanism              |
| Embedding           | 256-dimensional word embeddings            |
| Vocabulary          | English: 18,165 words, Hindi: 18,054 words |
| Total Parameters    | 30,402,950 (~116 MB)                       |
| Validation Accuracy | 84.6%                                      |

### Training Data Sources

- Helsinki-NLP/opus-100 (English-Hindi subset)
- CFILT IIT Bombay English-Hindi Corpus
- Curated conversational phrases

## 💻 Technical Specifications

| Component          | Technology                    |
| ------------------ | ----------------------------- |
| Language           | Python 3.11                   |
| Deep Learning      | TensorFlow 2.13, Keras        |
| Speech Recognition | Google Speech Recognition API |
| GUI Framework      | Tkinter (DPI-aware)           |
| Audio Processing   | PyAudio, SpeechRecognition    |

## ⏰ Time Restriction

The application operates during:

| Mode        | Hours              | Status                             |
| ----------- | ------------------ | ---------------------------------- |
| **Active**  | 9:00 PM - 10:00 PM | Full operation                     |
| **Testing** | Other hours        | Fully functional (for development) |

This restriction is implemented as per the internship project requirements.

## 🔧 Troubleshooting

### Microphone not detected

- Ensure microphone is properly connected
- Check Windows Sound Settings → Input
- Grant microphone permissions to Python

### Speech not recognized

- Speak clearly and at moderate pace
- Ensure minimal background noise
- Check internet connection (required for Google API)

### Model not loading

- Verify all files exist in `saved_model/` directory
- Check if TensorFlow is properly installed
- Run `pip install tensorflow==2.13.0`

### Poor translation quality

- Use simple, common English phrases
- Speak complete sentences
- Avoid slang and abbreviations

## 📚 Dependencies

```
tensorflow>=2.10.0
numpy>=1.21.0
SpeechRecognition>=3.8.1
PyAudio>=0.2.11
pillow>=8.0.0
matplotlib>=3.4.0
```

## 🎯 Key Achievements

- ✅ Custom-trained Seq2Seq model with 30M+ parameters
- ✅ 84.6% validation accuracy
- ✅ 19,423 phrase dictionary entries
- ✅ Real-time speech recognition
- ✅ Modern, responsive GUI
- ✅ Complete sentence capture
- ✅ Proper start/stop functionality

## 📄 License

This project is developed for educational purposes as part of an internship assignment.

## 👨‍💻 Author

**Shreyas**

_Internship Project - December 2024_

---

<p align="center">
  <b>Voice Translator v1.0</b><br>
  <i>Custom Seq2Seq Neural Network for English-Hindi Translation</i>
</p>
