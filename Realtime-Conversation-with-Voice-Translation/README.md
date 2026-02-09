# 🌐 Real-time Conversation with Voice Translation

A machine learning project that facilitates **real-time conversation between an English-speaking person and a Spanish-speaking person** using voice input, neural machine translation, and text-to-speech output.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📋 Project Description

This project implements a complete voice translation system that:

- **Extracts Spanish words from voice input** and translates them into English, then reads the translated text aloud
- **Takes English voice input** from the other user, translates it into Spanish, and reads the translated text aloud
- Uses a **custom-built machine learning model** (Seq2Seq with Attention mechanism)
- Provides both **web interface** and **command-line interface**

## ✨ Features

| Feature                | Description                                            |
| ---------------------- | ------------------------------------------------------ |
| 🎤 **Voice Input**     | Real-time speech recognition for English and Spanish   |
| 🤖 **Custom ML Model** | Seq2Seq with Bahdanau Attention (trained from scratch) |
| 🔊 **Voice Output**    | Text-to-speech for translated text                     |
| 🌐 **Web Interface**   | Beautiful, responsive web UI                           |
| ⚡ **Real-time**       | Instant translation with low latency                   |
| 🔄 **Bidirectional**   | English ↔ Spanish translation                          |

## 🏗️ Project Structure

```
Realtime-Conversation-with-Voice-Translation/
├── app.py                      # Flask web application
├── main.py                     # Command-line interface
├── voice_conversation.py       # Voice conversation system
├── Project_Analysis.ipynb      # Jupyter notebook with visualizations
├── requirements.txt            # Project dependencies
├── test_setup.py              # Installation verification
├── view_progress.py           # Training progress monitor
│
├── models/                    # Neural network models
│   ├── seq2seq.py            # Seq2Seq with Attention architecture
│   └── model_utils.py        # Model utilities
│
├── training/                  # Training scripts
│   ├── train_translator.py   # Main training script
│   ├── evaluate.py           # Model evaluation
│   └── config.py             # Training configuration
│
├── src/                       # Source modules
│   ├── translator.py         # Custom model translator
│   ├── translator_pretrained.py  # Pre-trained model translator
│   ├── speech_recognition_module.py  # Voice to text
│   ├── text_to_speech.py     # Text to voice
│   └── conversation.py       # Conversation flow
│
├── data/                      # Data processing
│   └── data_loader.py        # Dataset loading from Hugging Face
│
├── templates/                 # Web interface
│   └── index.html            # Main web page
│
└── checkpoints/              # Trained models
    ├── en_es/                # English → Spanish model
    └── es_en/                # Spanish → English model
```

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster training)
- Microphone (for voice input)
- Speakers (for voice output)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/Realtime-Conversation-with-Voice-Translation.git
cd Realtime-Conversation-with-Voice-Translation
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Windows Users - PyAudio Installation:**
If PyAudio fails to install, download the wheel file from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio) and install:

```bash
pip install PyAudio-0.2.11-cp311-cp311-win_amd64.whl
```

### Step 3: Verify Installation

```bash
python test_setup.py
```

## 🚀 Quick Start

### Option 1: Web Interface (Recommended)

```bash
python app.py
```

Open your browser and go to: **http://127.0.0.1:5000**

### Option 2: Command Line Interface

```bash
python main.py
```

### Option 3: Voice Conversation Mode

```bash
python voice_conversation.py
```

## 🧠 Model Architecture

### Custom Seq2Seq with Bahdanau Attention

```
┌─────────────────────────────────────────────────────────────┐
│                    ENCODER (Bidirectional LSTM)              │
│  Input: "Hello how are you"                                  │
│  ↓                                                           │
│  Embedding Layer (256 dim) → BiLSTM (512 hidden × 2 layers) │
│  ↓                                                           │
│  Encoder Outputs + Hidden States                             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    ATTENTION MECHANISM                       │
│  Bahdanau (Additive) Attention                              │
│  - Computes alignment scores                                 │
│  - Creates context vector from encoder outputs               │
│  - Focuses on relevant source words for each target word    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    DECODER (LSTM)                            │
│  Context Vector + Previous Token → LSTM (512 hidden)        │
│  ↓                                                           │
│  Output: "Hola cómo estás"                                   │
└─────────────────────────────────────────────────────────────┘
```

### Model Specifications

| Component               | Specification                      |
| ----------------------- | ---------------------------------- |
| **Encoder**             | 2-layer Bidirectional LSTM         |
| **Decoder**             | 2-layer LSTM with Attention        |
| **Embedding Dimension** | 256                                |
| **Hidden Dimension**    | 512                                |
| **Vocabulary Size**     | 15,000 words                       |
| **Total Parameters**    | ~16 million                        |
| **Training Data**       | 50,000 sentence pairs (opus_books) |
| **Dropout**             | 0.3                                |

## 📊 Model Performance

### Training Results

| Model       | Best Epoch | Training Loss | Validation Loss | Perplexity |
| ----------- | ---------- | ------------- | --------------- | ---------- |
| **EN → ES** | 21         | 4.53          | 5.93            | ~375       |
| **ES → EN** | 22         | 4.54          | 6.34            | ~566       |

### Training Details

- **Dataset**: opus_books (English-Spanish parallel corpus)
- **Training Samples**: 40,000 pairs
- **Validation Samples**: 5,000 pairs
- **Test Samples**: 5,000 pairs
- **Epochs**: 31 (with early stopping, patience=10)
- **Batch Size**: 128
- **Optimizer**: Adam (lr=0.001)
- **Device**: NVIDIA GeForce GTX 1650 (CUDA)

### 📈 Visualizations

For detailed training curves, model comparison, and analysis, see the **[Project_Analysis.ipynb](Project_Analysis.ipynb)** notebook which includes:

- Training and validation loss curves
- Model architecture visualization
- Custom model vs Pre-trained model comparison
- Translation examples
- System pipeline diagram

## 🎯 How It Works

### 1. Voice Input (Speech Recognition)

```python
# Using Google Speech Recognition API
recognizer = SpeechRecognizer(language='en-US')
text = recognizer.listen_from_microphone()
# → "Hello, how are you?"
```

### 2. Translation (Neural Machine Translation)

```python
# Using custom Seq2Seq or pre-trained Helsinki-NLP model
translator = PretrainedTranslator()
translation = translator.translate("Hello, how are you?", "en-es")
# → "Hola, ¿cómo estás?"
```

### 3. Voice Output (Text-to-Speech)

```python
# Using pyttsx3 or gTTS
tts = TextToSpeech(language='es')
tts.speak("Hola, ¿cómo estás?")
# → Audio plays: "Hola, ¿cómo estás?"
```

## 🛠️ Training Your Own Model

### Train English → Spanish Model

```bash
python training/train_translator.py --direction en-es
```

### Train Spanish → English Model

```bash
python training/train_translator.py --direction es-en
```

### Monitor Training Progress

```bash
python view_progress.py
```

### Evaluate Model

```bash
python training/evaluate.py --direction en-es --interactive
```

### Configuration

Edit `training/config.py` to adjust:

- `BATCH_SIZE`: Reduce if running out of memory
- `NUM_EPOCHS`: Increase for better quality
- `MAX_VOCAB_SIZE`: Vocabulary limit
- `HIDDEN_DIM`: Model capacity

## 🌐 Web Interface

The web interface provides:

- 🎤 **Voice Input Button**: Click to speak
- 🔄 **Language Switch**: Toggle between EN→ES and ES→EN
- ✨ **Translate Button**: Manual translation trigger
- 🔊 **Listen Button**: Hear the translation spoken aloud
- 📋 **Copy Button**: Copy translation to clipboard

### Screenshots

The interface features:

- Dark gradient theme
- Real-time voice wave animations
- Responsive design for mobile and desktop
- Instant translation feedback

## 📁 Files Description

| File                               | Purpose                                |
| ---------------------------------- | -------------------------------------- |
| `app.py`                           | Flask web server with translation API  |
| `main.py`                          | Command-line entry point               |
| `voice_conversation.py`            | Full voice conversation system         |
| `models/seq2seq.py`                | Neural network architecture definition |
| `training/train_translator.py`     | Training script with checkpointing     |
| `src/translator_pretrained.py`     | Helsinki-NLP MarianMT integration      |
| `src/speech_recognition_module.py` | Microphone input handling              |
| `src/text_to_speech.py`            | Audio output generation                |

## 🔌 API Endpoints

### Translation Endpoint

```http
POST /translate
Content-Type: application/json

{
    "text": "Hello, how are you?",
    "direction": "en-es"
}
```

**Response:**

```json
{
  "translation": "Hola, ¿cómo estás?",
  "original": "Hello, how are you?",
  "direction": "en-es"
}
```

## 🤝 Technologies Used

- **PyTorch**: Deep learning framework
- **Transformers**: Pre-trained models (Helsinki-NLP/MarianMT)
- **Flask**: Web framework
- **SpeechRecognition**: Voice input
- **pyttsx3 / gTTS**: Text-to-speech
- **Hugging Face Datasets**: Training data (opus_books)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for datasets and pre-trained models
- [Helsinki-NLP](https://huggingface.co/Helsinki-NLP) for MarianMT translation models
- [opus_books](https://opus.nlpl.eu/) for the parallel corpus dataset

## 👤 Author

**Shreyas** - Internship Project

---

⭐ If you found this project helpful, please give it a star!
