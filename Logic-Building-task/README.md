# English to Hindi Word Translator

A machine learning-based English to Hindi word translator with a modern GUI interface.

## Project Overview

This project uses a fine-tuned MarianMT transformer model to translate English words to Hindi. It includes a graphical user interface with time-based translation rules.

## Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 96% |
| Model Parameters | 74 million |
| Fine-tuning Dataset | 8,072 word pairs |
| Training Epochs | 8 |

## Features

- ML-Powered Translation: Fine-tuned transformer model for accurate word translation
- Time-Based Rules: Vowel words (A,E,I,O,U) only allowed between 9 PM - 10 PM
- Modern GUI: Dark-themed Tkinter interface with real-time status
- Hugging Face Dataset: Uses IITB English-Hindi parallel corpus

## Project Structure

```
Logic Building task/
├── app.py                    # Main GUI application
├── model/
│   ├── train.py             # Model training script
│   ├── translator.py        # Translation API
│   └── saved_model/         # Trained model weights
├── data/
│   ├── download_massive.py  # Dataset download script
│   ├── large_dictionary.py  # Dictionary word pairs
│   └── word_pairs.json      # Training data
├── utils/
│   └── time_validator.py    # Time and vowel validation
├── evaluate.py              # Accuracy evaluation
├── requirements.txt         # Python dependencies
└── README.md
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python app.py
```

## Usage

1. Enter an English word in the input field
2. Click TRANSLATE button or press Enter
3. View the Hindi translation

### Translation Rules

| Word Type | Time | Result |
|-----------|------|--------|
| Consonant words (book, water) | Anytime | Translated |
| Vowel words (apple, orange) | 9-10 PM | Translated |
| Vowel words (apple, orange) | Other times | Error shown |

## Technical Details

- Base Model: Helsinki-NLP/opus-mt-en-hi (MarianMT)
- Fine-tuning: Full fine-tuning on 8,072 word pairs
- Framework: PyTorch with Hugging Face Transformers
- GUI: Tkinter with custom dark theme

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Tkinter

## Sample Translations

| English | Hindi |
|---------|-------|
| book | किताब |
| water | पानी |
| computer | कंप्यूटर |
| beautiful | सुंदर |
| hello | नमस्ते |

---

Internship Project - Logic Building Task
