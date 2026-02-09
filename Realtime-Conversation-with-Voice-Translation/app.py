"""
Web Interface for Real-time Voice Translation System
Simple Flask-based UI for English-Spanish translation
Uses pre-trained Helsinki-NLP MarianMT models for high-quality translation.
"""

from flask import Flask, render_template, request, jsonify, after_this_request
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.translator_pretrained import PretrainedTranslator

app = Flask(__name__)

# Enable CORS for all routes
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response

# Initialize translator
print("Loading translation models...")
translator = None

def get_translator():
    global translator
    if translator is None:
        translator = PretrainedTranslator()
        print("Models loaded successfully!")
    return translator

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/translate', methods=['POST', 'OPTIONS'])
def translate():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'})
    try:
        print("Received translation request")
        data = request.get_json()
        print(f"Data: {data}")
        text = data.get('text', '').strip()
        direction = data.get('direction', 'en-es')
        
        if not text:
            return jsonify({'error': 'Please enter some text to translate', 'translation': ''})
        
        t = get_translator()
        translation = t.translate(text, direction)
        
        return jsonify({
            'translation': translation,
            'original': text,
            'direction': direction
        })
    except Exception as e:
        return jsonify({'error': str(e), 'translation': ''})

if __name__ == '__main__':
    # Pre-load models
    get_translator()
    print("\n" + "="*50)
    print("Translation Web Interface Started!")
    print("Open your browser and go to: http://127.0.0.1:5000")
    print("="*50 + "\n")
    app.run(debug=False, host='127.0.0.1', port=5000)
