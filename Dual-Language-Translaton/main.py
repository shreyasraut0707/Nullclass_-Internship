"""
Dual Language Translator - Main Application
Translates English to French and Hindi
"""

print("Starting Dual Language Translator...")
print()

from translation_model import DualLanguageTranslator
from validator import InputValidator

validator = InputValidator()
translator = DualLanguageTranslator()
print("Components loaded.")
print()

import gradio as gr


def translate(text):
    """Translate English text to French and Hindi."""
    if not text or not text.strip():
        return "Please enter some text to translate", "", ""
    
    is_valid, message = validator.validate_input(text)
    
    if not is_valid:
        return message, "", ""
    
    try:
        french = translator.translate_to_french(text)
        hindi = translator.translate_to_hindi(text)
        letter_count = validator.count_letters(text)
        
        status = f"Translation Successful! ({letter_count} letters translated)"
        return status, french, hindi
        
    except Exception as e:
        return f"Translation Error: {str(e)}", "", ""


def clear_all():
    """Clear all fields."""
    return "", "", "", ""


with gr.Blocks(
    title="Dual Language Translator",
    theme=gr.themes.Soft()
) as app:
    
    gr.Markdown("""
    # Dual Language Translator
    ### Translate English to French and Hindi
    
    Your input must contain at least 10 letters (spaces and punctuation are not counted)
    """)
    
    gr.Markdown("---")
    
    input_text = gr.Textbox(
        label="Enter English Text",
        placeholder="Type or paste your English text here (minimum 10 letters)...",
        lines=4,
        show_copy_button=True
    )
    
    with gr.Row():
        translate_btn = gr.Button("Translate", variant="primary", scale=2)
        clear_btn = gr.Button("Clear All", variant="secondary", scale=1)
    
    gr.Markdown("### Status")
    status = gr.Textbox(label="Translation Status", interactive=False)
    
    gr.Markdown("---")
    gr.Markdown("### Translation Results")
    
    with gr.Row():
        french_output = gr.Textbox(
            label="French Translation",
            lines=4,
            interactive=False,
            show_copy_button=True
        )
        hindi_output = gr.Textbox(
            label="Hindi Translation",
            lines=4,
            interactive=False,
            show_copy_button=True
        )
    
    gr.Markdown("---")
    gr.Markdown("### Examples (click to use)")
    
    gr.Examples(
        examples=[
            ["Good morning, how are you today?"],
            ["The weather is beautiful and sunny."],
            ["I am learning machine learning and artificial intelligence."],
            ["Technology is changing the world rapidly."],
            ["Education is the most powerful weapon to change the world."],
        ],
        inputs=input_text,
        label=""
    )
    
    translate_btn.click(
        fn=translate,
        inputs=input_text,
        outputs=[status, french_output, hindi_output]
    )
    
    clear_btn.click(
        fn=clear_all,
        inputs=None,
        outputs=[input_text, status, french_output, hindi_output]
    )


if __name__ == "__main__":
    print("Starting web server...")
    print("The application will open in your browser.")
    print("Press Ctrl+C to stop.")
    print()
    
    app.launch(
        server_name="0.0.0.0",
        server_port=None,
        share=False,
        inbrowser=True,
        show_api=False
    )
