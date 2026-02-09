"""
English to Hindi Word Translator - GUI Application
A machine learning-based translator with time-sensitive vowel word handling.

Features:
- Translates English words to Hindi using fine-tuned MarianMT model
- Words starting with vowels only allowed between 9 PM - 10 PM
- Modern, interactive Tkinter GUI
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.time_validator import is_translation_allowed, get_current_time_status, is_vowel_word
from model.translator import get_translator


class TranslatorApp:
    """
    Main application class for the English-Hindi translator GUI.
    """
    
    def __init__(self, root):
        """Initialize the translator application."""
        self.root = root
        self.root.title("English to Hindi Translator")
        self.root.geometry("700x600")
        self.root.resizable(True, True)
        self.root.minsize(600, 500)
        
        # Color scheme - Original theme with interactive enhancements
        self.colors = {
            'bg_dark': '#1a1a2e',
            'bg_medium': '#16213e',
            'bg_light': '#0f3460',
            'accent': '#e94560',
            'accent_hover': '#ff2e63',
            'accent_active': '#c73e54',
            'success': '#00d4aa',
            'warning': '#ffc107',
            'error': '#ff6b6b',
            'text_light': '#ffffff',
            'text_muted': '#a0aec0',
            'input_bg': '#2d3748',
            'input_focus': '#3d4a5c',
            'button_hover': '#ff2e63'
        }
        
        # Configure root
        self.root.configure(bg=self.colors['bg_dark'])
        
        # Load translator in background
        self.translator = None
        self.is_loading = True
        self.current_translation = ""
        
        # Setup UI
        self.setup_ui()
        
        # Load model in background thread
        self.load_model_async()
    
    def setup_ui(self):
        """Setup the main user interface."""
        # Main container with padding
        main_frame = tk.Frame(self.root, bg=self.colors['bg_dark'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)
        
        # ===== HEADER SECTION =====
        self.create_header(main_frame)
        
        # ===== INPUT SECTION =====
        self.create_input_section(main_frame)
        
        # ===== TRANSLATE BUTTON =====
        self.create_translate_button(main_frame)
        
        # ===== OUTPUT SECTION =====
        self.create_output_section(main_frame)
    
    def create_header(self, parent):
        """Create the header section with title."""
        header_frame = tk.Frame(parent, bg=self.colors['bg_dark'])
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Main title
        self.title_label = tk.Label(
            header_frame,
            text="English to Hindi Translator",
            font=("Segoe UI", 28, "bold italic"),
            fg=self.colors['text_light'],
            bg=self.colors['bg_dark'],
            cursor='hand2'
        )
        self.title_label.pack()
        
        # Title hover effect
        self.title_label.bind('<Enter>', lambda e: self.title_label.configure(fg=self.colors['accent']))
        self.title_label.bind('<Leave>', lambda e: self.title_label.configure(fg=self.colors['text_light']))
        
        # Subtitle
        self.subtitle_label = tk.Label(
            header_frame,
            text="Machine Learning Powered Word Translation",
            font=("Segoe UI", 12),
            fg=self.colors['text_muted'],
            bg=self.colors['bg_dark']
        )
        self.subtitle_label.pack(pady=(5, 0))
    
    def create_input_section(self, parent):
        """Create the input section for English words."""
        input_frame = tk.Frame(parent, bg=self.colors['bg_dark'])
        input_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Input label
        input_label = tk.Label(
            input_frame,
            text="Enter English Word:",
            font=("Segoe UI", 14, "bold"),
            fg=self.colors['text_light'],
            bg=self.colors['bg_dark'],
            anchor='w'
        )
        input_label.pack(fill=tk.X, pady=(0, 10))
        
        # Input container for border effect
        self.input_container = tk.Frame(input_frame, bg=self.colors['accent'], padx=2, pady=2)
        self.input_container.pack(fill=tk.X)
        
        # Input text field
        self.input_entry = tk.Entry(
            self.input_container,
            font=("Segoe UI", 18),
            fg=self.colors['text_muted'],
            bg=self.colors['input_bg'],
            insertbackground=self.colors['text_light'],
            relief=tk.FLAT,
            justify='center'
        )
        self.input_entry.pack(fill=tk.X, ipady=15)
        
        # Placeholder text
        self.placeholder_text = "Type a word here..."
        self.input_entry.insert(0, self.placeholder_text)
        
        # Bind events for interactivity
        self.input_entry.bind('<Return>', lambda e: self.translate())
        self.input_entry.bind('<FocusIn>', self.on_entry_focus_in)
        self.input_entry.bind('<FocusOut>', self.on_entry_focus_out)
        self.input_entry.bind('<KeyRelease>', self.on_key_release)
    
    def create_translate_button(self, parent):
        """Create the translate button with hover effects."""
        button_frame = tk.Frame(parent, bg=self.colors['bg_dark'])
        button_frame.pack(fill=tk.X, pady=15)
        
        # Button container for press effect
        self.button_container = tk.Frame(button_frame, bg=self.colors['bg_dark'])
        self.button_container.pack(fill=tk.X)
        
        self.translate_button = tk.Button(
            self.button_container,
            text="🔄 TRANSLATE",
            font=("Segoe UI", 14, "bold"),
            fg=self.colors['text_light'],
            bg=self.colors['accent'],
            activebackground=self.colors['accent_active'],
            activeforeground=self.colors['text_light'],
            relief=tk.FLAT,
            cursor='hand2',
            command=self.translate
        )
        self.translate_button.pack(fill=tk.X, ipady=12)
        
        # Hover effects
        self.translate_button.bind('<Enter>', self.on_button_enter)
        self.translate_button.bind('<Leave>', self.on_button_leave)
        self.translate_button.bind('<ButtonPress-1>', self.on_button_press)
        self.translate_button.bind('<ButtonRelease-1>', self.on_button_release)
    
    def on_button_enter(self, event):
        """Button hover enter effect."""
        if self.translate_button['state'] != tk.DISABLED:
            self.translate_button.configure(bg=self.colors['button_hover'])
            # Slight scale effect simulation with padding
            self.translate_button.pack_configure(ipady=14)
    
    def on_button_leave(self, event):
        """Button hover leave effect."""
        if self.translate_button['state'] != tk.DISABLED:
            self.translate_button.configure(bg=self.colors['accent'])
            self.translate_button.pack_configure(ipady=12)
    
    def on_button_press(self, event):
        """Button press effect."""
        if self.translate_button['state'] != tk.DISABLED:
            self.translate_button.configure(bg=self.colors['accent_active'])
            self.translate_button.pack_configure(ipady=11)
    
    def on_button_release(self, event):
        """Button release effect."""
        if self.translate_button['state'] != tk.DISABLED:
            self.translate_button.configure(bg=self.colors['button_hover'])
            self.translate_button.pack_configure(ipady=14)
    
    def create_output_section(self, parent):
        """Create the output section for Hindi translation."""
        # Output frame with interactive border
        self.output_frame = tk.Frame(parent, bg=self.colors['bg_light'], padx=20, pady=20)
        self.output_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        self.output_frame.configure(highlightbackground=self.colors['bg_light'], highlightthickness=2)
        
        # Output label
        output_label = tk.Label(
            self.output_frame,
            text="Hindi Translation:",
            font=("Segoe UI", 14, "bold"),
            fg=self.colors['text_light'],
            bg=self.colors['bg_light'],
            anchor='w'
        )
        output_label.pack(fill=tk.X, pady=(0, 10))
        
        # Output display (larger font for Hindi)
        self.output_label = tk.Label(
            self.output_frame,
            text="Translation will appear here",
            font=("Nirmala UI", 42, "bold"),
            fg=self.colors['text_muted'],
            bg=self.colors['bg_light'],
            wraplength=550,
            cursor='hand2'
        )
        self.output_label.pack(fill=tk.BOTH, expand=True, pady=25)
        
        # Click to copy functionality
        self.output_label.bind('<Button-1>', self.copy_on_click)
        self.output_label.bind('<Enter>', self.on_output_enter)
        self.output_label.bind('<Leave>', self.on_output_leave)
        
        # Status message (for errors and success)
        self.status_message = tk.Label(
            self.output_frame,
            text="",
            font=("Segoe UI", 11),
            fg=self.colors['warning'],
            bg=self.colors['bg_light']
        )
        self.status_message.pack(fill=tk.X)
    
    def on_output_enter(self, event):
        """Output hover enter - show copy hint."""
        if self.current_translation:
            self.status_message.configure(
                text="📋 Click to copy translation",
                fg=self.colors['text_muted']
            )
    
    def on_output_leave(self, event):
        """Output hover leave - restore status."""
        if self.current_translation:
            vowel_indicator = "🔤 (Vowel word)" if self.last_word and is_vowel_word(self.last_word) else ""
            self.status_message.configure(
                text=f"✓ Translated: '{self.last_word}' {vowel_indicator}",
                fg=self.colors['success']
            )
    
    def copy_on_click(self, event):
        """Copy translation when output is clicked."""
        if self.current_translation:
            self.root.clipboard_clear()
            self.root.clipboard_append(self.current_translation)
            
            # Visual feedback
            original_text = self.status_message.cget("text")
            original_color = self.status_message.cget("fg")
            self.status_message.configure(text="✅ Copied to clipboard!", fg=self.colors['success'])
            
            # Flash the output
            self.output_label.configure(fg=self.colors['text_light'])
            self.root.after(150, lambda: self.output_label.configure(fg=self.colors['success']))
            
            # Restore after delay
            self.root.after(1500, lambda: self.status_message.configure(text=original_text, fg=original_color))
    
    def on_entry_focus_in(self, event):
        """Handle input field focus in with animation."""
        if self.input_entry.get() == self.placeholder_text:
            self.input_entry.delete(0, tk.END)
            self.input_entry.configure(fg=self.colors['text_light'])
        
        # Highlight border
        self.input_container.configure(bg=self.colors['button_hover'])
        self.input_entry.configure(bg=self.colors['input_focus'])
    
    def on_entry_focus_out(self, event):
        """Handle input field focus out."""
        if not self.input_entry.get():
            self.input_entry.insert(0, self.placeholder_text)
            self.input_entry.configure(fg=self.colors['text_muted'])
        
        # Reset border
        self.input_container.configure(bg=self.colors['accent'])
        self.input_entry.configure(bg=self.colors['input_bg'])
    
    def on_key_release(self, event):
        """Handle key release for visual feedback."""
        text = self.input_entry.get()
        if text and text != self.placeholder_text:
            # Visual feedback for vowel words
            if text[0].upper() in 'AEIOU':
                self.input_container.configure(bg=self.colors['warning'])
            else:
                self.input_container.configure(bg=self.colors['success'])
        else:
            self.input_container.configure(bg=self.colors['accent'])
    
    def load_model_async(self):
        """Load the translation model in a background thread."""
        def load():
            self.translator = get_translator()
            success = self.translator.load_model()
            self.is_loading = False
            self.root.after(0, lambda: self.update_model_status(success))
        
        thread = threading.Thread(target=load, daemon=True)
        thread.start()
    
    def update_model_status(self, success):
        """Update model status."""
        if success:
            self.status_message.configure(text="✅ Model loaded - Ready to translate!", fg=self.colors['success'])
            self.root.after(2000, lambda: self.status_message.configure(text=""))
    
    def translate(self):
        """Perform the translation with visual feedback."""
        # Get input word
        word = self.input_entry.get().strip()
        
        # Check for placeholder
        if word == self.placeholder_text or not word:
            self.show_error("Please enter a word to translate.")
            self.shake_input()
            return
        
        # Check if model is loaded
        if self.is_loading:
            self.show_error("Please wait, model is still loading...")
            return
        
        # Validate translation (vowel and time check)
        is_allowed, error_message = is_translation_allowed(word)
        
        if not is_allowed:
            self.show_error(error_message)
            return
        
        # Show loading state
        self.translate_button.configure(text="⏳ Translating...", state=tk.DISABLED, bg=self.colors['warning'])
        self.status_message.configure(text="", fg=self.colors['warning'])
        self.output_label.configure(text="...", fg=self.colors['text_muted'])
        
        def do_translate():
            try:
                hindi_translation = self.translator.translate(word)
                self.root.after(0, lambda: self.show_translation(word, hindi_translation))
            except Exception as e:
                self.root.after(0, lambda: self.show_error(f"Translation failed: {str(e)}"))
        
        thread = threading.Thread(target=do_translate, daemon=True)
        thread.start()
    
    def shake_input(self):
        """Shake animation for input field on error."""
        # Flash the border red
        self.input_container.configure(bg=self.colors['error'])
        self.root.after(100, lambda: self.input_container.configure(bg=self.colors['accent']))
        self.root.after(200, lambda: self.input_container.configure(bg=self.colors['error']))
        self.root.after(300, lambda: self.input_container.configure(bg=self.colors['accent']))
        self.root.after(400, lambda: self.input_container.configure(bg=self.colors['error']))
        self.root.after(500, lambda: self.input_container.configure(bg=self.colors['accent']))
    
    def show_translation(self, english_word, hindi_translation):
        """Display the translation result with animation."""
        self.translate_button.configure(text="🔄 TRANSLATE", state=tk.NORMAL, bg=self.colors['accent'])
        
        # Store for copy functionality
        self.current_translation = hindi_translation
        self.last_word = english_word
        
        # Animate output appearance
        self.output_label.configure(text="", fg=self.colors['success'])
        self.root.after(50, lambda: self.output_label.configure(text=hindi_translation))
        
        # Flash success border
        self.output_frame.configure(highlightbackground=self.colors['success'])
        self.root.after(500, lambda: self.output_frame.configure(highlightbackground=self.colors['bg_light']))
        
        # Show success status
        vowel_indicator = "🔤 (Vowel word)" if is_vowel_word(english_word) else ""
        self.status_message.configure(
            text=f"✓ Translated: '{english_word}' {vowel_indicator}",
            fg=self.colors['success']
        )
    
    def show_error(self, message):
        """Display an error message with animation."""
        self.translate_button.configure(text="🔄 TRANSLATE", state=tk.NORMAL, bg=self.colors['accent'])
        
        self.output_label.configure(text="✗", fg=self.colors['error'])
        
        # Flash error border
        self.output_frame.configure(highlightbackground=self.colors['error'])
        self.root.after(500, lambda: self.output_frame.configure(highlightbackground=self.colors['bg_light']))
        
        self.status_message.configure(text=message, fg=self.colors['error'])
        
        # Clear stored translation
        self.current_translation = ""


def main():
    """Main entry point for the application."""
    root = tk.Tk()
    
    # Initialize last_word attribute
    app = TranslatorApp(root)
    app.last_word = ""
    
    # Center window on screen
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f'+{x}+{y}')
    
    root.mainloop()


if __name__ == "__main__":
    main()
