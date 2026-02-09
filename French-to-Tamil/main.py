"""
French to Tamil Translation Application
A GUI application using a custom-trained Neural Network for French to Tamil translation.
Only translates French words with exactly 5 letters.

NO EXTERNAL API - Uses locally trained PyTorch model.
"""

import tkinter as tk
from tkinter import messagebox
import threading
from translator import FrenchToTamilTranslator


class TranslationApp:
    """Main application class for French to Tamil translation using ML model."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("French to Tamil ML Translator")
        self.root.geometry("750x700")
        self.root.resizable(True, True)
        self.root.minsize(650, 600)
        
        self.last_translation = ""
        self.translator = FrenchToTamilTranslator()
        self.model_loaded = False
        
        # Theme colors
        self.bg_color = "#f5f5f5"
        self.card_color = "#ffffff"
        self.primary_color = "#4CAF50"
        self.secondary_color = "#2196F3"
        self.text_color = "#333333"
        self.border_color = "#dddddd"
        self.warning_color = "#FF9800"
        
        self.root.configure(bg=self.bg_color)
        self.setup_ui()
        
        # Start model loading in background
        self.load_model_async()
    
    def setup_ui(self):
        """Create and arrange all UI components."""
        
        main_frame = tk.Frame(self.root, bg=self.bg_color)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)
        
        # Header
        header = tk.Frame(main_frame, bg=self.bg_color)
        header.pack(fill=tk.X, pady=(0, 15))
        
        title = tk.Label(
            header,
            text="French to Tamil ML Translator",
            font=("Segoe UI", 22, "bold"),
            fg=self.text_color,
            bg=self.bg_color
        )
        title.pack()
        
        # Info label about 5-letter constraint
        info_frame = tk.Frame(main_frame, bg="#FFF3E0", bd=1, relief=tk.SOLID)
        info_frame.pack(fill=tk.X, pady=(0, 15))
        
        info_label = tk.Label(
            info_frame,
            text="⚠️ Note: Only translates French words with exactly 5 letters",
            font=("Segoe UI", 10),
            fg="#E65100",
            bg="#FFF3E0",
            pady=8
        )
        info_label.pack()
        
        # Input section
        input_frame = tk.Frame(main_frame, bg=self.card_color, bd=1, relief=tk.SOLID)
        input_frame.pack(fill=tk.X, pady=(0, 15))
        
        input_inner = tk.Frame(input_frame, bg=self.card_color)
        input_inner.pack(fill=tk.X, padx=20, pady=20)
        
        input_label = tk.Label(
            input_inner,
            text="Enter French Word (5 letters):",
            font=("Segoe UI", 11, "bold"),
            fg=self.text_color,
            bg=self.card_color
        )
        input_label.pack(anchor=tk.W, pady=(0, 8))
        
        self.input_entry = tk.Entry(
            input_inner,
            font=("Segoe UI", 14),
            bg="#ffffff",
            fg=self.text_color,
            relief=tk.SOLID,
            bd=1
        )
        self.input_entry.pack(fill=tk.X, ipady=10)
        self.input_entry.bind("<Return>", lambda e: self.translate())
        self.input_entry.focus()
        
        # Buttons
        button_frame = tk.Frame(main_frame, bg=self.bg_color)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.translate_btn = tk.Button(
            button_frame,
            text="Translate",
            font=("Segoe UI", 12, "bold"),
            bg=self.primary_color,
            fg="white",
            relief=tk.FLAT,
            cursor="hand2",
            command=self.translate
        )
        self.translate_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5), ipady=10)
        
        self.clear_btn = tk.Button(
            button_frame,
            text="Clear",
            font=("Segoe UI", 12, "bold"),
            bg=self.secondary_color,
            fg="white",
            relief=tk.FLAT,
            cursor="hand2",
            command=self.clear_all
        )
        self.clear_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(5, 0), ipady=10)
        
        # Output section
        output_frame = tk.Frame(main_frame, bg=self.card_color, bd=1, relief=tk.SOLID)
        output_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 15))
        
        output_inner = tk.Frame(output_frame, bg=self.card_color)
        output_inner.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)
        
        output_header = tk.Frame(output_inner, bg=self.card_color)
        output_header.pack(fill=tk.X, pady=(0, 8))
        
        output_label = tk.Label(
            output_header,
            text="Tamil Translation:",
            font=("Segoe UI", 11, "bold"),
            fg=self.text_color,
            bg=self.card_color
        )
        output_label.pack(side=tk.LEFT)
        
        self.copy_btn = tk.Button(
            output_header,
            text="Copy",
            font=("Segoe UI", 9),
            bg="#FF7043",
            fg="white",
            relief=tk.FLAT,
            cursor="hand2",
            command=self.copy_translation
        )
        self.copy_btn.pack(side=tk.RIGHT, ipadx=10, ipady=3)
        
        # Output text area
        text_frame = tk.Frame(output_inner, bg=self.border_color)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.output_text = tk.Text(
            text_frame,
            font=("Segoe UI", 13),
            bg="#ffffff",
            fg=self.text_color,
            relief=tk.FLAT,
            wrap=tk.WORD,
            state="disabled",
            padx=12,
            pady=12
        )
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        
        scrollbar = tk.Scrollbar(self.output_text, command=self.output_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.output_text.config(yscrollcommand=scrollbar.set)
        
        # Status bar
        self.status_label = tk.Label(
            main_frame,
            text="Ready",
            font=("Segoe UI", 9),
            fg="#4CAF50",
            bg=self.bg_color
        )
        self.status_label.pack(anchor=tk.W)
    
    def translate(self):
        """Translate the input text from French to Tamil."""
        french_text = self.input_entry.get().strip()
        
        if not french_text:
            self.status_label.config(text="Please enter a French word", fg="#f44336")
            return
        
        self.status_label.config(text="Translating...", fg=self.secondary_color)
        self.root.update()
        
        try:
            tamil_text = self.translator.translate(french_text)
            self.last_translation = tamil_text
            
            self.output_text.config(state="normal")
            self.output_text.insert(tk.END, f"French: {french_text}\n")
            self.output_text.insert(tk.END, f"Tamil: {tamil_text}\n")
            self.output_text.insert(tk.END, "-" * 40 + "\n\n")
            self.output_text.config(state="disabled")
            self.output_text.see(tk.END)
            
            self.status_label.config(text="Translation completed", fg="#4CAF50")
            
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}", fg="#f44336")
            messagebox.showerror("Error", f"Translation failed: {str(e)}")
    
    def copy_translation(self):
        """Copy the last translation to clipboard."""
        if self.last_translation:
            self.root.clipboard_clear()
            self.root.clipboard_append(self.last_translation)
            self.status_label.config(text="Copied to clipboard", fg="#4CAF50")
        else:
            self.output_text.config(state="normal")
            text = self.output_text.get(1.0, tk.END).strip()
            self.output_text.config(state="disabled")
            
            if text:
                self.root.clipboard_clear()
                self.root.clipboard_append(text)
                self.status_label.config(text="Copied to clipboard", fg="#4CAF50")
            else:
                self.status_label.config(text="Nothing to copy", fg="#f44336")
    
    def clear_all(self):
        """Clear input and output fields."""
        self.input_entry.delete(0, tk.END)
        self.output_text.config(state="normal")
        self.output_text.delete(1.0, tk.END)
        self.output_text.config(state="disabled")
        self.last_translation = ""
        self.status_label.config(text="Cleared", fg=self.secondary_color)
        self.input_entry.focus()
    
    def load_model_async(self):
        """Load ML model in background thread."""
        self.status_label.config(text="Loading model...", fg=self.secondary_color)
        
        def load_in_thread():
            def progress_callback(message):
                self.root.after(0, lambda: self.status_label.config(text=message))
            
            success = self.translator.load_model(progress_callback)
            
            def update_ui():
                if success:
                    self.model_loaded = True
                    self.status_label.config(text="Model loaded - Ready to translate!", fg="#4CAF50")
                else:
                    self.status_label.config(text="Model loading failed", fg="#F44336")
            
            self.root.after(0, update_ui)
        
        thread = threading.Thread(target=load_in_thread, daemon=True)
        thread.start()


def main():
    """Entry point for the application."""
    root = tk.Tk()
    
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass
    
    app = TranslationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
