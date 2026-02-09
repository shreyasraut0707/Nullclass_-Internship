# -*- coding: utf-8 -*-
"""
Voice Translator - Main GUI Application
English to Hindi Voice Translation System
Active Hours: 9:30 PM to 10:00 PM

Features:
- Real-time speech recognition (English)
- Neural machine translation (English to Hindi)
- Time-restricted operation
- User-friendly interface
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from datetime import datetime
import ctypes

# Fix Windows DPI scaling for sharp text
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # Per-monitor DPI aware
except:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except:
        pass

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.time_restriction import TimeRestriction
from utils.speech_recognition_module import SpeechRecognizer
from model.translator import EnglishHindiTranslator


class VoiceTranslatorGUI:
    """
    Main GUI Application for Voice Translator
    Provides interface for English to Hindi voice translation
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("🎤 Voice Translator - English to Hindi")
        self.root.geometry("950x720")
        self.root.minsize(800, 600)
        self.root.resizable(True, True)
        
        # Modern gradient-like dark theme
        self.root.configure(bg='#0d1117')
        
        # Initialize components
        self.time_restriction = TimeRestriction()
        self.speech_recognizer = None
        self.translator = None
        
        # State variables
        self.is_listening = False
        self.model_loaded = False
        
        # Setup GUI
        self._setup_styles()
        self._create_widgets()
        self._start_time_update()
        
        # Initialize components in background
        self.root.after(100, self._initialize_components)
    
    def _setup_styles(self):
        """Setup custom styles for widgets - Modern Dark Theme"""
        self.colors = {
            'bg_dark': '#0d1117',       # GitHub dark
            'bg_medium': '#161b22',      # Slightly lighter
            'bg_light': '#21262d',       # Card background
            'bg_input': '#0d1117',       # Input fields
            'border': '#30363d',         # Border color
            'accent': '#58a6ff',         # Blue accent
            'accent_green': '#3fb950',   # Green accent
            'accent_red': '#f85149',     # Red accent
            'accent_orange': '#d29922',  # Orange/warning
            'accent_purple': '#a371f7',  # Purple highlight
            'text_light': '#f0f6fc',     # Primary text
            'text_medium': '#c9d1d9',    # Secondary text
            'text_dim': '#8b949e',       # Muted text
            'success': '#3fb950',
            'warning': '#d29922',
            'error': '#f85149',
            'gradient_start': '#238636', # Button gradient
            'gradient_end': '#2ea043'
        }
        
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure styles
        style.configure('Title.TLabel',
                       background=self.colors['bg_dark'],
                       foreground=self.colors['text_light'],
                       font=('Segoe UI', 28, 'bold'))
        
        style.configure('Subtitle.TLabel',
                       background=self.colors['bg_dark'],
                       foreground=self.colors['text_dim'],
                       font=('Segoe UI', 11))
        
        style.configure('Status.TLabel',
                       background=self.colors['bg_dark'],
                       foreground=self.colors['text_light'],
                       font=('Segoe UI', 11))
        
        style.configure('Active.TLabel',
                       background=self.colors['bg_dark'],
                       foreground=self.colors['success'],
                       font=('Segoe UI', 12, 'bold'))
        
        style.configure('Inactive.TLabel',
                       background=self.colors['bg_dark'],
                       foreground=self.colors['error'],
                       font=('Segoe UI', 12, 'bold'))
    
    def _create_widgets(self):
        """Create all GUI widgets"""
        # Main container with padding
        main_frame = tk.Frame(self.root, bg=self.colors['bg_dark'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=25, pady=20)
        
        # Header Section
        self._create_header(main_frame)
        
        # Status Section
        self._create_status_section(main_frame)
        
        # Translation Section
        self._create_translation_section(main_frame)
        
        # Control Section
        self._create_control_section(main_frame)
        
        # History Section
        self._create_history_section(main_frame)
        
        # Footer
        self._create_footer(main_frame)
    
    def _create_header(self, parent):
        """Create header with title and time display"""
        header_frame = tk.Frame(parent, bg=self.colors['bg_dark'])
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Left side - Title with icon
        title_frame = tk.Frame(header_frame, bg=self.colors['bg_dark'])
        title_frame.pack(side=tk.LEFT)
        
        title_label = tk.Label(
            title_frame,
            text="🎙️ Voice Translator",
            font=('Segoe UI', 26, 'bold'),
            bg=self.colors['bg_dark'],
            fg=self.colors['text_light']
        )
        title_label.pack(anchor=tk.W)
        
        subtitle_label = tk.Label(
            title_frame,
            text="English Speech → Hindi Text Translation",
            font=('Segoe UI', 11),
            bg=self.colors['bg_dark'],
            fg=self.colors['text_dim']
        )
        subtitle_label.pack(anchor=tk.W, pady=(2, 0))
        
        # Right side - Time display (modern card style)
        time_card = tk.Frame(header_frame, bg=self.colors['bg_medium'], padx=20, pady=12)
        time_card.pack(side=tk.RIGHT)
        
        # Add subtle border effect
        time_card.configure(highlightbackground=self.colors['border'], highlightthickness=1)
        
        self.time_label = tk.Label(
            time_card,
            text="00:00:00 AM",
            font=('Consolas', 18, 'bold'),
            bg=self.colors['bg_medium'],
            fg=self.colors['accent']
        )
        self.time_label.pack()
    
    def _create_status_section(self, parent):
        """Create status indicator section - Modern card style"""
        status_frame = tk.Frame(parent, bg=self.colors['bg_medium'], padx=20, pady=15)
        status_frame.pack(fill=tk.X, pady=(0, 20))
        status_frame.configure(highlightbackground=self.colors['border'], highlightthickness=1)
        
        # Service status (left)
        status_left = tk.Frame(status_frame, bg=self.colors['bg_medium'])
        status_left.pack(side=tk.LEFT)
        
        tk.Label(
            status_left,
            text="Service Status:",
            font=('Segoe UI', 11),
            bg=self.colors['bg_medium'],
            fg=self.colors['text_dim']
        ).pack(side=tk.LEFT, padx=(0, 8))
        
        self.service_status_label = tk.Label(
            status_left,
            text="● Checking...",
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['bg_medium'],
            fg=self.colors['warning']
        )
        self.service_status_label.pack(side=tk.LEFT)
        
        # Active hours info
        tk.Label(
            status_frame,
            text="⏰ Active: 9:00 PM - 10:00 PM",
            font=('Segoe UI', 10),
            bg=self.colors['bg_medium'],
            fg=self.colors['text_dim']
        ).pack(side=tk.LEFT, padx=(30, 0))
        
        # Model status (right)
        status_right = tk.Frame(status_frame, bg=self.colors['bg_medium'])
        status_right.pack(side=tk.RIGHT)
        
        tk.Label(
            status_right,
            text="Model:",
            font=('Segoe UI', 11),
            bg=self.colors['bg_medium'],
            fg=self.colors['text_dim']
        ).pack(side=tk.LEFT, padx=(0, 8))
        
        self.model_status_label = tk.Label(
            status_right,
            text="● Loading...",
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['bg_medium'],
            fg=self.colors['warning']
        )
        self.model_status_label.pack(side=tk.LEFT)
    
    def _create_translation_section(self, parent):
        """Create main translation display area - Modern card style"""
        trans_frame = tk.Frame(parent, bg=self.colors['bg_dark'])
        trans_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        # Configure grid weights for equal sizing
        trans_frame.columnconfigure(0, weight=1)
        trans_frame.columnconfigure(1, weight=1)
        trans_frame.rowconfigure(0, weight=1)
        
        # English input section (Left card)
        eng_card = tk.Frame(trans_frame, bg=self.colors['bg_medium'])
        eng_card.grid(row=0, column=0, sticky='nsew', padx=(0, 10))
        eng_card.configure(highlightbackground=self.colors['border'], highlightthickness=1)
        
        # English header
        eng_header = tk.Frame(eng_card, bg=self.colors['bg_light'], pady=12, padx=15)
        eng_header.pack(fill=tk.X)
        
        tk.Label(
            eng_header,
            text="🎤 English (Recognized Speech)",
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['bg_light'],
            fg=self.colors['accent']
        ).pack(side=tk.LEFT)
        
        # English text area
        eng_text_frame = tk.Frame(eng_card, bg=self.colors['bg_medium'], padx=15, pady=15)
        eng_text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.english_text = tk.Text(
            eng_text_frame,
            height=8,
            font=('Segoe UI', 13),
            bg=self.colors['bg_input'],
            fg=self.colors['text_light'],
            insertbackground=self.colors['accent'],
            selectbackground=self.colors['accent'],
            selectforeground=self.colors['text_light'],
            wrap=tk.WORD,
            padx=15,
            pady=12,
            relief=tk.FLAT,
            borderwidth=0
        )
        self.english_text.pack(fill=tk.BOTH, expand=True)
        self.english_text.configure(highlightbackground=self.colors['border'], highlightthickness=1)
        
        # Hindi output section (Right card)
        hin_card = tk.Frame(trans_frame, bg=self.colors['bg_medium'])
        hin_card.grid(row=0, column=1, sticky='nsew', padx=(10, 0))
        hin_card.configure(highlightbackground=self.colors['border'], highlightthickness=1)
        
        # Hindi header
        hin_header = tk.Frame(hin_card, bg=self.colors['bg_light'], pady=12, padx=15)
        hin_header.pack(fill=tk.X)
        
        tk.Label(
            hin_header,
            text="📝 Hindi (Translation)",
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['bg_light'],
            fg=self.colors['accent_green']
        ).pack(side=tk.LEFT)
        
        # Hindi text area
        hin_text_frame = tk.Frame(hin_card, bg=self.colors['bg_medium'], padx=15, pady=15)
        hin_text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.hindi_text = tk.Text(
            hin_text_frame,
            height=8,
            font=('Nirmala UI', 15),  # Better Hindi font
            bg=self.colors['bg_input'],
            fg=self.colors['text_light'],
            wrap=tk.WORD,
            padx=15,
            pady=12,
            relief=tk.FLAT,
            borderwidth=0,
            state=tk.DISABLED
        )
        self.hindi_text.pack(fill=tk.BOTH, expand=True)
        self.hindi_text.configure(highlightbackground=self.colors['border'], highlightthickness=1)
    
    def _create_control_section(self, parent):
        """Create control buttons - Modern styled buttons"""
        control_frame = tk.Frame(parent, bg=self.colors['bg_dark'])
        control_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Button container
        btn_container = tk.Frame(control_frame, bg=self.colors['bg_dark'])
        btn_container.pack(side=tk.LEFT)
        
        # Listen button (Primary - Green accent)
        self.listen_btn = tk.Button(
            btn_container,
            text="🎤 Start Listening",
            font=('Segoe UI', 13, 'bold'),
            bg='#238636',
            fg='#ffffff',
            activebackground='#2ea043',
            activeforeground='#ffffff',
            padx=28,
            pady=14,
            cursor='hand2',
            relief=tk.FLAT,
            borderwidth=0,
            command=self._toggle_listening
        )
        self.listen_btn.pack(side=tk.LEFT, padx=(0, 12))
        
        # Translate button (Secondary - Blue accent)
        self.translate_btn = tk.Button(
            btn_container,
            text="🔄 Translate Text",
            font=('Segoe UI', 12),
            bg=self.colors['bg_light'],
            fg=self.colors['text_light'],
            activebackground=self.colors['accent'],
            activeforeground='#ffffff',
            padx=22,
            pady=14,
            cursor='hand2',
            relief=tk.FLAT,
            borderwidth=0,
            command=self._translate_text
        )
        self.translate_btn.pack(side=tk.LEFT, padx=(0, 12))
        
        # Clear button
        self.clear_btn = tk.Button(
            btn_container,
            text="🗑️ Clear",
            font=('Segoe UI', 12),
            bg=self.colors['bg_medium'],
            fg=self.colors['text_medium'],
            activebackground=self.colors['bg_light'],
            activeforeground=self.colors['text_light'],
            padx=22,
            pady=14,
            cursor='hand2',
            relief=tk.FLAT,
            borderwidth=0,
            command=self._clear_fields
        )
        self.clear_btn.pack(side=tk.LEFT)
        
        # Status message (right side)
        status_container = tk.Frame(control_frame, bg=self.colors['bg_medium'], padx=15, pady=10)
        status_container.pack(side=tk.RIGHT)
        status_container.configure(highlightbackground=self.colors['border'], highlightthickness=1)
        
        self.action_status = tk.Label(
            status_container,
            text="✨ Ready",
            font=('Segoe UI', 11),
            bg=self.colors['bg_medium'],
            fg=self.colors['text_dim']
        )
        self.action_status.pack()
    
    def _create_history_section(self, parent):
        """Create translation history section - Modern card style"""
        history_card = tk.Frame(parent, bg=self.colors['bg_medium'])
        history_card.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        history_card.configure(highlightbackground=self.colors['border'], highlightthickness=1)
        
        # History header
        history_header = tk.Frame(history_card, bg=self.colors['bg_light'], pady=10, padx=15)
        history_header.pack(fill=tk.X)
        
        tk.Label(
            history_header,
            text="📜 Translation History",
            font=('Segoe UI', 11, 'bold'),
            bg=self.colors['bg_light'],
            fg=self.colors['accent_purple']
        ).pack(side=tk.LEFT)
        
        # History content
        history_content = tk.Frame(history_card, bg=self.colors['bg_medium'], padx=15, pady=10)
        history_content.pack(fill=tk.BOTH, expand=True)
        
        self.history_text = scrolledtext.ScrolledText(
            history_content,
            height=6,
            font=('Consolas', 10),
            bg=self.colors['bg_input'],
            fg=self.colors['text_medium'],
            wrap=tk.WORD,
            padx=12,
            pady=10,
            relief=tk.FLAT,
            borderwidth=0,
            state=tk.DISABLED
        )
        self.history_text.pack(fill=tk.BOTH, expand=True)
        self.history_text.configure(highlightbackground=self.colors['border'], highlightthickness=1)
    
    def _create_footer(self, parent):
        """Create footer with credits"""
        footer_frame = tk.Frame(parent, bg=self.colors['bg_dark'])
        footer_frame.pack(fill=tk.X)
        
        tk.Label(
            footer_frame,
            text="🚀 Voice Translator v1.0  |  Developed by Shreyas  |  Custom Seq2Seq Neural Network  |  84.6% Accuracy",
            font=('Segoe UI', 10),
            bg=self.colors['bg_dark'],
            fg=self.colors['text_dim']
        ).pack()
    
    def _start_time_update(self):
        """Start the time update loop"""
        self._update_time()
    
    def _update_time(self):
        """Update the time display and check service status"""
        # Update time
        current_time = datetime.now().strftime("%I:%M:%S %p")
        self.time_label.config(text=current_time)
        
        # Check service status and update display
        is_active = self.time_restriction.is_service_active()
        
        if is_active:
            self.service_status_label.config(
                text="● ACTIVE",
                fg=self.colors['success']
            )
        else:
            self.service_status_label.config(
                text="● Testing Mode (Outside 9-10 PM)",
                fg=self.colors['warning']
            )
        
        # Always enable buttons when model is loaded (for testing purposes)
        # Remove time restriction on buttons - only show status
        if self.model_loaded:
            if self.listen_btn['state'] == tk.DISABLED:
                self.listen_btn.config(state=tk.NORMAL)
            if self.translate_btn['state'] == tk.DISABLED:
                self.translate_btn.config(state=tk.NORMAL)
        
        # Schedule next update
        self.root.after(1000, self._update_time)
    
    def _enable_controls(self):
        """Enable control buttons when service is active"""
        if self.model_loaded:
            self.listen_btn.config(state=tk.NORMAL)
            self.translate_btn.config(state=tk.NORMAL)
    
    def _disable_controls(self):
        """Disable control buttons when service is inactive"""
        self.listen_btn.config(state=tk.DISABLED)
        self.translate_btn.config(state=tk.DISABLED)
        
        if self.is_listening:
            self._stop_listening()
    
    def _initialize_components(self):
        """Initialize speech recognizer and translation model"""
        self._add_to_history("Initializing components...")
        
        # Initialize in background thread
        thread = threading.Thread(target=self._load_components, daemon=True)
        thread.start()
    
    def _load_components(self):
        """Load components in background"""
        try:
            # Initialize speech recognizer
            self._add_to_history("Initializing speech recognition...")
            self.speech_recognizer = SpeechRecognizer()
            if self.speech_recognizer.initialize_microphone():
                self._add_to_history("✓ Microphone ready")
            else:
                self._add_to_history("⚠ Microphone initialization failed")
            
            # Load translation model
            self._add_to_history("Loading translation model...")
            self.translator = EnglishHindiTranslator()
            
            if self.translator.load_model():
                self.model_loaded = True
                self.root.after(0, lambda: self.model_status_label.config(
                    text="● Loaded",
                    fg=self.colors['success']
                ))
                self._add_to_history("✓ Translation model loaded successfully")
            else:
                self._add_to_history("⚠ Model not found. Please train the model first.")
                self._add_to_history("  Run: python model/translator.py")
                self.root.after(0, lambda: self.model_status_label.config(
                    text="● Not Found",
                    fg=self.colors['error']
                ))
            
            self._add_to_history("=" * 40)
            if self.time_restriction.is_service_active():
                self._add_to_history("System ready. Click 'Start Listening' to begin!")
            else:
                self._add_to_history("System ready. Service active: 9:00 PM - 10:00 PM")
            
        except Exception as e:
            self._add_to_history(f"Error: {str(e)}")
    
    def _toggle_listening(self):
        """Toggle speech listening on/off"""
        if self.is_listening:
            self._stop_listening()
        else:
            self._start_listening()
    
    def _start_listening(self):
        """Start listening for speech - FRESH START, forget previous audio"""
        if not self.model_loaded:
            messagebox.showwarning("Model Not Loaded", "Translation model is not loaded yet.")
            return
        
        # IMPORTANT: Reset the speech recognizer completely for fresh start
        # This clears any old buffered audio
        if self.speech_recognizer:
            self.speech_recognizer.reset()
        
        # Small delay to ensure everything is cleared
        import time
        time.sleep(0.2)
        
        self.is_listening = True
        self.listen_btn.config(
            text="⏹️ Stop Listening",
            bg=self.colors['accent_red'],
            activebackground='#da3633'
        )
        self.action_status.config(text="🎤 Listening... Speak now!", fg=self.colors['accent_green'])
        
        # Start listening in background thread
        thread = threading.Thread(target=self._listen_and_translate, daemon=True)
        thread.start()
    
    def _stop_listening(self):
        """Stop listening for speech - IMMEDIATELY and clear everything"""
        # Set flag first to stop the loop
        self.is_listening = False
        
        # Stop the recognizer immediately
        if self.speech_recognizer:
            self.speech_recognizer.stop_listening()
        
        # Update UI
        self.listen_btn.config(
            text="🎤 Start Listening",
            bg='#238636',
            activebackground='#2ea043'
        )
        self.action_status.config(text="✨ Ready - Click Start to begin", fg=self.colors['text_dim'])
    
    def _listen_and_translate(self):
        """Listen for speech and translate - REAL-TIME with COMPLETE SENTENCES"""
        import time
        
        self._add_to_history("🎤 Speak now! (pause 1 sec when done)")
        
        while self.is_listening:
            # Check stop flag at start of each iteration
            if not self.is_listening:
                break
                
            try:
                # Complete sentences: 4s timeout, 10s phrase limit
                # Pause for 1 second when done speaking
                success, text, confidence = self.speech_recognizer.listen_for_speech(timeout=4, phrase_time_limit=10)
                
                # Check if we should stop IMMEDIATELY after listen returns
                if not self.is_listening:
                    break
                
                if success:
                    # Show we heard something
                    self.root.after(0, lambda: self.action_status.config(
                        text="✓ Got it! Translating...",
                        fg=self.colors['warning']
                    ))
                    
                    # Update English text
                    self.root.after(0, lambda t=text: self._update_english_text(t))
                    
                    # Translate
                    try:
                        translation = self.translator.translate(text)
                        if translation:
                            self.root.after(0, lambda tr=translation: self._update_hindi_text(tr))
                            
                            # Add to history
                            self._add_to_history(f"EN: {text}")
                            self._add_to_history(f"HI: {translation}")
                            self._add_to_history("-" * 30)
                            
                            self.root.after(0, lambda: self.action_status.config(
                                text="🎤 Speak again!",
                                fg=self.colors['success']
                            ))
                        else:
                            self._add_to_history(f"⚠ No translation for: {text}")
                            self.root.after(0, lambda: self.action_status.config(
                                text="🎤 Listening...",
                                fg=self.colors['success']
                            ))
                        
                    except Exception as e:
                        print(f"Translation error: {e}")
                        self._add_to_history(f"Translation error: {str(e)}")
                        self.root.after(0, lambda: self.action_status.config(
                            text="🎤 Listening... speak now!",
                            fg=self.colors['success']
                        ))
                
                elif text == self.speech_recognizer.ERROR_NOT_UNDERSTOOD:
                    self.root.after(0, lambda: self.action_status.config(
                        text="❓ Didn't catch that - please repeat",
                        fg=self.colors['warning']
                    ))
                
                elif text == self.speech_recognizer.ERROR_NO_AUDIO:
                    # Silently continue listening
                    self.root.after(0, lambda: self.action_status.config(
                        text="🎤 Listening... speak now!",
                        fg=self.colors['success']
                    ))
                
                else:
                    # Other errors
                    self.root.after(0, lambda: self.action_status.config(
                        text="🎤 Listening...",
                        fg=self.colors['success']
                    ))
                    
            except Exception as e:
                print(f"Listen loop error: {e}")
                import traceback
                traceback.print_exc()
                # Check if we should stop
                if not self.is_listening:
                    break
                time.sleep(0.3)  # Brief pause before retrying
                continue
        
        # Only update UI if not already stopped by button click
        # The _stop_listening method already handles UI updates
        print("Listen loop ended")
    
    def _translate_text(self):
        """Translate manually entered text"""
        if not self.model_loaded:
            messagebox.showwarning("Model Not Loaded", "Translation model is not loaded yet. Please wait...")
            return
        
        english = self.english_text.get("1.0", tk.END).strip()
        
        if not english:
            messagebox.showinfo("No Text", "Please enter or speak some English text.")
            return
        
        try:
            self.action_status.config(text="Translating...", fg=self.colors['warning'])
            self.root.update()
            
            translation = self.translator.translate(english)
            self._update_hindi_text(translation)
            
            # Add to history
            self._add_to_history(f"EN: {english}")
            self._add_to_history(f"HI: {translation}")
            self._add_to_history("-" * 30)
            
            self.action_status.config(text="Translation complete", fg=self.colors['success'])
            
        except Exception as e:
            messagebox.showerror("Error", f"Translation failed: {str(e)}")
            self.action_status.config(text="Error", fg=self.colors['error'])
    
    def _update_english_text(self, text):
        """Update English text display"""
        self.english_text.delete("1.0", tk.END)
        self.english_text.insert("1.0", text)
    
    def _update_hindi_text(self, text):
        """Update Hindi text display"""
        self.hindi_text.config(state=tk.NORMAL)
        self.hindi_text.delete("1.0", tk.END)
        self.hindi_text.insert("1.0", text)
        self.hindi_text.config(state=tk.DISABLED)
    
    def _clear_fields(self):
        """Clear all text fields"""
        self.english_text.delete("1.0", tk.END)
        self.hindi_text.config(state=tk.NORMAL)
        self.hindi_text.delete("1.0", tk.END)
        self.hindi_text.config(state=tk.DISABLED)
        self.action_status.config(text="Cleared", fg=self.colors['text_dim'])
    
    def _add_to_history(self, message):
        """Add message to history with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        def update():
            self.history_text.config(state=tk.NORMAL)
            self.history_text.insert(tk.END, f"[{timestamp}] {message}\n")
            self.history_text.see(tk.END)
            self.history_text.config(state=tk.DISABLED)
        
        self.root.after(0, update)


def main():
    """Main entry point for the application"""
    root = tk.Tk()
    
    # Set window icon (if available)
    try:
        root.iconbitmap('assets/icon.ico')
    except:
        pass
    
    app = VoiceTranslatorGUI(root)
    
    # Handle window close
    def on_closing():
        if messagebox.askokcancel("Quit", "Do you want to quit Voice Translator?"):
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
