"""Desktop GUI Application for Hima TTS Nano (Telugu).

Built with CustomTkinter for a sleek, modern desktop user experience.
Runs inference in background threads, supports all voice modes,
real-time audio playback controls, and voice profile exports.
"""

from __future__ import annotations

import datetime
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any, Optional

import customtkinter as ctk

from engine import (
    BUILTIN_VOICES,
    DEFAULT_MODEL_DIR,
    OUTPUTS_DIR,
    SAMPLE_PROMPTS,
    SAMPLE_RATE,
    HimaTTSEngine,
)

# App Configuration
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

TELUGU_FONTS = ("Nirmala UI", "Gautami", "Segoe UI", "Arial")


class HimaTTSDesktopApp(ctk.CTk):
    """Main CustomTkinter window for Hima TTS Telugu."""

    def __init__(self) -> None:
        super().__init__()

        self.title("Hima TTS Nano - Telugu Text-to-Speech (తెలుగు)")
        self.geometry("1100x760")
        self.minsize(980, 680)

        # Engine initialization
        self.engine = HimaTTSEngine()
        self.hw_info = self.engine.get_hardware_info()

        # State tracking
        self.is_synthesizing = False
        self.audio_data = None
        self.audio_duration = 0.0
        self.timer_id: Optional[str] = None
        self.saved_custom_voice_path: Optional[str] = None
        self.clone_audio_path: Optional[str] = None

        self._build_ui()
        self._start_playback_poller()

    def _build_ui(self) -> None:
        """Construct the UI layout."""
        # Top Header Bar
        self.header_frame = ctk.CTkFrame(self, corner_radius=10, fg_color=("gray85", "gray17"))
        self.header_frame.pack(fill="x", padx=16, pady=(12, 8))

        title_box = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        title_box.pack(side="left", padx=12, pady=8)

        app_title = ctk.CTkLabel(
            title_box,
            text="హిమ TTS నానో • Hima TTS Nano",
            font=ctk.CTkFont(family=TELUGU_FONTS[0], size=22, weight="bold"),
            text_color=("gray10", "#38bdf8"),
        )
        app_title.pack(anchor="w")

        subtitle = ctk.CTkLabel(
            title_box,
            text="High-Fidelity On-Device Telugu Text-to-Speech (62M Parameters)",
            font=ctk.CTkFont(size=12),
            text_color=("gray40", "gray70"),
        )
        subtitle.pack(anchor="w")

        # Hardware Status Badges
        badges_box = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        badges_box.pack(side="right", padx=12, pady=8)

        dev_badge = ctk.CTkLabel(
            badges_box,
            text=f"⚡ {self.hw_info['device']} ({self.hw_info['gpu_name'] if self.hw_info['cuda_available'] else 'CPU'})",
            fg_color=("gray75", "#1e293b"),
            corner_radius=6,
            font=ctk.CTkFont(size=11, weight="bold"),
            padx=10,
            pady=4,
        )
        dev_badge.pack(side="left", padx=4)

        cpu_badge = ctk.CTkLabel(
            badges_box,
            text=f"⚙ CPU Cap: {self.hw_info['cpu_usage_cap']} ({self.hw_info['allowed_threads']} Threads)",
            fg_color=("gray75", "#1e293b"),
            corner_radius=6,
            font=ctk.CTkFont(size=11, weight="bold"),
            padx=10,
            pady=4,
        )
        cpu_badge.pack(side="left", padx=4)

        self.theme_switch = ctk.CTkSwitch(
            badges_box,
            text="Dark Mode",
            command=self._toggle_theme,
            font=ctk.CTkFont(size=11),
        )
        self.theme_switch.select()
        self.theme_switch.pack(side="left", padx=8)

        # Main Body Splitter
        self.body_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.body_frame.pack(fill="both", expand=True, padx=16, pady=4)

        # Left Column: Voice Settings & Controls
        self._build_left_sidebar(self.body_frame)

        # Right Column: Text Input, Action & Playback
        self._build_right_content(self.body_frame)

        # Bottom Status Bar
        self.status_bar = ctk.CTkFrame(self, height=28, fg_color=("gray90", "gray14"))
        self.status_bar.pack(fill="x", side="bottom")

        self.status_label = ctk.CTkLabel(
            self.status_bar,
            text="సిద్ధంగా ఉంది (Ready to synthesize Telugu speech)",
            font=ctk.CTkFont(family=TELUGU_FONTS[0], size=11),
            text_color=("gray30", "gray70"),
        )
        self.status_label.pack(side="left", padx=16)

        self.model_status_label = ctk.CTkLabel(
            self.status_bar,
            text="Model: Prahaladha/hima-tts-nano-te",
            font=ctk.CTkFont(size=11),
            text_color=("gray40", "gray60"),
        )
        self.model_status_label.pack(side="right", padx=16)

    def _build_left_sidebar(self, parent: ctk.CTkFrame) -> None:
        """Left panel containing voice selection, generation parameters, and tools."""
        left_panel = ctk.CTkFrame(parent, width=340, corner_radius=10)
        left_panel.pack(side="left", fill="y", padx=(0, 10), pady=4)
        left_panel.pack_propagate(False)

        scrollable = ctk.CTkScrollableFrame(left_panel, fg_color="transparent")
        scrollable.pack(fill="both", expand=True, padx=6, pady=6)

        # Voice Mode Header
        voice_title = ctk.CTkLabel(
            scrollable,
            text="కంఠస్వరం ఎంపిక (Voice Selection)",
            font=ctk.CTkFont(family=TELUGU_FONTS[0], size=14, weight="bold"),
            anchor="w",
        )
        voice_title.pack(fill="x", padx=6, pady=(4, 6))

        # Voice Mode Tabs
        self.voice_mode_var = ctk.StringVar(value="Built-in")
        self.voice_mode_seg = ctk.CTkSegmentedButton(
            scrollable,
            values=["Built-in", "Random", "Saved Voice", "Clone Voice"],
            variable=self.voice_mode_var,
            command=self._on_voice_mode_change,
            font=ctk.CTkFont(size=11),
        )
        self.voice_mode_seg.pack(fill="x", padx=6, pady=(0, 10))

        # Built-in Voice Dropdown Frame
        self.builtin_frame = ctk.CTkFrame(scrollable, fg_color="transparent")
        self.builtin_frame.pack(fill="x", padx=6, pady=2)

        builtin_label = ctk.CTkLabel(self.builtin_frame, text="Built-in Voice:", font=ctk.CTkFont(size=12))
        builtin_label.pack(anchor="w")

        voice_options = [f"{v.name} ({v.gender.capitalize()})" for v in BUILTIN_VOICES]
        self.builtin_voice_var = ctk.StringVar(value=voice_options[0])
        self.voice_dropdown = ctk.CTkOptionMenu(
            self.builtin_frame,
            values=voice_options,
            variable=self.builtin_voice_var,
            command=self._on_builtin_voice_select,
            font=ctk.CTkFont(size=12),
        )
        self.voice_dropdown.pack(fill="x", pady=(2, 6))

        self.voice_desc_label = ctk.CTkLabel(
            self.builtin_frame,
            text=BUILTIN_VOICES[0].description,
            font=ctk.CTkFont(family=TELUGU_FONTS[0], size=11),
            text_color=("gray40", "gray70"),
            anchor="w",
            wraplength=300,
        )
        self.voice_desc_label.pack(fill="x", pady=(0, 6))

        # Custom File Frame (For Saved or Cloned Voice)
        self.custom_file_frame = ctk.CTkFrame(scrollable, fg_color=("gray85", "gray20"), corner_radius=8)
        # Hidden by default until mode changes

        self.custom_file_label = ctk.CTkLabel(
            self.custom_file_frame,
            text="Select Voice File:",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.custom_file_label.pack(anchor="w", padx=8, pady=(6, 2))

        file_pick_row = ctk.CTkFrame(self.custom_file_frame, fg_color="transparent")
        file_pick_row.pack(fill="x", padx=8, pady=(0, 8))

        self.custom_file_entry = ctk.CTkEntry(
            file_pick_row,
            placeholder_text="No file selected...",
            font=ctk.CTkFont(size=11),
        )
        self.custom_file_entry.pack(side="left", fill="x", expand=True, padx=(0, 4))

        self.browse_btn = ctk.CTkButton(
            file_pick_row,
            text="Browse",
            width=70,
            command=self._browse_voice_file,
            font=ctk.CTkFont(size=11),
        )
        self.browse_btn.pack(side="right")

        # Separator
        sep1 = ctk.CTkProgressBar(scrollable, height=2)
        sep1.set(1.0)
        sep1.pack(fill="x", padx=6, pady=12)

        # Fine-tuning Parameters
        tune_title = ctk.CTkLabel(
            scrollable,
            text="సర్దుబాట్లు (Audio Settings)",
            font=ctk.CTkFont(family=TELUGU_FONTS[0], size=14, weight="bold"),
            anchor="w",
        )
        tune_title.pack(fill="x", padx=6, pady=(0, 6))

        # Seed Setting
        seed_box = ctk.CTkFrame(scrollable, fg_color="transparent")
        seed_box.pack(fill="x", padx=6, pady=3)

        seed_label = ctk.CTkLabel(seed_box, text="Random Seed:", font=ctk.CTkFont(size=12))
        seed_label.pack(side="left")

        self.randomize_seed_var = ctk.BooleanVar(value=True)
        self.randomize_seed_cb = ctk.CTkCheckBox(
            seed_box,
            text="Randomize",
            variable=self.randomize_seed_var,
            command=self._on_randomize_seed_toggle,
            font=ctk.CTkFont(size=11),
        )
        self.randomize_seed_cb.pack(side="right")

        self.seed_entry = ctk.CTkEntry(
            scrollable,
            placeholder_text="e.g. 42",
            font=ctk.CTkFont(size=12),
        )
        self.seed_entry.pack(fill="x", padx=6, pady=(2, 8))
        self.seed_entry.insert(0, "42")
        self.seed_entry.configure(state="disabled")

        # Sentence Pause Setting
        pause_head = ctk.CTkFrame(scrollable, fg_color="transparent")
        pause_head.pack(fill="x", padx=6, pady=(4, 0))

        ctk.CTkLabel(pause_head, text="Sentence Pause:", font=ctk.CTkFont(size=12)).pack(side="left")
        self.pause_val_label = ctk.CTkLabel(pause_head, text="0.15s", font=ctk.CTkFont(size=12, weight="bold"))
        self.pause_val_label.pack(side="right")

        self.pause_slider = ctk.CTkSlider(
            scrollable,
            from_=0.05,
            to=0.50,
            number_of_steps=45,
            command=lambda v: self.pause_val_label.configure(text=f"{v:.2f}s"),
        )
        self.pause_slider.set(0.15)
        self.pause_slider.pack(fill="x", padx=6, pady=(2, 8))

        # Retries Setting
        retry_head = ctk.CTkFrame(scrollable, fg_color="transparent")
        retry_head.pack(fill="x", padx=6, pady=(4, 0))

        ctk.CTkLabel(retry_head, text="Audio Retries Guard:", font=ctk.CTkFont(size=12)).pack(side="left")
        self.retry_val_label = ctk.CTkLabel(retry_head, text="2", font=ctk.CTkFont(size=12, weight="bold"))
        self.retry_val_label.pack(side="right")

        self.retry_slider = ctk.CTkSlider(
            scrollable,
            from_=0,
            to=5,
            number_of_steps=5,
            command=lambda v: self.retry_val_label.configure(text=f"{int(v)}"),
        )
        self.retry_slider.set(2)
        self.retry_slider.pack(fill="x", padx=6, pady=(2, 10))

        # Utilities / Advanced
        sep2 = ctk.CTkProgressBar(scrollable, height=2)
        sep2.set(1.0)
        sep2.pack(fill="x", padx=6, pady=10)

        ctk.CTkLabel(
            scrollable,
            text="టూల్స్ (Tools & Export)",
            font=ctk.CTkFont(family=TELUGU_FONTS[0], size=13, weight="bold"),
            anchor="w",
        ).pack(fill="x", padx=6, pady=(0, 6))

        self.export_voice_btn = ctk.CTkButton(
            scrollable,
            text="💾 Save Voice as .safetensors",
            fg_color=("gray75", "gray25"),
            text_color=("black", "white"),
            command=self._on_export_voice_profile,
            font=ctk.CTkFont(size=11),
        )
        self.export_voice_btn.pack(fill="x", padx=6, pady=3)

        self.open_outputs_btn = ctk.CTkButton(
            scrollable,
            text="📁 Open Outputs Folder",
            fg_color=("gray75", "gray25"),
            text_color=("black", "white"),
            command=self._open_outputs_folder,
            font=ctk.CTkFont(size=11),
        )
        self.open_outputs_btn.pack(fill="x", padx=6, pady=3)

        self.launch_web_btn = ctk.CTkButton(
            scrollable,
            text="🌐 Launch Browser Web UI",
            fg_color="#0284c7",
            hover_color="#0369a1",
            command=self._launch_web_ui,
            font=ctk.CTkFont(size=11, weight="bold"),
        )
        self.launch_web_btn.pack(fill="x", padx=6, pady=(8, 4))

    def _build_right_content(self, parent: ctk.CTkFrame) -> None:
        """Right panel containing text input, quick sample presets, and audio player."""
        right_panel = ctk.CTkFrame(parent, corner_radius=10)
        right_panel.pack(side="right", fill="both", expand=True, pady=4)

        # Quick Preset Chips
        presets_bar = ctk.CTkFrame(right_panel, fg_color="transparent")
        presets_bar.pack(fill="x", padx=12, pady=(10, 4))

        ctk.CTkLabel(
            presets_bar,
            text="శాంపిల్ టెక్స్ట్ (Presets):",
            font=ctk.CTkFont(family=TELUGU_FONTS[0], size=12, weight="bold"),
        ).pack(side="left", padx=(0, 6))

        presets_scroll = ctk.CTkScrollableFrame(presets_bar, height=36, orientation="horizontal", fg_color="transparent")
        presets_scroll.pack(side="left", fill="x", expand=True)

        for label, prompt in SAMPLE_PROMPTS:
            btn = ctk.CTkButton(
                presets_scroll,
                text=label,
                height=26,
                font=ctk.CTkFont(size=11),
                fg_color=("gray80", "gray25"),
                text_color=("gray10", "gray90"),
                command=lambda p=prompt: self._load_preset(p),
            )
            btn.pack(side="left", padx=3)

        # Telugu Text Input Frame
        text_header = ctk.CTkFrame(right_panel, fg_color="transparent")
        text_header.pack(fill="x", padx=12, pady=(6, 2))

        ctk.CTkLabel(
            text_header,
            text="తెలుగు టెక్స్ట్ నమోదు చేయండి (Enter Telugu Text):",
            font=ctk.CTkFont(family=TELUGU_FONTS[0], size=13, weight="bold"),
        ).pack(side="left")

        self.char_count_label = ctk.CTkLabel(
            text_header,
            text="0 characters • 0 words",
            font=ctk.CTkFont(size=11),
            text_color=("gray40", "gray60"),
        )
        self.char_count_label.pack(side="right")

        # Multi-line Textbox
        self.textbox = ctk.CTkTextbox(
            right_panel,
            font=ctk.CTkFont(family=TELUGU_FONTS[0], size=15),
            wrap="word",
            corner_radius=8,
        )
        self.textbox.pack(fill="both", expand=True, padx=12, pady=4)
        self.textbox.insert("1.0", SAMPLE_PROMPTS[0][1])
        self.textbox.bind("<KeyRelease>", self._update_char_count)
        self._update_char_count()

        # Text Actions Row
        text_actions = ctk.CTkFrame(right_panel, fg_color="transparent")
        text_actions.pack(fill="x", padx=12, pady=(0, 8))

        clear_btn = ctk.CTkButton(
            text_actions,
            text="Clear",
            width=65,
            height=24,
            fg_color=("gray80", "gray30"),
            command=self._clear_text,
            font=ctk.CTkFont(size=11),
        )
        clear_btn.pack(side="left", padx=(0, 4))

        paste_btn = ctk.CTkButton(
            text_actions,
            text="Paste Clipboard",
            width=100,
            height=24,
            fg_color=("gray80", "gray30"),
            command=self._paste_clipboard,
            font=ctk.CTkFont(size=11),
        )
        paste_btn.pack(side="left")

        # Synthesis Action Button & Progress
        action_box = ctk.CTkFrame(right_panel, fg_color="transparent")
        action_box.pack(fill="x", padx=12, pady=4)

        self.generate_btn = ctk.CTkButton(
            action_box,
            text="🎙️ Generate Speech (స్వర ఉత్పత్తి చేయండి)",
            height=46,
            font=ctk.CTkFont(family=TELUGU_FONTS[0], size=15, weight="bold"),
            fg_color="#10b981",
            hover_color="#059669",
            command=self._on_generate_clicked,
        )
        self.generate_btn.pack(fill="x", pady=(2, 6))

        self.progress_bar = ctk.CTkProgressBar(action_box, height=8)
        self.progress_bar.set(0.0)
        self.progress_bar.pack(fill="x", pady=(0, 4))

        # Audio Player Controls Card
        self.player_card = ctk.CTkFrame(right_panel, corner_radius=10, fg_color=("gray85", "gray18"))
        self.player_card.pack(fill="x", padx=12, pady=(4, 12))

        player_top = ctk.CTkFrame(self.player_card, fg_color="transparent")
        player_top.pack(fill="x", padx=12, pady=(8, 2))

        self.audio_info_label = ctk.CTkLabel(
            player_top,
            text="ఆడియో సిద్ధంగా లేదు (No audio generated yet)",
            font=ctk.CTkFont(family=TELUGU_FONTS[0], size=12, weight="bold"),
        )
        self.audio_info_label.pack(side="left")

        self.time_label = ctk.CTkLabel(
            player_top,
            text="00:00 / 00:00",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.time_label.pack(side="right")

        # Playback Progress Slider
        self.playback_slider = ctk.CTkSlider(
            self.player_card,
            from_=0.0,
            to=1.0,
            command=self._on_seek,
            height=14,
        )
        self.playback_slider.set(0.0)
        self.playback_slider.pack(fill="x", padx=12, pady=4)

        # Control Buttons
        ctrl_bar = ctk.CTkFrame(self.player_card, fg_color="transparent")
        ctrl_bar.pack(fill="x", padx=12, pady=(4, 10))

        self.play_btn = ctk.CTkButton(
            ctrl_bar,
            text="▶ Play",
            width=80,
            command=self._on_play,
            state="disabled",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.play_btn.pack(side="left", padx=(0, 6))

        self.pause_btn = ctk.CTkButton(
            ctrl_bar,
            text="⏸ Pause",
            width=80,
            fg_color=("gray75", "gray30"),
            command=self._on_pause,
            state="disabled",
            font=ctk.CTkFont(size=12),
        )
        self.pause_btn.pack(side="left", padx=6)

        self.stop_btn = ctk.CTkButton(
            ctrl_bar,
            text="⏹ Stop",
            width=80,
            fg_color=("gray75", "gray30"),
            command=self._on_stop,
            state="disabled",
            font=ctk.CTkFont(size=12),
        )
        self.stop_btn.pack(side="left", padx=6)

        self.save_wav_btn = ctk.CTkButton(
            ctrl_bar,
            text="💾 Save WAV File",
            fg_color="#3b82f6",
            hover_color="#2563eb",
            command=self._on_save_wav,
            state="disabled",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.save_wav_btn.pack(side="right")

    # Event Handlers & Logics
    def _toggle_theme(self) -> None:
        if self.theme_switch.get() == 1:
            ctk.set_appearance_mode("Dark")
        else:
            ctk.set_appearance_mode("Light")

    def _on_voice_mode_change(self, mode: str) -> None:
        if mode == "Built-in":
            self.builtin_frame.pack(fill="x", padx=6, pady=2)
            self.custom_file_frame.pack_forget()
        elif mode == "Random":
            self.builtin_frame.pack_forget()
            self.custom_file_frame.pack_forget()
        elif mode == "Saved Voice":
            self.builtin_frame.pack_forget()
            self.custom_file_label.configure(text="Select Saved Voice (.safetensors):")
            self.custom_file_frame.pack(fill="x", padx=6, pady=4)
        elif mode == "Clone Voice":
            self.builtin_frame.pack_forget()
            self.custom_file_label.configure(text="Select Reference Audio (.wav/.flac):")
            self.custom_file_frame.pack(fill="x", padx=6, pady=4)

    def _on_builtin_voice_select(self, chosen: str) -> None:
        name = chosen.split()[0]
        for v in BUILTIN_VOICES:
            if v.name == name:
                self.voice_desc_label.configure(text=v.description)
                break

    def _browse_voice_file(self) -> None:
        mode = self.voice_mode_var.get()
        if mode == "Saved Voice":
            path = filedialog.askopenfilename(
                title="Select Saved Voice Safetensors",
                filetypes=[("Safetensors Voice", "*.safetensors"), ("All Files", "*.*")],
            )
            if path:
                self.saved_custom_voice_path = path
                self.custom_file_entry.delete(0, "end")
                self.custom_file_entry.insert(0, Path(path).name)
        elif mode == "Clone Voice":
            path = filedialog.askopenfilename(
                title="Select Reference Audio (min 2s clear speech)",
                filetypes=[("Audio Files", "*.wav;*.flac;*.mp3;*.ogg"), ("All Files", "*.*")],
            )
            if path:
                self.clone_audio_path = path
                self.custom_file_entry.delete(0, "end")
                self.custom_file_entry.insert(0, Path(path).name)

    def _on_randomize_seed_toggle(self) -> None:
        if self.randomize_seed_var.get():
            self.seed_entry.configure(state="disabled")
        else:
            self.seed_entry.configure(state="normal")

    def _load_preset(self, text: str) -> None:
        self.textbox.delete("1.0", "end")
        self.textbox.insert("1.0", text)
        self._update_char_count()

    def _clear_text(self) -> None:
        self.textbox.delete("1.0", "end")
        self._update_char_count()

    def _paste_clipboard(self) -> None:
        try:
            content = self.clipboard_get()
            self.textbox.insert("insert", content)
            self._update_char_count()
        except Exception as e:
            messagebox.showwarning("Clipboard", f"Could not paste clipboard content: {e}")

    def _update_char_count(self, event: Any = None) -> None:
        content = self.textbox.get("1.0", "end-1c")
        chars = len(content)
        words = len(content.split())
        self.char_count_label.configure(text=f"{chars} characters • {words} words")

    def _on_generate_clicked(self) -> None:
        if self.is_synthesizing:
            return

        text = self.textbox.get("1.0", "end-1c").strip()
        if not text:
            messagebox.showwarning("ఖాళీ టెక్స్ట్ (Empty Text)", "దయచేసి తెలుగు టెక్స్ట్ నమోదు చేయండి.\n(Please enter text to synthesize.)")
            return

        mode = self.voice_mode_var.get()
        chosen_voice: Optional[str] = None
        clone_clip: Optional[str] = None
        saved_voice: Optional[str] = None

        if mode == "Built-in":
            chosen_voice = self.builtin_voice_var.get().split()[0]
        elif mode == "Random":
            chosen_voice = None
        elif mode == "Saved Voice":
            if not self.saved_custom_voice_path or not os.path.exists(self.saved_custom_voice_path):
                messagebox.showerror("Error", "Please select a valid .safetensors voice file.")
                return
            saved_voice = self.saved_custom_voice_path
        elif mode == "Clone Voice":
            if not self.clone_audio_path or not os.path.exists(self.clone_audio_path):
                messagebox.showerror("Error", "Please select a reference audio file (.wav) to clone.")
                return
            clone_clip = self.clone_audio_path

        # Seed resolution
        seed_val: Optional[int] = None
        if not self.randomize_seed_var.get():
            try:
                seed_val = int(self.seed_entry.get().strip())
            except ValueError:
                seed_val = 42

        pause_val = float(self.pause_slider.get())
        retry_val = int(self.retry_slider.get())

        # UI state during synthesis
        self.is_synthesizing = True
        self.generate_btn.configure(state="disabled", text="⏳ Synthesizing Speech... (ఉత్పత్తి అవుతోంది)")
        self.progress_bar.start()
        self.status_label.configure(text="నమూనా నడుస్తోంది... Synthesizing Telugu speech...")

        # Run synthesis in background thread
        thread = threading.Thread(
            target=self._run_synthesis_worker,
            args=(text, chosen_voice, seed_val, pause_val, retry_val, clone_clip, saved_voice),
            daemon=True,
        )
        thread.start()

    def _run_synthesis_worker(
        self,
        text: str,
        voice: Optional[str],
        seed: Optional[int],
        pause: float,
        retries: int,
        clone_clip: Optional[str],
        saved_voice: Optional[str],
    ) -> None:
        try:
            audio, gen_time = self.engine.synthesize(
                text=text,
                voice=voice,
                seed=seed,
                pause=pause,
                retries=retries,
                clone_audio_path=clone_clip,
                saved_voice_path=saved_voice,
                progress_callback=lambda msg: self.after(0, lambda: self.status_label.configure(text=msg)),
            )

            # Auto-save a copy in outputs directory
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            voice_tag = voice or "custom"
            auto_path = OUTPUTS_DIR / f"telugu_{voice_tag}_{timestamp}.wav"
            self.engine.save_audio(auto_path, audio)

            self.after(0, lambda: self._on_synthesis_success(audio, gen_time, auto_path))
        except Exception as exc:
            logger.exception("Synthesis error")
            self.after(0, lambda: self._on_synthesis_error(str(exc)))

    def _on_synthesis_success(self, audio: Any, gen_time: float, auto_saved_path: Path) -> None:
        self.is_synthesizing = False
        self.progress_bar.stop()
        self.progress_bar.set(1.0)
        self.generate_btn.configure(state="normal", text="🎙️ Generate Speech (స్వర ఉత్పత్తి చేయండి)")

        self.audio_data = audio
        self.audio_duration = self.engine.last_duration

        speedup = (self.audio_duration / gen_time) if gen_time > 0 else 0.0
        success_msg = f"పూర్తయింది! Generated {self.audio_duration:.2f}s audio in {gen_time:.2f}s ({speedup:.1f}x real-time)"
        self.status_label.configure(text=success_msg)

        self.audio_info_label.configure(
            text=f"ఆడియో సిద్ధంగా ఉంది • Duration: {self.audio_duration:.2f}s • 24,000 Hz",
        )
        self._format_time_label(0.0, self.audio_duration)

        # Enable player controls
        self.play_btn.configure(state="normal")
        self.pause_btn.configure(state="normal")
        self.stop_btn.configure(state="normal")
        self.save_wav_btn.configure(state="normal")

        # Automatically start playback
        self._on_play()

    def _on_synthesis_error(self, error_message: str) -> None:
        self.is_synthesizing = False
        self.progress_bar.stop()
        self.progress_bar.set(0.0)
        self.generate_btn.configure(state="normal", text="🎙️ Generate Speech (స్వర ఉత్పత్తి చేయండి)")
        self.status_label.configure(text="లోపం జరిగింది (Error occurred during synthesis)")
        messagebox.showerror("Synthesis Error", f"Failed to generate speech:\n\n{error_message}")

    def _on_play(self) -> None:
        if self.audio_data is not None:
            self.engine.player.play()
            self.play_btn.configure(text="▶ Playing...")

    def _on_pause(self) -> None:
        self.engine.player.pause()
        self.play_btn.configure(text="▶ Play")

    def _on_stop(self) -> None:
        self.engine.player.stop()
        self.playback_slider.set(0.0)
        self._format_time_label(0.0, self.audio_duration)
        self.play_btn.configure(text="▶ Play")

    def _on_seek(self, value: float) -> None:
        if self.audio_duration > 0:
            target_sec = value * self.audio_duration
            self.engine.player.seek(target_sec)
            self._format_time_label(target_sec, self.audio_duration)

    def _format_time_label(self, current: float, total: float) -> None:
        cur_m, cur_s = divmod(int(current), 60)
        tot_m, tot_s = divmod(int(total), 60)
        self.time_label.configure(text=f"{cur_m:02d}:{cur_s:02d} / {tot_m:02d}:{tot_s:02d}")

    def _start_playback_poller(self) -> None:
        """Poll audio player state and update slider and timer label."""
        if self.engine.player.is_playing and self.audio_duration > 0:
            pos = self.engine.player.position
            fraction = min(1.0, pos / self.audio_duration)
            self.playback_slider.set(fraction)
            self._format_time_label(pos, self.audio_duration)
        elif not self.engine.player.is_playing and self.play_btn.cget("text") != "▶ Play":
            self.play_btn.configure(text="▶ Play")

        self.after(100, self._start_playback_poller)

    def _on_save_wav(self) -> None:
        if self.audio_data is None:
            return
        dest = filedialog.asksaveasfilename(
            title="Save Synthesized Audio",
            defaultextension=".wav",
            filetypes=[("Wave Audio (*.wav)", "*.wav")],
            initialdir=str(OUTPUTS_DIR),
            initialfile=f"telugu_speech_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav",
        )
        if dest:
            self.engine.save_audio(dest)
            messagebox.showinfo("Saved", f"Audio successfully saved to:\n{dest}")

    def _on_export_voice_profile(self) -> None:
        dest = filedialog.asksaveasfilename(
            title="Export Voice Profile",
            defaultextension=".safetensors",
            filetypes=[("Voice Profile (*.safetensors)", "*.safetensors")],
            initialdir=str(OUTPUTS_DIR),
            initialfile="my_custom_telugu_voice.safetensors",
        )
        if dest:
            try:
                seed_val = int(self.seed_entry.get().strip()) if not self.randomize_seed_var.get() else None
                self.engine.export_voice_profile(dest, from_clip=self.clone_audio_path, seed=seed_val)
                messagebox.showinfo("Success", f"Voice profile saved successfully to:\n{dest}")
            except Exception as e:
                messagebox.showerror("Export Failed", f"Could not export voice profile:\n{e}")

    def _open_outputs_folder(self) -> None:
        if os.name == "nt":
            os.startfile(str(OUTPUTS_DIR))
        else:
            subprocess.Popen(["xdg-open", str(OUTPUTS_DIR)])

    def _launch_web_ui(self) -> None:
        web_script = Path(__file__).parent / "web_app.py"
        if not web_script.exists():
            messagebox.showerror("Error", "web_app.py not found.")
            return

        subprocess.Popen([sys.executable, str(web_script)])
        self.status_label.configure(text="Launching Gradio Web UI in background browser...")
        messagebox.showinfo("Web UI", "Gradio Web UI is starting in the background.\nOpen http://127.0.0.1:7860 in your browser.")


def main() -> None:
    app = HimaTTSDesktopApp()
    app.mainloop()


if __name__ == "__main__":
    main()
