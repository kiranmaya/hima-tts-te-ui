"""Hima TTS Nano Engine for Telugu Text-to-Speech.

Manages model loading, CPU resource allocation (80% processing cap),
synthesis, audio playback, and voice profile exports.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("HimaTTSEngine")

# Enforce user rule: use 80% CPU processing power, not 100%
TOTAL_CORES = os.cpu_count() or 4
ALLOWED_THREADS = max(1, int(TOTAL_CORES * 0.8))
torch.set_num_threads(ALLOWED_THREADS)
logger.info("Configured PyTorch thread limit to %d threads (80%% of %d cores)", ALLOWED_THREADS, TOTAL_CORES)

# Paths
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = BASE_DIR / "hima_model"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Add model directory to sys.path so 'hima_tts' can be imported cleanly
if str(DEFAULT_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(DEFAULT_MODEL_DIR))

REPO_ID = "Prahaladha/hima-tts-nano-te"
SAMPLE_RATE = 24000


@dataclass
class VoiceInfo:
    """Information about a supported voice."""
    name: str
    gender: str
    description: str


BUILTIN_VOICES: List[VoiceInfo] = [
    VoiceInfo(name="Bhavana", gender="female", description="Natural, expressive female voice (భావన)"),
    VoiceInfo(name="Lakshmi", gender="female", description="Warm, clear female voice (లక్ష్మి)"),
    VoiceInfo(name="Sravani", gender="female", description="Lively, melodic female voice (శ్రావణి)"),
    VoiceInfo(name="Srinivas", gender="male", description="Mature, authoritative male voice (శ్రీనివాస్)"),
    VoiceInfo(name="Karthik", gender="male", description="Youthful, conversational male voice (కార్తీక్)"),
    VoiceInfo(name="Pranay", gender="male", description="Calm, steady male voice (ప్రణయ్)"),
]

SAMPLE_PROMPTS: List[Tuple[str, str]] = [
    ("Friendly Greeting", "ఏమండీ, సాయంత్రం free గా ఉన్నారా? బయటికి వెళ్ళి coffee తాగుదాం."),
    ("Weekend Travel", "Weekend Vizag trip plan చేస్తున్నాం, నువ్వు కూడా వస్తావు కదా?"),
    ("News Broadcast", "నమస్కారం, నేటి ముఖ్యాంశాలు. భారతదేశంలో సాంకేతిక రంగం అత్యంత వేగంగా అభివృద్ధి చెందుతోంది."),
    ("Meeting Reminder", "ఈ రోజు సాయంత్రం 3:30 గంటలకు టీమ్ మీటింగ్ ఉంది, దయచేసి సమయానికి హాజరుకాండి."),
    ("Technology & AI", "కృత్రిమ మేధస్సు మరియు మెషిన్ లెర్నింగ్ రంగాల్లో తెలుగు సాంకేతికత కొత్త పుంతలు తొక్కుతోంది."),
    ("Casual Inquiry", "మీరు ఎలా ఉన్నారు? ఈ రోజు వాతావరణం చాలా ఆహ్లాదకరంగా ఉంది."),
]


class AudioPlaybackController:
    """Thread-safe non-blocking audio player using sounddevice."""

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate
        self.audio_data: Optional[np.ndarray] = None
        self._start_time: float = 0.0
        self._is_playing: bool = False
        self._is_paused: bool = False
        self._pause_offset: float = 0.0
        self._lock = threading.RLock()
        self._on_finish_callback: Optional[Callable[[], None]] = None

    @property
    def is_playing(self) -> bool:
        if not self._is_playing or self._is_paused:
            return False
        if time.time() - self._start_time >= self.duration:
            self._is_playing = False
            return False
        return True

    @property
    def is_paused(self) -> bool:
        return self._is_paused

    @property
    def duration(self) -> float:
        if self.audio_data is None or len(self.audio_data) == 0:
            return 0.0
        return float(len(self.audio_data) / self.sample_rate)

    @property
    def position(self) -> float:
        if self.audio_data is None or len(self.audio_data) == 0:
            return 0.0
        if self._is_paused:
            return min(self.duration, self._pause_offset)
        if self._is_playing:
            elapsed = (time.time() - self._start_time) + self._pause_offset
            return min(self.duration, max(0.0, elapsed))
        return 0.0

    def load(self, audio: np.ndarray, on_finish: Optional[Callable[[], None]] = None) -> None:
        """Load audio array for playback."""
        with self._lock:
            self.stop()
            self.audio_data = np.asarray(audio, dtype=np.float32).reshape(-1)
            self._on_finish_callback = on_finish

    def play(self) -> None:
        """Start or resume playback."""
        with self._lock:
            if self.audio_data is None or len(self.audio_data) == 0:
                return

            if self._is_paused:
                # Resume from pause offset
                start_frame = int(self._pause_offset * self.sample_rate)
                if start_frame < len(self.audio_data):
                    sd.play(self.audio_data[start_frame:], self.sample_rate)
                self._start_time = time.time()
                self._is_paused = False
                self._is_playing = True
                return

            self.stop()
            sd.play(self.audio_data, self.sample_rate)
            self._start_time = time.time()
            self._pause_offset = 0.0
            self._is_playing = True
            self._is_paused = False

    def pause(self) -> None:
        """Pause playback."""
        with self._lock:
            if self._is_playing and not self._is_paused:
                self._pause_offset = self.position
                sd.stop()
                self._is_paused = True
                self._is_playing = False

    def stop(self) -> None:
        """Stop playback and reset position."""
        with self._lock:
            self._is_playing = False
            self._is_paused = False
            self._pause_offset = 0.0
            try:
                sd.stop()
            except Exception as exc:
                logger.debug("Stopping audio error: %s", exc)

    def seek(self, position_seconds: float) -> None:
        """Seek to specific position in seconds."""
        with self._lock:
            if self.audio_data is None:
                return
            target_pos = max(0.0, min(position_seconds, self.duration))
            was_playing = self._is_playing
            self.stop()
            self._pause_offset = target_pos
            if was_playing:
                start_frame = int(target_pos * self.sample_rate)
                if start_frame < len(self.audio_data):
                    sd.play(self.audio_data[start_frame:], self.sample_rate)
                self._start_time = time.time()
                self._is_playing = True


class HimaTTSEngine:
    """High-level manager for Hima TTS model inference and resource control."""

    def __init__(self, model_dir: Optional[str | Path] = None, device: Optional[str] = None) -> None:
        self.model_dir = Path(model_dir or DEFAULT_MODEL_DIR)
        self.preferred_device = device
        self.device = self._resolve_device(device)
        self._tts_instance: Optional[Any] = None
        self._init_lock = threading.RLock()
        self.player = AudioPlaybackController(SAMPLE_RATE)
        self.last_audio: Optional[np.ndarray] = None
        self.last_duration: float = 0.0
        self.last_generation_time: float = 0.0

    @staticmethod
    def _resolve_device(preferred: Optional[str]) -> torch.device:
        if preferred:
            return torch.device(preferred)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def ensure_model_downloaded(self, progress_callback: Optional[Callable[[str], None]] = None) -> None:
        """Check if model files exist; download automatically from Hugging Face if missing."""
        required_files = [
            self.model_dir / "model.safetensors",
            self.model_dir / "codec.safetensors",
            self.model_dir / "config.yaml",
            self.model_dir / "tokenizer.model",
        ]
        all_exist = all(p.exists() for p in required_files)
        if all_exist:
            logger.info("All model files present in %s", self.model_dir)
            return

        if progress_callback:
            progress_callback(f"Downloading model from {REPO_ID}...")
        logger.info("Downloading model files from Hugging Face: %s to %s", REPO_ID, self.model_dir)

        from huggingface_hub import snapshot_download

        self.model_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=REPO_ID, local_dir=str(self.model_dir))
        logger.info("Model download finished successfully.")

    def load_model(self, force_reload: bool = False, progress_callback: Optional[Callable[[str], None]] = None) -> None:
        """Initialize the TTS instance if not loaded."""
        with self._init_lock:
            if self._tts_instance is not None and not force_reload:
                return

            self.ensure_model_downloaded(progress_callback=progress_callback)

            if str(self.model_dir) not in sys.path:
                sys.path.insert(0, str(self.model_dir))

            if progress_callback:
                progress_callback(f"Loading neural network on {self.device.type.upper()}...")

            # Enforce 80% CPU limit
            torch.set_num_threads(ALLOWED_THREADS)

            from hima_tts import TTS

            self._tts_instance = TTS(str(self.model_dir), device=str(self.device))
            logger.info("TTS instance successfully loaded on %s", self.device)

    @property
    def available_voices(self) -> List[str]:
        if self._tts_instance is not None and hasattr(self._tts_instance, "voices"):
            return list(self._tts_instance.voices)
        return [v.name for v in BUILTIN_VOICES]

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = "Bhavana",
        seed: Optional[int] = None,
        pause: float = 0.15,
        retries: int = 2,
        clone_audio_path: Optional[str] = None,
        saved_voice_path: Optional[str] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Tuple[np.ndarray, float]:
        """Synthesize Telugu speech from text.

        Returns:
            Tuple of (audio_numpy_array, generation_time_seconds)
        """
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("Input text cannot be empty.")

        self.load_model(progress_callback=progress_callback)
        assert self._tts_instance is not None

        # Resolve voice parameter
        chosen_voice: Any = None
        if clone_audio_path and os.path.exists(clone_audio_path):
            chosen_voice = clone_audio_path
        elif saved_voice_path and os.path.exists(saved_voice_path):
            chosen_voice = saved_voice_path
        elif voice and voice != "Random Voice" and voice != "None":
            chosen_voice = voice
        else:
            chosen_voice = None  # Random voice

        if progress_callback:
            voice_label = chosen_voice if isinstance(chosen_voice, str) else "Random Voice"
            progress_callback(f"Synthesizing with {voice_label}...")

        start_time = time.perf_counter()

        # Enforce CPU allocation before run
        torch.set_num_threads(ALLOWED_THREADS)

        audio: np.ndarray = self._tts_instance.say(
            clean_text,
            voice=chosen_voice,
            seed=seed,
            pause=pause,
            retries=retries,
        )

        gen_time = time.perf_counter() - start_time
        self.last_audio = audio
        self.last_duration = float(len(audio) / SAMPLE_RATE)
        self.last_generation_time = gen_time

        logger.info(
            "Synthesized %.2fs of audio in %.2fs (RTFX: %.2fx)",
            self.last_duration,
            gen_time,
            (self.last_duration / gen_time) if gen_time > 0 else 0.0,
        )

        self.player.load(audio)
        return audio, gen_time

    def save_audio(self, filepath: str | Path, audio: Optional[np.ndarray] = None) -> Path:
        """Save synthesized audio array to WAV file."""
        target_audio = audio if audio is not None else self.last_audio
        if target_audio is None or len(target_audio) == 0:
            raise ValueError("No audio available to save.")

        dest = Path(filepath)
        dest.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(dest), target_audio, SAMPLE_RATE)
        logger.info("Saved audio file: %s", dest)
        return dest

    def export_voice_profile(self, save_path: str | Path, from_clip: Optional[str] = None, seed: Optional[int] = None) -> Path:
        """Export a reusable voice profile (.safetensors)."""
        self.load_model()
        assert self._tts_instance is not None
        dest = Path(save_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        self._tts_instance.make_voice(from_clip, path=str(dest), seed=seed)
        logger.info("Saved voice profile: %s", dest)
        return dest

    def get_hardware_info(self) -> Dict[str, Any]:
        """Return system and hardware information for UI display."""
        cuda_avail = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "N/A"
        return {
            "device": self.device.type.upper(),
            "cuda_available": cuda_avail,
            "gpu_name": gpu_name,
            "total_cpu_cores": TOTAL_CORES,
            "allowed_threads": ALLOWED_THREADS,
            "cpu_usage_cap": "80%",
            "sample_rate": SAMPLE_RATE,
        }
