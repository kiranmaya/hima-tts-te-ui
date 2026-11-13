# 🎙️ Hima TTS Nano - Telugu (తెలుగు) UI Application

A modern, high-performance Python Text-to-Speech (TTS) desktop and web application powered by [`Prahaladha/hima-tts-nano-te`](https://huggingface.co/Prahaladha/hima-tts-nano-te) (62M parameters).

Provides state-of-the-art on-device Telugu speech synthesis with built-in voices, voice cloning, audio playback, waveform visualization, and voice profile export.

---

## ✨ Features

- **Built-in Telugu Voices**:
  - 👩 **Bhavana** (భావన - Natural, expressive female voice)
  - 👩 **Lakshmi** (లక్ష్మి - Warm, clear female voice)
  - 👩 **Sravani** (శ్రావణి - Lively, melodic female voice)
  - 👨 **Srinivas** (శ్రీనివాస్ - Mature, authoritative male voice)
  - 👨 **Karthik** (కార్తీక్ - Youthful, conversational male voice)
  - 👨 **Pranay** (ప్రణయ్ - Calm, steady male voice)
- **Generative & Custom Voices**:
  - 🎲 **Random Voice**: Generates unique natural voices on the fly.
  - 🧬 **Voice Cloning**: Reference short audio clips (>= 2s) to copy voices.
  - 💾 **Voice Profile Export**: Export custom or random voices to `.safetensors` files for future reuse.
- **Modern User Interfaces**:
  - 🖥️ **Native Desktop GUI**: Built with CustomTkinter (Dark/Light themes, responsive, audio timeline scrub slider).
  - 🌐 **Web UI Companion**: Interactive Gradio browser interface for local network or remote access.
  - ⌨️ **CLI Mode**: Fast one-line command synthesis directly from the terminal.
- **Resource Management**:
  - Strict CPU utilization limiter enforcing **80% CPU processing cap** (leaves CPU headroom for other tasks).
  - CUDA GPU acceleration enabled automatically with fallback to CPU.
- **Full Audio Controls**:
  - Integrated `sounddevice` playback engine (Play, Pause, Resume, Stop, Timeline seek).
  - Export to high-fidelity 24,000 Hz `.wav` format.
  - Instant sample presets (Greetings, Coffee chats, Vizag trip, News broadcasts, Tech & AI).

---

## 🚀 Quick Start

### 1. Requirements

- Python 3.10+ (tested on Python 3.12, Windows 10/11)
- PyTorch 2.1+ (CUDA or CPU)

Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Launch Native Desktop App

```bash
python run.py
```
*(or `python desktop_app.py`)*

### 3. Launch Gradio Web UI

```bash
python run.py --web
```
Access at `http://127.0.0.1:7860` in any web browser.

### 4. CLI One-Liner Synthesis

```bash
python run.py --cli "నమస్కారం, మీరు ఎలా ఉన్నారు?" --voice Bhavana -o namaskaram.wav
```

---

## 📁 Project Structure

```
hima-tts-te-ui/
├── desktop_app.py      # CustomTkinter Desktop GUI application
├── web_app.py          # Gradio Web UI browser application
├── engine.py           # Core TTS engine, PyTorch thread capping, audio playback
├── run.py              # Unified command runner (Desktop, Web, CLI)
├── test_app.py         # Automated test suite (audio integrity, determinism, CPU caps)
├── requirements.txt    # Project dependencies
├── hima_model/         # Downloaded model weights (Prahaladha/hima-tts-nano-te)
└── outputs/            # Synthesized .wav files and exported voice profiles
```

---

## ⚙️ Model Architecture & Technical Details

- **Base Architecture**: Flow-matching autoregressive acoustic model paired with Mimi neural audio codec.
- **Model Footprint**: ~61.8M parameters (F32).
- **Sampling Rate**: 24,000 Hz.
- **Inference Speed**: ~7x to 15x faster than real-time on CUDA RTX GPUs; faster than real-time on modern CPUs.
- **License**: Other (Model weights from [Prahaladha/hima-tts-nano-te](https://huggingface.co/Prahaladha/hima-tts-nano-te)).
