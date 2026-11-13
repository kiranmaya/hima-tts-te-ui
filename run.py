"""Unified Launcher for Hima TTS Nano (Telugu).

Usage:
    python run.py                     # Launch Native Desktop GUI (CustomTkinter)
    python run.py --web               # Launch Browser Web UI (Gradio)
    python run.py --cli "నమస్కారం"     # Run one-off CLI synthesis
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hima TTS Nano (Telugu) UI & CLI Application Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--web",
        "-w",
        action="store_true",
        help="Launch the Gradio Web UI in your default web browser",
    )
    parser.add_argument(
        "--cli",
        "-c",
        type=str,
        default=None,
        help="Run CLI synthesis with the provided Telugu text string",
    )
    parser.add_argument(
        "--voice",
        "-v",
        type=str,
        default="Bhavana",
        help="Voice to use (Bhavana, Lakshmi, Sravani, Srinivas, Karthik, Pranay, or path to .wav/.safetensors)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output.wav",
        help="Output WAV filepath for CLI mode",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=None,
        help="Random seed for reproducible voice generation",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=7860,
        help="Port for Gradio Web UI server (default: 7860)",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        choices=["cuda", "cpu"],
        default=None,
        help="Compute device (cuda or cpu)",
    )

    args = parser.parse_args()

    if args.cli:
        print("Initializing Hima TTS Engine...")
        from engine import HimaTTSEngine

        engine = HimaTTSEngine(device=args.device)
        print(f"Synthesizing: {args.cli}")
        audio, gen_time = engine.synthesize(text=args.cli, voice=args.voice, seed=args.seed)
        out_path = Path(args.output).resolve()
        engine.save_audio(out_path, audio)
        duration = len(audio) / 24000
        print(f"Success! Saved {duration:.2f}s audio to {out_path} in {gen_time:.2f}s")
        return

    if args.web:
        print("Launching Gradio Web UI on http://127.0.0.1:{}...".format(args.port))
        import web_app

        demo = web_app.build_interface()
        demo.queue().launch(server_name="127.0.0.1", server_port=args.port, inbrowser=True)
        return

    # Default: Native Desktop Application
    print("Starting Hima TTS Telugu Desktop GUI...")
    import desktop_app

    desktop_app.main()


if __name__ == "__main__":
    main()
