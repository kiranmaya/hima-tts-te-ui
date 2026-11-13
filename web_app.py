"""Web UI Application for Hima TTS Nano (Telugu).

Built with Gradio for modern browser-based synthesis and audio playback.
Supports built-in voices, custom seeds, pause duration, and audio downloads.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import numpy as np

from engine import (
    BUILTIN_VOICES,
    OUTPUTS_DIR,
    SAMPLE_PROMPTS,
    SAMPLE_RATE,
    HimaTTSEngine,
)

engine = HimaTTSEngine()
hw_info = engine.get_hardware_info()

VOICE_NAMES = [v.name for v in BUILTIN_VOICES] + ["Random Voice"]

CUSTOM_CSS = """
.gradio-container {
    max-width: 1050px !important;
    margin: auto !important;
    font-family: 'Segoe UI', 'Nirmala UI', sans-serif;
}
.header-box {
    text-align: center;
    padding: 20px;
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border-radius: 12px;
    color: white;
    margin-bottom: 20px;
}
.header-box h1 {
    margin: 0;
    font-size: 2.2rem;
    color: #38bdf8;
}
.badge {
    display: inline-block;
    background: rgba(56, 189, 248, 0.2);
    border: 1px solid #38bdf8;
    color: #bae6fd;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 0.85rem;
    margin: 4px;
}
"""


def generate_speech(
    text: str,
    voice: str,
    seed_mode: str,
    custom_seed: float,
    pause: float,
    retries: int,
) -> Tuple[Tuple[int, np.ndarray], str, Optional[str]]:
    """Synthesizes Telugu speech and returns Gradio audio tuple, status markdown, and file path."""
    clean_text = text.strip()
    if not clean_text:
        return (SAMPLE_RATE, np.zeros(0, dtype=np.float32)), "⚠️ దయచేసి తెలుగు టెక్స్ట్ నమోదు చేయండి (Please enter text).", None

    seed_val: Optional[int] = None if seed_mode == "Randomize" else int(custom_seed)
    chosen_voice = None if voice == "Random Voice" else voice

    audio, gen_time = engine.synthesize(
        text=clean_text,
        voice=chosen_voice,
        seed=seed_val,
        pause=pause,
        retries=retries,
    )

    duration = float(len(audio) / SAMPLE_RATE)
    speedup = (duration / gen_time) if gen_time > 0 else 0.0

    # Save output WAV file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    voice_tag = chosen_voice or "random"
    out_file = OUTPUTS_DIR / f"telugu_{voice_tag}_{timestamp}.wav"
    engine.save_audio(out_file, audio)

    status_md = (
        f"✅ **సక్సెస్! Synthesis Complete**\n\n"
        f"- **Voice**: `{voice}`\n"
        f"- **Audio Duration**: `{duration:.2f} seconds`\n"
        f"- **Inference Time**: `{gen_time:.2f} seconds` (`{speedup:.1f}x real-time`)\n"
        f"- **Sample Rate**: `{SAMPLE_RATE} Hz`\n"
        f"- **Saved to**: `{out_file.name}`"
    )

    return (SAMPLE_RATE, audio), status_md, str(out_file)


def build_interface() -> gr.Blocks:
    """Create Gradio Blocks interface."""
    with gr.Blocks(title="Hima TTS Nano - Telugu") as demo:
        with gr.Column(elem_classes=["header-box"]):
            gr.Markdown(
                """
                # 🎙️ హిమ TTS నానో • Hima TTS Nano (Telugu)
                ### On-Device Neural Text-to-Speech for Telugu (62M Parameters)
                """
            )
            gr.Markdown(
                f"""
                <span class="badge">⚡ Device: {hw_info['device']} ({hw_info['gpu_name'] if hw_info['cuda_available'] else 'CPU'})</span>
                <span class="badge">⚙️ CPU Thread Cap: {hw_info['cpu_usage_cap']} ({hw_info['allowed_threads']} Threads)</span>
                <span class="badge">🔊 24,000 Hz Hi-Fi Output</span>
                """
            )

        with gr.Row():
            with gr.Column(scale=6):
                text_input = gr.Textbox(
                    label="తెలుగు టెక్స్ట్ నమోదు చేయండి (Enter Telugu Text)",
                    placeholder="ఇక్కడ తెలుగు లేదా ఇంగ్లీష్ మిశ్రమ టెక్స్ట్ రాయండి...",
                    value=SAMPLE_PROMPTS[0][1],
                    lines=5,
                )

                gr.Markdown("#### 💡 Quick Examples (శాంపిల్ ప్రాంప్ట్స్):")
                with gr.Row():
                    for label, sample in SAMPLE_PROMPTS[:3]:
                        gr.Button(label, size="sm").click(
                            fn=lambda s=sample: s,
                            inputs=[],
                            outputs=[text_input],
                        )

                with gr.Row():
                    for label, sample in SAMPLE_PROMPTS[3:]:
                        gr.Button(label, size="sm").click(
                            fn=lambda s=sample: s,
                            inputs=[],
                            outputs=[text_input],
                        )

                generate_btn = gr.Button(
                    "🎙️ Generate Telugu Speech (స్వర ఉత్పత్తి చేయండి)",
                    variant="primary",
                    size="lg",
                )

            with gr.Column(scale=4):
                voice_select = gr.Dropdown(
                    choices=VOICE_NAMES,
                    value="Bhavana",
                    label="కంఠస్వరం (Voice)",
                    info="Pick a built-in regional voice or Random",
                )

                with gr.Row():
                    seed_mode = gr.Radio(
                        choices=["Randomize", "Fixed Seed"],
                        value="Randomize",
                        label="Seed Mode",
                    )
                    custom_seed = gr.Number(value=42, label="Seed Value", precision=0)

                pause_slider = gr.Slider(
                    minimum=0.05,
                    maximum=0.50,
                    value=0.15,
                    step=0.01,
                    label="Sentence Pause (సెకన్లలో విరామం)",
                )

                retry_slider = gr.Slider(
                    minimum=0,
                    maximum=5,
                    value=2,
                    step=1,
                    label="Audio Retries Guard",
                )

        gr.Markdown("---")
        with gr.Row():
            with gr.Column(scale=6):
                audio_output = gr.Audio(
                    label="🔊 ప్లేయర్ (Synthesized Audio Player)",
                    type="numpy",
                    autoplay=True,
                )
            with gr.Column(scale=4):
                status_box = gr.Markdown("సిద్ధంగా ఉంది (Ready)")
                download_file = gr.File(label="⬇️ Download WAV File")

        generate_btn.click(
            fn=generate_speech,
            inputs=[text_input, voice_select, seed_mode, custom_seed, pause_slider, retry_slider],
            outputs=[audio_output, status_box, download_file],
        )

    return demo


def main() -> None:
    demo = build_interface()
    demo.queue().launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, css=CUSTOM_CSS)


if __name__ == "__main__":
    main()
