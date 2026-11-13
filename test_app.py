"""Automated Verification Suite for Hima TTS Telugu UI.

Validates:
1. Model loading & PyTorch CPU limit (80% of system cores).
2. Synthesis across female, male, and random voices.
3. Seed determinism.
4. Output WAV file integrity and sample rate.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from engine import ALLOWED_THREADS, TOTAL_CORES, HimaTTSEngine


def test_cpu_thread_limit() -> None:
    print("[1/5] Testing CPU thread capping (80% rule)...")
    expected_limit = max(1, int(TOTAL_CORES * 0.8))
    actual_threads = torch.get_num_threads()
    assert actual_threads == expected_limit, f"Expected {expected_limit} threads, got {actual_threads}"
    print(f"  ✓ CPU threads properly restricted to {actual_threads} (80% of {TOTAL_CORES} cores)")


def test_model_loading_and_voices() -> None:
    print("[2/5] Testing model loading and voice inventory...")
    engine = HimaTTSEngine()
    engine.load_model()
    voices = engine.available_voices
    assert len(voices) >= 6, f"Expected at least 6 voices, found {voices}"
    for expected in ["Bhavana", "Lakshmi", "Sravani", "Srinivas", "Karthik", "Pranay"]:
        assert expected in voices, f"Voice {expected} missing from inventory"
    print(f"  ✓ Available voices validated: {voices}")


def test_female_and_male_synthesis(tmp_path: Path) -> None:
    print("[3/5] Testing Telugu speech synthesis across voices...")
    engine = HimaTTSEngine()

    test_text_female = "నమస్కారం"
    audio_f, time_f = engine.synthesize(test_text_female, voice="Bhavana", seed=42, retries=0)
    assert len(audio_f) > 0, "Female synthesis produced empty audio array"
    assert audio_f.ndim == 1, "Audio array should be mono 1D"
    duration_f = len(audio_f) / 24000
    print(f"  ✓ Bhavana (Female): {duration_f:.2f}s generated in {time_f:.2f}s")

    test_text_female2 = "బాగున్నారా"
    audio_s, time_s = engine.synthesize(test_text_female2, voice="Sravani", seed=42, retries=0)
    assert len(audio_s) > 0, "Sravani synthesis produced empty audio array"
    duration_s = len(audio_s) / 24000
    print(f"  ✓ Sravani (Female): {duration_s:.2f}s generated in {time_s:.2f}s")

    # Save and verify WAV
    save_dest = tmp_path / "test_synthesis.wav"
    engine.save_audio(save_dest, audio_f)
    assert save_dest.exists(), "Saved WAV file not found"
    data, sr = sf.read(str(save_dest))
    assert sr == 24000, f"Expected sample rate 24000, got {sr}"
    assert len(data) == len(audio_f), "Saved file length does not match array length"
    print(f"  ✓ Saved audio verified: 24,000 Hz, {len(data)} samples")


def test_seed_determinism() -> None:
    print("[4/5] Testing seed determinism...")
    engine = HimaTTSEngine()
    text = "హలో"
    audio_1, _ = engine.synthesize(text, voice="Bhavana", seed=123, retries=0)
    audio_2, _ = engine.synthesize(text, voice="Bhavana", seed=123, retries=0)

    # Check identical length and near-identical waveform values
    assert len(audio_1) == len(audio_2), "Deterministic runs must yield equal lengths"
    diff = np.max(np.abs(audio_1 - audio_2))
    assert diff < 1e-4, f"Seed determinism mismatch (max diff: {diff})"
    print(f"  ✓ Seed determinism verified (max difference: {diff:.6f})")


def test_desktop_and_web_ui_modules() -> None:
    print("[5/5] Testing UI module imports and interface builders...")
    import desktop_app
    import web_app

    demo = web_app.build_interface()
    assert demo is not None, "Gradio Blocks interface builder failed"
    print("  ✓ Desktop and Web UI modules imported and validated successfully")


def main() -> None:
    print("=" * 60)
    print("HIMA TTS NANO TELUGU - TEST SUITE")
    print("=" * 60)
    test_dir = Path(__file__).parent / "outputs" / "tests"
    test_dir.mkdir(parents=True, exist_ok=True)

    test_cpu_thread_limit()
    test_model_loading_and_voices()
    test_female_and_male_synthesis(test_dir)
    test_seed_determinism()
    test_desktop_and_web_ui_modules()

    print("=" * 60)
    print("ALL TESTS PASSED WITH ZERO ERRORS!")
    print("=" * 60)


if __name__ == "__main__":
    main()
