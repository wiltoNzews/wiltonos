#!/usr/bin/env python3
"""
WiltonOS Voice → Crystal
Speak. It listens. It stores. It sees you.

Usage:
    python wiltonos_voice.py                    # Record and store
    python wiltonos_voice.py --duration 120     # Record for 2 minutes
    python wiltonos_voice.py --file audio.wav   # Transcribe existing file
    python wiltonos_voice.py --watch ~/voice    # Watch folder for new audio
"""
import os
import sys
import sqlite3
import hashlib
import argparse
import subprocess
import tempfile
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

# Lazy load whisper (it's heavy)
_whisper_model = None

DB_PATH = Path.home() / "crystals_unified.db"
VOICE_INBOX = Path.home() / "voice_inbox"


def get_whisper_model():
    """Load whisper model on first use."""
    global _whisper_model
    if _whisper_model is None:
        print("Loading Whisper model (first time takes a moment)...")
        from faster_whisper import WhisperModel
        # Use small model for speed, or medium for accuracy
        _whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
        print("Whisper ready.")
    return _whisper_model


def record_audio(duration: int = 60, output_path: str = None) -> str:
    """Record audio from microphone."""
    if output_path is None:
        output_path = tempfile.mktemp(suffix=".wav")

    print(f"\n🎙️  Recording for {duration} seconds... (Ctrl+C to stop early)")
    print("    Speak now.\n")

    try:
        # Use arecord with RØDECaster Pro II (card 2) or default
        subprocess.run([
            "arecord",
            "-D", "plughw:2,0",  # RØDECaster Pro II
            "-d", str(duration),
            "-f", "S16_LE",
            "-r", "16000",
            "-c", "1",
            output_path
        ], check=True)
    except KeyboardInterrupt:
        print("\n    Stopped early.")
    except subprocess.CalledProcessError as e:
        print(f"Recording error: {e}")
        return None

    return output_path


def transcribe(audio_path: str) -> str:
    """Transcribe audio file to text."""
    model = get_whisper_model()

    print("Transcribing...")
    segments, info = model.transcribe(audio_path, language="pt")  # Portuguese default

    # Combine all segments
    text = " ".join([segment.text.strip() for segment in segments])

    print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")

    return text


def store_crystal(content: str, source: str = "voice", metadata: dict = None) -> int:
    """Store transcription as crystal."""
    if not content or len(content.strip()) < 10:
        print("Content too short, not storing.")
        return None

    content_hash = hashlib.md5(content.encode()).hexdigest()

    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    # Check for duplicate
    c.execute("SELECT id FROM crystals WHERE content_hash = ?", (content_hash,))
    if c.fetchone():
        print("Duplicate content, not storing.")
        conn.close()
        return None

    # Detect mode (simple keyword check)
    content_lower = content.lower()
    wilton_words = ["mãe", "pai", "família", "juliana", "medo", "dor", "trauma", "sinto"]
    psi_words = ["sistema", "código", "glyph", "coherence", "architecture", "module"]

    wilton_count = sum(1 for w in wilton_words if w in content_lower)
    psi_count = sum(1 for w in psi_words if w in content_lower)

    if wilton_count > psi_count:
        mode = "wiltonos"
    elif psi_count > wilton_count:
        mode = "psios"
    else:
        mode = "neutral"

    # Insert
    c.execute('''
        INSERT INTO crystals (content_hash, content, source, source_file, author, created_at, mode)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        content_hash,
        content,
        source,
        metadata.get('file', 'voice_recording') if metadata else 'voice_recording',
        'wilton',
        datetime.now().isoformat(),
        mode
    ))

    conn.commit()
    crystal_id = c.lastrowid
    conn.close()

    return crystal_id


def process_audio(audio_path: str, delete_after: bool = False) -> Optional[int]:
    """Full pipeline: transcribe and store."""
    if not Path(audio_path).exists():
        print(f"File not found: {audio_path}")
        return None

    # Transcribe
    text = transcribe(audio_path)

    if not text:
        print("No speech detected.")
        return None

    print(f"\n📝 Transcription:\n{'-' * 40}")
    print(text[:500] + ("..." if len(text) > 500 else ""))
    print(f"{'-' * 40}\n")

    # Store
    crystal_id = store_crystal(text, source="voice", metadata={"file": audio_path})

    if crystal_id:
        print(f"✓ Stored as crystal #{crystal_id}")

    if delete_after and crystal_id:
        Path(audio_path).unlink()
        print(f"  Deleted: {audio_path}")

    return crystal_id


def watch_folder(folder: Path, interval: int = 5):
    """Watch folder for new audio files."""
    folder = Path(folder)
    folder.mkdir(exist_ok=True)

    processed = set()
    extensions = {'.wav', '.mp3', '.m4a', '.ogg', '.flac', '.webm'}

    print(f"👁️  Watching {folder} for audio files...")
    print("   Drop audio files here to transcribe and store.")
    print("   Press Ctrl+C to stop.\n")

    try:
        while True:
            for f in folder.iterdir():
                if f.suffix.lower() in extensions and f not in processed:
                    print(f"\n🎵 Found: {f.name}")
                    process_audio(str(f), delete_after=True)
                    processed.add(f)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped watching.")


def main():
    parser = argparse.ArgumentParser(description="WiltonOS Voice → Crystal")
    parser.add_argument("--duration", "-d", type=int, default=60,
                       help="Recording duration in seconds (default: 60)")
    parser.add_argument("--file", "-f", type=str,
                       help="Transcribe existing audio file")
    parser.add_argument("--watch", "-w", type=str,
                       help="Watch folder for new audio files")
    parser.add_argument("--language", "-l", type=str, default="pt",
                       help="Language code (default: pt)")

    args = parser.parse_args()

    if args.watch:
        watch_folder(Path(args.watch))
    elif args.file:
        process_audio(args.file)
    else:
        # Record and process
        audio_path = record_audio(args.duration)
        if audio_path and Path(audio_path).exists():
            process_audio(audio_path, delete_after=True)


if __name__ == "__main__":
    main()
