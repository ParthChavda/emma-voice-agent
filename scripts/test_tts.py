#!/usr/bin/env python3
"""
Standalone script: convert a sample sentence to speech via Deepgram TTS and save it as output.mp3.

Usage:
    source venv/bin/activate
    python scripts/test_tts.py ["custom text to speak"]

Requirements: DEEPGRAM_API_KEY must be set in .env or environment.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.tts_deepgram import synthesize_speech

DEFAULT_TEXT = (
    "Hello, thank you for calling Elmwood Road Surgery. "
    "This is Emma, how can I help you today?"
)


async def main() -> None:
    text = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TEXT
    output_path = Path("output.mp3")

    print(f"Text     : {text}")
    print("Synthesizing speech via Deepgram...")

    audio = await synthesize_speech(text)
    output_path.write_bytes(audio)

    print(f"Saved {len(audio)} bytes to {output_path.resolve()}")
    print(f"Play it with: afplay {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
