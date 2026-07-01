#!/usr/bin/env python3
"""
Standalone script: transcribe a local .wav file via Deepgram STT.

Usage:
    python scripts/test_stt.py path/to/audio.wav

The WAV file should be mono. Common formats:
  - 8kHz, 8-bit unsigned (mulaw) — Twilio Media Stream format
  - 8kHz or 16kHz, 16-bit PCM (linear16) — standard recorded speech

The script streams the file to Deepgram at real-time speed (20ms chunks)
and prints each final transcript as it arrives.

Requirements: DEEPGRAM_API_KEY must be set in .env or environment.
"""
import asyncio
import sys
import wave
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.stt_deepgram import transcribe_stream


async def _wav_chunks(path: str, chunk_ms: int = 20):
    """Yield audio chunks from a WAV file paced at real-time speed."""
    with wave.open(path, "rb") as wf:
        sample_rate = wf.getframerate()
        frames_per_chunk = int(sample_rate * chunk_ms / 1000)
        while True:
            data = wf.readframes(frames_per_chunk)
            if not data:
                break
            yield data
            await asyncio.sleep(chunk_ms / 1000)


async def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    wav_path = sys.argv[1]
    if not Path(wav_path).exists():
        print(f"Error: file not found: {wav_path}")
        sys.exit(1)

    with wave.open(wav_path, "rb") as wf:
        sample_rate = wf.getframerate()
        sampwidth = wf.getsampwidth()
        channels = wf.getnchannels()
        n_frames = wf.getnframes()
        duration_s = n_frames / sample_rate
        # WAV sampwidth 1 = 8-bit (mulaw), 2 = 16-bit PCM
        encoding = "mulaw" if sampwidth == 1 else "linear16"

    print(f"File      : {wav_path}")
    print(f"Encoding  : {encoding}  ({sampwidth * 8}-bit)")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Channels  : {channels}")
    print(f"Duration  : {duration_s:.1f}s")
    print(f"Streaming to Deepgram...\n")

    transcripts: list[str] = []

    async def on_final(text: str) -> None:
        print(f"[FINAL]         {text}")
        transcripts.append(text)

    async def on_utterance_end() -> None:
        print("[UTTERANCE END] silence detected — caller finished speaking")

    await transcribe_stream(
        audio_iter=_wav_chunks(wav_path),
        on_final=on_final,
        on_utterance_end=on_utterance_end,
        encoding=encoding,
        sample_rate=sample_rate,
    )

    print(f"\n--- Full transcript ---")
    print(" ".join(transcripts) or "(no speech detected)")


if __name__ == "__main__":
    asyncio.run(main())
