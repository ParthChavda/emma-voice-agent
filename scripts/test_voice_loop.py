#!/usr/bin/env python3
"""
Standalone script: full STT -> LLM -> TTS loop test, driven by a local WAV file.

Streams a WAV file through Deepgram STT to get a transcript, sends that
transcript through the same RAG + function-calling pipeline the /chat
endpoint uses, then speaks the reply back via Deepgram TTS.

Usage:
    source venv/bin/activate
    python scripts/test_voice_loop.py path/to/question.wav

Record a question with Voice Memos/QuickTime (or generate one, see below),
export/convert to WAV, and pass the path in.

Generate a quick test WAV on macOS without a mic:
    say -o /tmp/question.wav --file-format=WAVE --data-format=LEI16@8000 \\
        "Hi, I'd like to book a routine appointment. My name is Alice Smith, date of birth 1990-05-20."

Does not touch Postgres session history — this is a single-turn smoke test,
not a full /chat call. Requires DEEPGRAM_API_KEY and OPENAI_API_KEY in .env,
and Qdrant to already be seeded (see scripts/seed_qdrant_schedules.py).
"""
import asyncio
import subprocess
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.prompts import build_messages
from app.services.llm_openai import TOOLS, chat_completion
from app.services.rag import retrieve
from app.services.stt_deepgram import transcribe_stream
from app.services.tts_deepgram import synthesize_speech

OUTPUT_PATH = Path("voice_loop_reply.mp3")


async def _wav_chunks(path: str, chunk_ms: int = 20):
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
        encoding = "mulaw" if sampwidth == 1 else "linear16"

    print(f"File     : {wav_path}")
    print(f"Encoding : {encoding} ({sampwidth * 8}-bit, {sample_rate} Hz)")
    print("Step 1/3 — transcribing via Deepgram STT...")

    transcripts: list[str] = []

    async def on_final(text: str) -> None:
        transcripts.append(text)
        print(f"  [STT] {text}")

    await transcribe_stream(
        audio_iter=_wav_chunks(wav_path),
        on_final=on_final,
        encoding=encoding,
        sample_rate=sample_rate,
    )

    user_message = " ".join(transcripts)
    if not user_message:
        print("No speech detected — nothing to send to the LLM.")
        return

    print("\nStep 2/3 — sending transcript through EMMA (RAG + function calling)...")
    chunks = await retrieve(user_message, top_k=3)
    messages = build_messages(chunks, [], user_message)
    reply, intent = await chat_completion(messages, TOOLS)
    print(f"  [LLM] intent: {intent}")
    print(f"  [LLM] reply : {reply}")

    print("\nStep 3/3 — synthesizing reply via Deepgram TTS...")
    audio = await synthesize_speech(reply)
    OUTPUT_PATH.write_bytes(audio)
    print(f"  Saved {len(audio)} bytes to {OUTPUT_PATH.resolve()}")

    try:
        subprocess.run(["afplay", str(OUTPUT_PATH)], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print(f"  Could not auto-play. Play it manually with: afplay {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
