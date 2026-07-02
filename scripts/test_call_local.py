#!/usr/bin/env python3
"""
Standalone script: simulate a full multi-turn Twilio Media Stream call against
your LOCAL server, entirely from the terminal — no phone, no ngrok, no Twilio
account needed.

Usage:
    source venv/bin/activate
    python scripts/test_call_local.py turn1.wav [turn2.wav ...]
    python scripts/test_call_local.py --port 8123 turn1.wav

Connects directly to ws://127.0.0.1:<port>/voice/stream (bypassing Twilio and
ngrok completely), sends a "start" event exactly like Twilio would, then
continuously streams audio for the whole call — real turn audio when you have
something to say, silence otherwise — exactly like a real Twilio call never
stops sending frames. (Without this, Deepgram's connection times out waiting
for audio during the gap while Emma is thinking — a real pitfall, not
hypothetical: this is the exact bug an earlier version of this script hit.)

Plays each of Emma's replies through your speakers via `afplay` before moving
to the next turn. This exercises the real STT -> LLM -> TTS -> DB pipeline in
app/core/call_handler.py, identically to a real call — the only thing it
doesn't test is the actual Twilio/PSTN telephony transport.

Record turns with Voice Memos or QuickTime (File > New Audio Recording),
export as a mono WAV, and pass the file(s) in conversation order. Or generate
repeatable no-mic test turns with macOS `say`:
    say -o turn1.wav --file-format=WAVE --data-format=LEI16@8000 \\
        "Hi, I'd like to book a routine appointment. My name is Alice Smith, date of birth 20th of May 1990."

Requires: the app server already running locally (`uvicorn app.main:app`),
Postgres + Qdrant reachable, and DEEPGRAM_API_KEY / OPENAI_API_KEY set in .env.
"""
import argparse
import asyncio
import audioop
import base64
import json
import sys
import wave
from pathlib import Path

import websockets
from websockets.exceptions import ConnectionClosed

FRAME_BYTES = 160  # 20ms of 8kHz mulaw
SILENCE_FRAME = b"\xff" * FRAME_BYTES
# Time-to-first-reply-byte is a real STT+RAG+LLM+TTS round trip and can take
# several seconds even when everything's working — be patient before Emma
# starts talking. Once frames are actually flowing (every ~20ms), a much
# shorter gap reliably means the reply is over.
REPLY_START_TIMEOUT_S = 20.0
REPLY_GAP_TIMEOUT_S = 2.0


def _load_mulaw(wav_path: str) -> bytes:
    with wave.open(wav_path, "rb") as wf:
        if wf.getframerate() != 8000 or wf.getsampwidth() != 2:
            raise ValueError(
                f"{wav_path}: expected 8kHz 16-bit mono WAV, got "
                f"{wf.getframerate()}Hz {wf.getsampwidth() * 8}-bit"
            )
        pcm = wf.readframes(wf.getnframes())
    return audioop.lin2ulaw(pcm, 2)


async def _play_mulaw(mulaw: bytes, label: str) -> None:
    pcm = audioop.ulaw2lin(mulaw, 2)
    wav_path = Path(f"call_reply_{label}.wav")
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(8000)
        wf.writeframes(pcm)

    print(f"  playing reply ({len(mulaw)} bytes)...")
    # Must not block the event loop here: the background sender_loop task
    # needs to keep streaming silence to the server while this plays, or
    # Deepgram's connection times out from inactivity during playback.
    try:
        proc = await asyncio.create_subprocess_exec("afplay", str(wav_path))
        await proc.wait()
    except FileNotFoundError:
        print(f"  could not auto-play — play it manually with: afplay {wav_path}")


async def _sender_loop(ws, audio_queue: "asyncio.Queue[bytes]", stop_event: asyncio.Event) -> None:
    """Stream audio to the server for the whole call: queued turn audio when
    available, silence otherwise. Never stops, matching real Twilio behavior —
    Deepgram's connection times out if it goes quiet for too long."""
    while not stop_event.is_set():
        try:
            frame = audio_queue.get_nowait()
        except asyncio.QueueEmpty:
            frame = SILENCE_FRAME
        try:
            await ws.send(json.dumps({
                "event": "media",
                "media": {"payload": base64.b64encode(frame).decode("ascii")},
            }))
        except ConnectionClosed:
            return
        await asyncio.sleep(0.02)


async def _queue_turn_audio(audio_queue: "asyncio.Queue[bytes]", mulaw: bytes) -> None:
    for i in range(0, len(mulaw), FRAME_BYTES):
        await audio_queue.put(mulaw[i:i + FRAME_BYTES])


async def _collect_and_play_reply(ws, label: str) -> bool:
    """Buffer Emma's reply audio until it goes quiet, then play it. Returns
    True if the call ended (server closed the socket — e.g. escalation/handoff)."""
    reply = bytearray()
    call_ended = False
    timeout = REPLY_START_TIMEOUT_S
    while True:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
        except asyncio.TimeoutError:
            break
        except ConnectionClosed:
            call_ended = True
            break
        timeout = REPLY_GAP_TIMEOUT_S  # got a message — switch to the short "still talking?" window

        msg = json.loads(raw)
        if msg.get("event") == "media":
            reply.extend(base64.b64decode(msg["media"]["payload"]))
        elif msg.get("event") == "mark":
            try:
                await ws.send(json.dumps(msg))  # echo the mark back, like Twilio does
            except ConnectionClosed:
                call_ended = True

    if reply:
        await _play_mulaw(bytes(reply), label)
    else:
        print("  (no reply audio received)")

    return call_ended


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("wav_files", nargs="+", help="WAV file(s), one per conversation turn, in order")
    parser.add_argument("--port", type=int, default=8000, help="Local server port (default 8000)")
    args = parser.parse_args()

    for wav_path in args.wav_files:
        if not Path(wav_path).exists():
            print(f"Error: file not found: {wav_path}")
            sys.exit(1)

    url = f"ws://127.0.0.1:{args.port}/voice/stream"
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps({
            "event": "start",
            "start": {"callSid": "CA_LOCAL_TEST", "streamSid": "MZ_LOCAL_TEST"},
        }))
        print(f"Connected to {url} — waiting for Emma's greeting...\n")

        audio_queue: "asyncio.Queue[bytes]" = asyncio.Queue()
        stop_event = asyncio.Event()
        sender_task = asyncio.create_task(_sender_loop(ws, audio_queue, stop_event))

        try:
            if await _collect_and_play_reply(ws, "greeting"):
                print("Call ended before any turns were sent.")
                return

            for i, wav_path in enumerate(args.wav_files, start=1):
                print(f"Turn {i}: speaking {wav_path}")
                await _queue_turn_audio(audio_queue, _load_mulaw(wav_path))

                if await _collect_and_play_reply(ws, str(i)):
                    print("\nCall ended by Emma (escalation or handoff) — stopping.")
                    return
                print()

            try:
                await ws.send(json.dumps({"event": "stop"}))
            except ConnectionClosed:
                pass
            print("Call ended.")
        finally:
            stop_event.set()
            sender_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
