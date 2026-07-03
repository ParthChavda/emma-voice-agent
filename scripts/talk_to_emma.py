#!/usr/bin/env python3
"""
Standalone script: have a live, real-time voice conversation with Emma from
your Mac's terminal — speak into your mic, hear her reply out loud, just like
a real phone call. No phone, no ngrok, no Twilio account needed.

Usage:
    source venv/bin/activate
    python scripts/talk_to_emma.py
    python scripts/talk_to_emma.py --port 8123

Connects directly to ws://127.0.0.1:<port>/voice/stream (bypassing Twilio and
ngrok completely) and exercises the real STT -> LLM -> TTS -> DB pipeline in
app/core/call_handler.py — identically to a real call, minus the actual
Twilio/PSTN telephony transport.

At each prompt, press Enter to start recording, speak, then press Enter again
to stop and send what you said. Type 'q' + Enter instead to hang up.

Requires: the app server already running locally (`uvicorn app.main:app`),
Postgres + Qdrant reachable, ELEVENLABS_API_KEY / OPENAI_API_KEY set in .env,
and microphone access granted to your terminal app (macOS will prompt for
this the first time; if you don't get a prompt, check System Settings ->
Privacy & Security -> Microphone).
"""
import argparse
import asyncio
import audioop
import base64
import json
import sys
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
import websockets
from websockets.exceptions import ConnectionClosed

SAMPLE_RATE = 8000  # matches Twilio's telephony audio format — no resampling needed
FRAME_BYTES = 160  # 20ms of 8kHz mulaw
SILENCE_FRAME = b"\xff" * FRAME_BYTES
REPLY_START_TIMEOUT_S = 20.0
REPLY_GAP_TIMEOUT_S = 2.0


def _record_until_enter_sync() -> bytes:
    """Blocking: records mic audio until Enter is pressed. Returns 16-bit PCM bytes."""
    frames: list[np.ndarray] = []

    def callback(indata, frame_count, time_info, status):
        frames.append(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", callback=callback):
        input()

    if not frames:
        return b""
    return np.concatenate(frames, axis=0).tobytes()


async def _record_turn() -> bytes:
    loop = asyncio.get_event_loop()
    print("  Recording... press Enter to stop.")
    pcm = await loop.run_in_executor(None, _record_until_enter_sync)
    return pcm


async def _prompt(text: str) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, input, text)


async def _play_mulaw(mulaw: bytes, label: str) -> None:
    pcm = audioop.ulaw2lin(mulaw, 2)
    wav_path = Path(f"call_reply_{label}.wav")
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm)

    print(f"  Emma: playing reply ({len(mulaw)} bytes)...")
    # Non-blocking: the sender_loop task must keep streaming silence to the
    # server while this plays, or the STT provider's connection times out.
    try:
        proc = await asyncio.create_subprocess_exec("afplay", str(wav_path))
        await proc.wait()
    except FileNotFoundError:
        print(f"  could not auto-play — play it manually with: afplay {wav_path}")


async def _sender_loop(ws, audio_queue: "asyncio.Queue[bytes]", stop_event: asyncio.Event) -> None:
    """Stream audio to the server for the whole call: queued turn audio when
    available, silence otherwise. Never stops, matching real Twilio behavior —
    the STT provider's connection times out if it goes quiet for too long."""
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
    parser.add_argument("--port", type=int, default=8000, help="Local server port (default 8000)")
    args = parser.parse_args()

    url = f"ws://127.0.0.1:{args.port}/voice/stream"
    # ping_interval=None disables the low-level websocket keepalive ping.
    # A real Twilio call never needs this (its own media stream keeps the
    # connection continuously busy); here, a human pacing the conversation
    # via "press Enter to speak" routinely takes longer than the library's
    # 20s default ping_interval/ping_timeout, which was silently killing the
    # connection mid-conversation (observed: CloseCode.INTERNAL_ERROR 1011,
    # "keepalive ping timeout").
    async with websockets.connect(url, ping_interval=None) as ws:
        await ws.send(json.dumps({
            "event": "start",
            "start": {"callSid": "CA_TALK_TO_EMMA", "streamSid": "MZ_TALK_TO_EMMA"},
        }))
        print(f"Connected to {url}. Call starting...\n")

        audio_queue: "asyncio.Queue[bytes]" = asyncio.Queue()
        stop_event = asyncio.Event()
        sender_task = asyncio.create_task(_sender_loop(ws, audio_queue, stop_event))

        try:
            if await _collect_and_play_reply(ws, "greeting"):
                print("Call ended before you could say anything.")
                return

            turn = 0
            while True:
                cmd = await _prompt("\nPress Enter to speak (or 'q' to hang up): ")
                if cmd.strip().lower() == "q":
                    break

                turn += 1
                pcm = await _record_turn()
                if not pcm:
                    print("  (no audio captured — try again)")
                    turn -= 1
                    continue

                mulaw = audioop.lin2ulaw(pcm, 2)
                await _queue_turn_audio(audio_queue, mulaw)

                if await _collect_and_play_reply(ws, str(turn)):
                    print("\nCall ended by Emma (escalation or handoff).")
                    return

            try:
                await ws.send(json.dumps({"event": "stop"}))
            except ConnectionClosed:
                pass
            print("Call ended.")
        except KeyboardInterrupt:
            print("\nHanging up.")
        finally:
            stop_event.set()
            sender_task.cancel()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
