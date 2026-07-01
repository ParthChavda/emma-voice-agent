# Phase 3 — Deepgram Streaming STT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `app/services/stt_deepgram.py` — a Deepgram live transcription client that streams audio bytes over WebSocket and delivers only final transcripts + utterance-end signals to its caller.

**Architecture:** A single async function `transcribe_stream(audio_iter, on_final, on_utterance_end)` opens a Deepgram WebSocket, runs a sender coroutine (forwards audio chunks + sends CloseStream when done) and a receiver coroutine (parses Deepgram JSON events, fires callbacks for final transcripts and utterance-end) concurrently via `asyncio.gather`. The caller controls audio sourcing; stt_deepgram controls the Deepgram protocol entirely.

**Tech Stack:** `websockets==16.0` (already installed), `asyncio`, `wave` (stdlib, for test script). No new packages.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `app/services/stt_deepgram.py` | fill (replace stub) | Deepgram WebSocket client — `transcribe_stream()` |
| `app/tests/test_stt.py` | create | Unit tests with `_FakeWS` — no real Deepgram needed |
| `scripts/test_stt.py` | create | Standalone manual test — streams a local WAV to Deepgram |

`requirements.txt` — no changes. `websockets==16.0` is already present.

---

## Deepgram WebSocket Protocol (reference)

**URL:** `wss://api.deepgram.com/v1/listen?<params>`

**Auth:** `Authorization: Token <DEEPGRAM_API_KEY>` header

**Key params used:**

| Param | Value | Why |
|---|---|---|
| `model` | `nova-2` | Best accuracy/speed for telephony |
| `language` | `en-GB` | NHS GP surgery context |
| `encoding` | `mulaw` (default) | Twilio audio format |
| `sample_rate` | `8000` (default) | Twilio default |
| `channels` | `1` | Mono |
| `interim_results` | `true` | Needed so we can distinguish is_final |
| `endpointing` | `300` | 300ms silence → endpoint |
| `utterance_end_ms` | `1000` | ms after last word → UtteranceEnd event |
| `smart_format` | `true` | Punctuation + capitalisation |

**Received message types:**

```
Results  { type, is_final, channel: { alternatives: [{ transcript }] } }
UtteranceEnd  { type: "UtteranceEnd" }
```

**Closing:** Send `{"type": "CloseStream"}` as a text frame to signal end of audio. Deepgram closes the connection; `async for` exits via `ConnectionClosedOK`.

---

## Task 1: Implement `app/services/stt_deepgram.py` (TDD)

**Files:**
- Create: `app/tests/test_stt.py`
- Fill: `app/services/stt_deepgram.py`

### Step 1: Write failing tests

Create `app/tests/test_stt.py`:

```python
import json
import pytest
from unittest.mock import patch

from app.services.stt_deepgram import transcribe_stream


class _FakeWS:
    """Fake WebSocket that returns pre-canned messages and records sends."""

    def __init__(self, messages: list[str]):
        self._messages = messages
        self.sent: list = []
        self._idx = 0

    async def send(self, data) -> None:
        self.sent.append(data)

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        if self._idx >= len(self._messages):
            raise StopAsyncIteration
        msg = self._messages[self._idx]
        self._idx += 1
        return msg

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


async def _one_chunk():
    yield b"\x00" * 160  # 20ms silence at 8kHz


@pytest.mark.anyio
async def test_final_transcript_forwarded_to_on_final():
    msg = json.dumps({
        "type": "Results",
        "is_final": True,
        "channel": {"alternatives": [{"transcript": "book an appointment"}]},
    })
    received = []

    async def on_final(text: str) -> None:
        received.append(text)

    with patch("app.services.stt_deepgram.websockets.connect", return_value=_FakeWS([msg])):
        await transcribe_stream(_one_chunk(), on_final)

    assert received == ["book an appointment"]


@pytest.mark.anyio
async def test_partial_transcript_not_forwarded():
    msg = json.dumps({
        "type": "Results",
        "is_final": False,
        "channel": {"alternatives": [{"transcript": "book"}]},
    })
    received = []

    async def on_final(text: str) -> None:
        received.append(text)

    with patch("app.services.stt_deepgram.websockets.connect", return_value=_FakeWS([msg])):
        await transcribe_stream(_one_chunk(), on_final)

    assert received == []


@pytest.mark.anyio
async def test_blank_final_transcript_not_forwarded():
    msg = json.dumps({
        "type": "Results",
        "is_final": True,
        "channel": {"alternatives": [{"transcript": "   "}]},
    })
    received = []

    async def on_final(text: str) -> None:
        received.append(text)

    with patch("app.services.stt_deepgram.websockets.connect", return_value=_FakeWS([msg])):
        await transcribe_stream(_one_chunk(), on_final)

    assert received == []


@pytest.mark.anyio
async def test_utterance_end_fires_callback():
    msg = json.dumps({"type": "UtteranceEnd"})
    ended = []

    async def on_final(text: str) -> None:
        pass

    async def on_end() -> None:
        ended.append(True)

    with patch("app.services.stt_deepgram.websockets.connect", return_value=_FakeWS([msg])):
        await transcribe_stream(_one_chunk(), on_final, on_utterance_end=on_end)

    assert ended == [True]


@pytest.mark.anyio
async def test_utterance_end_skipped_when_no_callback():
    msg = json.dumps({"type": "UtteranceEnd"})

    async def on_final(text: str) -> None:
        pass

    # Must not raise even with no on_utterance_end provided
    with patch("app.services.stt_deepgram.websockets.connect", return_value=_FakeWS([msg])):
        await transcribe_stream(_one_chunk(), on_final)  # no exception


@pytest.mark.anyio
async def test_audio_chunk_and_closestream_sent():
    chunk = b"\xff" * 320
    fake_ws = _FakeWS([])

    async def one_chunk():
        yield chunk

    async def on_final(text: str) -> None:
        pass

    with patch("app.services.stt_deepgram.websockets.connect", return_value=fake_ws):
        await transcribe_stream(one_chunk(), on_final)

    assert chunk in fake_ws.sent
    close_sent = [m for m in fake_ws.sent if isinstance(m, str) and "CloseStream" in m]
    assert len(close_sent) == 1
```

### Step 2: Run tests — verify they fail

```bash
source venv/bin/activate
python -m pytest app/tests/test_stt.py -v
```

Expected: `ImportError` — `transcribe_stream` not yet defined.

### Step 3: Implement `app/services/stt_deepgram.py`

Replace the entire file:

```python
import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable

import websockets
from websockets.exceptions import ConnectionClosed

from app.config import settings

_DEEPGRAM_URL = "wss://api.deepgram.com/v1/listen"


async def transcribe_stream(
    audio_iter: AsyncIterator[bytes],
    on_final: Callable[[str], Awaitable[None]],
    on_utterance_end: Callable[[], Awaitable[None]] | None = None,
    encoding: str = "mulaw",
    sample_rate: int = 8000,
    language: str = "en-GB",
) -> None:
    """
    Stream audio bytes to Deepgram live transcription.

    Calls on_final(transcript) for each final (non-partial) transcript.
    Calls on_utterance_end() when 300ms of silence triggers an endpoint.

    Default audio format matches Twilio Media Streams (mulaw, 8kHz, mono).
    """
    params = (
        f"model=nova-2"
        f"&language={language}"
        f"&encoding={encoding}"
        f"&sample_rate={sample_rate}"
        f"&channels=1"
        f"&smart_format=true"
        f"&interim_results=true"
        f"&endpointing=300"
        f"&utterance_end_ms=1000"
    )
    url = f"{_DEEPGRAM_URL}?{params}"
    headers = {"Authorization": f"Token {settings.deepgram_api_key}"}

    async with websockets.connect(url, additional_headers=headers) as ws:

        async def _send() -> None:
            async for chunk in audio_iter:
                await ws.send(chunk)
            await ws.send(json.dumps({"type": "CloseStream"}))

        async def _receive() -> None:
            try:
                async for raw in ws:
                    if isinstance(raw, bytes):
                        continue
                    msg = json.loads(raw)
                    msg_type = msg.get("type")

                    if msg_type == "Results":
                        alternatives = msg.get("channel", {}).get("alternatives", [])
                        transcript = (
                            alternatives[0].get("transcript", "").strip()
                            if alternatives
                            else ""
                        )
                        if msg.get("is_final") and transcript:
                            await on_final(transcript)

                    elif msg_type == "UtteranceEnd" and on_utterance_end:
                        await on_utterance_end()

            except ConnectionClosed:
                pass  # normal close after CloseStream

        await asyncio.gather(_send(), _receive())
```

### Step 4: Run tests — verify they pass

```bash
python -m pytest app/tests/test_stt.py -v
```

Expected:

```
app/tests/test_stt.py::test_final_transcript_forwarded_to_on_final[asyncio] PASSED
app/tests/test_stt.py::test_partial_transcript_not_forwarded[asyncio] PASSED
app/tests/test_stt.py::test_blank_final_transcript_not_forwarded[asyncio] PASSED
app/tests/test_stt.py::test_utterance_end_fires_callback[asyncio] PASSED
app/tests/test_stt.py::test_utterance_end_skipped_when_no_callback[asyncio] PASSED
app/tests/test_stt.py::test_audio_chunk_and_closestream_sent[asyncio] PASSED
6 passed
```

Also run the full suite to confirm no regressions:

```bash
python -m pytest app/tests/ -v --ignore=app/tests/test_chat.py
```

Expected: all existing tests pass.

---

## Task 2: Create `scripts/test_stt.py` standalone test script

**Files:**
- Create: `scripts/test_stt.py`

No unit test for this file — it is itself the test (manual verification against real Deepgram).

### Step 1: Create `scripts/test_stt.py`

```python
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
```

### Step 2: Smoke-test the script with a real WAV file

You need a mono WAV file. If you don't have one, create a 5-second test clip using macOS `say`:

```bash
# Generate a test WAV (macOS)
say -o /tmp/test_speech.aiff "I need to book an appointment for next Tuesday"
ffmpeg -i /tmp/test_speech.aiff -ar 8000 -ac 1 -acodec pcm_s16le /tmp/test_speech.wav
```

Then run:

```bash
source venv/bin/activate
python scripts/test_stt.py /tmp/test_speech.wav
```

Expected output:

```
File      : /tmp/test_speech.wav
Encoding  : linear16  (16-bit)
Sample rate: 8000 Hz
Channels  : 1
Duration  : ~5.0s
Streaming to Deepgram...

[FINAL]         I need to book an appointment for next Tuesday.
[UTTERANCE END] silence detected — caller finished speaking

--- Full transcript ---
I need to book an appointment for next Tuesday.
```

If you already have a `.wav` file, skip the ffmpeg step and pass it directly.

---

## Updated `requirements.txt`

No changes needed — `websockets==16.0` is already in `requirements.txt`. The implementation uses only stdlib (`asyncio`, `json`) and the already-installed `websockets`. For reference, the key entries:

```
websockets==16.0   # Deepgram WebSocket client (already present)
```

---

## Self-Review

**Spec coverage:**
- ✅ Connect to Deepgram live transcription WebSocket using DEEPGRAM_API_KEY from config — `_DEEPGRAM_URL` + `headers = {"Authorization": f"Token {settings.deepgram_api_key}"}` in Task 1
- ✅ Accept audio chunks (bytes) — `audio_iter: AsyncIterator[bytes]` parameter
- ✅ Return final transcripts only (not partials) — `if msg.get("is_final") and transcript` gate in `_receive`
- ✅ Endpointing/silence detection — `endpointing=300&utterance_end_ms=1000` params + `UtteranceEnd` handler
- ✅ Standalone test script for a local .wav file — `scripts/test_stt.py` in Task 2
- ✅ Matches existing async style + config via `settings` — same pattern as `rag.py` / `llm_openai.py`

**Placeholder scan:** None found. All steps have complete code.

**Type consistency:**
- `transcribe_stream` signature is identical in tests and implementation:  `(audio_iter: AsyncIterator[bytes], on_final: Callable[[str], Awaitable[None]], on_utterance_end: Callable[[], Awaitable[None]] | None = None, encoding: str, sample_rate: int, language: str) -> None`
- `_FakeWS.send` receives the same types `ws.send` sends: `bytes` (audio) and `str` (CloseStream JSON)
