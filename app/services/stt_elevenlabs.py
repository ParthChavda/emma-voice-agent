import asyncio
import base64
import time
from collections.abc import AsyncIterator, Awaitable, Callable

from elevenlabs import AsyncElevenLabs
from elevenlabs.realtime.scribe import AudioFormat, CommitStrategy

from app.config import settings

_MODEL_ID = "scribe_v2_realtime"
# ElevenLabs allows 0.3-3.0s here (verified against the SDK's own validation
# range) — Deepgram's utterance_end_ms has a hard floor at 1.0s (verified
# earlier by testing directly against their API, which rejects lower values
# with HTTP 400). 0.3s is their minimum, used here to test how far latency
# can actually be pushed with this provider.
_VAD_SILENCE_THRESHOLD_SECS = 0.3


async def transcribe_stream(
    audio_iter: AsyncIterator[bytes],
    on_final: Callable[[str], Awaitable[None]],
    on_utterance_end: Callable[[], Awaitable[None]] | None = None,
    encoding: str = "mulaw",
    sample_rate: int = 8000,
    language: str = "en",
) -> None:
    """
    Stream audio bytes to ElevenLabs Scribe Realtime v2 for live transcription.

    Calls on_final(transcript) for each committed (finalized) segment, then
    on_utterance_end() immediately after — unlike Deepgram's two-stage
    is_final + separate UtteranceEnd model, ElevenLabs' VAD commit_strategy
    means a committed_transcript event already represents a complete
    utterance (the server auto-commits once vad_silence_threshold_secs of
    silence is detected), so there's nothing to separately accumulate.

    Default audio format matches Twilio Media Streams (mulaw, 8kHz, mono).
    """
    client = AsyncElevenLabs(api_key=settings.elevenlabs_api_key)
    audio_format = AudioFormat.ULAW_8000 if encoding == "mulaw" else AudioFormat.PCM_8000

    connection = await client.speech_to_text.realtime.connect({
        "model_id": _MODEL_ID,
        "audio_format": audio_format,
        "sample_rate": sample_rate,
        "commit_strategy": CommitStrategy.VAD,
        "vad_silence_threshold_secs": _VAD_SILENCE_THRESHOLD_SECS,
        "language_code": language,
    })

    closed = asyncio.Event()
    # Local to this utterance: set on the first audio chunk sent after the
    # previous commit, reset to None right when committing — gives
    # [EMMA-TIMING] a pure STT-layer latency, independent of any scheduling
    # delay in the call handler that consumes on_final/on_utterance_end.
    # Mirrors the pattern the previous Deepgram STT module used.
    turn_start: float | None = None

    def _on_committed(data: dict) -> None:
        text = (data.get("text") or "").strip()
        if text:
            asyncio.create_task(_handle_committed(text))

    async def _handle_committed(text: str) -> None:
        nonlocal turn_start
        if turn_start is not None:
            elapsed = time.perf_counter() - turn_start
            print(f'[EMMA-TIMING] STT ElevenLabs committed: {elapsed:.2f}s | text: "{text}"')
        turn_start = None
        await on_final(text)
        if on_utterance_end:
            await on_utterance_end()

    def _on_close(*_args) -> None:
        closed.set()

    def _on_quota_exceeded(data: dict) -> None:
        print(f"[stt_elevenlabs] QUOTA EXCEEDED — no more transcription will happen this call: {data}")

    def _on_rate_limited(data: dict) -> None:
        print(f"[stt_elevenlabs] rate limited: {data}")

    def _on_auth_error(data: dict) -> None:
        print(f"[stt_elevenlabs] AUTH ERROR — check ELEVENLABS_API_KEY: {data}")

    def _on_error(data: dict) -> None:
        # Catch-all: fires for every error-shaped event (including the three
        # above, which get their own clearer message first) plus anything
        # not specifically handled — transcriber_error, input_error,
        # queue_overflow, resource_exhausted, session_time_limit_exceeded,
        # chunk_size_exceeded, insufficient_audio_activity, etc.
        print(f"[stt_elevenlabs] error: {data}")

    connection.on("committed_transcript", _on_committed)
    connection.on("close", _on_close)
    connection.on("quota_exceeded", _on_quota_exceeded)
    connection.on("rate_limited", _on_rate_limited)
    connection.on("auth_error", _on_auth_error)
    connection.on("error", _on_error)

    try:
        async for chunk in audio_iter:
            if closed.is_set():
                break
            if turn_start is None:
                turn_start = time.perf_counter()
            await connection.send({"audio_base_64": base64.b64encode(chunk).decode("ascii")})
    finally:
        if not closed.is_set():
            await connection.close()
