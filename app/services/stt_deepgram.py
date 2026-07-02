import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable

import anyio
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
        # Local to this utterance: set on the first audio chunk pulled from
        # Twilio after the previous UtteranceEnd, reset back to None there —
        # gives [EMMA-TIMING] a pure STT-layer latency, independent of any
        # scheduling delay in the call handler that consumes on_final/on_utterance_end.
        turn_start: float | None = None

        async def _send() -> None:
            nonlocal turn_start
            async for chunk in audio_iter:
                if turn_start is None:
                    turn_start = time.perf_counter()
                await ws.send(chunk)
            await ws.send(json.dumps({"type": "CloseStream"}))

        async def _receive() -> None:
            nonlocal turn_start
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
                            if turn_start is not None:
                                elapsed = time.perf_counter() - turn_start
                                print(f'[EMMA-TIMING] STT Deepgram is_final: {elapsed:.2f}s | text: "{transcript}"')
                            await on_final(transcript)

                    elif msg_type == "UtteranceEnd":
                        turn_start = None
                        if on_utterance_end:
                            await on_utterance_end()

            except ConnectionClosed:
                pass  # normal close after CloseStream

        async with anyio.create_task_group() as tg:
            tg.start_soon(_send)
            tg.start_soon(_receive)
