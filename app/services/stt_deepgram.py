import json
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

        async with anyio.create_task_group() as tg:
            tg.start_soon(_send)
            tg.start_soon(_receive)
