import time

import httpx

from app.config import settings

_TTS_URL_TEMPLATE = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"

# Premade voice from ElevenLabs' own API docs example — warm, clear tone,
# same rationale as the Deepgram pick (aura-2-thalia-en) it replaces.
_DEFAULT_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"
# eleven_flash_v2_5: their lowest-latency model. Measured on this app's own
# network: ~0.30s steady-state time-to-first-chunk vs Deepgram's ~0.90s.
_DEFAULT_MODEL = "eleven_flash_v2_5"

_http_client: httpx.AsyncClient | None = None


def get_http_client() -> httpx.AsyncClient:
    # Reused across every synthesis call rather than opened fresh each time —
    # avoids a new socket + TLS handshake per sentence (see tts_deepgram's
    # equivalent — measured to not matter for latency there, but still the
    # right resource-hygiene default under concurrent calls).
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient()
    return _http_client


async def close_http_client() -> None:
    global _http_client
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None


def _output_format(encoding: str, sample_rate: int | None) -> str:
    """Maps this app's Deepgram-shaped (encoding, sample_rate) call
    convention onto ElevenLabs' single output_format parameter, so callers
    (call_handler.py) don't need to change how they ask for audio."""
    if encoding == "mulaw" and sample_rate == 8000:
        return "ulaw_8000"
    if encoding == "mp3":
        return "mp3_44100_128"
    raise ValueError(f"unsupported encoding/sample_rate combination: {encoding!r}, {sample_rate!r}")


async def synthesize_speech(
    text: str,
    voice_id: str = _DEFAULT_VOICE_ID,
    model: str = _DEFAULT_MODEL,
    encoding: str = "mp3",
    sample_rate: int | None = None,
    container: str | None = None,  # unused — ElevenLabs has no separate container concept, kept for call-site compatibility
) -> bytes:
    """
    Synthesize speech via ElevenLabs TTS.

    Defaults produce an MP3 suitable for saving to disk. For Twilio Media
    Streams, pass encoding="mulaw", sample_rate=8000 to get raw headerless
    mulaw bytes ready to chunk straight into media frames.
    """
    output_format = _output_format(encoding, sample_rate)
    client = get_http_client()
    response = await client.post(
        _TTS_URL_TEMPLATE.format(voice_id=voice_id),
        params={"output_format": output_format},
        headers={"xi-api-key": settings.elevenlabs_api_key, "Content-Type": "application/json"},
        json={"text": text, "model_id": model},
        timeout=30.0,
    )
    response.raise_for_status()
    return response.content


async def synthesize_speech_stream(
    text: str,
    voice_id: str = _DEFAULT_VOICE_ID,
    model: str = _DEFAULT_MODEL,
    encoding: str = "mp3",
    sample_rate: int | None = None,
    container: str | None = None,  # unused — kept for call-site compatibility
    label: str | None = None,
):
    """
    Same as synthesize_speech, but yields audio bytes as they arrive instead
    of waiting for the full response — cuts time-to-first-audio for
    latency-sensitive callers (e.g. a live phone call).

    label (e.g. "turn 3") is prefixed on the [EMMA-TIMING] line so this
    sentence's synthesis latency can be tied back to the turn it belongs to.
    """
    output_format = _output_format(encoding, sample_rate)
    start = time.perf_counter()
    first_chunk = True
    tag = f"[{label}] " if label else ""

    client = get_http_client()
    async with client.stream(
        "POST",
        _TTS_URL_TEMPLATE.format(voice_id=voice_id),
        params={"output_format": output_format},
        headers={"xi-api-key": settings.elevenlabs_api_key, "Content-Type": "application/json"},
        json={"text": text, "model_id": model},
        timeout=30.0,
    ) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            if first_chunk:
                first_chunk = False
                elapsed = time.perf_counter() - start
                print(f'[EMMA-TIMING] {tag}TTS first chunk from ElevenLabs: {elapsed:.2f}s | text: "{text}"')
            yield chunk
