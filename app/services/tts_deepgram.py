import time

import httpx

from app.config import settings

_TTS_URL = "https://api.deepgram.com/v1/speak"

# aura-2-thalia-en: warm, clear, professional voice — well suited to a healthcare receptionist.
_DEFAULT_MODEL = "aura-2-thalia-en"

_http_client: httpx.AsyncClient | None = None


def get_http_client() -> httpx.AsyncClient:
    # Reused across every synthesis call rather than opened fresh each time —
    # avoids a new socket + TLS handshake per sentence. Measured to make no
    # difference to time-to-first-byte here (Deepgram's own synthesis time
    # dominates), so this is a resource/connection-churn cleanup, not a
    # latency fix.
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient()
    return _http_client


async def close_http_client() -> None:
    global _http_client
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None


def _build_request(model: str, encoding: str, sample_rate: int | None, container: str | None):
    headers = {
        "Authorization": f"Token {settings.deepgram_api_key}",
        "Content-Type": "application/json",
    }
    params = {"model": model, "encoding": encoding}
    if sample_rate is not None:
        params["sample_rate"] = sample_rate
    if container is not None:
        params["container"] = container
    return headers, params


async def synthesize_speech(
    text: str,
    model: str = _DEFAULT_MODEL,
    encoding: str = "mp3",
    sample_rate: int | None = None,
    container: str | None = None,
) -> bytes:
    """
    Synthesize speech via Deepgram TTS.

    Defaults produce an MP3 suitable for saving to disk. For Twilio Media
    Streams, pass encoding="mulaw", sample_rate=8000, container="none" to get
    raw headerless mulaw bytes ready to chunk straight into media frames.
    """
    headers, params = _build_request(model, encoding, sample_rate, container)

    client = get_http_client()
    response = await client.post(
        _TTS_URL,
        params=params,
        headers=headers,
        json={"text": text},
        timeout=30.0,
    )
    response.raise_for_status()
    return response.content


async def synthesize_speech_stream(
    text: str,
    model: str = _DEFAULT_MODEL,
    encoding: str = "mp3",
    sample_rate: int | None = None,
    container: str | None = None,
    label: str | None = None,
):
    """
    Same as synthesize_speech, but yields audio bytes as they arrive instead of
    waiting for the full response — Deepgram streams progressively, so this cuts
    time-to-first-audio substantially for latency-sensitive callers (e.g. a live
    phone call), versus buffering the entire synthesis before sending anything.

    label (e.g. "turn 3") is prefixed on the [EMMA-TIMING] line so this
    sentence's synthesis latency can be tied back to the turn it belongs to.
    """
    headers, params = _build_request(model, encoding, sample_rate, container)

    start = time.perf_counter()
    first_chunk = True
    tag = f"[{label}] " if label else ""

    client = get_http_client()
    async with client.stream(
        "POST",
        _TTS_URL,
        params=params,
        headers=headers,
        json={"text": text},
        timeout=30.0,
    ) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            if first_chunk:
                first_chunk = False
                elapsed = time.perf_counter() - start
                print(f'[EMMA-TIMING] {tag}TTS first chunk from Deepgram: {elapsed:.2f}s | text: "{text}"')
            yield chunk
