import httpx

from app.config import settings

_TTS_URL = "https://api.deepgram.com/v1/speak"

# aura-2-thalia-en: warm, clear, professional voice — well suited to a healthcare receptionist.
_DEFAULT_MODEL = "aura-2-thalia-en"


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

    async with httpx.AsyncClient() as client:
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
):
    """
    Same as synthesize_speech, but yields audio bytes as they arrive instead of
    waiting for the full response — Deepgram streams progressively, so this cuts
    time-to-first-audio substantially for latency-sensitive callers (e.g. a live
    phone call), versus buffering the entire synthesis before sending anything.
    """
    headers, params = _build_request(model, encoding, sample_rate, container)

    async with httpx.AsyncClient() as client:
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
                yield chunk
