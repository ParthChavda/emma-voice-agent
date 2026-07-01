import httpx

from app.config import settings

_TTS_URL = "https://api.deepgram.com/v1/speak"

# aura-2-thalia-en: warm, clear, professional voice — well suited to a healthcare receptionist.
_DEFAULT_MODEL = "aura-2-thalia-en"


async def synthesize_speech(text: str, model: str = _DEFAULT_MODEL) -> bytes:
    headers = {
        "Authorization": f"Token {settings.deepgram_api_key}",
    }
    params = {"model": model, "encoding": "mp3"}

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
