import pytest
from unittest.mock import AsyncMock, patch


class _FakeResponse:
    def __init__(self, content: bytes, status_code: int = 200):
        self.content = content
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeAsyncClient:
    def __init__(self, response: _FakeResponse):
        self.post = AsyncMock(return_value=response)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


@pytest.mark.anyio
async def test_synthesize_speech_returns_audio_bytes():
    fake_client = _FakeAsyncClient(_FakeResponse(b"\xff\xfbaudio-bytes"))

    with patch("app.services.tts_deepgram.httpx.AsyncClient", return_value=fake_client):
        from app.services.tts_deepgram import synthesize_speech
        audio = await synthesize_speech("Hello, this is a test.")

    assert audio == b"\xff\xfbaudio-bytes"


@pytest.mark.anyio
async def test_synthesize_speech_sends_correct_request():
    fake_client = _FakeAsyncClient(_FakeResponse(b"audio"))

    with patch("app.services.tts_deepgram.httpx.AsyncClient", return_value=fake_client):
        from app.services.tts_deepgram import synthesize_speech
        await synthesize_speech("Book an appointment")

    fake_client.post.assert_called_once()
    args, kwargs = fake_client.post.call_args
    assert args[0] == "https://api.deepgram.com/v1/speak"
    assert kwargs["json"] == {"text": "Book an appointment"}
    assert kwargs["params"] == {"model": "aura-2-thalia-en", "encoding": "mp3"}
    assert kwargs["headers"]["Authorization"].startswith("Token ")


@pytest.mark.anyio
async def test_synthesize_speech_uses_custom_model_when_given():
    fake_client = _FakeAsyncClient(_FakeResponse(b"audio"))

    with patch("app.services.tts_deepgram.httpx.AsyncClient", return_value=fake_client):
        from app.services.tts_deepgram import synthesize_speech
        await synthesize_speech("Hello", model="aura-2-luna-en")

    _, kwargs = fake_client.post.call_args
    assert kwargs["params"]["model"] == "aura-2-luna-en"


@pytest.mark.anyio
async def test_synthesize_speech_raises_on_http_error():
    fake_client = _FakeAsyncClient(_FakeResponse(b"", status_code=401))

    with patch("app.services.tts_deepgram.httpx.AsyncClient", return_value=fake_client):
        from app.services.tts_deepgram import synthesize_speech
        with pytest.raises(RuntimeError):
            await synthesize_speech("Hello")
