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


@pytest.mark.anyio
async def test_synthesize_speech_requests_raw_mulaw_for_telephony():
    fake_client = _FakeAsyncClient(_FakeResponse(b"\x00\x01mulaw-bytes"))

    with patch("app.services.tts_deepgram.httpx.AsyncClient", return_value=fake_client):
        from app.services.tts_deepgram import synthesize_speech
        audio = await synthesize_speech(
            "This sounds like an emergency.",
            encoding="mulaw",
            sample_rate=8000,
            container="none",
        )

    assert audio == b"\x00\x01mulaw-bytes"
    _, kwargs = fake_client.post.call_args
    assert kwargs["params"] == {
        "model": "aura-2-thalia-en",
        "encoding": "mulaw",
        "sample_rate": 8000,
        "container": "none",
    }


class _FakeStreamResponse:
    def __init__(self, chunks: list[bytes], status_code: int = 200):
        self._chunks = chunks
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk


class _FakeStreamContext:
    def __init__(self, response: _FakeStreamResponse):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *args):
        return False


class _FakeStreamingAsyncClient:
    def __init__(self, response: _FakeStreamResponse):
        from unittest.mock import MagicMock
        self.stream = MagicMock(return_value=_FakeStreamContext(response))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


@pytest.mark.anyio
async def test_synthesize_speech_stream_yields_chunks_as_received():
    fake_client = _FakeStreamingAsyncClient(_FakeStreamResponse([b"chunk1", b"chunk2", b"chunk3"]))

    with patch("app.services.tts_deepgram.httpx.AsyncClient", return_value=fake_client):
        from app.services.tts_deepgram import synthesize_speech_stream
        chunks = [chunk async for chunk in synthesize_speech_stream("Hello there")]

    assert chunks == [b"chunk1", b"chunk2", b"chunk3"]


@pytest.mark.anyio
async def test_synthesize_speech_stream_sends_correct_request():
    fake_client = _FakeStreamingAsyncClient(_FakeStreamResponse([b"audio"]))

    with patch("app.services.tts_deepgram.httpx.AsyncClient", return_value=fake_client):
        from app.services.tts_deepgram import synthesize_speech_stream
        async for _ in synthesize_speech_stream(
            "Book an appointment", encoding="mulaw", sample_rate=8000, container="none"
        ):
            pass

    fake_client.stream.assert_called_once()
    args, kwargs = fake_client.stream.call_args
    assert args[0] == "POST"
    assert args[1] == "https://api.deepgram.com/v1/speak"
    assert kwargs["json"] == {"text": "Book an appointment"}
    assert kwargs["params"] == {
        "model": "aura-2-thalia-en",
        "encoding": "mulaw",
        "sample_rate": 8000,
        "container": "none",
    }
    assert kwargs["headers"]["Authorization"].startswith("Token ")


@pytest.mark.anyio
async def test_synthesize_speech_stream_raises_on_http_error():
    fake_client = _FakeStreamingAsyncClient(_FakeStreamResponse([], status_code=401))

    with patch("app.services.tts_deepgram.httpx.AsyncClient", return_value=fake_client):
        from app.services.tts_deepgram import synthesize_speech_stream
        with pytest.raises(RuntimeError):
            async for _ in synthesize_speech_stream("Hello"):
                pass
