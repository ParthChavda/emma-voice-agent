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
