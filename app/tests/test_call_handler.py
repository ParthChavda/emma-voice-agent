import base64
import math

import anyio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.call_handler import CallSession

_DISCONNECT = object()


class _FakeTwilioWebSocket:
    """Simulates Twilio's Media Stream WebSocket: scripted incoming events,
    plus auto-echoing any 'mark' event we send, like Twilio does once
    playback of that mark completes."""

    def __init__(self, scripted_events: list[dict]):
        self._send_stream, self._receive_stream = anyio.create_memory_object_stream(
            max_buffer_size=math.inf
        )
        for event in scripted_events:
            self._send_stream.send_nowait(event)
        self.sent: list[dict] = []
        self.closed = False

    async def receive_json(self) -> dict:
        try:
            event = await self._receive_stream.receive()
        except anyio.EndOfStream:
            raise RuntimeError("disconnected")
        if event is _DISCONNECT:
            raise RuntimeError("disconnected")
        return event

    async def send_json(self, message: dict) -> None:
        self.sent.append(message)
        if message.get("event") == "mark":
            await self._send_stream.send({
                "event": "mark",
                "streamSid": message["streamSid"],
                "mark": message["mark"],
            })

    async def close(self) -> None:
        self.closed = True
        await self._send_stream.aclose()


def _media_event(payload: bytes) -> dict:
    return {"event": "media", "media": {"payload": base64.b64encode(payload).decode("ascii")}}


def _scripted_stt(turns: list[str]):
    """Fake transcribe_stream: fires on_final/on_utterance_end for each scripted
    turn, then drains audio_iter so the call task winds down once the audio
    stream is closed by the receive loop."""

    async def fake(audio_iter, on_final, on_utterance_end=None):
        await anyio.sleep(0.05)  # let the receive loop process "start" first, like a real STT round-trip would
        for text in turns:
            await on_final(text)
            if on_utterance_end:
                await on_utterance_end()
        async for _ in audio_iter:
            pass

    return fake


def _fake_tts_stream(*chunk_lists: bytes) -> MagicMock:
    """Mock usable as synthesize_speech_stream: each call yields (as an async
    generator) the next entry in chunk_lists, in order; the last entry repeats
    for any further calls. Still a MagicMock, so call assertions work."""
    remaining = list(chunk_lists)

    def make_stream(*args, **kwargs):
        chunks = remaining.pop(0) if len(remaining) > 1 else remaining[0]

        async def gen():
            yield chunks

        return gen()

    return MagicMock(side_effect=make_stream)


def _fake_chat_stream(reply: str, intent: str | None, sentences: list[str] | None = None) -> MagicMock:
    """Mock usable as chat_completion_stream: calls on_sentence once per
    entry in `sentences` (or once with the full reply if not given), then
    returns (reply, intent) — mirroring the real function's contract. Still
    a MagicMock, so call assertions (e.g. on the messages argument) work."""

    async def fake(messages, tools, on_sentence):
        for sentence in (sentences or [reply]):
            await on_sentence(sentence)
        return reply, intent

    return MagicMock(side_effect=fake)


@pytest.mark.anyio
async def test_start_event_speaks_greeting():
    ws = _FakeTwilioWebSocket([
        {"event": "start", "start": {"callSid": "CA0", "streamSid": "MZ0"}},
        {"event": "stop"},
    ])
    mock_tts = _fake_tts_stream(b"\x02" * 160)

    with (
        patch("app.core.call_handler.transcribe_stream", _scripted_stt([])),
        patch("app.core.call_handler.synthesize_speech_stream", mock_tts),
        patch(
            "app.core.call_handler.generate_call_summary",
            AsyncMock(return_value={"call_sid": "CA0"}),
        ),
        patch("app.core.call_handler.db.save_turn", AsyncMock()) as mock_save_turn,
        patch("app.core.call_handler.db.save_call_summary", AsyncMock()),
    ):
        session = CallSession(ws)
        await session.run()

    from app.core.call_handler import GREETING

    mock_tts.assert_called_once_with(GREETING, encoding="mulaw", sample_rate=8000, container="none")
    media_sent = [m for m in ws.sent if m["event"] == "media"]
    assert len(media_sent) == 1
    mock_save_turn.assert_called_once_with("CA0", "assistant", GREETING)


@pytest.mark.anyio
async def test_normal_turn_speaks_reply_and_saves_history():
    ws = _FakeTwilioWebSocket([
        {"event": "start", "start": {"callSid": "CA1", "streamSid": "MZ1"}},
        _media_event(b"\x00" * 160),
        {"event": "stop"},
    ])

    with (
        patch("app.core.call_handler.transcribe_stream", _scripted_stt(["Hi, I need an appointment."])),
        patch("app.core.call_handler.retrieve", AsyncMock(return_value=[])),
        patch(
            "app.core.call_handler.chat_completion_stream",
            _fake_chat_stream("Sure, how can I help?", None),
        ),
        patch(
            "app.core.call_handler.synthesize_speech_stream",
            _fake_tts_stream(b"\x02" * 160, b"\x01" * 320),
        ),
        patch(
            "app.core.call_handler.generate_call_summary",
            AsyncMock(return_value={"call_sid": "CA1"}),
        ) as mock_notes,
        patch("app.core.call_handler.db.save_turn", AsyncMock()) as mock_save_turn,
        patch("app.core.call_handler.db.save_call_summary", AsyncMock()) as mock_save_summary,
    ):
        session = CallSession(ws)
        await session.run()

    media_sent = [m for m in ws.sent if m["event"] == "media"]
    assert len(media_sent) == 3  # 1 greeting frame + 2 reply frames
    assert all(m["streamSid"] == "MZ1" for m in media_sent)
    assert base64.b64decode(media_sent[0]["media"]["payload"]) == b"\x02" * 160
    assert base64.b64decode(media_sent[1]["media"]["payload"]) == b"\x01" * 160

    mark_sent = [m for m in ws.sent if m["event"] == "mark"]
    assert mark_sent == []  # normal replies don't wait for playback ack

    assert mock_save_turn.call_count == 3  # greeting + user + assistant
    mock_save_turn.assert_any_call("CA1", "user", "Hi, I need an appointment.")
    mock_save_turn.assert_any_call("CA1", "assistant", "Sure, how can I help?")

    mock_notes.assert_called_once()
    mock_save_summary.assert_called_once()
    assert ws.closed is False  # Twilio owns the socket lifecycle on a graceful stop


@pytest.mark.anyio
async def test_multi_sentence_reply_speaks_each_sentence_separately():
    ws = _FakeTwilioWebSocket([
        {"event": "start", "start": {"callSid": "CA1e", "streamSid": "MZ1e"}},
        _media_event(b"\x00" * 160),
        {"event": "stop"},
    ])
    mock_tts = _fake_tts_stream(b"\x02" * 160, b"\x03" * 160, b"\x04" * 160)

    with (
        patch("app.core.call_handler.transcribe_stream", _scripted_stt(["Book it"])),
        patch("app.core.call_handler.retrieve", AsyncMock(return_value=[])),
        patch(
            "app.core.call_handler.chat_completion_stream",
            _fake_chat_stream(
                "You're all set. See you Monday.",
                None,
                sentences=["You're all set.", "See you Monday."],
            ),
        ),
        patch("app.core.call_handler.synthesize_speech_stream", mock_tts),
        patch("app.core.call_handler.generate_call_summary", AsyncMock(return_value={"call_sid": "CA1e"})),
        patch("app.core.call_handler.db.save_turn", AsyncMock()),
        patch("app.core.call_handler.db.save_call_summary", AsyncMock()),
    ):
        session = CallSession(ws)
        await session.run()

    # one synthesize_speech_stream call for the greeting + one per sentence
    assert mock_tts.call_count == 3
    mock_tts.assert_any_call("You're all set.", encoding="mulaw", sample_rate=8000, container="none")
    mock_tts.assert_any_call("See you Monday.", encoding="mulaw", sample_rate=8000, container="none")


@pytest.mark.anyio
async def test_short_turn_skips_rag_retrieval():
    ws = _FakeTwilioWebSocket([
        {"event": "start", "start": {"callSid": "CA1b", "streamSid": "MZ1b"}},
        _media_event(b"\x00" * 160),
        {"event": "stop"},
    ])

    with (
        patch("app.core.call_handler.transcribe_stream", _scripted_stt(["Book it"])),  # 2 words
        patch("app.core.call_handler.retrieve", AsyncMock(return_value=["should not be used"])) as mock_retrieve,
        patch(
            "app.core.call_handler.chat_completion_stream",
            _fake_chat_stream("Sure, one moment.", None),
        ) as mock_chat,
        patch(
            "app.core.call_handler.synthesize_speech_stream",
            _fake_tts_stream(b"\x02" * 160, b"\x01" * 160),
        ),
        patch("app.core.call_handler.generate_call_summary", AsyncMock(return_value={"call_sid": "CA1b"})),
        patch("app.core.call_handler.db.save_turn", AsyncMock()),
        patch("app.core.call_handler.db.save_call_summary", AsyncMock()),
    ):
        session = CallSession(ws)
        await session.run()

    mock_retrieve.assert_not_called()
    # build_messages still gets called with an empty chunks list, not the mocked chunks
    messages_arg = mock_chat.call_args[0][0]
    assert not any("should not be used" in m.get("content", "") for m in messages_arg)


@pytest.mark.anyio
async def test_longer_turn_still_calls_rag_retrieval():
    ws = _FakeTwilioWebSocket([
        {"event": "start", "start": {"callSid": "CA1c", "streamSid": "MZ1c"}},
        _media_event(b"\x00" * 160),
        {"event": "stop"},
    ])

    with (
        patch(
            "app.core.call_handler.transcribe_stream",
            _scripted_stt(["What are your opening hours on a Saturday?"]),  # 8 words
        ),
        patch("app.core.call_handler.retrieve", AsyncMock(return_value=["hours info"])) as mock_retrieve,
        patch(
            "app.core.call_handler.chat_completion_stream",
            _fake_chat_stream("We're open 9 to 12.", None),
        ),
        patch(
            "app.core.call_handler.synthesize_speech_stream",
            _fake_tts_stream(b"\x02" * 160, b"\x01" * 160),
        ),
        patch("app.core.call_handler.generate_call_summary", AsyncMock(return_value={"call_sid": "CA1c"})),
        patch("app.core.call_handler.db.save_turn", AsyncMock()),
        patch("app.core.call_handler.db.save_call_summary", AsyncMock()),
    ):
        session = CallSession(ws)
        await session.run()

    mock_retrieve.assert_called_once_with("What are your opening hours on a Saturday?", top_k=3)


@pytest.mark.anyio
async def test_db_save_turn_happens_after_speaking_not_before():
    ws = _FakeTwilioWebSocket([
        {"event": "start", "start": {"callSid": "CA1d", "streamSid": "MZ1d"}},
        _media_event(b"\x00" * 160),
        {"event": "stop"},
    ])

    order: list[str] = []

    async def fake_save_turn(call_sid, role, content):
        order.append(f"save_turn:{role}:{content}")

    real_send_json = ws.send_json

    async def tracking_send_json(message):
        if message.get("event") == "media":
            order.append("media")
        await real_send_json(message)

    ws.send_json = tracking_send_json

    with (
        patch("app.core.call_handler.transcribe_stream", _scripted_stt(["Book it"])),
        patch("app.core.call_handler.retrieve", AsyncMock(return_value=[])),
        patch(
            "app.core.call_handler.chat_completion_stream",
            _fake_chat_stream("Sure, one moment.", None),
        ),
        patch(
            "app.core.call_handler.synthesize_speech_stream",
            _fake_tts_stream(b"\x02" * 160, b"\x01" * 160),
        ),
        patch("app.core.call_handler.generate_call_summary", AsyncMock(return_value={"call_sid": "CA1d"})),
        patch("app.core.call_handler.db.save_turn", fake_save_turn),
        patch("app.core.call_handler.db.save_call_summary", AsyncMock()),
    ):
        session = CallSession(ws)
        await session.run()

    # the LAST "media" send (the turn's reply, not the earlier greeting) must
    # precede the fire-and-forget save_turn calls for that same turn
    last_media_index = max(i for i, entry in enumerate(order) if entry == "media")
    save_user_index = order.index("save_turn:user:Book it")
    save_assistant_index = order.index("save_turn:assistant:Sure, one moment.")
    assert last_media_index < save_user_index
    assert last_media_index < save_assistant_index


@pytest.mark.anyio
async def test_escalate_urgent_speaks_then_ends_call():
    ws = _FakeTwilioWebSocket([
        {"event": "start", "start": {"callSid": "CA2", "streamSid": "MZ2"}},
        _media_event(b"\x00" * 160),
    ])

    with (
        patch("app.core.call_handler.transcribe_stream", _scripted_stt(["I have chest pain."])),
        patch("app.core.call_handler.retrieve", AsyncMock(return_value=[])),
        patch(
            "app.core.call_handler.chat_completion_stream",
            _fake_chat_stream("This sounds like an emergency. Please call 999 now.", "escalate_urgent"),
        ),
        patch(
            "app.core.call_handler.synthesize_speech_stream",
            _fake_tts_stream(b"\x01" * 160),
        ),
        patch(
            "app.core.call_handler.generate_call_summary",
            AsyncMock(return_value={"call_sid": "CA2"}),
        ),
        patch("app.core.call_handler.db.save_turn", AsyncMock()),
        patch("app.core.call_handler.db.save_call_summary", AsyncMock()) as mock_save_summary,
    ):
        session = CallSession(ws)
        await session.run()

    mark_sent = [m for m in ws.sent if m["event"] == "mark"]
    assert len(mark_sent) == 1  # escalation waits for playback ack before hanging up
    assert ws.closed is True
    mock_save_summary.assert_called_once()


@pytest.mark.anyio
async def test_escalate_human_redirects_call_and_closes_socket():
    ws = _FakeTwilioWebSocket([
        {"event": "start", "start": {"callSid": "CA3", "streamSid": "MZ3"}},
        _media_event(b"\x00" * 160),
    ])

    with (
        patch("app.core.call_handler.transcribe_stream", _scripted_stt(["Can I speak to a person?"])),
        patch("app.core.call_handler.retrieve", AsyncMock(return_value=[])),
        patch(
            "app.core.call_handler.chat_completion_stream",
            _fake_chat_stream("Transferring you now.", "escalate_human"),
        ),
        patch(
            "app.core.call_handler.synthesize_speech_stream",
            _fake_tts_stream(b"\x01" * 160),
        ),
        patch(
            "app.core.call_handler._redirect_call_to_human", AsyncMock()
        ) as mock_redirect,
        patch(
            "app.core.call_handler.generate_call_summary",
            AsyncMock(return_value={"call_sid": "CA3"}),
        ),
        patch("app.core.call_handler.db.save_turn", AsyncMock()),
        patch("app.core.call_handler.db.save_call_summary", AsyncMock()),
    ):
        session = CallSession(ws)
        await session.run()

    mock_redirect.assert_called_once_with("CA3")
    assert ws.closed is True


@pytest.mark.anyio
async def test_escalate_human_speaks_apology_and_still_ends_call_when_redirect_fails():
    ws = _FakeTwilioWebSocket([
        {"event": "start", "start": {"callSid": "CA3b", "streamSid": "MZ3b"}},
        _media_event(b"\x00" * 160),
    ])

    with (
        patch("app.core.call_handler.transcribe_stream", _scripted_stt(["Can I speak to a person?"])),
        patch("app.core.call_handler.retrieve", AsyncMock(return_value=[])),
        patch(
            "app.core.call_handler.chat_completion_stream",
            _fake_chat_stream("Transferring you now.", "escalate_human"),
        ),
        patch(
            "app.core.call_handler.synthesize_speech_stream",
            _fake_tts_stream(b"\x01" * 160),
        ),
        patch(
            "app.core.call_handler._redirect_call_to_human",
            AsyncMock(side_effect=RuntimeError("404 Not Found")),
        ) as mock_redirect,
        patch(
            "app.core.call_handler.generate_call_summary",
            AsyncMock(return_value={"call_sid": "CA3b"}),
        ),
        patch("app.core.call_handler.db.save_turn", AsyncMock()),
        patch("app.core.call_handler.db.save_call_summary", AsyncMock()),
    ):
        session = CallSession(ws)
        await session.run()  # must not raise

    mock_redirect.assert_called_once_with("CA3b")
    media_sent = [m for m in ws.sent if m["event"] == "media"]
    assert len(media_sent) >= 2  # greeting + "transferring" reply + apology, at minimum 2
    mark_sent = [m for m in ws.sent if m["event"] == "mark"]
    assert len(mark_sent) == 2  # "transferring" reply AND the apology both wait for playback ack
    assert mark_sent[0]["mark"]["name"] != mark_sent[1]["mark"]["name"]  # distinct mark names
    assert ws.closed is True


@pytest.mark.anyio
async def test_call_drop_still_generates_and_saves_summary():
    ws = _FakeTwilioWebSocket([
        {"event": "start", "start": {"callSid": "CA4", "streamSid": "MZ4"}},
        _media_event(b"\x00" * 160),
        _DISCONNECT,  # simulates the call dropping without a graceful "stop" event
    ])

    with (
        patch("app.core.call_handler.transcribe_stream", _scripted_stt([])),
        patch("app.core.call_handler.synthesize_speech_stream", _fake_tts_stream(b"\x02" * 160)),
        patch(
            "app.core.call_handler.generate_call_summary",
            AsyncMock(return_value={"call_sid": "CA4"}),
        ) as mock_notes,
        patch("app.core.call_handler.db.save_turn", AsyncMock()),
        patch("app.core.call_handler.db.save_call_summary", AsyncMock()) as mock_save_summary,
    ):
        session = CallSession(ws)
        await session.run()

    mock_notes.assert_called_once()
    mock_save_summary.assert_called_once()


@pytest.mark.anyio
async def test_finalize_call_swallows_notes_generation_errors():
    ws = _FakeTwilioWebSocket([
        {"event": "start", "start": {"callSid": "CA5", "streamSid": "MZ5"}},
        {"event": "stop"},
    ])

    with (
        patch("app.core.call_handler.transcribe_stream", _scripted_stt([])),
        patch("app.core.call_handler.synthesize_speech_stream", _fake_tts_stream(b"\x02" * 160)),
        patch("app.core.call_handler.db.save_turn", AsyncMock()),
        patch(
            "app.core.call_handler.generate_call_summary",
            AsyncMock(side_effect=RuntimeError("openai down")),
        ),
        patch("app.core.call_handler.db.save_call_summary", AsyncMock()) as mock_save_summary,
    ):
        session = CallSession(ws)
        await session.run()  # must not raise

    mock_save_summary.assert_not_called()
