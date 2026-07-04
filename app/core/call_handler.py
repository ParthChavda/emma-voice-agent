import base64
import math
import time
from collections.abc import Awaitable, Callable

import anyio
import httpx

from app import db
from app.config import settings
from app.core.notes import generate_call_summary
from app.core.prompts import build_messages
from app.services.llm_openai import TOOLS, chat_completion_stream
from app.services.rag import retrieve
from app.services.stt_elevenlabs import transcribe_stream
from app.services.tts_elevenlabs import synthesize_speech_stream

FRAME_BYTES = 160  # 20ms of 8kHz 8-bit mulaw, Twilio's expected media frame size
FRAME_DURATION_S = 0.02
MARK_TIMEOUT_S = 10
# Below this word count, skip RAG entirely — short turns ("yes", "book it",
# "cancel") are confirmations/commands, not the kind of question RAG context
# would help answer, and the Qdrant+embedding round trip isn't free.
SHORT_TURN_WORD_THRESHOLD = 6

GREETING = (
    "Hello, thank you for calling QUANTUMLOOPAI. "
    "This is Emma, how can I help you today?"
)


async def _redirect_call_to_human(call_sid: str) -> None:
    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f"<Response><Dial>{settings.human_handoff_number}</Dial></Response>"
    )
    url = f"https://api.twilio.com/2010-04-01/Accounts/{settings.twilio_account_sid}/Calls/{call_sid}.json"

    async with httpx.AsyncClient() as client:
        response = await client.post(
            url,
            auth=(settings.twilio_account_sid, settings.twilio_auth_token),
            data={"Twiml": twiml},
            timeout=10.0,
        )
        response.raise_for_status()


class CallSession:
    """Orchestrates one live Twilio Media Stream call: audio in -> STT -> LLM -> TTS -> audio out."""

    def __init__(self, websocket):
        self.websocket = websocket
        self.call_sid: str | None = None
        self.stream_sid: str | None = None
        self.history: list[dict[str, str]] = []
        self.turn_parts: list[str] = []
        self.start_time = time.monotonic()
        self._mark_counter = 0
        self._mark_events: dict[str, anyio.Event] = {}
        self._audio_closed = False
        self._audio_send, self._audio_receive = anyio.create_memory_object_stream(max_buffer_size=math.inf)
        self._turn_lock = anyio.Lock()
        self._turn_counter = 0
        self._task_group: anyio.abc.TaskGroup | None = None
        # Reset to None at the end of each turn (see _process_turn's finally);
        # set on the next Twilio audio frame that arrives, marking t=0 for
        # that turn's [EMMA-TIMING] logs.
        self._turn_timer_start: float | None = None

    async def run(self) -> None:
        try:
            async with anyio.create_task_group() as tg:
                self._task_group = tg
                tg.start_soon(self._stt_task)
                await self._receive_loop()
        except Exception as exc:
            print(f"[call_handler] call ended with error: {exc!r}")
        finally:
            await self._finalize_call()

    async def _audio_iter(self):
        async with self._audio_receive:
            async for chunk in self._audio_receive:
                yield chunk

    async def _stt_task(self) -> None:
        await transcribe_stream(
            audio_iter=self._audio_iter(),
            on_final=self._on_final,
            on_utterance_end=self._on_utterance_end,
        )

    async def _receive_loop(self) -> None:
        try:
            while True:
                message = await self.websocket.receive_json()
                event = message.get("event")

                if event == "start":
                    self.call_sid = message["start"]["callSid"]
                    self.stream_sid = message["start"]["streamSid"]
                    self._task_group.start_soon(self._speak_greeting)

                elif event == "media":
                    if self._turn_timer_start is None:
                        self._turn_timer_start = time.perf_counter()
                    payload = base64.b64decode(message["media"]["payload"])
                    await self._audio_send.send(payload)

                elif event == "mark":
                    name = message["mark"]["name"]
                    mark_event = self._mark_events.get(name)
                    if mark_event is not None:
                        mark_event.set()

                elif event == "stop":
                    break
        except Exception as exc:
            print(f"[call_handler] receive loop ended: {exc!r}")
        finally:
            await self._close_audio_stream()

    async def _close_audio_stream(self) -> None:
        if not self._audio_closed:
            self._audio_closed = True
            await self._audio_send.aclose()

    async def _speak_greeting(self) -> None:
        # Shares _turn_lock with turn processing so the greeting and a reply
        # can never interleave into garbled overlapping audio.
        async with self._turn_lock:
            self.history.append({"role": "assistant", "content": GREETING})
            if self.call_sid:
                await db.save_turn(self.call_sid, "assistant", GREETING)
            await self._speak(GREETING, label="greeting")

    async def _on_final(self, text: str) -> None:
        self.turn_parts.append(text)

    async def _on_utterance_end(self) -> None:
        if not self.turn_parts:
            return
        user_message = " ".join(self.turn_parts)
        self.turn_parts = []
        self._turn_counter += 1
        turn_num = self._turn_counter
        if self._turn_timer_start is not None:
            elapsed = time.perf_counter() - self._turn_timer_start
            print(f'[EMMA-TIMING] [turn {turn_num}] STT final transcript: {elapsed:.2f}s | text: "{user_message}"')
        # Spawn rather than await: the STT receive loop must keep pulling
        # audio into the transcription service while this turn is processed,
        # or the provider closes the connection for inactivity if a turn
        # takes more than a few seconds.
        # Note STT keeps listening the whole time, so the NEXT turn's lines can
        # start appearing in the log before this turn's finish — turn_num
        # tags every line below so overlapping turns can still be told apart.
        self._task_group.start_soon(self._process_turn_locked, user_message, turn_num)

    async def _process_turn_locked(self, user_message: str, turn_num: int) -> None:
        async with self._turn_lock:
            await self._process_turn(user_message, turn_num)

    async def _process_turn(self, user_message: str, turn_num: int) -> None:
        # Falls back to now() if somehow unset (e.g. tests that skip the
        # media-event path) so the TOTAL log below never explodes/goes negative.
        turn_start = self._turn_timer_start or time.perf_counter()
        label = f"turn {turn_num}"
        tag = f"[{label}] "
        try:
            # Short turns ("yes", "book it", "cancel") are confirmations/commands,
            # not questions RAG context would help answer — skip the Qdrant round
            # trip entirely rather than pay for a lookup the model won't use.
            if len(user_message.split()) < SHORT_TURN_WORD_THRESHOLD:
                chunks: list[str] = []
            else:
                rag_start = time.perf_counter()
                chunks = await retrieve(user_message, top_k=3)
                print(f"[EMMA-TIMING] {tag}RAG retrieve: {time.perf_counter() - rag_start:.2f}s")
            messages = build_messages(chunks, self.history, user_message)

            # Speaks each sentence as soon as it's generated rather than waiting
            # for the whole reply — the caller hears the first sentence while the
            # model is still composing the rest of it.
            is_first_sentence = True

            async def on_sentence(sentence: str) -> None:
                nonlocal is_first_sentence
                if is_first_sentence:
                    is_first_sentence = False
                    tts_start = time.perf_counter()

                    async def _on_first_frame() -> None:
                        print(f"[EMMA-TIMING] {tag}TTS first audio sent: {time.perf_counter() - tts_start:.2f}s")
                        print(f"[EMMA-TIMING] {tag}TIME TO FIRST AUDIO (mic -> speaker): {time.perf_counter() - turn_start:.2f}s")

                    await self._speak(sentence, on_first_frame=_on_first_frame, label=label)
                else:
                    await self._speak(sentence, label=label)

            reply, intent = await chat_completion_stream(messages, TOOLS, on_sentence, turn_label=label)

            self.history.append({"role": "user", "content": user_message})
            self.history.append({"role": "assistant", "content": reply})

            if intent == "escalate_urgent":
                await self._wait_for_playback_ack()
                self._save_turn_async(user_message, reply)
                await self._end_call()
            elif intent == "escalate_human":
                await self._wait_for_playback_ack()
                self._save_turn_async(user_message, reply)
                await self._transfer_to_human()
            else:
                self._save_turn_async(user_message, reply)

            print(f"[EMMA-TIMING] {tag}TURN FULLY PROCESSED (mic -> reply generated): {time.perf_counter() - turn_start:.2f}s")
        finally:
            # Ready for the next turn's first Twilio audio frame to mark a fresh t=0.
            self._turn_timer_start = None

    def _save_turn_async(self, user_message: str, reply: str) -> None:
        # Fire-and-forget: the caller hears the reply without waiting on two
        # Postgres writes first. The enclosing task group still waits for
        # these before the call is considered finished (anyio guarantees
        # spawned tasks complete before `async with tg:` exits), so nothing
        # is lost even if the call ends right after this.
        if not self.call_sid:
            return
        self._task_group.start_soon(db.save_turn, self.call_sid, "user", user_message)
        self._task_group.start_soon(db.save_turn, self.call_sid, "assistant", reply)

    async def _speak(
        self,
        text: str,
        on_first_frame: Callable[[], Awaitable[None]] | None = None,
        label: str | None = None,
    ) -> None:
        # Streamed rather than buffered in full first: the TTS provider sends
        # audio progressively, so relaying frames as they arrive gets sound to
        # the caller far sooner than waiting for the entire synthesis to finish.
        # Still paced to real time (one frame per FRAME_DURATION_S) regardless
        # of how fast the provider delivers — sending faster than real time is
        # what causes choppy/broken playback on Twilio's end. Called once per
        # sentence during streaming replies, so this only speaks — callers
        # that need to know playback has actually finished (before hanging up
        # or transferring) call _wait_for_playback_ack() afterward.
        frame_sent = False

        async def _send(frame: bytes) -> None:
            nonlocal frame_sent
            await self.websocket.send_json({
                "event": "media",
                "streamSid": self.stream_sid,
                "media": {"payload": base64.b64encode(frame).decode("ascii")},
            })
            if not frame_sent:
                frame_sent = True
                if on_first_frame is not None:
                    await on_first_frame()
            await anyio.sleep(FRAME_DURATION_S)

        buffer = bytearray()
        async for chunk in synthesize_speech_stream(
            text, encoding="mulaw", sample_rate=8000, container="none", label=label
        ):
            buffer.extend(chunk)
            while len(buffer) >= FRAME_BYTES:
                frame = bytes(buffer[:FRAME_BYTES])
                del buffer[:FRAME_BYTES]
                await _send(frame)

        if buffer:
            await _send(bytes(buffer))

    async def _wait_for_playback_ack(self) -> None:
        """Sends a mark and waits for Twilio's echo, confirming everything
        spoken so far has actually finished playing. Used before ending or
        transferring a call, so the caller isn't cut off mid-sentence."""
        self._mark_counter += 1
        mark_name = f"turn-{self._mark_counter}"
        mark_event = anyio.Event()
        self._mark_events[mark_name] = mark_event
        await self.websocket.send_json({
            "event": "mark",
            "streamSid": self.stream_sid,
            "mark": {"name": mark_name},
        })
        with anyio.move_on_after(MARK_TIMEOUT_S):
            await mark_event.wait()
        del self._mark_events[mark_name]

    async def _end_call(self) -> None:
        await self._close_audio_stream()
        await self.websocket.close()

    async def _transfer_to_human(self) -> None:
        try:
            await _redirect_call_to_human(self.call_sid)
        except Exception as exc:
            print(f"[call_handler] human transfer failed: {exc!r}")
            await self._speak(
                "I'm sorry, I wasn't able to transfer you right now. "
                "Please call us on 0161 234 5678 to speak to a receptionist."
            )
            await self._wait_for_playback_ack()
        await self._close_audio_stream()
        await self.websocket.close()

    async def _finalize_call(self) -> None:
        duration = time.monotonic() - self.start_time
        try:
            summary = await generate_call_summary(self.history, self.call_sid or "unknown", duration)
            await db.save_call_summary(summary)
        except Exception as exc:
            print(f"[call_handler] failed to generate/save call summary: {exc}")


async def handle_call(websocket) -> None:
    await CallSession(websocket).run()
