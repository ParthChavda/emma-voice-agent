import base64
import math
import time

import anyio
import httpx

from app import db
from app.config import settings
from app.core.notes import generate_call_summary
from app.core.prompts import build_messages
from app.services.llm_openai import TOOLS, chat_completion
from app.services.rag import retrieve
from app.services.stt_deepgram import transcribe_stream
from app.services.tts_deepgram import synthesize_speech_stream

FRAME_BYTES = 160  # 20ms of 8kHz 8-bit mulaw, Twilio's expected media frame size
FRAME_DURATION_S = 0.02
MARK_TIMEOUT_S = 10

GREETING = (
    "Hello, thank you for calling Elmwood Road Surgery. "
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
        self._task_group: anyio.abc.TaskGroup | None = None

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
            await self._speak(GREETING, wait_for_playback=False)

    async def _on_final(self, text: str) -> None:
        self.turn_parts.append(text)

    async def _on_utterance_end(self) -> None:
        if not self.turn_parts:
            return
        user_message = " ".join(self.turn_parts)
        self.turn_parts = []
        # Spawn rather than await: the STT receive loop must keep pulling
        # audio into Deepgram while this turn is processed, or Deepgram closes
        # the connection for inactivity if a turn takes more than a few seconds.
        self._task_group.start_soon(self._process_turn_locked, user_message)

    async def _process_turn_locked(self, user_message: str) -> None:
        async with self._turn_lock:
            await self._process_turn(user_message)

    async def _process_turn(self, user_message: str) -> None:
        chunks = await retrieve(user_message, top_k=3)
        messages = build_messages(chunks, self.history, user_message)
        reply, intent = await chat_completion(messages, TOOLS)

        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": reply})
        if self.call_sid:
            await db.save_turn(self.call_sid, "user", user_message)
            await db.save_turn(self.call_sid, "assistant", reply)

        if intent == "escalate_urgent":
            await self._speak(reply, wait_for_playback=True)
            await self._end_call()
        elif intent == "escalate_human":
            await self._speak(reply, wait_for_playback=True)
            await self._transfer_to_human()
        else:
            await self._speak(reply, wait_for_playback=False)

    async def _speak(self, text: str, wait_for_playback: bool) -> None:
        # Streamed rather than buffered in full first: Deepgram sends audio
        # progressively, so relaying frames as they arrive gets sound to the
        # caller far sooner than waiting for the entire synthesis to finish.
        # Still paced to real time (one frame per FRAME_DURATION_S) regardless
        # of how fast Deepgram delivers — sending faster than real time is what
        # causes choppy/broken playback on Twilio's end.
        buffer = bytearray()
        async for chunk in synthesize_speech_stream(text, encoding="mulaw", sample_rate=8000, container="none"):
            buffer.extend(chunk)
            while len(buffer) >= FRAME_BYTES:
                frame = bytes(buffer[:FRAME_BYTES])
                del buffer[:FRAME_BYTES]
                await self.websocket.send_json({
                    "event": "media",
                    "streamSid": self.stream_sid,
                    "media": {"payload": base64.b64encode(frame).decode("ascii")},
                })
                await anyio.sleep(FRAME_DURATION_S)

        if buffer:
            await self.websocket.send_json({
                "event": "media",
                "streamSid": self.stream_sid,
                "media": {"payload": base64.b64encode(bytes(buffer)).decode("ascii")},
            })
            await anyio.sleep(FRAME_DURATION_S)

        if not wait_for_playback:
            return

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
                "Please call us on 0161 234 5678 to speak to a receptionist.",
                wait_for_playback=True,
            )
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
