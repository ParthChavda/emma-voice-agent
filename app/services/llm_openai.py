import json
import re
import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone

from openai import AsyncOpenAI

from app.config import settings
from app.services import appointments

_client: AsyncOpenAI | None = None

URGENT_REPLY = (
    "This sounds like an emergency. Please call 999 now, "
    "or 111 if it is not immediately life-threatening. Do not wait."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "book_appointment",
            "description": (
                "Book a new appointment for the patient. preferred_time accepts "
                "any natural spoken phrasing (e.g. 'tomorrow at 3pm', 'next "
                "Tuesday afternoon', 'the 15th at 10am') — never require the "
                "caller to give a specific format."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_name": {"type": "string"},
                    "phone_number": {"type": "string"},
                    "service": {
                        "type": "string",
                        "enum": ["routine", "urgent", "telephone", "nurse"],
                    },
                    "preferred_time": {
                        "type": "string",
                        "description": "The patient's preferred appointment date/time, in whatever words they used.",
                    },
                },
                "required": ["patient_name", "phone_number", "service", "preferred_time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_test_results",
            "description": "Check whether the patient's test results are ready.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_name": {"type": "string"},
                },
                "required": ["patient_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_urgent",
            "description": (
                "Route patient to emergency services. "
                "Use when any life-threatening symptom is mentioned."
            ),
            "parameters": {
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": ["reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_human",
            "description": "Transfer the patient to a human receptionist.",
            "parameters": {
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": ["reason"],
            },
        },
    },
]

URGENT_KEYWORDS = [
    "chest pain",
    "can't breathe",
    "cannot breathe",
    "difficulty breathing",
    "trouble breathing",
    "struggling to breathe",
    "severe bleeding",
    "bleeding heavily",
    "unconscious",
    "loss of consciousness",
    "passed out",
    "collapsed",
    "suicidal",
    "kill myself",
    "want to die",
    "end my life",
]


def _mentions_urgent_symptom(messages: list[dict]) -> bool:
    """Match the same hard-safety symptom list as EMMA_SYSTEM_PROMPT, so escalate_urgent
    is forced via tool_choice rather than left to the model's discretion."""
    if not messages or messages[-1].get("role") != "user":
        return False
    text = (messages[-1].get("content") or "").lower()
    return any(keyword in text for keyword in URGENT_KEYWORDS)


# Output-side safety net: the model can occasionally speak the correct
# escalation wording as plain text without also emitting the matching tool
# call (finish_reason == "stop", not "tool_calls") — this catches that
# failure mode by checking what was actually said, complementing
# _mentions_urgent_symptom's input-side bypass with an output-side one.
# escalate_urgent has an exact required phrase (system prompt rule 2), so
# that check is high-confidence. escalate_human has no mandated wording, so
# only affirmative "I'm doing this now" phrasing is matched — deliberately
# excluding question forms like "would you like me to transfer you?", which
# is Emma asking permission, not a decision already made.
_URGENT_REPLY_MARKERS = ("sounds like an emergency", "call 999 now")
# "i'll transfer you" / "i will transfer you" / "i can transfer you",
# allowing a short affirmative infill like "need to" or "now" between the
# modal and "transfer you" (real model phrasing seen: "I'll need to
# transfer you..."). "i can transfer you" is ambiguous on its own (it can be
# a capability statement offered as a question), so it only counts when the
# reply as a whole isn't phrased as a question — genuine offers like "Would
# you like me to transfer you?" or "...would you like that?" are excluded
# by the trailing "?" check below, not by the wording of this pattern.
_AFFIRMATIVE_TRANSFER_RE = re.compile(
    r"\bi(?:'ll| will)\b(?:\s+\w+){0,3}\s+transfer(?:ring)? you\b"
    r"|\bi(?:'m| am)\b\s+transferring you\b"
    r"|\bi (?:can|could) transfer you\b",
    re.IGNORECASE,
)


def _infer_intent_from_reply(reply: str | None) -> str | None:
    if not reply:
        return None
    low = reply.lower()
    if any(marker in low for marker in _URGENT_REPLY_MARKERS):
        return "escalate_urgent"
    if _AFFIRMATIVE_TRANSFER_RE.search(low) and not low.rstrip().endswith("?"):
        return "escalate_human"
    return None


MOCK_RESPONSES: dict = {
    "check_test_results": lambda _: {
        "status": "available",
        "message": "Results are ready. Please call after 2pm.",
    },
    "escalate_urgent": lambda _: {"action": "999_redirect"},
    "escalate_human": lambda _: {"action": "transfer", "queue_position": 2},
}

# Tools whose result is already final, canned text — for these, a second
# completion would only spend ~1-2s re-phrasing a string we already have, so
# the reply is built directly instead of round-tripping through the model
# again. Any tool NOT listed here (e.g. a future one backed by real, variable
# data) still goes through the normal second-completion synthesis path below.
MOCK_REPLY_TEMPLATES: dict = {
    "check_test_results": lambda fn_args, mock_result: mock_result["message"],
    "escalate_human": lambda fn_args, mock_result: (
        "I'll transfer you to a human receptionist now — please hold for a moment."
    ),
}


async def _handle_book_appointment(args: dict) -> dict:
    now = datetime.now(timezone.utc)
    parsed = appointments.parse_preferred_time(args["preferred_time"], now=now)

    if parsed is None:
        result = {"status": "unclear_time"}
    elif parsed < now:
        result = {"status": "time_in_past"}
    elif not appointments.is_date_supported(parsed, now):
        result = {"status": "date_out_of_range"}  # only today/tomorrow have generated slots
    elif not appointments.is_within_opening_hours(parsed):
        result = {"status": "outside_hours"}
    else:
        slot = appointments.round_to_slot(parsed)
        row = await appointments.book_slot_and_create_appointment(
            args["patient_name"], args["phone_number"], args["service"], slot
        )
        if row is None:
            result = {"status": "slot_taken", "requested_time": slot.isoformat()}
        else:
            result = {
                "status": "booked",
                "appointment_time": row["appointment_time"].isoformat(),
                "ref": str(row["id"])[:8],
            }

    # Logged separately from the tool-call timing line — this is the *what*
    # (which branch fired and why), not the *how long*, so it's visible
    # without needing a DB query to understand a booking outcome.
    print(f'[EMMA-TIMING] book_appointment result | preferred_time={args["preferred_time"]!r} -> {result}')
    return result


ASYNC_HANDLERS: dict = {
    "book_appointment": _handle_book_appointment,
}


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _client


async def chat_completion(
    messages: list[dict],
    tools: list[dict],
) -> tuple[str, str | None]:
    # Bypass the model entirely for known emergency phrasing — relying on the
    # model to reliably emit a tool call for this exact wording isn't safe enough.
    if _mentions_urgent_symptom(messages):
        print(f'[EMMA-TIMING] Emma reply (keyword short-circuit) | intent: escalate_urgent | text: "{URGENT_REPLY}"')
        return URGENT_REPLY, "escalate_urgent"

    llm_start = time.perf_counter()
    response = await get_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    print(f"[EMMA-TIMING] LLM first completion: {time.perf_counter() - llm_start:.2f}s")
    choice = response.choices[0]

    if choice.finish_reason != "tool_calls" or not choice.message.tool_calls:
        reply = choice.message.content
        inferred_intent = _infer_intent_from_reply(reply)
        if inferred_intent:
            print(f'[EMMA-TIMING] Emma reply (no tool call, intent inferred from wording) | intent: {inferred_intent} | text: "{reply}"')
            return reply, inferred_intent
        print(f'[EMMA-TIMING] Emma reply (no tool call) | text: "{reply}"')
        return reply, None

    tool_call = choice.message.tool_calls[0]
    fn_name = tool_call.function.name
    fn_args = json.loads(tool_call.function.arguments)

    if fn_name == "escalate_urgent":
        print(f'[EMMA-TIMING] Emma reply | intent: {fn_name} | text: "{URGENT_REPLY}"')
        return URGENT_REPLY, fn_name

    tool_start = time.perf_counter()
    if fn_name in ASYNC_HANDLERS:
        mock_result = await ASYNC_HANDLERS[fn_name](fn_args)
    elif fn_name in MOCK_RESPONSES:
        mock_result = MOCK_RESPONSES[fn_name](fn_args)
    else:
        reply = "I'm sorry, I wasn't able to complete that request. Please call us on 0161 234 5678."
        print(f'[EMMA-TIMING] Emma reply | intent: {fn_name} (unhandled) | text: "{reply}"')
        return reply, fn_name
    print(f"[EMMA-TIMING] tool '{fn_name}': {time.perf_counter() - tool_start:.2f}s")

    if fn_name in MOCK_REPLY_TEMPLATES:
        reply = MOCK_REPLY_TEMPLATES[fn_name](fn_args, mock_result)
        print(f'[EMMA-TIMING] Emma reply (templated, no second completion) | intent: {fn_name} | text: "{reply}"')
        return reply, fn_name

    messages = messages + [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": fn_name,
                        "arguments": tool_call.function.arguments,
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(mock_result),
        },
    ]
    second_start = time.perf_counter()
    response2 = await get_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="none",
    )
    print(f"[EMMA-TIMING] LLM second completion: {time.perf_counter() - second_start:.2f}s")
    reply = response2.choices[0].message.content
    print(f'[EMMA-TIMING] Emma reply | intent: {fn_name} | text: "{reply}"')
    return reply, fn_name


# Requires whitespace already present after the punctuation — deliberately
# does NOT match end-of-buffer, since mid-stream that just means "more
# characters may still be coming" (e.g. "3." before a "5" arrives).
_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?]+\s+")


def _extract_ready_sentences(buffer: str) -> tuple[list[str], str]:
    """Split complete sentences off the front of buffer. Returns (sentences, remainder)."""
    sentences = []
    while True:
        match = _SENTENCE_BOUNDARY_RE.search(buffer)
        if not match:
            break
        sentence = buffer[: match.end()].strip()
        if sentence:
            sentences.append(sentence)
        buffer = buffer[match.end() :]
    return sentences, buffer


async def _stream_text_by_sentence(stream, on_sentence: Callable[[str], Awaitable[None]]) -> str:
    buffer = ""
    parts: list[str] = []
    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            buffer += delta.content
            ready, buffer = _extract_ready_sentences(buffer)
            for sentence in ready:
                parts.append(sentence)
                await on_sentence(sentence)
    remaining = buffer.strip()
    if remaining:
        parts.append(remaining)
        await on_sentence(remaining)
    return " ".join(parts)


async def _stream_first_completion(
    messages: list[dict],
    tools: list[dict],
    on_sentence: Callable[[str], Awaitable[None]],
):
    """Streams the tool_choice="auto" completion. If the model replies with
    plain text, streams it sentence-by-sentence via on_sentence and returns
    (full_text, None, None, None). If it calls a tool instead, returns
    (None, fn_name, fn_args, tool_call_id) without calling on_sentence — tool
    calls don't produce user-facing text on this first round."""
    stream = await get_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto",
        stream=True,
    )

    buffer = ""
    parts: list[str] = []
    tool_call_id: str | None = None
    tool_call_name: str | None = None
    tool_call_args = ""
    saw_tool_call = False

    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.tool_calls:
            saw_tool_call = True
            for tc_delta in delta.tool_calls:
                if tc_delta.id:
                    tool_call_id = tc_delta.id
                if tc_delta.function and tc_delta.function.name:
                    tool_call_name = tc_delta.function.name
                if tc_delta.function and tc_delta.function.arguments:
                    tool_call_args += tc_delta.function.arguments
        elif delta.content:
            buffer += delta.content
            ready, buffer = _extract_ready_sentences(buffer)
            for sentence in ready:
                parts.append(sentence)
                await on_sentence(sentence)

    if saw_tool_call:
        return None, tool_call_name, json.loads(tool_call_args or "{}"), tool_call_id

    remaining = buffer.strip()
    if remaining:
        parts.append(remaining)
        await on_sentence(remaining)
    return " ".join(parts), None, None, None


async def chat_completion_stream(
    messages: list[dict],
    tools: list[dict],
    on_sentence: Callable[[str], Awaitable[None]],
    turn_label: str | None = None,
) -> tuple[str, str | None]:
    """Like chat_completion(), but calls on_sentence(text) for each complete
    sentence of the reply as soon as it's available, instead of only
    returning the full reply once generation has finished. Lets the caller
    (the live call handler) start speaking a reply before the model has
    finished composing the rest of it.

    turn_label (e.g. "turn 3") is prefixed on every [EMMA-TIMING] line so logs
    from overlapping turns — STT keeps listening while a previous turn's
    reply is still being spoken — can be told apart in the raw log stream.
    """
    tag = f"[{turn_label}] " if turn_label else ""

    if _mentions_urgent_symptom(messages):
        await on_sentence(URGENT_REPLY)
        print(f'[EMMA-TIMING] {tag}Emma reply (keyword short-circuit) | intent: escalate_urgent | text: "{URGENT_REPLY}"')
        return URGENT_REPLY, "escalate_urgent"

    # Marks when the (first) completion call begins; wrapping on_sentence lets
    # us log time-to-first-sentence without the caller needing to know
    # whether that sentence comes from this completion or, for tool-call
    # turns, the second one below.
    llm_start = time.perf_counter()
    first_sentence_logged = False

    async def timed_on_sentence(sentence: str) -> None:
        nonlocal first_sentence_logged
        if not first_sentence_logged:
            first_sentence_logged = True
            elapsed = time.perf_counter() - llm_start
            # Text isn't printed here — the full reply is logged once, below,
            # once it's known; printing per-sentence text here as well as
            # there just repeats the same words across two log lines.
            print(f"[EMMA-TIMING] {tag}LLM first sentence: {elapsed:.2f}s")
        await on_sentence(sentence)

    reply, fn_name, fn_args, tool_call_id = await _stream_first_completion(messages, tools, timed_on_sentence)

    if reply is not None:
        inferred_intent = _infer_intent_from_reply(reply)
        if inferred_intent:
            print(f'[EMMA-TIMING] {tag}Emma reply (no tool call, intent inferred from wording) | intent: {inferred_intent} | text: "{reply}"')
            return reply, inferred_intent
        print(f'[EMMA-TIMING] {tag}Emma reply (no tool call) | text: "{reply}"')
        return reply, None

    if fn_name == "escalate_urgent":
        await timed_on_sentence(URGENT_REPLY)
        print(f'[EMMA-TIMING] {tag}Emma reply | intent: {fn_name} | text: "{URGENT_REPLY}"')
        return URGENT_REPLY, fn_name

    tool_start = time.perf_counter()
    if fn_name in ASYNC_HANDLERS:
        mock_result = await ASYNC_HANDLERS[fn_name](fn_args)
    elif fn_name in MOCK_RESPONSES:
        mock_result = MOCK_RESPONSES[fn_name](fn_args)
    else:
        reply = "I'm sorry, I wasn't able to complete that request. Please call us on 0161 234 5678."
        await timed_on_sentence(reply)
        print(f'[EMMA-TIMING] {tag}Emma reply | intent: {fn_name} (unhandled) | text: "{reply}"')
        return reply, fn_name
    print(f"[EMMA-TIMING] {tag}tool '{fn_name}': {time.perf_counter() - tool_start:.2f}s")

    if fn_name in MOCK_REPLY_TEMPLATES:
        reply = MOCK_REPLY_TEMPLATES[fn_name](fn_args, mock_result)
        await timed_on_sentence(reply)
        print(f'[EMMA-TIMING] {tag}Emma reply (templated, no second completion) | intent: {fn_name} | text: "{reply}"')
        return reply, fn_name

    messages = messages + [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": fn_name,
                        "arguments": json.dumps(fn_args),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": json.dumps(mock_result),
        },
    ]
    stream2 = await get_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="none",
        stream=True,
    )
    full_reply = await _stream_text_by_sentence(stream2, timed_on_sentence)
    print(f'[EMMA-TIMING] {tag}Emma reply | intent: {fn_name} | text: "{full_reply}"')
    return full_reply, fn_name
