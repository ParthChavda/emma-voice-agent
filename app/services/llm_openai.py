import json
from datetime import datetime

from openai import AsyncOpenAI

from app.config import settings
from app.services import appointments, patients, prescriptions

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
            "description": "Book or request an appointment for the patient.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_name": {"type": "string"},
                    "patient_dob": {
                        "type": "string",
                        "description": "Patient's date of birth, format YYYY-MM-DD",
                    },
                    "appointment_type": {
                        "type": "string",
                        "enum": ["routine", "urgent", "telephone", "nurse"],
                    },
                    "preferred_date": {
                        "type": "string",
                        "description": "Free-text date/time preference",
                    },
                },
                "required": ["patient_name", "patient_dob", "appointment_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "repeat_prescription",
            "description": "Request a repeat prescription for the patient.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_name": {"type": "string"},
                    "patient_dob": {
                        "type": "string",
                        "description": "Patient's date of birth, format YYYY-MM-DD",
                    },
                    "medication_name": {"type": "string"},
                },
                "required": ["patient_name", "patient_dob", "medication_name"],
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


MOCK_RESPONSES: dict = {
    "check_test_results": lambda _: {
        "status": "available",
        "message": "Results are ready. Please call after 2pm.",
    },
    "escalate_urgent": lambda _: {"action": "999_redirect"},
    "escalate_human": lambda _: {"action": "transfer", "queue_position": 2},
}


def _format_slot_time(iso_str: str) -> str:
    """Render an ISO timestamp as unambiguous natural language for the model to quote verbatim."""
    return datetime.fromisoformat(iso_str).strftime("%A %d %B %Y at %H:%M")


async def _handle_book_appointment(args: dict) -> dict:
    patient = await patients.find_patient(args["patient_name"], args["patient_dob"])
    if patient is None:
        return {"error": "patient_not_found"}
    slots = await appointments.list_available_slots(args["appointment_type"])
    if not slots:
        return {"error": "no_slots_available"}
    booking = await appointments.create_booking(patient["id"], slots[0]["id"])
    return {
        # The verified record's name, not necessarily what the caller's
        # speech transcribed to — so Emma confirms the correct name back.
        "patient_name": patient["full_name"],
        "slot": _format_slot_time(booking["start_time"]),
        "doctor": booking["doctor_name"],
        "ref": booking["ref"],
    }


async def _handle_repeat_prescription(args: dict) -> dict:
    patient = await patients.find_patient(args["patient_name"], args["patient_dob"])
    if patient is None:
        return {"error": "patient_not_found"}
    result = await prescriptions.request_repeat(patient["id"], args["medication_name"])
    return {
        "patient_name": patient["full_name"],
        "status": "requested",
        "ready_in": "48 hours",
        "ref": result["ref"],
    }


ASYNC_HANDLERS = {
    "book_appointment": _handle_book_appointment,
    "repeat_prescription": _handle_repeat_prescription,
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
        return URGENT_REPLY, "escalate_urgent"

    response = await get_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    choice = response.choices[0]

    if choice.finish_reason != "tool_calls" or not choice.message.tool_calls:
        return choice.message.content, None

    tool_call = choice.message.tool_calls[0]
    fn_name = tool_call.function.name
    fn_args = json.loads(tool_call.function.arguments)

    if fn_name == "escalate_urgent":
        return URGENT_REPLY, fn_name

    if fn_name in ASYNC_HANDLERS:
        mock_result = await ASYNC_HANDLERS[fn_name](fn_args)
    elif fn_name in MOCK_RESPONSES:
        mock_result = MOCK_RESPONSES[fn_name](fn_args)
    else:
        return f"I'm sorry, I wasn't able to complete that request. Please call us on 0161 234 5678.", fn_name

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
    response2 = await get_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="none",
    )
    return response2.choices[0].message.content, fn_name
