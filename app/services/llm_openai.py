import json

from openai import AsyncOpenAI

from app.config import settings

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
                    "appointment_type": {
                        "type": "string",
                        "enum": ["routine", "urgent", "telephone", "nurse"],
                    },
                    "preferred_date": {
                        "type": "string",
                        "description": "Free-text date/time preference",
                    },
                },
                "required": ["patient_name", "appointment_type"],
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
                    "medication_name": {"type": "string"},
                },
                "required": ["patient_name", "medication_name"],
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

MOCK_RESPONSES: dict = {
    "book_appointment": lambda args: {
        "slot": "Tuesday 15 Jul 10:30",
        "ref": f"APT-{abs(hash(args.get('patient_name', ''))) % 9000 + 1000}",
    },
    "repeat_prescription": lambda args: {
        "status": "requested",
        "ready_in": "48 hours",
        "ref": f"RX-{abs(hash(args.get('medication_name', ''))) % 9000 + 1000}",
    },
    "check_test_results": lambda _: {
        "status": "available",
        "message": "Results are ready. Please call after 2pm.",
    },
    "escalate_urgent": lambda _: {"action": "999_redirect"},
    "escalate_human": lambda _: {"action": "transfer", "queue_position": 2},
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

    if fn_name not in MOCK_RESPONSES:
        return f"I'm sorry, I wasn't able to complete that request. Please call us on 0161 234 5678.", fn_name

    mock_result = MOCK_RESPONSES[fn_name](fn_args)

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
