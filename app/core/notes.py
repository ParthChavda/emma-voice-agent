import json

from app.services.llm_openai import get_client

NOTES_TOOL = {
    "type": "function",
    "function": {
        "name": "record_call_summary",
        "description": "Record a structured summary of a completed patient call.",
        "parameters": {
            "type": "object",
            "properties": {
                "patient_name": {
                    "type": ["string", "null"],
                    "description": "Patient's full name if mentioned, else null.",
                },
                "intent": {
                    "type": "string",
                    "enum": [
                        "book_appointment",
                        "repeat_prescription",
                        "check_test_results",
                        "escalate_urgent",
                        "escalate_human",
                        "general_enquiry",
                        "no_action",
                    ],
                },
                "key_details": {
                    "type": "string",
                    "description": "One or two sentence summary of what happened on the call.",
                },
                "escalation_flag": {
                    "type": "boolean",
                    "description": "True if the call involved an urgent/emergency escalation.",
                },
                "next_action": {
                    "type": "string",
                    "description": "Any follow-up action staff need to take, or 'none'.",
                },
            },
            "required": ["patient_name", "intent", "key_details", "escalation_flag", "next_action"],
        },
    },
}


async def generate_call_summary(
    transcript: list[dict[str, str]],
    call_sid: str,
    call_duration: float,
) -> dict:
    transcript_text = "\n".join(f"{turn['role']}: {turn['content']}" for turn in transcript)
    messages = [
        {
            "role": "system",
            "content": (
                "You are summarizing a completed call to an NHS GP surgery's AI receptionist. "
                "Record a structured summary using the record_call_summary tool."
            ),
        },
        {"role": "user", "content": transcript_text or "(no speech was captured on this call)"},
    ]

    response = await get_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=[NOTES_TOOL],
        tool_choice={"type": "function", "function": {"name": "record_call_summary"}},
    )
    args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)

    return {
        "call_sid": call_sid,
        "patient_name": args["patient_name"],
        "intent": args["intent"],
        "key_details": args["key_details"],
        "escalation_flag": args["escalation_flag"],
        "next_action": args["next_action"],
        "call_duration": call_duration,
    }
