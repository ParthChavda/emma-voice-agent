import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.anyio
async def test_generate_call_summary_returns_structured_dict():
    tool_call = MagicMock()
    tool_call.function.arguments = json.dumps({
        "patient_name": "Alice Smith",
        "intent": "book_appointment",
        "key_details": "Booked a routine appointment with Dr. Ahmed",
        "escalation_flag": False,
        "next_action": "none",
    })

    mock_choice = MagicMock()
    mock_choice.message.tool_calls = [tool_call]

    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]

    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = mock_resp

    transcript = [
        {"role": "user", "content": "I'd like to book an appointment, Alice Smith, DOB 1990-05-20"},
        {"role": "assistant", "content": "I've booked you in with Dr. Ahmed."},
    ]

    with patch("app.core.notes.get_client", return_value=mock_client):
        from app.core.notes import generate_call_summary
        summary = await generate_call_summary(transcript, call_sid="CA123", call_duration=42.5)

    assert summary == {
        "call_sid": "CA123",
        "patient_name": "Alice Smith",
        "intent": "book_appointment",
        "key_details": "Booked a routine appointment with Dr. Ahmed",
        "escalation_flag": False,
        "next_action": "none",
        "call_duration": 42.5,
    }
    mock_client.chat.completions.create.assert_called_once()
    _, kwargs = mock_client.chat.completions.create.call_args
    assert kwargs["tool_choice"] == {"type": "function", "function": {"name": "record_call_summary"}}


@pytest.mark.anyio
async def test_generate_call_summary_handles_empty_transcript():
    tool_call = MagicMock()
    tool_call.function.arguments = json.dumps({
        "patient_name": None,
        "intent": "no_action",
        "key_details": "Call ended with no speech captured",
        "escalation_flag": False,
        "next_action": "none",
    })

    mock_choice = MagicMock()
    mock_choice.message.tool_calls = [tool_call]

    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]

    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = mock_resp

    with patch("app.core.notes.get_client", return_value=mock_client):
        from app.core.notes import generate_call_summary
        summary = await generate_call_summary([], call_sid="CA999", call_duration=1.0)

    assert summary["patient_name"] is None
    assert summary["intent"] == "no_action"
    assert summary["call_sid"] == "CA999"
    assert summary["call_duration"] == 1.0
