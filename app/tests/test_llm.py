import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.llm_openai import MOCK_RESPONSES, URGENT_REPLY, TOOLS


def test_tools_list_has_five_entries():
    assert len(TOOLS) == 5


def test_tools_all_have_function_type():
    for tool in TOOLS:
        assert tool["type"] == "function"


def test_tool_names():
    names = {t["function"]["name"] for t in TOOLS}
    assert names == {
        "book_appointment",
        "repeat_prescription",
        "check_test_results",
        "escalate_urgent",
        "escalate_human",
    }


def test_mock_check_test_results():
    result = MOCK_RESPONSES["check_test_results"]({"patient_name": "Carol"})
    assert result["status"] == "available"


def test_mock_escalate_urgent():
    result = MOCK_RESPONSES["escalate_urgent"]({"reason": "chest pain"})
    assert result["action"] == "999_redirect"


def test_mock_escalate_human():
    result = MOCK_RESPONSES["escalate_human"]({"reason": "patient request"})
    assert result["action"] == "transfer"


def test_urgent_reply_contains_999():
    assert "999" in URGENT_REPLY


@pytest.mark.anyio
async def test_chat_completion_no_tool_call_returns_content():
    mock_choice = MagicMock()
    mock_choice.finish_reason = "stop"
    mock_choice.message.content = "Hello! How can I help you today?"
    mock_choice.message.tool_calls = None

    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]

    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = mock_resp

    with patch("app.services.llm_openai._client", mock_client):
        from app.services.llm_openai import chat_completion
        reply, intent = await chat_completion(
            [{"role": "user", "content": "hello"}], TOOLS
        )

    assert reply == "Hello! How can I help you today?"
    assert intent is None


@pytest.mark.anyio
async def test_chat_completion_escalate_urgent_returns_hardcoded_reply():
    # Wording chosen to NOT match URGENT_KEYWORDS, so this exercises the
    # model-driven tool-call path rather than the keyword short-circuit.
    tool_call = MagicMock()
    tool_call.id = "call_abc"
    tool_call.function.name = "escalate_urgent"
    tool_call.function.arguments = json.dumps({"reason": "possible stroke"})

    mock_choice = MagicMock()
    mock_choice.finish_reason = "tool_calls"
    mock_choice.message.tool_calls = [tool_call]

    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]

    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = mock_resp

    with patch("app.services.llm_openai._client", mock_client):
        from app.services.llm_openai import chat_completion
        reply, intent = await chat_completion(
            [{"role": "user", "content": "I think I'm having a stroke"}], TOOLS
        )

    assert intent == "escalate_urgent"
    assert "999" in reply
    mock_client.chat.completions.create.assert_called_once()


@pytest.mark.anyio
async def test_chat_completion_short_circuits_on_keyword_match_without_calling_api():
    mock_client = AsyncMock()

    with patch("app.services.llm_openai._client", mock_client):
        from app.services.llm_openai import chat_completion, URGENT_REPLY
        reply, intent = await chat_completion(
            [{"role": "user", "content": "I have really bad chest pain"}], TOOLS
        )

    assert reply == URGENT_REPLY
    assert intent == "escalate_urgent"
    mock_client.chat.completions.create.assert_not_called()


@pytest.mark.anyio
async def test_chat_completion_uses_auto_tool_choice_without_urgent_keyword():
    mock_choice = MagicMock()
    mock_choice.finish_reason = "stop"
    mock_choice.message.content = "Sure, I can help with that."
    mock_choice.message.tool_calls = None

    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]

    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = mock_resp

    with patch("app.services.llm_openai._client", mock_client):
        from app.services.llm_openai import chat_completion
        await chat_completion(
            [{"role": "user", "content": "What time do you open?"}], TOOLS
        )

    _, kwargs = mock_client.chat.completions.create.call_args
    assert kwargs["tool_choice"] == "auto"


@pytest.mark.anyio
async def test_chat_completion_book_appointment_makes_second_call():
    tool_call = MagicMock()
    tool_call.id = "call_xyz"
    tool_call.function.name = "book_appointment"
    tool_call.function.arguments = json.dumps(
        {"patient_name": "Alice", "patient_dob": "1990-05-20", "appointment_type": "routine"}
    )

    mock_first_choice = MagicMock()
    mock_first_choice.finish_reason = "tool_calls"
    mock_first_choice.message.tool_calls = [tool_call]

    mock_second_choice = MagicMock()
    mock_second_choice.finish_reason = "stop"
    mock_second_choice.message.content = "I've booked you in for Tuesday 15 Jul at 10:30."
    mock_second_choice.message.tool_calls = None

    mock_client = AsyncMock()
    mock_client.chat.completions.create.side_effect = [
        MagicMock(choices=[mock_first_choice]),
        MagicMock(choices=[mock_second_choice]),
    ]

    with (
        patch("app.services.llm_openai._client", mock_client),
        patch(
            "app.services.patients.find_patient",
            AsyncMock(return_value={"id": 1, "full_name": "Alice Smith"}),
        ),
        patch(
            "app.services.appointments.list_available_slots",
            AsyncMock(return_value=[
                {"id": 5, "doctor_name": "Dr. Ahmed", "start_time": "2026-07-14T10:30:00"}
            ]),
        ),
        patch(
            "app.services.appointments.create_booking",
            AsyncMock(return_value={
                "appointment_id": 42,
                "ref": "APT-ABC123",
                "doctor_name": "Dr. Ahmed",
                "start_time": "2026-07-14T10:30:00",
            }),
        ),
    ):
        from app.services.llm_openai import chat_completion
        reply, intent = await chat_completion(
            [{"role": "user", "content": "book appointment"}], TOOLS
        )

    assert intent == "book_appointment"
    assert "10:30" in reply
    assert mock_client.chat.completions.create.call_count == 2


@pytest.mark.anyio
async def test_handle_book_appointment_returns_error_when_patient_not_found():
    from app.services.llm_openai import _handle_book_appointment

    with patch("app.services.patients.find_patient", AsyncMock(return_value=None)):
        result = await _handle_book_appointment(
            {"patient_name": "Nobody", "patient_dob": "2000-01-01", "appointment_type": "routine"}
        )

    assert result == {"error": "patient_not_found"}


@pytest.mark.anyio
async def test_handle_book_appointment_returns_error_when_no_slots():
    from app.services.llm_openai import _handle_book_appointment

    with (
        patch("app.services.patients.find_patient", AsyncMock(return_value={"id": 1})),
        patch("app.services.appointments.list_available_slots", AsyncMock(return_value=[])),
    ):
        result = await _handle_book_appointment(
            {"patient_name": "Alice", "patient_dob": "1990-05-20", "appointment_type": "routine"}
        )

    assert result == {"error": "no_slots_available"}


@pytest.mark.anyio
async def test_handle_book_appointment_returns_booking_on_success():
    from app.services.llm_openai import _handle_book_appointment

    with (
        patch(
            "app.services.patients.find_patient",
            AsyncMock(return_value={"id": 1, "full_name": "Alice Smith"}),
        ),
        patch(
            "app.services.appointments.list_available_slots",
            AsyncMock(return_value=[
                {"id": 5, "doctor_name": "Dr. Ahmed", "start_time": "2026-07-06T09:00:00"}
            ]),
        ),
        patch(
            "app.services.appointments.create_booking",
            AsyncMock(return_value={
                "appointment_id": 42,
                "ref": "APT-ABC123",
                "doctor_name": "Dr. Ahmed",
                "start_time": "2026-07-06T09:00:00",
            }),
        ),
    ):
        result = await _handle_book_appointment(
            {"patient_name": "Elias", "patient_dob": "1990-05-20", "appointment_type": "routine"}
        )

    assert result == {
        "patient_name": "Alice Smith",
        "slot": "Monday 06 July 2026 at 09:00",
        "doctor": "Dr. Ahmed",
        "ref": "APT-ABC123",
    }


@pytest.mark.anyio
async def test_handle_repeat_prescription_returns_error_when_patient_not_found():
    from app.services.llm_openai import _handle_repeat_prescription

    with patch("app.services.patients.find_patient", AsyncMock(return_value=None)):
        result = await _handle_repeat_prescription(
            {"patient_name": "Nobody", "patient_dob": "2000-01-01", "medication_name": "metformin"}
        )

    assert result == {"error": "patient_not_found"}


@pytest.mark.anyio
async def test_handle_repeat_prescription_returns_requested_status_on_success():
    from app.services.llm_openai import _handle_repeat_prescription

    with (
        patch(
            "app.services.patients.find_patient",
            AsyncMock(return_value={"id": 1, "full_name": "Bob Jones"}),
        ),
        patch(
            "app.services.prescriptions.request_repeat",
            AsyncMock(return_value={"ref": "RX-XYZ999"}),
        ),
    ):
        result = await _handle_repeat_prescription(
            {"patient_name": "Bob", "patient_dob": "1985-02-14", "medication_name": "metformin"}
        )

    assert result == {
        "patient_name": "Bob Jones",
        "status": "requested",
        "ready_in": "48 hours",
        "ref": "RX-XYZ999",
    }
