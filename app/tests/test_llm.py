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


def test_mock_book_appointment():
    result = MOCK_RESPONSES["book_appointment"](
        {"patient_name": "Alice", "appointment_type": "routine"}
    )
    assert "slot" in result
    assert result["ref"].startswith("APT-")


def test_mock_repeat_prescription():
    result = MOCK_RESPONSES["repeat_prescription"](
        {"patient_name": "Bob", "medication_name": "metformin"}
    )
    assert result["status"] == "requested"
    assert result["ready_in"] == "48 hours"
    assert result["ref"].startswith("RX-")


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
    tool_call = MagicMock()
    tool_call.id = "call_abc"
    tool_call.function.name = "escalate_urgent"
    tool_call.function.arguments = json.dumps({"reason": "chest pain"})

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
            [{"role": "user", "content": "chest pain"}], TOOLS
        )

    assert intent == "escalate_urgent"
    assert "999" in reply
    mock_client.chat.completions.create.assert_called_once()


@pytest.mark.anyio
async def test_chat_completion_book_appointment_makes_second_call():
    tool_call = MagicMock()
    tool_call.id = "call_xyz"
    tool_call.function.name = "book_appointment"
    tool_call.function.arguments = json.dumps(
        {"patient_name": "Alice", "appointment_type": "routine"}
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

    with patch("app.services.llm_openai._client", mock_client):
        from app.services.llm_openai import chat_completion
        reply, intent = await chat_completion(
            [{"role": "user", "content": "book appointment"}], TOOLS
        )

    assert intent == "book_appointment"
    assert "10:30" in reply
    assert mock_client.chat.completions.create.call_count == 2
