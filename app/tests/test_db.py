import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class _FakeAcquire:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *args):
        return False


@pytest.mark.anyio
async def test_load_history_returns_empty_list_for_new_session():
    mock_conn = AsyncMock()
    mock_conn.fetch.return_value = []

    mock_pool = MagicMock()
    mock_pool.acquire.return_value = _FakeAcquire(mock_conn)

    with patch("app.db._pool", mock_pool):
        from app.db import load_history
        result = await load_history("new-session-xyz")

    assert result == []
    mock_conn.fetch.assert_called_once_with(
        "SELECT role, content FROM conversations "
        "WHERE session_id = $1 ORDER BY created_at",
        "new-session-xyz",
    )


@pytest.mark.anyio
async def test_load_history_returns_ordered_turns():
    mock_conn = AsyncMock()
    mock_conn.fetch.return_value = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]

    mock_pool = MagicMock()
    mock_pool.acquire.return_value = _FakeAcquire(mock_conn)

    with patch("app.db._pool", mock_pool):
        from app.db import load_history
        result = await load_history("existing-session")

    assert result == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]


@pytest.mark.anyio
async def test_save_turn_inserts_row():
    mock_conn = AsyncMock()

    mock_pool = MagicMock()
    mock_pool.acquire.return_value = _FakeAcquire(mock_conn)

    with patch("app.db._pool", mock_pool):
        from app.db import save_turn
        await save_turn("sess-1", "user", "I need an appointment")

    mock_conn.execute.assert_called_once()
    sql, session_id, role, content = mock_conn.execute.call_args[0]
    assert "INSERT INTO conversations" in sql
    assert session_id == "sess-1"
    assert role == "user"
    assert content == "I need an appointment"


@pytest.mark.anyio
async def test_save_call_summary_inserts_row():
    mock_conn = AsyncMock()

    mock_pool = MagicMock()
    mock_pool.acquire.return_value = _FakeAcquire(mock_conn)

    summary = {
        "call_sid": "CA123",
        "patient_name": "Alice Smith",
        "intent": "book_appointment",
        "key_details": "Booked routine appointment for Monday morning",
        "escalation_flag": False,
        "next_action": "none",
        "call_duration": 42.5,
    }

    with patch("app.db._pool", mock_pool):
        from app.db import save_call_summary
        await save_call_summary(summary)

    mock_conn.execute.assert_called_once()
    sql, call_sid, patient_name, intent, key_details, escalation_flag, next_action, call_duration = (
        mock_conn.execute.call_args[0]
    )
    assert "INSERT INTO call_summaries" in sql
    assert call_sid == "CA123"
    assert patient_name == "Alice Smith"
    assert intent == "book_appointment"
    assert escalation_flag is False
    assert call_duration == 42.5


def test_get_pool_raises_when_not_initialised():
    with patch("app.db._pool", None):
        from app.db import get_pool
        with pytest.raises(RuntimeError):
            get_pool()


def test_get_pool_returns_pool_when_initialised():
    mock_pool = MagicMock()
    with patch("app.db._pool", mock_pool):
        from app.db import get_pool
        assert get_pool() is mock_pool
