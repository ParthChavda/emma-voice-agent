import pytest
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch


class _FakeAcquire:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *args):
        return False


@pytest.mark.anyio
async def test_find_patient_returns_patient_dict_on_match():
    mock_conn = AsyncMock()
    mock_conn.fetchrow.return_value = {
        "id": 1,
        "full_name": "Alice Smith",
        "date_of_birth": date(1990, 5, 20),
        "phone": "07700900001",
    }

    mock_pool = MagicMock()
    mock_pool.acquire.return_value = _FakeAcquire(mock_conn)

    with patch("app.services.patients.get_pool", return_value=mock_pool):
        from app.services.patients import find_patient
        result = await find_patient("Alice Smith", "1990-05-20")

    assert result == {
        "id": 1,
        "full_name": "Alice Smith",
        "date_of_birth": "1990-05-20",
        "phone": "07700900001",
    }
    mock_conn.fetchrow.assert_called_once_with(
        "SELECT id, full_name, date_of_birth, phone FROM patients "
        "WHERE lower(full_name) = lower($1) AND date_of_birth = $2",
        "Alice Smith",
        date(1990, 5, 20),
    )


@pytest.mark.anyio
async def test_find_patient_returns_none_when_no_match():
    mock_conn = AsyncMock()
    mock_conn.fetchrow.return_value = None

    mock_pool = MagicMock()
    mock_pool.acquire.return_value = _FakeAcquire(mock_conn)

    with patch("app.services.patients.get_pool", return_value=mock_pool):
        from app.services.patients import find_patient
        result = await find_patient("Nobody Here", "2000-01-01")

    assert result is None
