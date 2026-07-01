import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch


class _FakeAcquire:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *args):
        return False


class _FakeTransaction:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *args):
        return False


def _mock_pool(mock_conn) -> MagicMock:
    mock_conn.transaction = MagicMock(return_value=_FakeTransaction())
    mock_pool = MagicMock()
    mock_pool.acquire.return_value = _FakeAcquire(mock_conn)
    return mock_pool


@pytest.mark.anyio
async def test_list_available_slots_returns_slot_dicts():
    mock_conn = AsyncMock()
    mock_conn.fetch.return_value = [
        {
            "id": 1,
            "doctor_name": "Dr. Ahmed",
            "appointment_type": "routine",
            "start_time": datetime(2026, 7, 6, 9, 0),
        }
    ]
    mock_pool = _mock_pool(mock_conn)

    with patch("app.services.appointments.get_pool", return_value=mock_pool):
        from app.services.appointments import list_available_slots
        result = await list_available_slots("routine", limit=3)

    assert result == [
        {
            "id": 1,
            "doctor_name": "Dr. Ahmed",
            "appointment_type": "routine",
            "start_time": "2026-07-06T09:00:00",
        }
    ]
    mock_conn.fetch.assert_called_once_with(
        "SELECT id, doctor_name, appointment_type, start_time FROM slots "
        "WHERE appointment_type = $1 AND is_booked = FALSE "
        "ORDER BY start_time LIMIT $2",
        "routine",
        3,
    )


@pytest.mark.anyio
async def test_create_booking_returns_none_when_slot_unavailable():
    mock_conn = AsyncMock()
    mock_conn.fetchrow.return_value = None
    mock_pool = _mock_pool(mock_conn)

    with patch("app.services.appointments.get_pool", return_value=mock_pool):
        from app.services.appointments import create_booking
        result = await create_booking(patient_id=1, slot_id=99)

    assert result is None


@pytest.mark.anyio
async def test_create_booking_returns_booking_dict_on_success():
    mock_conn = AsyncMock()
    mock_conn.fetchrow.side_effect = [
        {"id": 5, "doctor_name": "Dr. Ahmed", "start_time": datetime(2026, 7, 6, 9, 0)},
        {"id": 42},
    ]
    mock_pool = _mock_pool(mock_conn)

    with patch("app.services.appointments.get_pool", return_value=mock_pool):
        from app.services.appointments import create_booking
        result = await create_booking(patient_id=1, slot_id=5)

    assert result["appointment_id"] == 42
    assert result["doctor_name"] == "Dr. Ahmed"
    assert result["start_time"] == "2026-07-06T09:00:00"
    assert result["ref"].startswith("APT-")
    mock_conn.execute.assert_any_call("UPDATE slots SET is_booked = TRUE WHERE id = $1", 5)


@pytest.mark.anyio
async def test_cancel_booking_returns_false_when_not_found():
    mock_conn = AsyncMock()
    mock_conn.fetchrow.return_value = None
    mock_pool = _mock_pool(mock_conn)

    with patch("app.services.appointments.get_pool", return_value=mock_pool):
        from app.services.appointments import cancel_booking
        result = await cancel_booking(appointment_id=999)

    assert result is False


@pytest.mark.anyio
async def test_cancel_booking_returns_false_when_already_cancelled():
    mock_conn = AsyncMock()
    mock_conn.fetchrow.return_value = {"slot_id": 5, "status": "cancelled"}
    mock_pool = _mock_pool(mock_conn)

    with patch("app.services.appointments.get_pool", return_value=mock_pool):
        from app.services.appointments import cancel_booking
        result = await cancel_booking(appointment_id=7)

    assert result is False


@pytest.mark.anyio
async def test_cancel_booking_returns_true_and_frees_slot_on_success():
    mock_conn = AsyncMock()
    mock_conn.fetchrow.return_value = {"slot_id": 5, "status": "booked"}
    mock_pool = _mock_pool(mock_conn)

    with patch("app.services.appointments.get_pool", return_value=mock_pool):
        from app.services.appointments import cancel_booking
        result = await cancel_booking(appointment_id=7)

    assert result is True
    mock_conn.execute.assert_any_call(
        "UPDATE appointments SET status = 'cancelled' WHERE id = $1", 7
    )
    mock_conn.execute.assert_any_call(
        "UPDATE slots SET is_booked = FALSE WHERE id = $1", 5
    )
