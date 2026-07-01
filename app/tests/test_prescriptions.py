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
async def test_request_repeat_inserts_row_and_returns_ref():
    mock_conn = AsyncMock()

    mock_pool = MagicMock()
    mock_pool.acquire.return_value = _FakeAcquire(mock_conn)

    with patch("app.services.prescriptions.get_pool", return_value=mock_pool):
        from app.services.prescriptions import request_repeat
        result = await request_repeat(patient_id=1, medication_name="metformin")

    assert result["ref"].startswith("RX-")
    mock_conn.execute.assert_called_once()
    sql, patient_id, medication_name, ref = mock_conn.execute.call_args[0]
    assert "INSERT INTO prescriptions" in sql
    assert patient_id == 1
    assert medication_name == "metformin"
    assert ref == result["ref"]
