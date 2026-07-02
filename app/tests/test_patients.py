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


def _mock_pool(rows: list[dict]) -> MagicMock:
    mock_conn = AsyncMock()
    mock_conn.fetch.return_value = rows
    mock_pool = MagicMock()
    mock_pool.acquire.return_value = _FakeAcquire(mock_conn)
    return mock_pool


@pytest.mark.anyio
async def test_find_patient_returns_patient_dict_on_exact_match():
    rows = [{
        "id": 1,
        "full_name": "Alice Smith",
        "date_of_birth": date(1990, 5, 20),
        "phone": "07700900001",
    }]

    with patch("app.services.patients.get_pool", return_value=_mock_pool(rows)):
        from app.services.patients import find_patient
        result = await find_patient("Alice Smith", "1990-05-20")

    assert result == {
        "id": 1,
        "full_name": "Alice Smith",
        "date_of_birth": "1990-05-20",
        "phone": "07700900001",
    }


@pytest.mark.anyio
async def test_find_patient_matches_stt_garbled_name_with_correct_dob():
    # Deepgram is far more reliable transcribing digits (DOB) than proper
    # nouns — "Alice" commonly gets misheard as "Elias" or similar. As long
    # as the DOB is an exact match, a close-enough name should still resolve.
    rows = [{
        "id": 1,
        "full_name": "Alice Smith",
        "date_of_birth": date(1990, 5, 20),
        "phone": "07700900001",
    }]

    with patch("app.services.patients.get_pool", return_value=_mock_pool(rows)):
        from app.services.patients import find_patient
        result = await find_patient("Elias Smith", "1990-05-20")

    assert result is not None
    assert result["full_name"] == "Alice Smith"


@pytest.mark.anyio
async def test_find_patient_returns_none_when_dob_has_no_match():
    with patch("app.services.patients.get_pool", return_value=_mock_pool([])):
        from app.services.patients import find_patient
        result = await find_patient("Alice Smith", "2000-01-01")

    assert result is None


@pytest.mark.anyio
async def test_find_patient_returns_none_when_name_too_different_for_matching_dob():
    # Same DOB, but a genuinely different person's name — must not match.
    rows = [{
        "id": 1,
        "full_name": "Alice Smith",
        "date_of_birth": date(1990, 5, 20),
        "phone": "07700900001",
    }]

    with patch("app.services.patients.get_pool", return_value=_mock_pool(rows)):
        from app.services.patients import find_patient
        result = await find_patient("Bob Jones", "1990-05-20")

    assert result is None


@pytest.mark.anyio
async def test_find_patient_picks_best_name_match_among_same_dob_candidates():
    rows = [
        {"id": 1, "full_name": "Alice Smith", "date_of_birth": date(1990, 5, 20), "phone": "07700900001"},
        {"id": 2, "full_name": "Bob Jones", "date_of_birth": date(1990, 5, 20), "phone": "07700900002"},
    ]

    with patch("app.services.patients.get_pool", return_value=_mock_pool(rows)):
        from app.services.patients import find_patient
        result = await find_patient("Elias Smith", "1990-05-20")

    assert result["id"] == 1
    assert result["full_name"] == "Alice Smith"
