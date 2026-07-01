import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    with (
        patch("app.db.init_pool", new_callable=AsyncMock),
        patch("app.db.close_pool", new_callable=AsyncMock),
        patch("app.services.rag.ensure_ingested", new_callable=AsyncMock),
    ):
        from app.main import app
        with TestClient(app) as c:
            yield c
