from unittest.mock import AsyncMock, patch


def test_chat_generates_session_id_when_omitted(client):
    with (
        patch("app.db.load_history", new_callable=AsyncMock, return_value=[]),
        patch("app.db.save_turn", new_callable=AsyncMock),
        patch("app.routes.chat.retrieve", new_callable=AsyncMock, return_value=[]),
        patch(
            "app.routes.chat.chat_completion",
            new_callable=AsyncMock,
            return_value=("Hello! How can I help?", None),
        ),
    ):
        resp = client.post("/chat", json={"message": "hi"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"]
    import uuid
    uuid.UUID(data["session_id"])  # raises ValueError if not a valid UUID


def test_chat_reuses_provided_session_id(client):
    with (
        patch("app.db.load_history", new_callable=AsyncMock, return_value=[]) as mock_load,
        patch("app.db.save_turn", new_callable=AsyncMock) as mock_save,
        patch("app.routes.chat.retrieve", new_callable=AsyncMock, return_value=[]),
        patch(
            "app.routes.chat.chat_completion",
            new_callable=AsyncMock,
            return_value=("Hello again!", None),
        ),
    ):
        resp = client.post("/chat", json={"message": "hi", "session_id": "my-session-1"})

    assert resp.status_code == 200
    assert resp.json()["session_id"] == "my-session-1"
    mock_load.assert_called_once_with("my-session-1")
    assert mock_save.call_args_list[0].args[0] == "my-session-1"
