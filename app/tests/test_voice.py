from unittest.mock import AsyncMock, patch


def test_incoming_call_returns_twiml_with_stream_url(client):
    resp = client.post("/voice/incoming")

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/xml")
    assert "<Connect>" in resp.text
    assert 'url="wss://testserver/voice/stream"' in resp.text


def test_media_stream_websocket_delegates_to_call_handler(client):
    with patch("app.routes.voice.handle_call", new_callable=AsyncMock) as mock_handle_call:
        with client.websocket_connect("/voice/stream") as ws:
            ws.close()

    mock_handle_call.assert_called_once()
