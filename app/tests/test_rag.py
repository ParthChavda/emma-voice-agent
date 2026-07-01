import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.anyio
async def test_retrieve_returns_text_chunks():
    mock_hit_1 = MagicMock()
    mock_hit_1.payload = {"text": "Elmwood Road Surgery open Mon-Fri 8am-6:30pm"}
    mock_hit_2 = MagicMock()
    mock_hit_2.payload = {"text": "Saturday 9am-12pm appointment only"}

    mock_embed_resp = MagicMock()
    mock_embed_resp.data = [MagicMock(embedding=[0.1] * 1536)]

    mock_openai = AsyncMock()
    mock_openai.embeddings.create.return_value = mock_embed_resp

    mock_response = MagicMock()
    mock_response.points = [mock_hit_1, mock_hit_2]

    mock_qdrant = AsyncMock()
    mock_qdrant.query_points.return_value = mock_response

    with (
        patch("app.services.rag._qdrant", mock_qdrant),
        patch("app.services.rag._openai", mock_openai),
    ):
        from app.services.rag import retrieve
        chunks = await retrieve("what time do you open", top_k=2)

    assert chunks == [
        "Elmwood Road Surgery open Mon-Fri 8am-6:30pm",
        "Saturday 9am-12pm appointment only",
    ]
    mock_qdrant.query_points.assert_called_once_with(
        collection_name="emma_knowledge",
        query=[0.1] * 1536,
        limit=2,
    )


@pytest.mark.anyio
async def test_retrieve_returns_empty_list_when_no_results():
    mock_embed_resp = MagicMock()
    mock_embed_resp.data = [MagicMock(embedding=[0.0] * 1536)]

    mock_openai = AsyncMock()
    mock_openai.embeddings.create.return_value = mock_embed_resp

    mock_response = MagicMock()
    mock_response.points = []

    mock_qdrant = AsyncMock()
    mock_qdrant.query_points.return_value = mock_response

    with (
        patch("app.services.rag._qdrant", mock_qdrant),
        patch("app.services.rag._openai", mock_openai),
    ):
        from app.services.rag import retrieve
        chunks = await retrieve("unrelated query", top_k=3)

    assert chunks == []
