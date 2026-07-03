import uuid
from pathlib import Path

from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.config import settings

COLLECTION = "emma_knowledge"
VECTOR_SIZE = 1536

_qdrant: AsyncQdrantClient | None = None
_openai: AsyncOpenAI | None = None


def get_qdrant() -> AsyncQdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )
    return _qdrant


def get_openai() -> AsyncOpenAI:
    global _openai
    if _openai is None:
        _openai = AsyncOpenAI(api_key=settings.openai_api_key)
    return _openai


async def _embed(text: str) -> list[float]:
    resp = await get_openai().embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return resp.data[0].embedding


async def ingest_docs(docs_dir: str = "app/knowledge_base") -> None:
    client = get_qdrant()
    existing = {c.name for c in (await client.get_collections()).collections}
    if COLLECTION not in existing:
        await client.create_collection(
            COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )

    points: list[PointStruct] = []
    for path in Path(docs_dir).glob("*.md"):
        text = path.read_text()
        chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
        for chunk in chunks:
            vector = await _embed(chunk)
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={"source": path.name, "text": chunk},
                )
            )

    if points:
        await client.upsert(collection_name=COLLECTION, points=points)


async def ensure_ingested(docs_dir: str = "app/knowledge_base") -> None:
    client = get_qdrant()
    try:
        info = await client.get_collection(COLLECTION)
        if info.points_count and info.points_count > 0:
            return
    except Exception as exc:
        print(f"[rag] could not check collection, re-ingesting: {exc}")
    await ingest_docs(docs_dir)


async def retrieve(query: str, top_k: int = 3) -> list[str]:
    vector = await _embed(query)
    response = await get_qdrant().query_points(
        collection_name=COLLECTION,
        query=vector,
        limit=top_k,
    )
    return [r.payload["text"] for r in response.points]
