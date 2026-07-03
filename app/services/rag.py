import re
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


def _chunk_markdown(text: str) -> list[str]:
    """Split on '## ' section headers, keeping each heading together with
    the paragraphs under it. A naive blank-line split produces near-empty
    fragments (a heading alone, a lone sentence) that can crowd out the
    actually informative chunk once top_k caps how many are retrieved —
    section-level chunks keep facts attached to the context that gives them
    meaning (e.g. "48 hours" stays inside the same chunk as "before
    collecting" and the rest of that paragraph).
    """
    parts = [p.strip() for p in re.split(r"(?=^## )", text, flags=re.MULTILINE) if p.strip()]
    if not parts or parts[0].startswith("## "):
        return parts

    preamble = parts[0]
    preamble_body = preamble.split("\n", 1)
    remainder = preamble_body[1].strip() if len(preamble_body) > 1 else ""
    if not remainder and len(parts) > 1:
        # Preamble is just a bare "# Title" line with nothing else — merge
        # into the first real section rather than keep it as its own
        # near-empty chunk.
        parts[1] = preamble + "\n\n" + parts[1]
        return parts[1:]

    # Preamble has real content beyond the title (e.g. a contact-info card)
    # — keep it as its own chunk. Merging it into the next section would
    # dilute that section's embedding with unrelated text (confirmed: this
    # pushed the actual "Opening Hours" chunk out of top_k=3 for "what time
    # do you open" because it got merged with practice name/address/phone/
    # email/website, none of which relates to hours).
    return parts


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
        chunks = _chunk_markdown(text)
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


async def warm_up() -> None:
    """Fires one throwaway retrieve() at startup. A freshly-created OpenAI/
    Qdrant client pays a one-time connection/TLS setup cost on its first
    call — measured at ~5s cold vs ~0.5s warm — so this moves that cost to
    boot instead of the first real caller's turn. Best-effort: a failure
    here just means the first real call pays the cold-start cost instead,
    so it must never block startup.
    """
    try:
        await retrieve("warm up", top_k=1)
    except Exception as exc:
        print(f"[rag] warm-up failed (non-fatal): {exc}")
