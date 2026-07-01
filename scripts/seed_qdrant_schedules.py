#!/usr/bin/env python3
"""
Standalone script: embed app/data/knowledge_base/doctor_schedules.md into Qdrant.

Usage:
    source venv/bin/activate
    python scripts/seed_qdrant_schedules.py

Only embeds doctor_schedules.md — it does not re-touch chunks already ingested
from the other knowledge_base files. Safe to re-run: it first deletes any
previously-seeded points for this source file, then re-embeds it fresh.
"""
import asyncio
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
)

from app.services.rag import COLLECTION, _embed, get_qdrant

SOURCE_FILE = "doctor_schedules.md"
DOCS_DIR = Path("app/data/knowledge_base")


async def main() -> None:
    client = get_qdrant()

    # "source" needs a keyword index before it can be used in a delete filter.
    await client.create_payload_index(
        collection_name=COLLECTION,
        field_name="source",
        field_schema=PayloadSchemaType.KEYWORD,
    )

    await client.delete(
        collection_name=COLLECTION,
        points_selector=Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=SOURCE_FILE))]
        ),
    )

    text = (DOCS_DIR / SOURCE_FILE).read_text()
    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]

    points = []
    for chunk in chunks:
        vector = await _embed(chunk)
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"source": SOURCE_FILE, "text": chunk},
            )
        )

    if points:
        await client.upsert(collection_name=COLLECTION, points=points)

    print(f"Embedded {len(points)} chunks from {SOURCE_FILE} into Qdrant collection '{COLLECTION}'.")


if __name__ == "__main__":
    asyncio.run(main())
