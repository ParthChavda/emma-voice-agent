#!/usr/bin/env python3
"""
Standalone script: wipe and rebuild the entire Qdrant knowledge base.

Why this exists: app.services.rag.ensure_ingested() only (re-)embeds
app/knowledge_base/*.md when the Qdrant collection is completely empty
(points_count == 0). Once seeded, editing a knowledge_base file has NO
EFFECT on retrieval until the collection is rebuilt -- and re-running
ingest_docs() directly would just upsert new points alongside the old
ones (it never deletes stale points for a source), producing duplicates.
scripts/seed_qdrant_schedules.py already does delete-then-reupsert for
doctor_schedules.md alone; this script does the same for every file at
once by dropping the whole collection first.

Usage:
    source venv/bin/activate
    python scripts/reseed_knowledge_base.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.rag import COLLECTION, get_qdrant, ingest_docs


async def main() -> None:
    client = get_qdrant()
    existing = {c.name for c in (await client.get_collections()).collections}
    if COLLECTION in existing:
        await client.delete_collection(COLLECTION)
        print(f"Deleted existing '{COLLECTION}' collection.")

    await ingest_docs()

    info = await client.get_collection(COLLECTION)
    print(f"Re-ingested knowledge base: {info.points_count} chunks in '{COLLECTION}'.")


if __name__ == "__main__":
    asyncio.run(main())
