---
name: qdrant-seed-check
description: Verify Postgres and Qdrant are up and healthy, and safely re-embed the knowledge base into Qdrant after editing any app/knowledge_base/*.md file. Make sure to use this before running the app, running evals, or testing RAG-dependent replies, and whenever the user asks "is qdrant seeded", "do I need to reseed", "why isn't emma using the updated info", or after any knowledge-base content edit (renames, new services, schedule changes) -- even if the user doesn't mention Qdrant by name.
---

## The gotcha this exists to catch

`app.services.rag.ensure_ingested()` -- called on every app startup -- only (re-)embeds `app/knowledge_base/*.md` when the Qdrant collection is completely empty (`points_count == 0`). Once seeded, **editing a knowledge_base file has zero effect on what Emma retrieves** until the collection is rebuilt. There's no error or warning when this happens -- RAG just quietly keeps answering from the old text, which is easy to mistake for a prompt bug instead of a stale-embeddings problem.

Separately, `ingest_docs()` (the function that does the actual embedding) always *upserts* new points and never deletes old ones for a source file -- calling it again after an edit adds duplicate/stale chunks alongside the new ones instead of replacing them. `scripts/seed_qdrant_schedules.py` handles this correctly, but only for `doctor_schedules.md` alone.

## Step 1: check containers are up

```bash
docker-compose ps
```

If postgres/qdrant aren't `Up`/healthy:

```bash
docker-compose up -d postgres qdrant
```

Quick health checks:

```bash
docker-compose exec postgres pg_isready -U emma -d emma
curl -s http://localhost:6333/collections/emma_knowledge
```

## Step 2: decide if a reseed is needed

Reseed if any `app/knowledge_base/*.md` file changed since the Qdrant collection was last built. If unsure, reseeding is cheap and safe to do anyway -- prefer reseeding over guessing.

## Step 3: reseed cleanly

Use `scripts/reseed_knowledge_base.py` (deletes the whole `emma_knowledge` collection, then rebuilds it from every file in `app/knowledge_base/` -- the safe equivalent of what `seed_qdrant_schedules.py` does for just `doctor_schedules.md`):

```bash
source venv/bin/activate
python scripts/reseed_knowledge_base.py
```

If only `doctor_schedules.md` changed, `scripts/seed_qdrant_schedules.py` is a lighter-weight option (it deletes and re-embeds just that file's points, leaving everything else untouched).

## Step 4: confirm

```bash
curl -s http://localhost:6333/collections/emma_knowledge | python3 -m json.tool
```

Check `points_count` is nonzero and roughly proportional to the number of `## ` section headers across all knowledge_base files. The app (or `run-emma-evals`) will pick up the new content on its next RAG retrieval automatically -- no restart needed, since `retrieve()` queries Qdrant directly on every turn.
