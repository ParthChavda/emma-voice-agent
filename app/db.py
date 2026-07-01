import asyncpg

_pool: asyncpg.Pool | None = None


async def init_pool(dsn: str) -> None:
    global _pool
    if _pool is not None:
        return
    _pool = await asyncpg.create_pool(dsn)
    async with _pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id          SERIAL PRIMARY KEY,
                session_id  TEXT NOT NULL,
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                created_at  TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS conversations_session_idx
            ON conversations (session_id, created_at)
        """)


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


async def load_history(session_id: str) -> list[dict[str, str]]:
    if _pool is None:
        raise RuntimeError("Database pool not initialised — call init_pool() first")
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT role, content FROM conversations "
            "WHERE session_id = $1 ORDER BY created_at",
            session_id,
        )
    return [{"role": r["role"], "content": r["content"]} for r in rows]


async def save_turn(session_id: str, role: str, content: str) -> None:
    if _pool is None:
        raise RuntimeError("Database pool not initialised — call init_pool() first")
    async with _pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO conversations (session_id, role, content) VALUES ($1, $2, $3)",
            session_id,
            role,
            content,
        )
