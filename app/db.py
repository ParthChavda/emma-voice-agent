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
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                id              SERIAL PRIMARY KEY,
                full_name       TEXT NOT NULL,
                date_of_birth   DATE NOT NULL,
                phone           TEXT,
                created_at      TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS patients_name_dob_idx
            ON patients (full_name, date_of_birth)
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS slots (
                id                SERIAL PRIMARY KEY,
                doctor_name       TEXT NOT NULL,
                appointment_type  TEXT NOT NULL,
                start_time        TIMESTAMPTZ NOT NULL,
                is_booked         BOOLEAN NOT NULL DEFAULT FALSE
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS slots_lookup_idx
            ON slots (appointment_type, is_booked, start_time)
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS appointments (
                id          SERIAL PRIMARY KEY,
                patient_id  INTEGER NOT NULL REFERENCES patients(id),
                slot_id     INTEGER NOT NULL REFERENCES slots(id),
                status      TEXT NOT NULL DEFAULT 'booked',
                ref         TEXT NOT NULL,
                created_at  TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS prescriptions (
                id              SERIAL PRIMARY KEY,
                patient_id      INTEGER NOT NULL REFERENCES patients(id),
                medication_name TEXT NOT NULL,
                status          TEXT NOT NULL DEFAULT 'requested',
                ref             TEXT NOT NULL,
                created_at      TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS call_summaries (
                id              SERIAL PRIMARY KEY,
                call_sid        TEXT NOT NULL,
                patient_name    TEXT,
                intent          TEXT,
                key_details     TEXT,
                escalation_flag BOOLEAN NOT NULL DEFAULT FALSE,
                next_action     TEXT,
                call_duration   DOUBLE PRECISION,
                created_at      TIMESTAMPTZ DEFAULT NOW()
            )
        """)


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("Database pool not initialised — call init_pool() first")
    return _pool


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


async def save_call_summary(summary: dict) -> None:
    pool = get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO call_summaries "
            "(call_sid, patient_name, intent, key_details, escalation_flag, next_action, call_duration) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7)",
            summary["call_sid"],
            summary["patient_name"],
            summary["intent"],
            summary["key_details"],
            summary["escalation_flag"],
            summary["next_action"],
            summary["call_duration"],
        )
