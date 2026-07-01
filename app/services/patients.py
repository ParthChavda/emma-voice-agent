from datetime import date

from app.db import get_pool


async def find_patient(full_name: str, date_of_birth: str) -> dict | None:
    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, full_name, date_of_birth, phone FROM patients "
            "WHERE lower(full_name) = lower($1) AND date_of_birth = $2",
            full_name,
            date.fromisoformat(date_of_birth),
        )
    if row is None:
        return None
    return {
        "id": row["id"],
        "full_name": row["full_name"],
        "date_of_birth": row["date_of_birth"].isoformat(),
        "phone": row["phone"],
    }
