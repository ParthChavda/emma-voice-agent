import difflib
from datetime import date

from app.db import get_pool

# STT transcribes dates far more reliably than proper nouns — "Alice" commonly
# comes back as "Elias" or similar. DOB is the primary identity key; the name
# only needs to be a plausible match, not an exact one.
NAME_MATCH_THRESHOLD = 0.5


async def find_patient(full_name: str, date_of_birth: str) -> dict | None:
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, full_name, date_of_birth, phone FROM patients WHERE date_of_birth = $1",
            date.fromisoformat(date_of_birth),
        )
    if not rows:
        return None

    best_row = None
    best_score = 0.0
    for row in rows:
        score = difflib.SequenceMatcher(None, row["full_name"].lower(), full_name.lower()).ratio()
        if score > best_score:
            best_score = score
            best_row = row

    if best_row is None or best_score < NAME_MATCH_THRESHOLD:
        return None

    return {
        "id": best_row["id"],
        "full_name": best_row["full_name"],
        "date_of_birth": best_row["date_of_birth"].isoformat(),
        "phone": best_row["phone"],
    }
