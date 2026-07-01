import uuid

from app.db import get_pool


async def request_repeat(patient_id: int, medication_name: str) -> dict:
    pool = get_pool()
    ref = f"RX-{uuid.uuid4().hex[:6].upper()}"
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO prescriptions (patient_id, medication_name, ref) "
            "VALUES ($1, $2, $3)",
            patient_id,
            medication_name,
            ref,
        )
    return {"ref": ref}
