import uuid

from app.db import get_pool


async def list_available_slots(appointment_type: str, limit: int = 3) -> list[dict]:
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, doctor_name, appointment_type, start_time FROM slots "
            "WHERE appointment_type = $1 AND is_booked = FALSE "
            "ORDER BY start_time LIMIT $2",
            appointment_type,
            limit,
        )
    return [
        {
            "id": r["id"],
            "doctor_name": r["doctor_name"],
            "appointment_type": r["appointment_type"],
            "start_time": r["start_time"].isoformat(),
        }
        for r in rows
    ]


async def create_booking(patient_id: int, slot_id: int) -> dict | None:
    pool = get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            slot = await conn.fetchrow(
                "SELECT id, doctor_name, start_time FROM slots "
                "WHERE id = $1 AND is_booked = FALSE FOR UPDATE",
                slot_id,
            )
            if slot is None:
                return None
            ref = f"APT-{uuid.uuid4().hex[:6].upper()}"
            await conn.execute(
                "UPDATE slots SET is_booked = TRUE WHERE id = $1", slot_id
            )
            appt_row = await conn.fetchrow(
                "INSERT INTO appointments (patient_id, slot_id, ref) "
                "VALUES ($1, $2, $3) RETURNING id",
                patient_id,
                slot_id,
                ref,
            )
    return {
        "appointment_id": appt_row["id"],
        "ref": ref,
        "doctor_name": slot["doctor_name"],
        "start_time": slot["start_time"].isoformat(),
    }


async def cancel_booking(appointment_id: int) -> bool:
    pool = get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            appt = await conn.fetchrow(
                "SELECT slot_id, status FROM appointments WHERE id = $1 FOR UPDATE",
                appointment_id,
            )
            if appt is None or appt["status"] == "cancelled":
                return False
            await conn.execute(
                "UPDATE appointments SET status = 'cancelled' WHERE id = $1",
                appointment_id,
            )
            await conn.execute(
                "UPDATE slots SET is_booked = FALSE WHERE id = $1", appt["slot_id"]
            )
    return True
