#!/usr/bin/env python3
"""
Standalone script: seed Postgres with dummy patients and appointment slots for local testing.

Usage:
    source venv/bin/activate
    python scripts/seed_data.py

Truncates patients/slots/appointments/prescriptions first, so it's safe to re-run.
Requires Postgres to be reachable via the DSN in app.config.settings (POSTGRES_* env vars).
"""
import asyncio
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import db
from app.config import settings

PATIENTS = [
    ("Alice Smith", date(1990, 5, 20), "07700900001"),
    ("Bob Jones", date(1985, 2, 14), "07700900002"),
    ("Carol White", date(1978, 11, 3), "07700900003"),
    ("David Brown", date(2001, 7, 9), "07700900004"),
    ("Emma Davies", date(1995, 3, 30), "07700900005"),
]

DOCTORS = [
    ("Dr. Ahmed", "routine"),
    ("Dr. Ahmed", "telephone"),
    ("Dr. Chen", "routine"),
    ("Dr. Chen", "urgent"),
    ("Nurse Patel", "nurse"),
]


def _next_weekday_mornings(days_ahead: int, count: int) -> list[datetime]:
    """Return `count` 30-minute morning slots on weekdays, starting `days_ahead` from today."""
    slots: list[datetime] = []
    current = date.today() + timedelta(days=days_ahead)
    while len(slots) < count:
        if current.weekday() < 5:  # Monday-Friday
            for hour, minute in [(9, 0), (9, 30), (10, 0), (10, 30)]:
                if len(slots) >= count:
                    break
                slots.append(
                    datetime(current.year, current.month, current.day, hour, minute, tzinfo=timezone.utc)
                )
        current += timedelta(days=1)
    return slots


async def main() -> None:
    await db.init_pool(settings.postgres_dsn)
    pool = db.get_pool()

    async with pool.acquire() as conn:
        await conn.execute(
            "TRUNCATE patients, slots, appointments, prescriptions RESTART IDENTITY CASCADE"
        )

        for full_name, dob, phone in PATIENTS:
            await conn.execute(
                "INSERT INTO patients (full_name, date_of_birth, phone) VALUES ($1, $2, $3)",
                full_name,
                dob,
                phone,
            )

        slot_count = 0
        for doctor_name, appointment_type in DOCTORS:
            for start_time in _next_weekday_mornings(days_ahead=1, count=8):
                await conn.execute(
                    "INSERT INTO slots (doctor_name, appointment_type, start_time) "
                    "VALUES ($1, $2, $3)",
                    doctor_name,
                    appointment_type,
                    start_time,
                )
                slot_count += 1

    print(f"Seeded {len(PATIENTS)} patients and {slot_count} slots.")
    await db.close_pool()


if __name__ == "__main__":
    asyncio.run(main())
