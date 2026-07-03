import uuid
from datetime import date, datetime, time, timedelta, timezone

import parsedatetime

from app.db import get_pool

_calendar = parsedatetime.Calendar()

SLOT_MINUTES = 30

# Matches app/knowledge_base/practice_info.md's Opening Hours table.
# weekday(): Monday=0 ... Sunday=6.
_OPENING_HOURS: dict[int, tuple[time, time] | None] = {
    0: (time(8, 0), time(18, 30)),
    1: (time(8, 0), time(18, 30)),
    2: (time(8, 0), time(18, 30)),
    3: (time(8, 0), time(18, 30)),
    4: (time(8, 0), time(18, 30)),
    5: (time(9, 0), time(12, 0)),
    6: None,
}


def parse_preferred_time(text: str, now: datetime) -> datetime | None:
    """Parses free-text like "tomorrow at 3pm", "next Tuesday afternoon", or
    "in three days at 10am" into a UTC datetime. parsedatetime (not
    dateutil — verified dateutil doesn't understand relative phrases like
    "tomorrow" at all, silently defaulting to today) handles these directly.
    Never raises — status 0 means it couldn't confidently extract anything,
    which becomes None, so a caller's unusual phrasing turns into a normal
    "please clarify" conversational turn instead of a crashed tool call.
    """
    try:
        parsed, status = _calendar.parseDT(text, sourceTime=now, tzinfo=timezone.utc)
    except (ValueError, OverflowError, TypeError):
        return None
    if status == 0:
        return None
    return parsed


def is_within_opening_hours(dt: datetime) -> bool:
    hours = _OPENING_HOURS.get(dt.weekday())
    if hours is None:
        return False
    start_time, end_time = hours
    return start_time <= dt.time() < end_time


def round_to_slot(dt: datetime) -> datetime:
    floored_minute = (dt.minute // SLOT_MINUTES) * SLOT_MINUTES
    return dt.replace(minute=floored_minute, second=0, microsecond=0)


def is_date_supported(dt: datetime, now: datetime) -> bool:
    """Slots are only pre-generated for today and tomorrow (see
    ensure_slots_for_days) — anything further out has no row to book
    against, so check this before even looking at the slots table."""
    return dt.date() in (now.date(), (now + timedelta(days=1)).date())


async def ensure_slots_for_days(days: list[date]) -> None:
    """Idempotently generates half-hour slot rows for the given days, based
    on opening hours. Safe to call repeatedly — ON CONFLICT DO NOTHING means
    re-running this never resets an already-booked slot back to available.
    """
    times: list[datetime] = []
    for day in days:
        hours = _OPENING_HOURS.get(day.weekday())
        if hours is None:
            continue
        start_time, end_time = hours
        current = datetime.combine(day, start_time, tzinfo=timezone.utc)
        end = datetime.combine(day, end_time, tzinfo=timezone.utc)
        while current < end:
            times.append(current)
            current += timedelta(minutes=SLOT_MINUTES)

    if not times:
        return
    pool = get_pool()
    async with pool.acquire() as conn:
        await conn.executemany(
            "INSERT INTO slots (slot_time) VALUES ($1) ON CONFLICT (slot_time) DO NOTHING",
            [(t,) for t in times],
        )


async def book_slot_and_create_appointment(
    patient_name: str, phone_number: str, service: str, slot: datetime
) -> dict | None:
    """Atomically claims the slot and creates the appointment row in one
    transaction. Returns None if the slot doesn't exist (outside the
    pre-generated window) or was already booked — including by a concurrent
    request that claimed it between this caller's availability check and
    this call, which the old exact-datetime-match approach against
    `appointments` had no protection against at all.
    """
    pool = get_pool()
    appointment_id = uuid.uuid4()
    async with pool.acquire() as conn:
        async with conn.transaction():
            claimed = await conn.execute(
                "UPDATE slots SET is_booked = TRUE WHERE slot_time = $1 AND is_booked = FALSE",
                slot,
            )
            if claimed == "UPDATE 0":
                return None
            row = await conn.fetchrow(
                "INSERT INTO appointments (id, patient_name, phone_number, service, appointment_time) "
                "VALUES ($1, $2, $3, $4, $5) "
                "RETURNING id, patient_name, phone_number, service, appointment_time, status",
                appointment_id,
                patient_name,
                phone_number,
                service,
                slot,
            )
    return dict(row)
