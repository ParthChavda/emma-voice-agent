import uuid
from datetime import datetime, time, timezone

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


async def is_slot_taken(slot: datetime) -> bool:
    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT 1 FROM appointments WHERE appointment_time = $1 AND status = 'booked'",
            slot,
        )
    return row is not None


async def create_appointment(
    patient_name: str, phone_number: str, service: str, slot: datetime
) -> dict:
    pool = get_pool()
    appointment_id = uuid.uuid4()
    async with pool.acquire() as conn:
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
