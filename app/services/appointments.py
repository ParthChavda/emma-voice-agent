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


def parse_preferred_time(text: str, now: datetime) -> tuple[datetime | None, bool]:
    """Parses free-text like "tomorrow at 3pm", "next Tuesday afternoon", or
    "in three days at 10am" into a UTC datetime. parsedatetime (not
    dateutil — verified dateutil doesn't understand relative phrases like
    "tomorrow" at all, silently defaulting to today) handles these directly.
    Never raises — status 0 means it couldn't confidently extract anything,
    which becomes None, so a caller's unusual phrasing turns into a normal
    "please clarify" conversational turn instead of a crashed tool call.

    Returns (parsed_datetime, has_explicit_time). parsedatetime's status
    code distinguishes a real time-of-day (status 2 or 3) from a date-only
    result (status 1) — and status 1 silently fills the time with a
    hardcoded 09:00 default that has nothing to do with what the caller
    said (verified directly: "today", "tomorrow", and "today, anytime" all
    return status 1 with parsed defaulting to 09:00:00 regardless of the
    actual current time). has_explicit_time is False in that case so a
    caller who gave no real time preference doesn't get silently booked
    into that meaningless placeholder hour.
    """
    try:
        parsed, status = _calendar.parseDT(text, sourceTime=now, tzinfo=timezone.utc)
    except (ValueError, OverflowError, TypeError):
        return None, False
    if status == 0:
        return None, False
    return parsed, status != 1


def ceil_to_slot(dt: datetime) -> datetime:
    """Rounds up to the next half-hour boundary — the slot starting at or
    after dt, as opposed to round_to_slot's floor (used when the caller did
    state an explicit time and that exact slot is what's being requested)."""
    floored = round_to_slot(dt)
    if floored < dt:
        return floored + timedelta(minutes=SLOT_MINUTES)
    return floored


def is_within_opening_hours(dt: datetime) -> bool:
    hours = _OPENING_HOURS.get(dt.weekday())
    if hours is None:
        return False
    start_time, end_time = hours
    return start_time <= dt.time() < end_time


def round_to_slot(dt: datetime) -> datetime:
    floored_minute = (dt.minute // SLOT_MINUTES) * SLOT_MINUTES
    return dt.replace(minute=floored_minute, second=0, microsecond=0)


# TEMPORARY one-off test override (2026-07-04) — lets the conversational
# booking flow accept Monday 2026-07-06 for a demo/recording, without
# widening the real today/tomorrow-only rule for every other date. Remove
# this constant and its use in is_date_supported once testing is done.
_TEMP_EXTRA_SUPPORTED_DATE = date(2026, 7, 6)


def is_date_supported(dt: datetime, now: datetime) -> bool:
    """Slots are only pre-generated for today and tomorrow (see
    ensure_slots_for_days) — anything further out has no row to book
    against, so check this before even looking at the slots table."""
    return dt.date() in (now.date(), (now + timedelta(days=1)).date(), _TEMP_EXTRA_SUPPORTED_DATE)


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


async def find_next_available_slot(day: date, not_before: datetime) -> datetime | None:
    """Earliest still-open slot on `day` at or after `not_before`, rounded
    up to the next half-hour boundary — used when the caller gave no real
    time preference ("today", "anytime", "whenever works") instead of
    trusting parsedatetime's meaningless 09:00 default. Only rows inside
    opening hours ever exist in `slots` (see ensure_slots_for_days), so no
    separate opening-hours check is needed here. Returns None if every slot
    on that day at or after `not_before` is already booked, or the day is
    over/closed (no rows left to match).
    """
    lower_bound = max(
        datetime.combine(day, time.min, tzinfo=timezone.utc),
        ceil_to_slot(not_before),
    )
    upper_bound = datetime.combine(day, time.max, tzinfo=timezone.utc)
    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT slot_time FROM slots WHERE slot_time >= $1 AND slot_time <= $2 "
            "AND is_booked = FALSE ORDER BY slot_time LIMIT 1",
            lower_bound,
            upper_bound,
        )
    return row["slot_time"] if row else None


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
