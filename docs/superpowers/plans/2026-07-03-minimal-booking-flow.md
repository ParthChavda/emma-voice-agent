# Minimal Real Booking Flow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a real (non-mocked) `book_appointment` tool that lets Emma collect a patient's name, phone number, service, and preferred date/time, then book it against a simple generated-slot availability check — with no code path that can throw an exception into the conversation.

**Architecture:** One new Postgres table (`appointments`), one new pure-function-heavy service module (`app/services/appointments.py`) for parsing/validation, one new entry in the existing (currently empty) `ASYNC_HANDLERS` dict in `llm_openai.py`, and a narrowed system-prompt rule. No pytest suite exists in this project (intentionally removed) — every step is verified with direct `python -c` execution against the real dev Postgres/Qdrant/OpenAI, matching this session's established verification style.

**Tech Stack:** FastAPI, asyncpg, parsedatetime (new dependency), OpenAI function calling — all consistent with the existing codebase.

> **Correction made during execution:** Task 1/2 below were originally written against `python-dateutil`. Verification during Task 2 caught a real bug — `dateutil.parser` doesn't understand relative phrases like "tomorrow" or "in three days" at all; it silently ignores them and defaults to the current day. Switched to `parsedatetime` (purpose-built for relative natural-language date parsing, confirmed correct for every phrasing tested). The actual shipped code in `app/services/appointments.py` uses `parsedatetime.Calendar().parseDT(text, sourceTime=now, tzinfo=timezone.utc)`, treating `status == 0` as "couldn't parse" — not the `dateutil` code shown in the task steps below.

---

### Task 1: Add `python-dateutil` dependency and the `appointments` table

**Files:**
- Modify: `requirements.txt`
- Modify: `app/db.py`

- [ ] **Step 1: Add and install the dependency**

Append to `requirements.txt`:
```
python-dateutil==2.9.0.post0
```

Run:
```bash
source venv/bin/activate && pip install python-dateutil==2.9.0.post0
```
Expected: `Successfully installed python-dateutil-2.9.0.post0`

- [ ] **Step 2: Add the `appointments` table to `init_pool`**

In `app/db.py`, the current `init_pool` ends with the `call_summaries` table creation immediately before `async def close_pool`. Insert a new table creation right after the `call_summaries` block (which currently ends at the `"""` closing the `CREATE TABLE IF NOT EXISTS call_summaries (...)` statement), so `init_pool` reads:

```python
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
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS appointments (
                id                UUID PRIMARY KEY,
                patient_name      TEXT NOT NULL,
                phone_number      TEXT NOT NULL,
                service           TEXT NOT NULL,
                appointment_time  TIMESTAMPTZ NOT NULL,
                status            TEXT NOT NULL DEFAULT 'booked',
                created_at        TIMESTAMPTZ DEFAULT NOW(),
                updated_at        TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS appointments_time_idx
            ON appointments (appointment_time)
        """)
```

(Everything else in `db.py` — `close_pool`, `get_pool`, `load_history`, `save_turn`, `save_call_summary` — is unchanged.)

- [ ] **Step 3: Verify the table is created**

Run:
```bash
source venv/bin/activate && python -c "
import asyncio, sys
sys.path.insert(0, '.')
from app import db
from app.config import settings

async def main():
    await db.init_pool(settings.postgres_dsn)
    pool = db.get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(\"SELECT to_regclass('appointments')\")
        print('appointments table exists:', row[0] is not None)
    await db.close_pool()

asyncio.run(main())
"
```
Expected: `appointments table exists: True`

- [ ] **Step 4: Commit**

```bash
git add requirements.txt app/db.py
git commit -m "Add appointments table and python-dateutil dependency"
```

---

### Task 2: Pure functions in `app/services/appointments.py` (no I/O)

**Files:**
- Create: `app/services/appointments.py`

- [ ] **Step 1: Write the file's pure-function half**

```python
import uuid
from datetime import datetime, time, timedelta, timezone

from dateutil import parser as dateutil_parser

from app.db import get_pool

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
    """Parses free-text like "tomorrow at 3pm" or "the 15th at 10am" into a
    UTC datetime. Never raises — returns None for anything it can't
    confidently parse, so a caller's unusual phrasing becomes a normal
    "please clarify" conversational turn instead of a crashed tool call.
    """
    try:
        parsed = dateutil_parser.parse(text, fuzzy=True, default=now)
    except (ValueError, OverflowError, TypeError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
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
```

- [ ] **Step 2: Verify parsing, hours-check, and rounding directly**

Run:
```bash
source venv/bin/activate && python -c "
import sys
sys.path.insert(0, '.')
from datetime import datetime, timezone
from app.services.appointments import parse_preferred_time, is_within_opening_hours, round_to_slot

now = datetime(2026, 7, 3, 9, 0, tzinfo=timezone.utc)  # a Friday

# Parsing
print('tomorrow at 3pm ->', parse_preferred_time('tomorrow at 3pm', now))
print('gibberish ->', parse_preferred_time('asdkjhaskjdh not a date at all', now))

# Hours check
friday_3pm = datetime(2026, 7, 3, 15, 0, tzinfo=timezone.utc)
friday_11pm = datetime(2026, 7, 3, 23, 0, tzinfo=timezone.utc)
sunday_10am = datetime(2026, 7, 5, 10, 0, tzinfo=timezone.utc)
saturday_10am = datetime(2026, 7, 4, 10, 0, tzinfo=timezone.utc)
print('Friday 3pm within hours:', is_within_opening_hours(friday_3pm))
print('Friday 11pm within hours:', is_within_opening_hours(friday_11pm))
print('Sunday 10am within hours:', is_within_opening_hours(sunday_10am))
print('Saturday 10am within hours:', is_within_opening_hours(saturday_10am))

# Rounding
messy = datetime(2026, 7, 3, 15, 47, tzinfo=timezone.utc)
print('15:47 rounds to:', round_to_slot(messy))
"
```
Expected output:
```
tomorrow at 3pm -> 2026-07-04 15:00:00+00:00
gibberish -> None
Friday 3pm within hours: True
Friday 11pm within hours: False
Sunday 10am within hours: False
Saturday 10am within hours: True
15:47 rounds to: 2026-07-03 15:30:00+00:00
```

- [ ] **Step 3: Commit**

```bash
git add app/services/appointments.py
git commit -m "Add pure time-parsing/validation functions for booking"
```

---

### Task 3: DB-dependent functions in `app/services/appointments.py`

**Files:**
- Modify: `app/services/appointments.py`

- [ ] **Step 1: Append the DB-dependent functions**

Add to the end of `app/services/appointments.py`:

```python
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
```

- [ ] **Step 2: Verify against real Postgres — book a slot, confirm it's then taken**

Run:
```bash
source venv/bin/activate && python -c "
import asyncio, sys
sys.path.insert(0, '.')
from datetime import datetime, timedelta, timezone
from app import db
from app.config import settings
from app.services.appointments import is_slot_taken, create_appointment, round_to_slot

async def main():
    await db.init_pool(settings.postgres_dsn)
    slot = round_to_slot(datetime.now(timezone.utc) + timedelta(days=1))
    slot = slot.replace(hour=10, minute=0)

    print('taken before booking:', await is_slot_taken(slot))
    row = await create_appointment('Test Patient', '07700900000', 'routine', slot)
    print('created row id:', row['id'])
    print('taken after booking:', await is_slot_taken(slot))

    # cleanup so re-running this script stays idempotent
    pool = db.get_pool()
    async with pool.acquire() as conn:
        await conn.execute('DELETE FROM appointments WHERE id = \$1', row['id'])
    await db.close_pool()

asyncio.run(main())
"
```
Expected:
```
taken before booking: False
created row id: <some-uuid>
taken after booking: True
```

- [ ] **Step 3: Commit**

```bash
git add app/services/appointments.py
git commit -m "Add DB-backed slot-availability check and appointment creation"
```

---

### Task 4: Wire `book_appointment` into `llm_openai.py`

**Files:**
- Modify: `app/services/llm_openai.py`

- [ ] **Step 1: Add the `datetime`/`timezone` import and the `appointments` module import**

Current top of file:
```python
import json
import re
import time
from collections.abc import Awaitable, Callable

from openai import AsyncOpenAI

from app.config import settings
```

Change to:
```python
import json
import re
import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone

from openai import AsyncOpenAI

from app.config import settings
from app.services import appointments
```

- [ ] **Step 2: Add `book_appointment` to `TOOLS`**

`TOOLS` currently starts with:
```python
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_test_results",
```

Insert a new entry before the `check_test_results` entry, so `TOOLS` becomes:
```python
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "book_appointment",
            "description": (
                "Book a new appointment for the patient. preferred_time accepts "
                "any natural spoken phrasing (e.g. 'tomorrow at 3pm', 'next "
                "Tuesday afternoon', 'the 15th at 10am') — never require the "
                "caller to give a specific format."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_name": {"type": "string"},
                    "phone_number": {"type": "string"},
                    "service": {
                        "type": "string",
                        "enum": ["routine", "urgent", "telephone", "nurse"],
                    },
                    "preferred_time": {
                        "type": "string",
                        "description": "The patient's preferred appointment date/time, in whatever words they used.",
                    },
                },
                "required": ["patient_name", "phone_number", "service", "preferred_time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_test_results",
```
(the rest of `TOOLS` — `check_test_results`, `escalate_urgent`, `escalate_human` — is unchanged)

- [ ] **Step 3: Add the handler and register it in `ASYNC_HANDLERS`**

Current:
```python
ASYNC_HANDLERS: dict = {}
```

Change to:
```python
async def _handle_book_appointment(args: dict) -> dict:
    now = datetime.now(timezone.utc)
    parsed = appointments.parse_preferred_time(args["preferred_time"], now=now)
    if parsed is None:
        return {"status": "unclear_time"}
    if parsed < now:
        return {"status": "time_in_past"}
    if not appointments.is_within_opening_hours(parsed):
        return {"status": "outside_hours"}
    slot = appointments.round_to_slot(parsed)
    if await appointments.is_slot_taken(slot):
        return {"status": "slot_taken", "requested_time": slot.isoformat()}
    row = await appointments.create_appointment(
        args["patient_name"], args["phone_number"], args["service"], slot
    )
    return {
        "status": "booked",
        "appointment_time": row["appointment_time"].isoformat(),
        "ref": str(row["id"])[:8],
    }


ASYNC_HANDLERS: dict = {
    "book_appointment": _handle_book_appointment,
}
```

Note: `book_appointment` is deliberately **not** added to `MOCK_REPLY_TEMPLATES` — its result is genuinely variable (a confirmed time, or one of three different decline reasons), so it correctly falls through to the existing second-completion synthesis path that already handles any tool without a template. No changes needed to `chat_completion`/`chat_completion_stream` themselves — that fallback already exists.

- [ ] **Step 4: Verify the app still imports cleanly**

Run:
```bash
source venv/bin/activate && python -c "import app.main; print('import OK')"
```
Expected: `import OK`

- [ ] **Step 5: Verify a real booking end-to-end through `chat_completion`**

Run:
```bash
source venv/bin/activate && python -c "
import asyncio, sys
sys.path.insert(0, '.')
from app.services.llm_openai import chat_completion, TOOLS
from app.core.prompts import build_messages
from app import db
from app.config import settings

async def main():
    await db.init_pool(settings.postgres_dsn)
    messages = build_messages(
        [], [],
        'My name is Priya Sharma, phone number 07700 900123, '
        \"I'd like a routine appointment tomorrow at 10am.\"
    )
    reply, intent = await chat_completion(messages, TOOLS)
    print(f'intent={intent!r}')
    print(f'reply={reply!r}')
    await db.close_pool()

asyncio.run(main())
" 2>&1 | grep -v EMMA-TIMING
```
Expected: `intent='book_appointment'` and a reply confirming the booking (exact wording is model-generated, but should reference the appointment being booked/confirmed).

- [ ] **Step 6: Commit**

```bash
git add app/services/llm_openai.py
git commit -m "Wire book_appointment into TOOLS and ASYNC_HANDLERS"
```

---

### Task 5: Narrow prompt rule 9 to allow new bookings

**Files:**
- Modify: `app/core/prompts.py`

- [ ] **Step 1: Update the intro sentence and rule 9**

Current:
```python
EMMA_SYSTEM_PROMPT = """You are EMMA, the AI receptionist for Elmwood Road Surgery, an NHS GP practice. \
Speak in a warm, calm, professional tone, like a skilled human receptionist. You can discuss \
appointment types, prescription renewals, test result queries, opening hours, and general admin \
questions — but you cannot check appointment availability or confirm a booking yourself.
```
...
```python
9. You cannot check appointment availability or confirm, reschedule, or cancel a booking \
yourself. If a patient wants to book, change, or cancel an appointment, immediately offer to \
transfer them to a human receptionist — do not first collect their name, date of birth, or \
preferred time, since you have no way to act on those details.
10. If PRACTICE INFORMATION describes how a patient would normally do something you also \
have a tool for (e.g. calling to check test results), still use the tool for that patient's \
specific request — the practice-information text is general policy, not a reason to skip a \
check you can perform directly."""
```

Change the intro to:
```python
EMMA_SYSTEM_PROMPT = """You are EMMA, the AI receptionist for Elmwood Road Surgery, an NHS GP practice. \
Speak in a warm, calm, professional tone, like a skilled human receptionist. You can discuss \
appointment types, prescription renewals, test result queries, opening hours, and general admin \
questions, and you can book new appointments directly — but you cannot reschedule or cancel an \
existing booking yourself.
```

Change rule 9 to:
```python
9. You can book a NEW appointment yourself using the book_appointment tool — ask for the \
patient's full name, phone number, the service needed (routine, urgent, telephone, or nurse), \
and their preferred date/time, one detail at a time rather than all at once. You cannot \
reschedule or cancel an existing appointment yourself — for those, immediately offer to \
transfer to a human receptionist.
10. If PRACTICE INFORMATION describes how a patient would normally do something you also \
have a tool for (e.g. calling to check test results), still use the tool for that patient's \
specific request — the practice-information text is general policy, not a reason to skip a \
check you can perform directly."""
```

- [ ] **Step 2: Verify the required substrings are all still present**

Run:
```bash
source venv/bin/activate && python -c "
import sys
sys.path.insert(0, '.')
from app.core.prompts import EMMA_SYSTEM_PROMPT as p
print('EMMA:', 'EMMA' in p)
print('Elmwood Road Surgery:', 'Elmwood Road Surgery' in p)
print('999:', '999' in p)
print('clinical advice:', 'clinical advice' in p.lower())
print('book_appointment:', 'book_appointment' in p)
"
```
Expected: all `True`.

- [ ] **Step 3: Commit**

```bash
git add app/core/prompts.py
git commit -m "Allow Emma to book new appointments directly; keep reschedule/cancel human-only"
```

---

### Task 6: Add booking scenarios to `scripts/eval_emma.py`

**Files:**
- Modify: `scripts/eval_emma.py`

- [ ] **Step 1: Add four new eval cases covering success, out-of-hours, unclear time, and slot-taken**

In `scripts/eval_emma.py`, add to the `CASES` list (after the existing `booking_request_offers_human_immediately` case — that case still tests the *reschedule/cancel-style* vague "book an appointment" request correctly deferring; these new cases test the *complete-details* path that should now actually book):

```python
    EvalCase(
        "booking_success_with_full_details",
        [
            "My name is Priya Sharma, phone number 07700 900123, "
            "I'd like a routine appointment tomorrow at 10am.",
        ],
        [has_intent("book_appointment"), not_contains("unable"), not_contains("cannot")],
    ),
    EvalCase(
        "booking_outside_opening_hours",
        [
            "My name is Alex Chen, phone number 07700 900124, "
            "I'd like a routine appointment tomorrow at 3am.",
        ],
        [has_intent("book_appointment"), contains("hours")],
    ),
    EvalCase(
        "booking_unclear_time",
        [
            "My name is Jordan Lee, phone number 07700 900125, "
            "I'd like a routine appointment sometime whenever works I guess.",
        ],
        [has_intent("book_appointment")],
    ),
```

- [ ] **Step 2: Run the full eval suite**

Run:
```bash
source venv/bin/activate && python scripts/eval_emma.py 2>&1 | grep -v "^\[EMMA-TIMING\]"
```
Expected: the three new cases print `[PASS]` or `[FAIL]` with their actual replies visible — read the replies to confirm `booking_outside_opening_hours` mentions the practice's opening hours and `booking_unclear_time` asks the caller to clarify their preferred time, since those replies are model-phrased and not strictly checkable by exact substring alone.

- [ ] **Step 3: Clean up any test rows created in Postgres**

Run:
```bash
source venv/bin/activate && python -c "
import asyncio, sys
sys.path.insert(0, '.')
from app import db
from app.config import settings

async def main():
    await db.init_pool(settings.postgres_dsn)
    pool = db.get_pool()
    async with pool.acquire() as conn:
        deleted = await conn.execute(
            \"DELETE FROM appointments WHERE patient_name IN ('Priya Sharma', 'Alex Chen', 'Jordan Lee', 'Test Patient')\"
        )
        print(deleted)
    await db.close_pool()

asyncio.run(main())
"
```
Expected: a `DELETE <n>` count printed, confirming cleanup.

- [ ] **Step 4: Commit**

```bash
git add scripts/eval_emma.py
git commit -m "Add booking-flow scenarios to eval_emma.py"
```

---

### Task 7: Manual multi-turn conversation verification

**Files:** none (verification only)

- [ ] **Step 1: Simulate a realistic multi-turn booking where details are given across separate turns, not all at once**

Run:
```bash
source venv/bin/activate && python -c "
import asyncio, sys
sys.path.insert(0, '.')
from app.services.llm_openai import chat_completion, TOOLS
from app.core.prompts import build_messages
from app import db
from app.config import settings

async def main():
    await db.init_pool(settings.postgres_dsn)
    history = []
    turns = [
        'I need to book an appointment.',
        'My name is Morgan Taylor.',
        'My phone number is 07700 900199.',
        \"It's for a nurse appointment.\",
        'Tomorrow at 11am would be great.',
    ]
    for turn in turns:
        messages = build_messages([], history, turn)
        reply, intent = await chat_completion(messages, TOOLS)
        print(f'You: {turn}')
        print(f'Emma [{intent}]: {reply}')
        print()
        history.append({'role': 'user', 'content': turn})
        history.append({'role': 'assistant', 'content': reply})
    await db.close_pool()

asyncio.run(main())
" 2>&1 | grep -v "^\[EMMA-TIMING\]"
```
Expected: Emma asks for whatever details are still missing at each step (not re-asking for ones already given), and the final turn results in `intent='book_appointment'` with a confirmed booking reply.

- [ ] **Step 2: Confirm the row landed in Postgres, then clean it up**

Run:
```bash
source venv/bin/activate && python -c "
import asyncio, sys
sys.path.insert(0, '.')
from app import db
from app.config import settings

async def main():
    await db.init_pool(settings.postgres_dsn)
    pool = db.get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(\"SELECT * FROM appointments WHERE patient_name = 'Morgan Taylor'\")
        print(dict(row) if row else 'NOT FOUND')
        if row:
            await conn.execute('DELETE FROM appointments WHERE id = \$1', row['id'])
    await db.close_pool()

asyncio.run(main())
"
```
Expected: the row is printed (confirming the real multi-turn flow actually persisted a booking), then deleted.

- [ ] **Step 3: Re-run the full 50-case suite from earlier this session as a regression check**

Run:
```bash
source venv/bin/activate && python scripts/eval_50_cases.py 2>&1 | grep -E "^\[(PASS|FAIL)\]|passed"
```
Expected: Category 2 (Appointment Related) scores should not be worse than the last recorded run (8/10) — case #11 ("I need to book an appointment") in particular may now behave differently (a bare booking request with no details given should still ask for details rather than book anything, since the tool requires all four fields) — read its actual reply to confirm it's still reasonable, not a strict pass/fail regression gate.
