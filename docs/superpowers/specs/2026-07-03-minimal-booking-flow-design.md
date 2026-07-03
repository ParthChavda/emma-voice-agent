# Minimal real booking flow — design

> **Superseded, 2026-07-03 (later same day):** the "explicitly out of scope"
> section below ruled out a pre-seeded `slots` table and concurrency-safe
> booking, and rule 9 originally required reading details back for
> confirmation before booking. Both were revisited per user request: a real
> `slots` table (half-hour increments, generated for today/tomorrow from
> opening hours) now backs `book_appointment`, claimed atomically inside a
> transaction to close the double-booking race this doc calls out as
> acceptable-for-POC below, and the confirmation step was removed from the
> prompt as too repetitive. See `app/services/appointments.py`
> (`is_date_supported`, `ensure_slots_for_days`,
> `book_slot_and_create_appointment`) and `app/core/prompts.py` rule 9 for
> the current implementation — the module/function names and code samples
> below reflect the original (superseded) design, not what's running now.

## Context

Emma previously had a full booking system (`patients`/`slots`/`appointments` tables,
identity verification by name+DOB) that was deliberately removed earlier in this
project's development for being out of scope and, per user feedback, for lacking
input validation and clear separation of concerns — a bad combination for a
voice interface, where a thrown validation error has no good way to surface to
the caller.

This spec adds back a deliberately minimal, real (non-mocked) booking capability,
built with voice-safe error handling from the start: no code path in the new
tool may raise an exception into the conversation flow. Every "bad" outcome
(unparseable time, outside opening hours, slot already taken) is a normal,
structured return value that the existing tool-calling mechanism already knows
how to hand back to the model for natural-language phrasing.

## Goals

- Caller can book a new appointment by giving their name, phone number, the
  service/appointment type, and a preferred date/time in any spoken format.
- No validation error ever crashes or breaks the call.
- Zero regression to existing, already-measured latency (RAG/FAQ turns,
  `check_test_results`, `escalate_human`, `escalate_urgent` are untouched).
- Follows existing code conventions exactly — this is an extension, not a
  rewrite.

## Explicitly out of scope

- Reschedule / cancel (still defers to human handoff, per existing rule 9,
  narrowed rather than removed)
- Patient identity verification / lookup (no DOB, no `patients` table — every
  booking is accepted as a new request under the given name+phone)
- Concurrency-safe double-booking prevention beyond a simple existence check
  (acceptable for a POC's traffic level; would need a DB constraint/transaction
  for real concurrent load)
- A pre-seeded `slots` table (slots are computed algorithmically, not stored)

## Data model

New table in `app/db.py`, alongside `conversations`/`call_summaries`:

```sql
CREATE TABLE IF NOT EXISTS appointments (
    id                UUID PRIMARY KEY,
    patient_name      TEXT NOT NULL,
    phone_number      TEXT NOT NULL,
    service           TEXT NOT NULL,
    appointment_time  TIMESTAMPTZ NOT NULL,
    status            TEXT NOT NULL DEFAULT 'booked',
    created_at        TIMESTAMPTZ DEFAULT NOW(),
    updated_at        TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS appointments_time_idx ON appointments (appointment_time);
```

`id` is generated in Python (`uuid.uuid4()`) and passed in on insert, not a
DB-side default — `gen_random_uuid()` needs the `pgcrypto` extension enabled,
which this codebase doesn't otherwise depend on. Matches the existing
convention already used elsewhere (`rag.py`'s Qdrant point IDs, the old
appointments.py's ref numbers) of generating UUIDs application-side.

`service` is one of the values already used by the old system's tool schema:
`routine`, `urgent`, `telephone`, `nurse` (matches `doctor_schedules.md`'s
existing vocabulary).

## New service module: `app/services/appointments.py`

Mirrors the shape of the old (removed) module, but simpler:

- `parse_preferred_time(text: str, now: datetime) -> datetime | None` — uses
  `parsedatetime.Calendar().parseDT(text, sourceTime=now, tzinfo=timezone.utc)`
  wrapped in `try/except (ValueError, OverflowError, TypeError)`, treating a
  returned `status == 0` (parsedatetime's "couldn't extract anything"
  signal) as `None`. **Correction from the original plan:** `dateutil.parser`
  was tried first but verification caught it doesn't understand relative
  phrases ("tomorrow", "in three days") at all — it silently defaults to
  the current day instead. `parsedatetime` is purpose-built for exactly
  this and was confirmed correct against every phrasing tested. This
  function is directly responsible for the "never breaks on a weird
  format" requirement.
- `is_within_opening_hours(dt: datetime) -> bool` — checks the parsed time's
  weekday + time-of-day against a static opening-hours table (Mon–Fri
  8:00–18:30, Sat 9:00–12:00, Sun none, from `practice_info.md`). No slot
  enumeration needed for this — a direct boolean check is simpler (YAGNI).
- `round_to_slot(dt: datetime) -> datetime` — rounds down to the nearest
  30-minute boundary.
- `async def is_slot_taken(slot: datetime) -> bool` — `SELECT 1 FROM
  appointments WHERE appointment_time = $1 AND status = 'booked'`.
- `async def create_appointment(patient_name, phone_number, service,
  slot: datetime) -> dict` — INSERT, returns the row as a dict.

## Tool wiring in `app/services/llm_openai.py`

- New entry in `TOOLS`: `book_appointment`, parameters `patient_name` (string),
  `phone_number` (string), `service` (enum: routine/urgent/telephone/nurse),
  `preferred_time` (string, free text — explicitly documented in the
  function description as accepting any natural phrasing, not a required
  format).
- New async handler registered in `ASYNC_HANDLERS["book_appointment"]`
  (currently an empty dict — this is its first real entry):

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
      return {"status": "booked", "appointment_time": row["appointment_time"].isoformat(),
              "ref": str(row["id"])[:8]}
  ```

- `book_appointment` is **not** added to `MOCK_REPLY_TEMPLATES` — its result is
  genuinely variable (confirmed time, or one of three different decline
  reasons), so it correctly falls through to the existing second-completion
  synthesis path that already handles any non-templated tool. No change
  needed to that mechanism — it was already built to support this case.

## Prompt change in `app/core/prompts.py`

Rule 9 currently blocks all appointment actions. Narrow it to distinguish new
bookings (now possible) from reschedule/cancel (still not):

> 9. You can book a NEW appointment yourself using the book_appointment tool
> — ask for the patient's name, phone number, service needed, and preferred
> date/time, one at a time. You cannot reschedule or cancel an existing
> appointment; for those, offer to transfer to a human receptionist
> immediately.

Also remove/adjust the intro line's blanket "you cannot check appointment
availability or confirm a booking yourself" claim, since it's no longer true
for new bookings.

## Error handling philosophy (the core requirement)

| Failure | Where caught | What happens |
|---|---|---|
| Unparseable date/time text | `parse_preferred_time`, try/except | Returns `None` → tool returns `{"status": "unclear_time"}` → model asks caller to repeat the time |
| Requested time already in the past | plain comparison against `now` | Tool returns `{"status": "time_in_past"}` → model asks for a future date/time |
| Requested time outside opening hours | `is_within_opening_hours`, plain boolean check | Tool returns `{"status": "outside_hours"}` → model explains hours, asks again |
| Slot already booked | `is_slot_taken` query result | Tool returns `{"status": "slot_taken", ...}` → model says so, asks for another time |
| DB unavailable | Not specifically handled — same as every other DB call in this codebase (`db.save_turn`, etc.), which already assume Postgres is up | Out of scope; consistent with existing code's assumptions elsewhere |

No new exception types, no new global error handling — the entire safety
property comes from the handler functions returning structured "reasons"
instead of raising.

## Testing

Since `app/tests/` was intentionally removed for this POC, verification follows
the pattern established this session: manual scripted verification via
`python -c` snippets exercising the real function against real Postgres, plus
adding representative new cases to `scripts/eval_emma.py` covering: a
successful booking, an out-of-hours request, a double-booked slot, and an
unparseable time phrase.
