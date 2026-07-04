# Challenges Faced & How Latency Is Explained

Interview-prep notes: real problems hit while building this voice AI
receptionist + booking system, how each was actually diagnosed and fixed,
and a ready answer for "how do you think about latency here?"

---

## 1. Challenges & Fixes

### RAG was losing facts it should have retrieved
**Problem:** Chunking the knowledge-base Markdown by blank lines produced
near-empty fragments (a bare heading, one stray sentence). Once retrieval
was capped at `top_k=3`, these fragments crowded out the chunk that
actually had the answer — e.g. asking "what time do you open?" sometimes
missed the Opening Hours section entirely because it got merged with an
unrelated contact-info block.
**Fix:** Chunk by `## ` section headers instead, so each chunk is one
complete, coherent topic. Verified by testing the exact failing query
before and after.

### The model would "say" an escalation without calling the tool
**Problem:** For borderline emergency phrasing, the LLM would sometimes
reply with the correct safety wording as plain text but not actually invoke
the `escalate_urgent`/`escalate_human` function call — so the call would
sound right but no structured escalation flag was ever set.
**Fix:** Two-sided safety net. Input side: known emergency keywords force
`tool_choice` so the tool call isn't left to the model's discretion at all.
Output side: a regex checks the model's final reply text for escalation
language it should have paired with a tool call, and fires it anyway if the
model only spoke the words. Belt and suspenders — never rely on the model
alone for a safety-critical decision.

### RAG content and tool-calling contradicted each other
**Problem:** The knowledge base described how a patient would *normally*
check test results (e.g. "call after 2pm"), and the model sometimes just
answered from that text instead of calling the `check_test_results` tool
for a specific patient's request — even though the tool existed for exactly
this.
**Fix:** Added an explicit prompt rule: general practice-information text
never overrides a tool that exists for a specific request — if a tool
covers it, use the tool.

### Local test harness kept disconnecting mid-conversation (`1011 keepalive ping timeout`)
**Problem:** The terminal-based test client (`talk_to_emma.py`) would
randomly drop connection during slower, human-paced test conversations.
**Root cause:** the `websockets` library's default ping interval/timeout
(20s/20s) is fine for machine-speed traffic but too aggressive once a human
is pausing to think or speak — confirmed via DB history showing a
conversation had genuinely stalled, not just "felt slow."
**Fix:** Disabled the client-side ping (`ping_interval=None`) for the local
test harness. (Note: the real server also needs `--ws-ping-interval`/
`--ws-ping-timeout` tuned via `uvicorn` flags for the same reason under real
call conditions.)

### `dateutil` silently ignored relative dates
**Problem:** Needed to parse natural spoken time phrases like "tomorrow at
3pm" for booking. `python-dateutil`'s fuzzy parser was tried first — it
does *not* understand "tomorrow" at all, and silently defaults to today
instead of raising an error. This would have caused wrong bookings with
zero warning.
**Fix:** Switched to `parsedatetime`, which is purpose-built for relative
natural-language date/time parsing. Verified correct against multiple real
phrasings before trusting it.

### A booking race condition (two callers, same slot)
**Problem:** The original booking check was "does an appointment already
exist at this time?" — a classic check-then-act race: two simultaneous
callers could both pass the check before either one's booking was written.
**Fix:** Replaced with a real `slots` table and an atomic claim:
`UPDATE slots SET is_booked = TRUE WHERE slot_time = $1 AND is_booked =
FALSE`, inside a transaction, checking Postgres's own row-count result
(`"UPDATE 0"` vs `"UPDATE 1"`) to know if *this* caller actually won the
slot. No separate "check" step needed — the update itself is the check.

### A full-day timezone bug in slot generation
**Problem:** Slots were generated for "today and tomorrow" using
`date.today()` — which returns the *server's local* calendar date — while
every booking validation elsewhere in the app used
`datetime.now(timezone.utc)`. When local time and UTC disagree about what
day it is (e.g. IST is 5.5 hours ahead of UTC), the slots table and the
validation logic were working against two different "todays," a full day
apart — bookings for a real, valid time would fail with no slots found.
**Root cause found by:** a live test unexpectedly returned `time_in_past`
for a same-day booking; checking the actual server UTC time vs. the
locally-generated date exposed the mismatch directly.
**Fix:** Generate slots using `datetime.now(timezone.utc).date()`
consistently, matching the rest of the app's UTC-everywhere convention.

### Removing the booking confirmation step
**Problem:** Once name/phone/service/time were collected, the original
flow read every detail back and asked "is that correct?" before booking —
an extra full turn (extra STT+LLM+TTS round-trip) on every single booking,
for something that rarely needed correcting.
**Fix:** `book_appointment` is now called immediately once all four details
are known; every failure mode (unclear time, past time, outside hours,
date too far out, slot already taken) returns a plain structured status
instead of ever needing a "confirm first" safety net — the validation
itself is the safety net.

### A benchmark that made the better provider look worse
**Problem:** While comparing ElevenLabs STT against the existing (tuned)
Deepgram setup, a first-pass timing script measured "speech ended" as an
*estimated* timestamp (`start_time + expected_audio_duration`) rather than
an actual one, and it also accumulated real per-frame network-send delay
across ~90 frames into the same number. This made ElevenLabs look slower
(~1.6–1.8s) than Deepgram (1.47s) — the opposite of what later, more
careful testing showed.
**Fix:** Rewrote the benchmark to record `speech_end_at` as a real
post-send timestamp. The corrected number: ElevenLabs STT is consistently
**~1.0s — a genuine ~32% improvement**, not a regression. Caught and
corrected the wrong conclusion before it drove a bad decision, instead of
quietly updating the number and moving on.

---

## 2. If asked: "How do you think about latency in a voice AI system?"

**The core idea to lead with:** a phone call has one hard constraint text
chat doesn't — silence is felt immediately. The goal isn't just "make each
component fast," it's "never make the caller wait for a component to
*fully finish* before the next one starts."

**Structure the answer around the pipeline, and how each stage overlaps
with the next instead of waiting for it:**

```
Caller speaks → STT (streams partial transcript while speaking)
             → RAG (skipped entirely for short filler turns like "okay")
             → LLM (streams its reply sentence-by-sentence)
             → TTS (starts synthesizing sentence 1 while LLM writes sentence 2)
             → Caller hears audio
```

Concretely:
- **STT is real-time/streaming**, not "record then transcribe" — partial
  results arrive while the caller is still talking, so processing starts
  before they've even finished their sentence.
- **RAG is skipped for very short turns** (under ~6 words) — a vector
  search adds latency for zero benefit on a one-word "yes."
- **The LLM response is streamed and split into sentences** as they
  complete, so TTS can start speaking sentence one while the LLM is still
  generating sentence two — the caller starts hearing Emma's reply well
  before the full reply exists.
- **TTS is also streamed** (audio chunks, not one finished MP3) so playback
  starts on the first chunk, not after the whole utterance is synthesized.
- **DB writes are fire-and-forget** where safe (saving conversation turns)
  — they don't block the caller's next turn.
- **Cold-start costs are paid once, at boot, not on the first real call** —
  a throwaway RAG query runs at startup specifically because a fresh
  OpenAI/Qdrant client's first request pays a one-time connection/TLS setup
  cost (measured ~5s cold vs ~0.5s warm); moving that to boot means the
  first real caller never feels it.

**Then back it with real, measured numbers** (this is the important part —
it shows the latency work was *evidence-driven*, not guessed):
- STT: ElevenLabs Scribe Realtime measured at **~1.0s**, vs. a tuned
  Deepgram baseline at **~1.47s** — a genuine improvement, only trusted
  after finding and fixing a bug in the *first* benchmark that had wrongly
  made ElevenLabs look slower.
- TTS: ElevenLabs' time-to-first-audio-chunk measured at **~0.30s**
  steady-state, vs. **~0.90s** on the previous provider.
- Removing the booking confirmation step cut a full extra
  STT→LLM→TTS round-trip off of *every single booking* — the highest-value
  latency fix in the whole project, because it didn't just make one stage
  faster, it eliminated an entire stage.

**The meta-point worth making explicitly:** every one of these changes came
from adding structured `[EMMA-TIMING]` logs first and measuring, not from
assuming what was slow — and one of those measurements was itself found to
be wrong and had to be corrected before trusting the conclusion. That
willingness to distrust and re-verify your own benchmark is usually a more
convincing signal in an interview than the numbers themselves.
