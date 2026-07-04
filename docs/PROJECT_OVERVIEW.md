# EMMA Voice Agent — Project Overview

A voice-based AI receptionist for an NHS GP surgery. Patients call a phone
number (via Twilio), talk to "Emma," and she answers questions, checks test
results, escalates urgent/human requests, and books appointments — all in
real time, in a natural spoken conversation.

There's also a text-only `/chat` endpoint that exercises the exact same
brain (RAG + LLM + tools) without the voice pipeline, useful for fast
testing.

---

## 1. The Big Picture — What Happens on a Call

```
Caller speaks
     │
     ▼
Twilio (phone network) ──── streams raw audio over a WebSocket ────▶ our server
     │
     ▼
STT (Speech-to-Text)  →  turns audio into text, in real time, as the caller talks
     │
     ▼
RAG (Retrieval)  →  looks up relevant surgery info (opening hours, services, policies)
     │
     ▼
LLM (OpenAI gpt-4o-mini)  →  decides what to say, or calls a "tool" (e.g. book an appointment)
     │
     ▼
TTS (Text-to-Speech)  →  turns Emma's reply back into audio, streamed sentence-by-sentence
     │
     ▼
Twilio plays it back to the caller
     │
     ▼
Postgres  →  every turn is saved (for context in the next turn, and for records)
```

Everything is **streamed**, not batched — Emma starts speaking her first
sentence before she's finished thinking of the rest, and STT gives partial
transcripts while the caller is still talking. This is what makes it feel
like a real phone call instead of a walkie-talkie.

---

## 2. Tech Stack — What and Why

| Piece | What we use | Why |
|---|---|---|
| API server | **FastAPI** | Async-native, plays well with WebSockets (needed for the live audio stream) and background tasks. |
| Telephony | **Twilio Media Streams** | Handles the actual phone call / PSTN connection; streams raw call audio to us over a WebSocket instead of us needing to speak SIP/telephony protocols directly. |
| Speech-to-Text | **ElevenLabs Scribe (Realtime)** | Converts caller's speech to text live, as they talk. Chosen after a head-to-head test against Deepgram (previous provider) — ElevenLabs came out faster (~1.0s latency vs ~1.47s) once measured correctly. |
| Text-to-Speech | **ElevenLabs TTS (streaming)** | Converts Emma's text reply into audio, streamed in small chunks so playback can start almost immediately instead of waiting for the whole sentence to be synthesized. |
| LLM / reasoning | **OpenAI gpt-4o-mini**, with function calling | The "brain" — decides what to say and when to invoke a tool (book an appointment, escalate, etc). Chosen for speed/cost — this is a phone call, latency matters more than using the biggest model. |
| Vector search | **Qdrant** | Stores embeddings of the surgery's knowledge base (opening hours, services, policies) so relevant facts can be retrieved and injected into the prompt — this is RAG (see below). |
| Embeddings | **OpenAI text-embedding-3-small** | Turns text into vectors for Qdrant to search against. |
| Database | **Postgres (via asyncpg, raw SQL)** | Stores conversation history, call summaries, and appointments. No ORM — just plain async SQL, kept simple on purpose. |
| Date/time parsing | **parsedatetime** | Understands natural spoken phrases like "tomorrow at 3pm" or "in three days" — the more common `python-dateutil` library was tested and found to silently ignore relative phrases like "tomorrow", so it was dropped. |

---

## 3. What is RAG, and How Is It Used Here?

**RAG = Retrieval-Augmented Generation.** Instead of hard-coding surgery
facts into the prompt (which would make it huge and hard to update), the
facts live as plain Markdown files in `app/knowledge_base/` (opening hours,
services offered, prescription policy, etc.). At startup, each file is:

1. **Chunked** — split by `## ` section headers, so each chunk is one
   coherent topic (e.g. "Opening Hours") rather than a random blob of text.
2. **Embedded** — turned into a vector via OpenAI's embedding model.
3. **Stored in Qdrant** — a vector database, so it can be searched by
   *meaning*, not just keyword matching.

Then, on every caller turn:

1. The caller's message is embedded the same way.
2. Qdrant returns the **top 3 most relevant chunks** (semantic search).
3. Those chunks get inserted into the system prompt as "PRACTICE
   INFORMATION" for that one turn.
4. The LLM answers using that context — so it can correctly say "we're open
   9-12 on Saturdays" without that fact being permanently baked into the
   prompt.

**Why this matters for latency:** for very short turns (under 6 words, e.g.
"yes" or "okay"), RAG retrieval is skipped entirely — there's nothing
meaningful to search for, and it would just add latency for no benefit.

---

## 4. Tools (Function Calling) — What Emma Can *Do*, Not Just Say

The LLM isn't just generating text — it can call structured "tools" (OpenAI
function calling) when the conversation needs a real action:

| Tool | What it does |
|---|---|
| `book_appointment` | Books a real appointment. Validates the requested date/time against a real Postgres `slots` table (see below) and claims it atomically — no double-booking possible. |
| `check_test_results` | Templated response — since the answer never varies, it skips a second LLM call entirely for speed (see "MOCK_REPLY_TEMPLATES" below). |
| `escalate_urgent` | Triggered for medical emergencies (chest pain, breathing difficulty, etc.) — tells the caller to call 999/111 and flags the call. |
| `escalate_human` | Hands off to a human receptionist (Twilio transfer) when the caller is distressed, confused, or explicitly asks for a person. |

**Two response paths after a tool call**, chosen per-tool for speed:
- **Templated** (`check_test_results`, `escalate_human`): the reply text is
  fixed/deterministic, so it's returned directly — no second LLM call needed.
- **Synthesized** (`book_appointment`): the outcome genuinely varies
  (booked / slot taken / outside hours / etc.), so the result goes back to
  the LLM for a natural-language reply based on what actually happened.

**Safety net:** even if the model *describes* an escalation in words without
actually calling the tool, a lightweight regex-based check
(`_infer_intent_from_reply`) catches that and fires the tool anyway — so an
emergency never gets missed just because the model phrased it as prose
instead of a function call.

---

## 5. The Appointment Booking Flow (Slots Table)

- A Postgres `slots` table is pre-generated at startup for **today and
  tomorrow only**, in 30-minute increments, based on the surgery's opening
  hours (Mon–Fri 8:00–18:30, Sat 9:00–12:00, Sun closed).
- When `book_appointment` is called: the caller's spoken time (e.g. "tomorrow
  at 3pm") is parsed, rounded to the nearest half-hour slot, and checked
  against that table.
- **Atomic claim:** the slot is claimed with a single `UPDATE ... WHERE
  is_booked = FALSE` inside a transaction — if two callers ask for the same
  slot at once, only one wins; the other gets a normal "already taken"
  response instead of a race condition.
- No confirmation step ("is that correct?") — as soon as all 4 details
  (name, phone, service, time) are collected, it books immediately. This was
  a deliberate change to cut conversation turns and latency.
- **Never crashes:** every failure mode (unparseable time, past time, outside
  hours, date too far out, slot taken) returns a plain structured status like
  `{"status": "slot_taken"}` — never an exception — so the model can always
  turn it into a natural spoken reply.

---

## 6. Observability — `[EMMA-TIMING]` Logs

Every stage of the pipeline prints a `[EMMA-TIMING]` log line with how long
it took: STT commit time, RAG retrieval time, LLM first-token/full-reply
time, TTS first-chunk time, and total turn latency. Turns are tagged (e.g.
`[turn 3]`) so overlapping/interleaved turns in the log stay readable. This
was essential for the two changes that shaped this project the most:
proving where latency actually went (not guessing), and catching a real STT
provider comparison bug (a bad benchmark script made ElevenLabs look worse
than it was, until the measurement itself was fixed).

---

## 7. Key Design Principles Followed Throughout

- **Never throw exceptions into the conversation.** Every tool/validation
  path returns a structured status, not an error — a voice call has no good
  way to "show an error message."
- **Evidence over assumption.** Every latency/provider decision (prompt
  trimming, Deepgram → ElevenLabs switch for both STT and TTS) was made
  from real measured timing data, not marketing claims or guesses.
- **UTC everywhere.** All date/time logic uses UTC consistently, to avoid
  local-timezone bugs (one such bug — slot generation using local time while
  validation used UTC — was found and fixed during this project).
- **YAGNI.** The re-added booking flow is deliberately smaller in scope than
  the original one it replaced (no patient identity lookup, no
  reschedule/cancel — those still hand off to a human).
