# EMMA Voice AI — How It Works (RAG, STT/TTS, Vector DB) + Interview Guide

This file explains, with exact code references, how the four things you asked
about actually work in this repo, then gives an interview-ready Q&A so you
can explain the system out loud without opening the code.

---

## 1. The Whole Pipeline in One Picture

```
Twilio call ─▶ WebSocket (/voice/stream) ─▶ CallSession (app/core/call_handler.py)
                                                  │
                    ┌─────────────────────────────┼─────────────────────────────┐
                    ▼                              ▼                             ▼
        STT: ElevenLabs Scribe          RAG: Qdrant + OpenAI embeddings   TTS: ElevenLabs
        (stt_elevenlabs.py)             (rag.py)                          (tts_elevenlabs.py)
                    │                              │                             │
                    └────────────▶ LLM: OpenAI gpt-4o-mini (llm_openai.py) ◀─────┘
                                          │
                                   Postgres (app/db.py)
                          conversation history, appointments, call summaries
```

Two entry points share the exact same "brain":
- **`/voice/stream`** (`app/routes/voice.py`) — the real phone call over a Twilio
  Media Streams WebSocket, fully streaming both directions.
- **`/chat`** (`app/routes/chat.py`) — a plain HTTP JSON endpoint that runs the
  same RAG → LLM → tools logic without audio, used for fast text-based testing.

---

## 2. How RAG Works Here

RAG = **Retrieval-Augmented Generation**: instead of baking every fact about
the surgery into the system prompt, facts live as Markdown files and are
pulled in *only when relevant*, per turn.

**Source of truth:** `app/knowledge_base/*.md` — `practice_info.md`,
`services.md`, `appointments.md`, `prescriptions.md`, `test_results.md`,
`doctor_schedules.md`.

### 2.1 Ingestion (happens once, at startup)

`ensure_ingested()` (`app/services/rag.py:102`) runs from the FastAPI
`lifespan` in `app/main.py:17`. It checks whether the Qdrant collection
already has points; if not, it calls `ingest_docs()`:

1. **Chunk** each `.md` file by `_chunk_markdown()` (`rag.py:43`) — splits on
   `## ` section headers, so each chunk is one complete topic (e.g. "Opening
   Hours"), not a random blob of text.
   - This was a deliberate fix: an earlier blank-line-based split produced
     near-empty fragments (a lone heading, a stray sentence) that crowded out
     the actually useful chunk once retrieval is capped at `top_k=3`.
2. **Embed** each chunk with OpenAI's `text-embedding-3-small` (`_embed()`,
   `rag.py:35`) → a 1536-dimension vector.
3. **Upsert** into Qdrant as a `PointStruct` with `payload = {"source": filename, "text": chunk}`.

### 2.2 Retrieval (happens every conversational turn)

`retrieve(query, top_k=3)` (`rag.py:113`):
1. Embeds the caller's message the same way (same model → same vector space).
2. Calls `qdrant.query_points(...)` for the top-3 nearest chunks by cosine
   similarity.
3. Returns their raw text.

`build_messages()` (`app/core/prompts.py:43`) inserts those chunks into the
system prompt as a clearly delimited **"PRACTICE INFORMATION"** block, so the
LLM only *sees* the facts relevant to this one question — not the entire
knowledge base every time.

### 2.3 The latency shortcut

In `call_handler.py:174`, RAG is **skipped entirely** for turns under 6 words
(`SHORT_TURN_WORD_THRESHOLD`). Replies like "yes", "book it", "cancel" are
confirmations, not questions — there's nothing to search for, so the
Qdrant + embedding round-trip is skipped for free latency.

### 2.4 Why RAG over just fine-tuning or a giant prompt?

- Facts (opening hours, policies) can be **edited in a `.md` file** and
  re-ingested — no retraining, no redeploy of prompt logic.
- Keeps the system prompt small and cheap per-call instead of stuffing every
  fact into every request regardless of relevance.
- Retrieval is **semantic**, not keyword match — "when can I come in?" still
  finds the "Opening Hours" chunk even without that exact wording.

---

## 3. How STT (Speech-to-Text) Works Here

**Provider:** ElevenLabs Scribe Realtime v2 (`app/services/stt_elevenlabs.py`),
chosen after a measured head-to-head against Deepgram (previous provider) —
ElevenLabs came out at **~1.0s** vs Deepgram's **~1.47s**, once a bug in the
first benchmark script was found and fixed (see §7 below).

**Flow (`transcribe_stream()`, `stt_elevenlabs.py:20`):**
1. Twilio streams raw call audio (mulaw, 8kHz, mono — the format Twilio Media
   Streams always sends) over the WebSocket, 20ms frames at a time.
2. Each audio chunk is base64-encoded and sent to ElevenLabs over a
   **persistent realtime connection** (`client.speech_to_text.realtime.connect`) —
   not "record N seconds, then transcribe."
3. ElevenLabs runs **server-side VAD** (voice activity detection) with
   `commit_strategy=VAD` and `vad_silence_threshold_secs=0.3` — it auto-detects
   when the caller has stopped talking (0.3s of silence) and fires a
   `committed_transcript` event with the finalized text for that utterance.
4. `on_final(text)` is called for each committed segment, `on_utterance_end()`
   right after — the call handler joins these into one user turn
   (`_on_final` / `_on_utterance_end`, `call_handler.py:138`).

**Why realtime/streaming matters:** processing can start the instant the
caller stops talking, instead of "record the whole call, then transcribe it
after" — this is what makes it feel like a live phone call rather than a
walkie-talkie.

---

## 4. How TTS (Text-to-Speech) Works Here

**Provider:** ElevenLabs TTS, streaming mode
(`app/services/tts_elevenlabs.py`), model `eleven_flash_v2_5` — their
lowest-latency model, measured at **~0.30s** time-to-first-audio-chunk vs
~0.90s on the previous provider (Deepgram Aura).

**Flow:**
1. `chat_completion_stream()` (`llm_openai.py:413`) streams the LLM's reply
   **sentence-by-sentence** — as soon as a complete sentence (`.`/`!`/`?`
   followed by whitespace) is detected in the token stream, it's handed off.
2. Each sentence goes to `synthesize_speech_stream()` (`tts_elevenlabs.py:76`),
   which POSTs to ElevenLabs' streaming endpoint and yields audio bytes
   **as they arrive** — not buffered until the whole clip is done.
3. `CallSession._speak()` (`call_handler.py:233`) re-chunks that audio stream
   into exact 160-byte / 20ms mulaw frames (Twilio's required frame size) and
   sends them to Twilio, **paced to real time** (`anyio.sleep(0.02)` per
   frame) — sending faster than real time causes choppy playback on Twilio's
   end.

**Net effect:** the caller hears Emma's first sentence *while the LLM is
still generating the second one*, and hears the first *word* of that
sentence before the whole sentence has finished synthesizing.

---

## 5. How the Vector Database (Qdrant) Connects

**Client setup** (`rag.py:18`):
```python
_qdrant = AsyncQdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
```
- `qdrant_url` / `qdrant_api_key` come from `app/config.py` (env vars via
  `pydantic-settings`, default `http://localhost:6333` for local Docker).
- Locally it runs via `docker-compose.yml` (`qdrant/qdrant:v1.14.1` image,
  port 6333, persisted to a `qdrant_data` volume). In production it can point
  at Qdrant Cloud by swapping the URL + API key — no code change needed.

**Collection:** one collection, `emma_knowledge` (`rag.py:11`), created with
```python
VectorParams(size=1536, distance=Distance.COSINE)
```
1536 matches OpenAI's `text-embedding-3-small` output dimension; cosine
similarity is the standard choice for OpenAI embeddings.

**Every point stored:**
```json
{
  "id": "<uuid4>",
  "vector": [1536 floats],
  "payload": {"source": "practice_info.md", "text": "## Opening Hours\n..."}
}
```

**Query path:** `client.query_points(collection_name, query=vector, limit=top_k)`
— an approximate-nearest-neighbor search over the stored vectors, returning
the closest chunks by cosine distance.

**Connection lifecycle:** the client is a module-level singleton
(`get_qdrant()`), created once and reused for the app's lifetime — not
reconnected per request. `warm_up()` (`rag.py:123`) fires one throwaway
`retrieve()` call at startup specifically to pay the first-request TLS/connection
cost (~5s cold) at boot instead of during a real caller's first turn (~0.5s warm
after that).

There's also a standalone maintenance script, `scripts/seed_qdrant_schedules.py`,
that can re-embed just one knowledge-base file (`doctor_schedules.md`) without
re-touching everything else already ingested — it deletes old points for that
`source` via a Qdrant payload filter, then re-embeds fresh.

---

## 6. What This System Can Actually Do (Capabilities)

| Capability | How | Where |
|---|---|---|
| Answer general questions (hours, services, prescriptions, test-result policy) | RAG retrieval + LLM | `rag.py`, `prompts.py` |
| **Book a new appointment** | Function calling → `book_appointment` tool → parses natural time phrases ("tomorrow at 3pm") → checks a real Postgres `slots` table → atomically claims the slot | `llm_openai.py`, `appointments.py` |
| Check test results | Function calling → `check_test_results` (currently a mocked/templated response — no real lab-system integration) | `llm_openai.py` |
| Escalate a medical emergency | Keyword short-circuit **and** an output-side regex safety net force `escalate_urgent`, speaking a fixed 999/111 message | `llm_openai.py` |
| Transfer to a human receptionist | `escalate_human` tool → live Twilio call redirect (`<Dial>`) to `human_handoff_number` | `call_handler.py:32`, `llm_openai.py` |
| Hold a natural multi-turn conversation | Full turn history kept in memory per call and persisted to Postgres, replayed into every LLM call | `call_handler.py`, `db.py` |
| Generate a structured call summary after hangup | A dedicated forced-tool-call LLM request (`record_call_summary`) extracts patient name, intent, key details, escalation flag, next action | `app/core/notes.py` |
| Real-time voice conversation over a phone call | Twilio Media Streams WebSocket + streaming STT/LLM/TTS | `voice.py`, `call_handler.py` |
| Text-only equivalent for testing | `/chat` HTTP endpoint, same brain minus audio | `chat.py` |
| Never double-book a slot under concurrent callers | Atomic `UPDATE slots SET is_booked=TRUE WHERE ... AND is_booked=FALSE` inside a transaction | `appointments.py:143` |
| Never crash the conversation on a bad/ambiguous input | Every tool/validation path returns a structured status (`unclear_time`, `slot_taken`, etc.), never an exception | `llm_openai.py`, `appointments.py` |

**What it does *not* do (current code, not the marketing pitch in
`emma-about.md`):** book more than 1 day ahead (only today/tomorrow have
generated slots), reschedule or cancel an existing appointment, give clinical
advice, or integrate with a real NHS/EHR system — those are explicitly
out of scope and hand off to a human. Note `emma-about.md` describes a
broader commercial product vision (multi-language, NHS system integrations,
Azure hosting, DTAC certification, SMS follow-up) — that is product/marketing
copy, not a description of what this codebase currently implements.

---

## 7. Real Problems Hit While Building This (good interview material)

See `docs/CHALLENGES_AND_LATENCY.md` for the full write-up. Highlights:
- RAG lost facts to blank-line chunking → fixed by chunking on `## ` headers.
- The LLM sometimes *said* an escalation without calling the tool → fixed with
  a two-sided safety net (keyword force on input, regex check on output).
- `python-dateutil` silently ignored "tomorrow" → switched to `parsedatetime`.
- A booking race condition → fixed with an atomic `UPDATE ... WHERE
  is_booked=FALSE`.
- A local-vs-UTC date mismatch caused valid same-day bookings to fail.
- A flawed STT benchmark made ElevenLabs look *slower* than Deepgram until the
  timing methodology itself was fixed — then it was ~32% faster.

---

## 8. Interview Q&A — How to Explain This System Out Loud

**Q: Walk me through what happens end-to-end when someone calls.**
> A: Twilio answers the call and opens a WebSocket streaming raw mulaw audio
> to our server. That audio is streamed live into ElevenLabs' realtime STT,
> which uses server-side VAD to detect when the caller stops talking and
> emits a finalized transcript. If the message is more than a few words, we
> embed it and query Qdrant for the top-3 most relevant knowledge-base
> chunks — that's the RAG step. Those chunks get inserted into the system
> prompt, and we call OpenAI's gpt-4o-mini with function-calling tools
> enabled. It either replies in plain text or calls a tool like
> `book_appointment`. The reply is streamed sentence-by-sentence into
> ElevenLabs' streaming TTS, and audio frames are sent back to Twilio in real
> time, paced to actual playback speed. Every turn is saved to Postgres for
> context on the next turn.

**Q: What is RAG and why did you use it instead of just a bigger prompt?**
> A: RAG is retrieval-augmented generation — you keep your facts in an
> external store and only pull in what's relevant per query, instead of
> hard-coding everything into the prompt. Here the facts are Markdown files
> per topic; each is chunked, embedded, and stored in Qdrant. At query time
> we embed the caller's message and do a semantic nearest-neighbor search.
> This keeps the prompt small and cheap, lets non-engineers update facts by
> editing a Markdown file with no redeploy of logic, and finds relevant
> content by meaning rather than requiring exact keyword matches.

**Q: How is the vector database actually wired into the pipeline?**
> A: Qdrant is a single async client created once at startup and reused. We
> store one collection with 1536-dim vectors (matching OpenAI's
> `text-embedding-3-small`) using cosine distance. Ingestion embeds each
> knowledge-base chunk and upserts it with the source filename and raw text
> as payload. At query time, we embed the user's message with the exact same
> model — so it lands in the same vector space — and call `query_points` for
> the nearest chunks. It's decoupled from the LLM entirely; swapping Qdrant
> for another vector store would only touch `rag.py`.

**Q: Why chunk by header instead of by paragraph or fixed token count?**
> A: We tried blank-line splitting first and it produced near-empty
> fragments — a bare heading, or one stray sentence — that could outrank the
> chunk that actually had the answer once retrieval is capped at top_k=3.
> Chunking on `## ` section headers keeps every fact attached to the
> surrounding context that gives it meaning, so a fact like "48 hours" stays
> in the same chunk as what it's 48 hours *of*.

**Q: How do STT and TTS differ from a "record then transcribe" batch approach?**
> A: Both are streaming/realtime, not batch. STT keeps a persistent
> connection open and gets partial/final transcripts as the caller talks,
> using server-side VAD to detect utterance boundaries — we don't wait for
> the whole call to end. TTS is the mirror image: we don't wait for the LLM's
> full reply or the full synthesized clip before sending audio — we split the
> LLM output into sentences as they complete and stream each sentence's audio
> to the caller as it's generated. That overlap is what removes dead air.

**Q: How do you keep latency low in a voice AI system specifically?**
> A: The core insight is that a phone call has a hard constraint text chat
> doesn't — silence is felt immediately — so the goal is to overlap every
> stage with the next instead of waiting for full completion. STT streams
> partials while the caller talks. RAG is skipped entirely for short
> confirmation turns like "yes". The LLM reply streams sentence-by-sentence
> so TTS can start on sentence one while sentence two is still being
> generated. TTS itself streams audio chunks. DB writes are fire-and-forget.
> And cold-start connection costs are paid once at boot via a throwaway
> warm-up call, not on the first real caller.

**Q: How do you handle a model hallucinating or contradicting your tools?**
> A: Two techniques. First, the system prompt has an explicit rule that
> general RAG text never overrides a tool that exists for a specific
> request — e.g. the knowledge base says "call after 2pm to check results",
> but if a patient asks about *their* results, the model must still call
> `check_test_results` rather than just answering from the retrieved text.
> Second, for safety-critical outputs like emergency escalation, we don't
> rely on the model alone — an emergency-keyword match on the input forces
> the tool call directly, and a regex check on the output catches the case
> where the model spoke the right words without invoking the tool.

**Q: How do you avoid a double-booking race condition?**
> A: The naive approach — "check if an appointment exists at this time, if
> not, create one" — is a classic check-then-act race between two
> simultaneous callers. Instead there's a real `slots` table, pre-generated
> for today/tomorrow from opening hours, and booking does a single atomic
> `UPDATE slots SET is_booked=TRUE WHERE slot_time=$1 AND is_booked=FALSE`
> inside a transaction. Postgres's own row-count tells us whether *this*
> caller actually won the slot — the update itself is the check, so there's
> no window for two callers to both pass a check before either write lands.

**Q: Why gpt-4o-mini instead of a bigger model?**
> A: This is a phone call — latency matters as much as raw capability, and
> the task (structured tool calls, short grounded replies) doesn't need a
> frontier model. It was chosen for the speed/cost tradeoff, not because
> it's the most capable option available.

**Q: What's the embedding model and why that one?**
> A: `text-embedding-3-small` from OpenAI — 1536 dimensions. It's used for
> both indexing the knowledge base and embedding each incoming query, which
> is required: you must embed with the same model on both sides for the
> vectors to be comparable in the same space.

**Q: How is conversation history/context managed?**
> A: Per call, the `CallSession` keeps an in-memory list of `{role,
> content}` turns and also persists every turn to Postgres
> (`conversations` table) keyed by `call_sid`/`session_id`. Each new turn's
> LLM call is built from system prompt + RAG context + full history + the
> new user message. The `/chat` endpoint reloads history from Postgres by
> `session_id` instead of keeping it in memory, since it's stateless between
> HTTP requests.

**Q: What would you improve or add next?**
> A: A few honest next steps: real multi-day/rescheduling booking support
> instead of today/tomorrow-only; a real lab-results/EHR integration instead
> of the current mocked `check_test_results`; moving the emergency-keyword
> list and RAG chunking parameters to config rather than constants; and
> adding automatic re-ingestion when a knowledge-base file changes instead of
> requiring the collection to be empty for `ensure_ingested` to trigger it.

**Q: How would this scale to many concurrent calls?**
> A: Each call is one `CallSession` coroutine with its own STT connection,
> Qdrant/OpenAI/ElevenLabs calls, and Postgres pool checkout — there's no
> shared mutable state between calls except the connection pools themselves
> (Postgres pool, Qdrant client, OpenAI client, HTTP client), which are all
> async and safe for concurrent use. Scaling further is mostly a matter of
> running more instances behind a load balancer and sizing the connection
> pools — the per-call logic itself doesn't assume single-tenancy.

---

## 9. Quick Reference — File Map

| Concern | File |
|---|---|
| RAG (chunk/embed/store/retrieve) | `app/services/rag.py` |
| STT (ElevenLabs realtime) | `app/services/stt_elevenlabs.py` |
| TTS (ElevenLabs streaming) | `app/services/tts_elevenlabs.py` |
| LLM + tools + streaming | `app/services/llm_openai.py` |
| Live call orchestration | `app/core/call_handler.py` |
| System prompt + RAG injection | `app/core/prompts.py` |
| Appointment/slot logic | `app/services/appointments.py` |
| Call summary generation | `app/core/notes.py` |
| DB schema + queries | `app/db.py` |
| Config/env | `app/config.py` |
| Voice webhook + WS route | `app/routes/voice.py` |
| Text-only chat route | `app/routes/chat.py` |
| Knowledge base content | `app/knowledge_base/*.md` |
| Local infra (Postgres+Qdrant) | `docker-compose.yml` |
