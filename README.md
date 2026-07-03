# EMMA Voice Agent (PoC)

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in OPENAI_API_KEY, ELEVENLABS_API_KEY, TWILIO_AUTH_TOKEN
```

## Run

```bash
uvicorn app.main:app --reload
```

`GET /health` should return `{"status": "ok"}`.

## Structure

| Path | Purpose |
|---|---|
| `app/main.py` | FastAPI entrypoint |
| `app/config.py` | env vars / settings |
| `app/routes/voice.py` | Twilio webhook + media stream endpoints (Phase 3) |
| `app/routes/chat.py` | text-only test endpoint (Phase 2) |
| `app/services/stt_elevenlabs.py` | ElevenLabs STT (Scribe Realtime) streaming client (Phase 3) |
| `app/services/tts_elevenlabs.py` | ElevenLabs TTS streaming client (Phase 3) |
| `app/services/llm_openai.py` | OpenAI chat + function calling (Phase 2) |
| `app/services/rag.py` | embeddings + vector search (Phase 4) |
| `app/core/prompts.py` | EMMA system prompt + safety rules (Phase 2) |
| `app/core/call_handler.py` | STT -> RAG -> LLM -> TTS turn orchestration (Phase 3) |
| `app/core/notes.py` | post-call structured note generator (Phase 4) |
| `app/knowledge_base/` | surgery info docs (txt/md) |
| `docker-compose.yml` | optional containerized run (Phase 5) |
