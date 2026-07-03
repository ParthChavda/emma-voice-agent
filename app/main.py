from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI

from app import db
from app.config import settings
from app.routes import chat, voice
from app.services import appointments
from app.services.rag import ensure_ingested, warm_up
from app.services.tts_elevenlabs import close_http_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init_pool(settings.postgres_dsn)
    await ensure_ingested()
    await warm_up()
    # UTC, not date.today() (which uses the server's local timezone) — every
    # other date/time decision in this app (parse_preferred_time,
    # is_date_supported) is UTC-based, and a local-vs-UTC mismatch here would
    # silently generate slots for the wrong day whenever local time and UTC
    # disagree on what day it is (verified: they do right now, IST is ahead
    # of UTC by 5:30 so "today" in IST can already be UTC's "tomorrow").
    today_utc = datetime.now(timezone.utc).date()
    await appointments.ensure_slots_for_days([today_utc, today_utc + timedelta(days=1)])
    yield
    await db.close_pool()
    await close_http_client()


app = FastAPI(lifespan=lifespan)
app.include_router(chat.router)
app.include_router(voice.router)


@app.get("/health")
def health():
    return {"status": "ok"}
