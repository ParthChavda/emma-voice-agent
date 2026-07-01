from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import db
from app.config import settings
from app.routes import chat, voice
from app.services.rag import ensure_ingested


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init_pool(settings.postgres_dsn)
    await ensure_ingested()
    yield
    await db.close_pool()


app = FastAPI(lifespan=lifespan)
app.include_router(chat.router)
app.include_router(voice.router)


@app.get("/health")
def health():
    return {"status": "ok"}
