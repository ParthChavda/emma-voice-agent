from fastapi import FastAPI

from app.routes import chat, voice

app = FastAPI()
app.include_router(chat.router)
app.include_router(voice.router)


@app.get("/health")
def health():
    return {"status": "ok"}
