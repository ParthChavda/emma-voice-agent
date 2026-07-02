import time
import uuid

from fastapi import APIRouter
from pydantic import BaseModel

from app import db
from app.core.prompts import build_messages
from app.services.llm_openai import TOOLS, chat_completion
from app.services.rag import retrieve

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    reply: str
    intent: str | None
    session_id: str


@router.post("", response_model=ChatResponse)
async def chat(body: ChatRequest) -> ChatResponse:
    turn_start = time.perf_counter()
    print(f'[EMMA-TIMING] User said | text: "{body.message}"')

    # First message of a new conversation omits session_id — mint one here
    # and hand it back so the caller reuses it for subsequent turns.
    session_id = body.session_id or str(uuid.uuid4())
    history = await db.load_history(session_id)

    rag_start = time.perf_counter()
    chunks = await retrieve(body.message, top_k=3)
    print(f"[EMMA-TIMING] RAG retrieve: {time.perf_counter() - rag_start:.2f}s")

    messages = build_messages(chunks, history, body.message)
    reply, intent = await chat_completion(messages, TOOLS)
    await db.save_turn(session_id, "user", body.message)
    await db.save_turn(session_id, "assistant", reply)

    print(f"[EMMA-TIMING] TOTAL turn latency: {time.perf_counter() - turn_start:.2f}s")
    return ChatResponse(reply=reply, intent=intent, session_id=session_id)
