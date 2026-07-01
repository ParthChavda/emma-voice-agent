from fastapi import APIRouter
from pydantic import BaseModel

from app import db
from app.core.prompts import build_messages
from app.services.llm_openai import TOOLS, chat_completion
from app.services.rag import retrieve

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    session_id: str


class ChatResponse(BaseModel):
    reply: str
    intent: str | None
    session_id: str


@router.post("", response_model=ChatResponse)
async def chat(body: ChatRequest) -> ChatResponse:
    history = await db.load_history(body.session_id)
    chunks = await retrieve(body.message, top_k=3)
    messages = build_messages(chunks, history, body.message)
    reply, intent = await chat_completion(messages, TOOLS)
    await db.save_turn(body.session_id, "user", body.message)
    await db.save_turn(body.session_id, "assistant", reply)
    return ChatResponse(reply=reply, intent=intent, session_id=body.session_id)
