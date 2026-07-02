#!/usr/bin/env python3
"""
Standalone script: talk to Emma as plain typed text in your terminal.

Exercises the same RAG + function-calling pipeline as the /chat endpoint
(app/routes/chat.py) and prints the [EMMA-TIMING] logs live, without needing
a running server, Postgres, or Qdrant. RAG lookup is best-effort: if Qdrant
isn't reachable, it's skipped and Emma answers from the system prompt alone.
Conversation history is kept in memory only — nothing is written to Postgres.

Usage:
    source venv/bin/activate
    python scripts/chat_terminal.py

Type a message and press Enter. Type 'q' or 'quit' to exit.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.prompts import build_messages
from app.services.llm_openai import TOOLS, chat_completion
from app.services.rag import retrieve


async def _rag_lookup(message: str) -> list[str]:
    try:
        return await retrieve(message, top_k=3)
    except Exception as exc:
        print(f"[chat_terminal] RAG unavailable, answering without it: {exc!r}")
        return []


async def main() -> None:
    print("Talk to Emma (text mode). Type 'q' to quit.\n")
    history: list[dict[str, str]] = []

    while True:
        try:
            user_message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_message:
            continue
        if user_message.lower() in {"q", "quit", "exit"}:
            break

        chunks = await _rag_lookup(user_message)
        messages = build_messages(chunks, history, user_message)
        reply, intent = await chat_completion(messages, TOOLS)

        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": reply})

        print(f"Emma [{intent}]: {reply}\n")


if __name__ == "__main__":
    asyncio.run(main())
