---
name: run-emma-evals
description: Run Emma's conversation-quality eval suites (scripts/eval_50_cases.py and scripts/eval_emma.py) and summarize pass/fail results in plain language. Make sure to use this whenever the user wants to check for regressions after changing prompts.py, call_handler.py, appointments.py, rag.py, or any knowledge_base/*.md file, or asks things like "run the evals", "did I break anything", "test emma", "check for regressions", "run eval_50_cases", or "is emma still behaving correctly" -- even if they don't name the scripts directly.
---

## What these scripts actually test

Both scripts drive the real pipeline end-to-end (real RAG retrieval, real OpenAI completions, real tool dispatch, real Postgres writes for bookings) -- no mocking. A failure means Emma's actual behavior diverged from what a specific conversation should produce, not a broken import or type error.

- `scripts/eval_50_cases.py` -- the 50-case plan from `emma-test-case.md`, one turn per case, grouped by category (General, Appointment, Prescriptions, Urgent, Handoff, Edge). Cases 49-50 are always skipped (they need live audio/silence handling -- see `talk_to_emma.py`).
- `scripts/eval_emma.py` -- a smaller, more adversarial/multi-turn set (jailbreaks, privacy probes, multi-turn memory, garbled STT text). Better for testing prompt changes aimed at safety/robustness.

Run both unless the change is clearly scoped to one area -- e.g. a RAG-content-only change makes `eval_50_cases.py`'s General/Prescriptions categories most relevant; a safety-prompt change makes `eval_emma.py`'s jailbreak/privacy cases matter more.

## Prerequisites

1. Postgres + Qdrant must be reachable: `docker-compose up -d postgres qdrant`, wait for healthy.
2. `.env` needs `OPENAI_API_KEY` set -- real API calls are made, which costs money and takes real wall-clock time (expect roughly 1-2 minutes for the 50-case run).
3. If any `app/knowledge_base/*.md` file was edited recently, the Qdrant collection may still hold embeddings from *before* the edit -- `ensure_ingested()` only (re-)embeds when the collection is empty, so a stale collection won't reflect new text and RAG-dependent cases (General, Prescriptions) can fail, or pass for the wrong reason. Use the `qdrant-seed-check` skill first if a knowledge-base file changed.

## Running

```bash
source venv/bin/activate
python scripts/eval_50_cases.py
python scripts/eval_emma.py
```

Both scripts clean up their own DB writes (booking test rows) at the end -- safe to rerun repeatedly without manual cleanup.

## Reporting results

Don't just paste raw stdout. For each FAIL:
- Show the case name/number, what was said, what was expected, and Emma's actual `intent` + `reply`.
- Check `case.note` in `eval_50_cases.py` when present -- several cases are intentionally loose ("original spec superseded", "no RAG content exists, checking for hallucination"), so a technically-failing check there may not indicate a real bug.
- Point at the likely source: RAG misses -> check the relevant `knowledge_base/*.md` file and whether Qdrant needs reseeding; wrong intent/tool routing -> `app/core/prompts.py` HARD RULES or `app/services/llm_openai.py` TOOLS; wrong appointment behavior -> `app/services/appointments.py`.

End with the pass-count summary and by-category breakdown so the user can see at a glance whether a change caused a narrow or broad regression.
