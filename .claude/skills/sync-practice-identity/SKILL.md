---
name: sync-practice-identity
description: Keep the practice's identity details (name, address, phone, email, website) consistent across every file that mentions them whenever one changes. Make sure to use this whenever the user asks to rename the practice/business (e.g. "call it X instead", "rebrand to X"), change its address, phone number, or contact email, or asks "did I update everything" / "check if this is consistent everywhere" after such a change -- even for a change that looks like a one-line edit.
---

## Why this needs a sweep, not a single edit

The practice's identity is duplicated across code and content rather than defined once, so editing only the obvious file (e.g. `app/knowledge_base/practice_info.md`) silently leaves the rest inconsistent -- Emma would state one name on the phone greeting and a different one when asked "what's your address."

## Every location that needs checking

| File | What it holds |
|---|---|
| `app/core/prompts.py` | `EMMA_SYSTEM_PROMPT` opening line -- "AI receptionist for X, an NHS GP practice" |
| `app/core/call_handler.py` | `GREETING` constant -- spoken on every call |
| `app/knowledge_base/practice_info.md` | Title, Practice Name, Address, Phone, Email, Website, and the "Getting Here" paragraph (street name) |
| `app/knowledge_base/appointments.md`, `prescriptions.md`, `services.md`, `test_results.md`, `doctor_schedules.md` | Each has a `# X at <name>` heading |
| `scripts/eval_50_cases.py`, `scripts/eval_emma.py` | Hard-coded assertions checking for the *old* name/address/phone in Emma's replies (e.g. `contains_any("Elmwood Road", "M14")`, `contains_any("0161 234 5678")`) -- these start failing, or passing for the wrong reason, if the identity changes but the assertion doesn't |
| `README.md` | May reference the practice name in setup/description text |

Grep for the current value across the repo before declaring the sweep done -- don't rely on this table alone, since it can go stale as the codebase changes:

```bash
grep -rn "<old value>" --include="*.py" --include="*.md" .
```

## What NOT to touch

Historical artifacts shouldn't be rewritten to match a new identity -- they're records of what actually happened, not live content: `logs.md` (real call transcripts), anything under `docs/superpowers/plans/` (planning docs written against the identity at the time), and `recordings/*.wav`. If the user wants a full historical rewrite anyway, confirm explicitly first since it changes the meaning of those records.

## After the sweep

Any change to `app/knowledge_base/*.md` needs the Qdrant collection re-embedded before RAG reflects it -- run the `qdrant-seed-check` skill next. Then consider `run-emma-evals` to confirm the new identity doesn't break assertions that check for it by name.
