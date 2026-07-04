# Demo Call Scripts (for Recording)

Three full conversations, increasing in depth, meant to be spoken out loud
into `scripts/talk_to_emma.py` (see "How to record" at the bottom). Each one
is a real, natural phone call — not a list of test inputs — so an
interviewer watching the recording sees how the system actually behaves in
context, not just that it "works."

**Timing note:** today's opening hours are only 9:00 AM–12:00 PM (this
changes day to day per the practice's real hours). To keep all three
scripts robust regardless of exactly when you record, the booking line in
each one uses flexible phrasing ("today, whenever you have availability")
rather than a fixed clock time — Emma finds the next real open slot
automatically instead of you having to guess a valid time in advance.

---

## Recording 1 — Simple: A Straightforward Booking

**What it shows:** the core booking capability end-to-end — natural
one-detail-at-a-time collection, immediate booking with no confirmation
step, a real reference number.

> **You:** "Hi there, I'd like to book an appointment please."
> *(Emma asks for your full name)*
>
> **You:** "Sure, it's [your name]."
> *(Emma asks for your phone number)*
>
> **You:** "It's 07700 900123."
> *(Emma asks what type of appointment you need)*
>
> **You:** "Just a routine appointment, please."
> *(Emma asks for your preferred date and time)*
>
> **You:** "Today, whenever you have availability."
> *(Emma books it immediately and reads back a reference number — no "is that correct?" step)*
>
> **You:** "That's great, thank you — that's all I needed."

---

## Recording 2 — Medium: A Question, Then a Booking

**What it shows:** the same booking flow, but opening with a real question
first — demonstrating RAG (retrieval-augmented generation) answering from
the practice's actual knowledge base before the conversation moves on to a
tool-calling task, and that Emma carries context smoothly between the two.

> **You:** "Hi, quick question first — what are your opening hours on Saturdays?"
> *(Emma answers with the real hours, pulled from the knowledge base)*
>
> **You:** "Perfect. And actually, while I've got you — how do repeat prescriptions work here?"
> *(Emma explains the repeat-prescription process, again from real practice info)*
>
> **You:** "Great, thanks. Actually, could you book me in for an appointment as well?"
> *(Emma pivots into the booking flow and asks for your name)*
>
> **You:** "It's [your name]."
> *(Emma asks for your phone number)*
>
> **You:** "07700 900124."
> *(Emma asks what type of appointment)*
>
> **You:** "A nurse appointment, please."
> *(Emma asks for your preferred date and time)*
>
> **You:** "Today, any time that works."
> *(Emma books it immediately and gives you a reference number)*
>
> **You:** "Brilliant, thank you very much."

---

## Recording 3 — Complex: Edge Cases, Validation, and Human Handoff

**What it shows:** the full range in one call — RAG on two different
topics, the system correctly rejecting an invalid request instead of
silently accepting it (real validation, not just a happy-path demo),
recovering gracefully within the same booking attempt, and escalating to a
human receptionist on request at the end.

> **You:** "Hello, hi — first off, do you offer cervical screening at your surgery?"
> *(Emma answers from the knowledge base — who does it, what it involves)*
>
> **You:** "Good to know. And one more thing — how far in advance do I need to order a repeat prescription?"
> *(Emma answers with the real turnaround time from practice info)*
>
> **You:** "Okay, thank you. I'd actually like to book an appointment now, if that's alright."
> *(Emma starts the booking flow, asks for your name)*
>
> **You:** "It's [your name]."
> *(Emma asks for your phone number)*
>
> **You:** "07700 900125."
> *(Emma asks what type of appointment you need)*
>
> **You:** "An urgent appointment please."
> *(Emma asks for your preferred date and time)*
>
> **You:** "Today at 11pm, if that's possible."
> *(This is outside opening hours every single day — Emma should decline gracefully and explain she can't book that time, without ever crashing or giving a broken reply)*
>
> **You:** "Ah, sorry — let's just say today, whenever you've got availability."
> *(Emma recovers cleanly and books using the earliest real open slot, giving you a reference number — same booking, corrected on the second try, in one continuous call)*
>
> **You:** "Perfect. Actually, one more thing before I go — could you transfer me to a real person at reception?"
> *(Emma recognizes this as a human-handoff request and responds accordingly, rather than trying to keep handling it herself)*

---

## How to record

1. Start the server: `source venv/bin/activate && uvicorn app.main:app --reload`
2. In another terminal: `python scripts/talk_to_emma.py`
3. Speak each line at its prompt (press Enter to start/stop recording per turn)
4. Type `q` + Enter to hang up when the script ends

Each call is automatically saved as one combined `.wav` file — both your
voice and Emma's replies, in order — at
`recordings/emma_call_<timestamp>.wav`, ready to use directly in your
presentation.
