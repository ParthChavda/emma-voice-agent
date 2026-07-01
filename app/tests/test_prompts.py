from app.core.prompts import build_messages


def test_first_message_is_system():
    messages = build_messages([], [], "hello")
    assert messages[0]["role"] == "system"


def test_system_prompt_contains_emma_persona():
    messages = build_messages([], [], "hello")
    system = messages[0]["content"]
    assert "EMMA" in system
    assert "Elmwood Road Surgery" in system


def test_system_prompt_contains_hard_safety_rules():
    messages = build_messages([], [], "hello")
    system = messages[0]["content"]
    assert "999" in system
    assert "clinical advice" in system.lower()


def test_rag_chunks_injected_into_system_prompt():
    chunks = ["Opening hours: Mon-Fri 8am-6:30pm", "Saturday 9am-12pm"]
    messages = build_messages(chunks, [], "what time do you open?")
    system = messages[0]["content"]
    assert "Opening hours: Mon-Fri 8am-6:30pm" in system
    assert "Saturday 9am-12pm" in system


def test_no_rag_context_block_when_chunks_empty():
    messages = build_messages([], [], "hello")
    system = messages[0]["content"]
    assert "PRACTICE INFORMATION" not in system


def test_rag_block_wrapped_in_separator_markers():
    chunks = ["We are open Mon-Fri 8am-6:30pm"]
    messages = build_messages(chunks, [], "hours?")
    system = messages[0]["content"]
    assert "--- PRACTICE INFORMATION" in system
    assert "--- END PRACTICE INFORMATION ---" in system


def test_history_turns_appear_after_system():
    history = [
        {"role": "user", "content": "My name is Sarah"},
        {"role": "assistant", "content": "Hello Sarah"},
    ]
    messages = build_messages([], history, "I need help")
    assert messages[1] == {"role": "user", "content": "My name is Sarah"}
    assert messages[2] == {"role": "assistant", "content": "Hello Sarah"}


def test_user_message_is_last():
    messages = build_messages([], [], "book me an appointment")
    assert messages[-1] == {"role": "user", "content": "book me an appointment"}


def test_full_message_order_with_history_and_rag():
    chunks = ["We have routine and urgent slots."]
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    messages = build_messages(chunks, history, "book appointment")

    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "hi"}
    assert messages[2] == {"role": "assistant", "content": "hello"}
    assert messages[3] == {"role": "user", "content": "book appointment"}
