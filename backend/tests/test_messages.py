# name: test_messages.py
# description: Unit tests for core message abstractions, serialization, and conversions.

import pytest
from app.core.primitives.messages import (
    AIMessage,
    BaseMessage,
    Correction,
    HumanMessage,
    MessageRole,
    SystemMessage,
    ToolMessage,
    messages_to_ollama,
)


def test_base_message_with_voice_fields():
    msg = HumanMessage(
        content="I go to school yesterday.",
        audio_url="/uploads/audio/user1_msg1.wav",
        audio_duration_ms=2500,
        stt_confidence=0.92,
    )
    assert msg.role == MessageRole.HUMAN
    assert msg.content == "I go to school yesterday."
    assert msg.audio_url == "/uploads/audio/user1_msg1.wav"
    assert msg.audio_duration_ms == 2500
    assert msg.stt_confidence == 0.92


def test_ai_message_with_corrections():
    correction = Correction(
        original="I go to school yesterday.",
        suggestion="I went to school yesterday.",
        type="grammar",
        explanation="Use past tense 'went' when referring to 'yesterday'.",
    )
    ai_msg = AIMessage(
        content="That sounds like a busy day! What did you do at school?",
        corrections=[correction],
    )
    assert ai_msg.role == MessageRole.AI
    assert len(ai_msg.corrections) == 1
    assert ai_msg.corrections[0].original == "I go to school yesterday."
    assert ai_msg.corrections[0].type == "grammar"


def test_serialization_and_deserialization():
    msg = HumanMessage(
        content="Hello tutor",
        audio_url="https://s3.example.com/audio.wav",
        stt_confidence=0.88,
    )
    serialized = msg.to_dict()
    assert serialized["role"] == "user"
    assert serialized["stt_confidence"] == 0.88

    restored = BaseMessage.from_dict(serialized)
    assert isinstance(restored, HumanMessage)
    assert restored.content == "Hello tutor"
    assert restored.stt_confidence == 0.88


def test_messages_to_ollama():
    messages = [
        SystemMessage(content="You are a friendly tutor."),
        HumanMessage(content="Hi!"),
        AIMessage(content="Hello! How are you feeling today?"),
    ]
    ollama_formatted = messages_to_ollama(messages)
    assert ollama_formatted == [
        {"role": "system", "content": "You are a friendly tutor."},
        {"role": "user", "content": "Hi!"},
        {"role": "assistant", "content": "Hello! How are you feeling today?"},
    ]
