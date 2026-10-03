# name: test_prompts.py
# description: Unit tests for ChatPromptTemplate and speaking personas.

from app.core.primitives.messages import HumanMessage, AIMessage, MessageRole
from app.core.primitives.prompts import (
    ChatPromptTemplate,
    SPEAKING_PARTNER_PROMPT,
)


def test_chat_prompt_template_basic_formatting():
    template = ChatPromptTemplate.from_messages([
        ("system", "You are an English tutor for level {user_level}."),
        ("human", "My sentence is: {user_input}"),
    ])
    messages = template.format_messages(user_level="B2", user_input="I want to improve fluency.")
    assert len(messages) == 2
    assert messages[0].role == MessageRole.SYSTEM
    assert "level B2" in messages[0].content
    assert messages[1].role == MessageRole.HUMAN
    assert "I want to improve fluency." in messages[1].content


def test_chat_prompt_template_with_history_injection():
    template = ChatPromptTemplate.from_messages([
        ("system", "Speaking tutor persona."),
        ("history", "{chat_history}"),
        ("human", "{user_input}"),
    ])
    history = [
        HumanMessage(content="Hello!"),
        AIMessage(content="Hi there! What are we talking about today?"),
    ]
    messages = template.format_messages(chat_history=history, user_input="Let's talk about travel.")
    assert len(messages) == 4
    assert messages[0].role == MessageRole.SYSTEM
    assert messages[1].content == "Hello!"
    assert messages[2].content == "Hi there! What are we talking about today?"
    assert messages[3].content == "Let's talk about travel."


def test_speaking_partner_prompt():
    messages = SPEAKING_PARTNER_PROMPT.format_messages(
        user_level="B1",
        user_weaknesses="past tense, prepositions",
        topic="Job Interview",
        correction_style="Silent tracking",
        chat_history=[],
        user_input="I am applying for a software job.",
    )
    assert len(messages) == 3
    assert "Alex" in messages[0].content
    assert "Job Interview" in messages[1].content
    assert messages[2].content == "I am applying for a software job."
