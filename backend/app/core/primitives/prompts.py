# name: prompts.py
# description: ChatPromptTemplate engine with speaking partner persona and dynamic variable formatting.

from __future__ import annotations

import re
from typing import Any, Sequence

from app.core.primitives.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    MessageRole,
    SystemMessage,
)


class ChatPromptTemplate:
    """
    Template engine for structured multi-turn chat prompts.

    Supports templated system, human, and assistant messages,
    as well as inserting conversation history buffers.
    """

    def __init__(self, message_specs: list[tuple[str, str]]) -> None:
        """
        Initialize with a list of (role, template_text) tuples.

        Roles supported:
        - "system": Generates SystemMessage
        - "human" or "user": Generates HumanMessage
        - "assistant" or "ai": Generates AIMessage
        - "history": Injects a list of BaseMessage instances directly
        """
        self.message_specs = message_specs

    @classmethod
    def from_messages(cls, message_specs: list[tuple[str, str]]) -> ChatPromptTemplate:
        """Factory constructor matching LangChain naming conventions."""
        return cls(message_specs=message_specs)

    def format_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """
        Render all message templates with the provided keyword variables.

        Args:
            **kwargs: Dictionary of variable values matching {placeholder} in templates.

        Returns:
            List of concrete BaseMessage instances ready for the LLM.
        """
        formatted_messages: list[BaseMessage] = []

        for role, template_str in self.message_specs:
            norm_role = role.lower()

            if norm_role == "history":
                # Expecting kwargs to have the key matching the placeholder (e.g. chat_history)
                # or match the template_str if specified like "{chat_history}"
                var_name = template_str.strip("{}")
                history_value = kwargs.get(var_name, [])

                if isinstance(history_value, Sequence):
                    for item in history_value:
                        if isinstance(item, BaseMessage):
                            formatted_messages.append(item)
                        elif isinstance(item, dict):
                            formatted_messages.append(BaseMessage.from_dict(item))
                elif isinstance(history_value, str) and history_value.strip():
                    formatted_messages.append(SystemMessage(content=f"Previous conversation:\n{history_value}"))
                continue

            # Regular template string substitution using safe format
            try:
                rendered_text = template_str.format(**kwargs)
            except KeyError as exc:
                # If a key is missing, leave the unformatted placeholder or default to empty
                missing_key = str(exc).strip("'")
                rendered_text = re.sub(rf"\{{{missing_key}\}}", "", template_str)

            if not rendered_text.strip():
                continue

            if norm_role in ("system",):
                formatted_messages.append(SystemMessage(content=rendered_text))
            elif norm_role in ("human", "user"):
                formatted_messages.append(HumanMessage(content=rendered_text))
            elif norm_role in ("assistant", "ai"):
                formatted_messages.append(AIMessage(content=rendered_text))
            else:
                formatted_messages.append(BaseMessage(role=MessageRole(norm_role), content=rendered_text))

        return formatted_messages


# ── Specialized Personas for English Speaking Practice ────────────────────────

SPEAKING_PARTNER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are Alex, a friendly, encouraging English-speaking partner. "
            "Talk to the learner naturally like a close friend having a coffee chat. "
            "Keep your responses concise, punchy, and conversational (1 to 3 sentences max) "
            "just like spoken dialogue — avoid writing long essays.\n"
            "Ask one follow-up question at the end to keep the conversation flowing smoothly."
        ),
    ),
    (
        "system",
        (
            "Learner Profile:\n"
            "- Current Level: {user_level}\n"
            "- Known Weaknesses to track: {user_weaknesses}\n"
            "- Topic / Context: {topic}\n"
            "- Feedback Strategy: {correction_style}"
        ),
    ),
    ("history", "{chat_history}"),
    ("human", "{user_input}"),
])

GRAMMAR_ANALYZER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are an expert English grammar coach. Analyze the user's sentence carefully. "
            "Identify any grammatical, lexical, or preposition slips. "
            "Provide:\n"
            "1. Concise Vietnamese explanation of the issue.\n"
            "2. Three natural, native-level ways to express the same idea.\n"
            "Keep the feedback warm and easy to understand."
        ),
    ),
    ("human", "{user_input}"),
])
