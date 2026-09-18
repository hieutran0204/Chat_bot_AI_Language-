# name: __init__.py
# description: Core primitives package containing LangChain-standard messages, prompt templates, and tools.

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
from app.core.primitives.prompts import (
    GRAMMAR_ANALYZER_PROMPT,
    SPEAKING_PARTNER_PROMPT,
    ChatPromptTemplate,
)
from app.core.primitives.tools import (
    BaseTool,
    DictionaryLookupTool,
    GrammarCheckerTool,
)

__all__ = [
    "AIMessage",
    "BaseMessage",
    "BaseTool",
    "ChatPromptTemplate",
    "Correction",
    "DictionaryLookupTool",
    "GRAMMAR_ANALYZER_PROMPT",
    "GrammarCheckerTool",
    "HumanMessage",
    "MessageRole",
    "SPEAKING_PARTNER_PROMPT",
    "SystemMessage",
    "ToolMessage",
    "messages_to_ollama",
]
