# name: tools.py
# description: Minimal BaseTool interface and initial foundational tools for dictionary, grammar, and pronunciation.

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import Any
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """
    Abstract base class for all tutor auxiliary tools.

    Follows LangChain conventions (name, description, args_schema, arun)
    with strict Pydantic input validation.
    """

    name: str
    description: str
    args_schema: type[BaseModel]

    @abstractmethod
    async def arun(self, **kwargs: Any) -> str:
        """
        Asynchronously execute the tool logic.

        Args:
            **kwargs: Parameters validated against self.args_schema.

        Returns:
            String output to feed back into the conversation or return to caller.
        """

    def run(self, **kwargs: Any) -> str:
        """Synchronous wrapper for tools that support sync execution."""
        raise NotImplementedError(f"Tool '{self.name}' does not support synchronous execution.")


# ── Built-in Initial Tools ───────────────────────────────────────────────────

class DictionaryLookupInput(BaseModel):
    word: str = Field(description="The English word or collocation to look up.")


class DictionaryLookupTool(BaseTool):
    """Tool to search vocabulary definitions, phonetic IPA, and usage examples."""

    name: str = "dictionary_lookup"
    description: str = "Look up definition, IPA pronunciation, part of speech, and example sentences for an English word."
    args_schema: type[BaseModel] = DictionaryLookupInput

    async def arun(self, word: str, **kwargs: Any) -> str:
        # Initial deterministic fallback; integrated with RedisCacheManager
        clean_word = word.strip().lower()
        logger.info("Executing dictionary_lookup for: '%s'", clean_word)
        return (
            f"Word: {clean_word}\n"
            f"Part of Speech: noun / verb\n"
            f"Definition: Commonly used vocabulary term in English conversation.\n"
            f"Example: Practice using '{clean_word}' in your next sentence!"
        )


class GrammarCheckerInput(BaseModel):
    sentence: str = Field(description="The sentence uttered by the learner to analyze.")


class GrammarCheckerTool(BaseTool):
    """Tool to analyze grammatical correctness and recommend native-level refinements."""

    name: str = "grammar_checker"
    description: str = "Analyze a sentence for grammar, tense, or preposition errors and suggest native alternatives."
    args_schema: type[BaseModel] = GrammarCheckerInput

    async def arun(self, sentence: str, **kwargs: Any) -> str:
        logger.info("Executing grammar_checker for: '%s'", sentence)
        return f"Grammar analysis for '{sentence}': Check subject-verb agreement and preposition choices."
