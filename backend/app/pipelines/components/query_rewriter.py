# name: query_rewriter.py
# description: Conditional conversational query rewriter with gatekeeper and independent timeout.

import asyncio
import logging
import re
import unicodedata

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

TRIGGER_PRONOUNS: list[str] = [
    # English
    "it", "its", "this", "that", "they", "these", "those",
    "the rule", "the concept", "the example",
    # Ellipsis / continuation markers
    "what about", "and that", "how about", "what if",
    # Vietnamese
    "nó", "đó", "này", "kia", "cái này", "cái đó",
    "vậy", "thế", "ở trên", "tiếp theo", "ngoài ra", "trường hợp đó",
]

REWRITE_SYSTEM_PROMPT = """You are a conversational query rewriter.
Given recent chat history and a follow-up user query, rewrite the follow-up query into a complete, standalone question.
RULES:
1. Preserve the user's original language (English or Vietnamese).
2. Resolve ambiguous pronouns or ellipses using specific nouns/topics from the history.
3. If the query is already standalone or clear, output it unchanged.
4. Output ONLY the rewritten question. Do NOT answer the question. Do NOT include explanations, quotes, or formatting."""


def should_trigger_rewriter(query: str, history: list[dict]) -> bool:
    """
    Evaluate the gatekeeper conditions to decide whether query rewriting is needed.

    Parameters:
        query: Raw incoming query string.
        history: List of prior message turns (excluding fallback turns).

    Returns:
        bool: True if gatekeeper conditions are satisfied, False otherwise.

    Gatekeeper Conditions:
        1. At least 1 valid (non-fallback) prior turn in history.
        2. AND either:
           a) Query word count <= 4 words, OR
           b) Query contains any trigger pronoun matching whole-word / phrase boundaries.
    """
    if not getattr(settings, "query_rewriting_enabled", True):
        return False

    if not history:
        return False

    query_nfc = unicodedata.normalize("NFC", query.strip().lower())
    words = [w for w in re.split(r"\s+", query_nfc) if w]

    # Condition 2a: query <= 4 words
    if len(words) <= 4:
        return True

    # Condition 2b: contains trigger pronouns with boundary checking
    for phrase in TRIGGER_PRONOUNS:
        phrase_nfc = unicodedata.normalize("NFC", phrase)
        # Match phrase surrounded by start/end of string or non-word characters
        pattern = rf"(?:^|[^\w]){re.escape(phrase_nfc)}(?:[^\w]|$)"
        if re.search(pattern, query_nfc):
            return True

    return False


async def rewrite_query(
    query: str,
    history: list[dict],
    base_url: str | None = None,
    model: str | None = None,
    timeout_seconds: float | None = None,
) -> tuple[str, bool, bool]:
    """
    Rewrite a contextual or ambiguous query into a standalone query.

    Parameters:
        query: Incoming user query.
        history: Prior conversation turns (clean, without fallback turns).
        base_url: Ollama base URL (defaults to settings.ollama_base_url).
        model: Ollama chat model (defaults to settings.ollama_chat_model).
        timeout_seconds: Strict timeout for rewriting (defaults to 2.5s).

    Returns:
        tuple[str, bool, bool]:
            - result_query: Rewritten query, or original query on fallback.
            - triggered: Whether gatekeeper triggered the rewrite.
            - timed_out: Whether rewriting timed out.

    Logic Flow:
        1. Gatekeeper check via should_trigger_rewriter(). If False, return (query, False, False).
        2. Take the 2 most recent turns (up to 4 messages).
        3. Send request to Ollama with strict 2.5s timeout, num_predict=80, and num_ctx=8192.
        4. Validate response: if timed out, error, or < 3 words, fallback to original query.
    """
    if not should_trigger_rewriter(query, history):
        return query, False, False

    ollama_url = (base_url or settings.ollama_base_url).rstrip("/")
    chat_model = model or settings.ollama_chat_model
    timeout = timeout_seconds or getattr(settings, "query_rewriter_timeout_seconds", 2.5)
    num_ctx = getattr(settings, "ollama_num_ctx", 8192)

    # Take last 2 turns (up to 4 messages: user, assistant, user, assistant)
    recent_history = history[-4:] if len(history) > 4 else history

    messages = [{"role": "system", "content": REWRITE_SYSTEM_PROMPT}]
    for msg in recent_history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": f"Follow-up query: {query}"})

    payload = {
        "model": chat_model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 80,
            "num_ctx": num_ctx,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
            response = await client.post(f"{ollama_url}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
            rewritten = data.get("message", {}).get("content", "").strip()

            # Clean any wrapping quotes or artifacts
            rewritten = re.sub(r'^["\']|["\']$', "", rewritten).strip()

            # Fallback if result is too short (< 3 words)
            if len(rewritten.split()) < 3:
                logger.info("Rewriter output too short (%s), using original: %s", rewritten, query)
                return query, True, False

            logger.info("Query rewritten: '%s' -> '%s'", query, rewritten)
            return rewritten, True, False

    except (httpx.TimeoutException, asyncio.TimeoutError):
        logger.warning("Query rewriter timed out after %ss, falling back to original query", timeout)
        return query, True, True
    except Exception as exc:
        logger.warning("Query rewriter failed (%s), falling back to original query", exc)
        return query, True, False
