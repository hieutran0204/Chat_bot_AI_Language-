# name: intent_classifier.py
# description: Rule-based intent classifier for greetings and summarization requests.

import re
import unicodedata
from enum import Enum

from app.core.config import settings


class IntentType(str, Enum):
    """Classified user intent type."""

    GREETING = "greeting"
    SUMMARIZATION = "summarization"
    STANDARD = "standard"


GREETING_WHITELIST: set[str] = {
    "hi",
    "hello",
    "hey",
    "xin chào",
    "chào",
    "chào bạn",
    "ừ",
    "ok",
    "được",
    "cảm ơn",
    "thanks",
    "thank you",
}

SUMMARIZATION_KEYWORDS: list[str] = [
    "tóm tắt",
    "tóm lại",
    "nội dung chính",
    "summary",
    "summarize",
    "give me an overview",
    "what is this document about",
    "tài liệu này nói gì",
]

SUMMARIZATION_STOP_WORDS: set[str] = {
    # English — articles, prepositions, conjunctions, aux verbs, pronouns
    "a", "an", "the", "this", "that", "these", "those",
    "in", "on", "at", "to", "for", "of", "from", "with",
    "by", "about", "as", "into", "through", "between", "among",
    "and", "or", "but", "if", "so", "yet", "nor",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "i", "you", "he", "she", "it", "we", "they",
    "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their",
    "what", "which", "who", "when", "where", "how",
    "can", "could", "will", "would", "should", "may", "might",
    "not", "no", "please", "give", "tell",
    # Vietnamese — common function words
    "của", "về", "trong", "với", "là", "có", "và",
    "một", "những", "các", "tôi", "bạn", "này", "đó",
    "cho", "từ", "đến", "trên", "dưới", "được",
}


def classify_intent(query: str) -> IntentType:
    """
    Classify the incoming query into Greeting, Summarization, or Standard.

    Parameters:
        query: Raw user message string.

    Returns:
        IntentType: GREETING, SUMMARIZATION, or STANDARD.

    Logic Flow:
        1. Normalize query using Unicode NFC, lowercase, and trim.
        2. Check greeting condition: length <= 3 words AND matches greeting whitelist.
        3. Check summarization condition: contains any summarization keyword.
           - Strip matched keyword.
           - Remove punctuation and count content words (excluding SUMMARIZATION_STOP_WORDS).
           - If content words <= summarization_content_word_threshold, classify as SUMMARIZATION.
           - Otherwise, classify as STANDARD (user asked about a specific topic).
        4. Default to STANDARD.
    """
    if not query:
        return IntentType.STANDARD

    # 1. Normalize
    normalized = unicodedata.normalize("NFC", query.strip().lower())
    clean_text = re.sub(r"[^\w\s]", "", normalized).strip()
    words = clean_text.split()

    # 2. Greeting check (<= 3 words and in whitelist)
    if len(words) <= 3 and clean_text in GREETING_WHITELIST:
        return IntentType.GREETING

    # 3. Summarization check
    for kw in SUMMARIZATION_KEYWORDS:
        kw_nfc = unicodedata.normalize("NFC", kw)
        if kw_nfc in normalized:
            # Strip keyword from query
            remainder = normalized.replace(kw_nfc, " ")
            remainder_clean = re.sub(r"[^\w\s]", " ", remainder).strip()
            remainder_words = [w for w in remainder_clean.split() if w]
            content_words = [w for w in remainder_words if w not in SUMMARIZATION_STOP_WORDS]

            threshold = getattr(settings, "summarization_content_word_threshold", 3)
            if len(content_words) <= threshold:
                return IntentType.SUMMARIZATION
            # If > threshold, has specific subject -> route normally (STANDARD)
            return IntentType.STANDARD

    return IntentType.STANDARD
