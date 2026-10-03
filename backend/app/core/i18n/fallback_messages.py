# name: fallback_messages.py
# description: Localized canned and fallback messages for RAG modes and conversational flows.

from enum import Enum


class FallbackReason(str, Enum):
    """Enumeration of canned fallback and fast-path response reasons."""

    NO_DOCUMENTS = "NO_DOCUMENTS"
    PROCESSING = "PROCESSING"
    FAILED_DOC = "FAILED_DOC"
    NOT_FOUND = "NOT_FOUND"
    GREETING = "GREETING"


FALLBACK_MESSAGES: dict[str, dict[str, str]] = {
    FallbackReason.NO_DOCUMENTS.value: {
        "en": "You haven't uploaded any study materials yet.",
        "vi": "Bạn chưa tải lên tài liệu học tập nào.",
    },
    FallbackReason.PROCESSING.value: {
        "en": "Your selected document is still being processed.",
        "vi": "Tài liệu bạn chọn vẫn đang trong quá trình xử lý.",
    },
    FallbackReason.FAILED_DOC.value: {
        "en": "There was a problem processing your document. Please try re-uploading.",
        "vi": "Đã xảy ra lỗi khi xử lý tài liệu của bạn. Vui lòng thử tải lên lại.",
    },
    FallbackReason.NOT_FOUND.value: {
        "en": "I couldn't find relevant information about this in your documents.",
        "vi": "Tôi không tìm thấy thông tin liên quan đến câu hỏi này trong tài liệu của bạn.",
    },
    FallbackReason.GREETING.value: {
        "en": "Hello! How can I assist you with your English learning today?",
        "vi": "Xin chào! Tôi có thể giúp gì cho bạn trong việc học tiếng Anh hôm nay?",
    },
}


def get_fallback_message(reason: str, lang: str = "en") -> str:
    """
    Get localized fallback message by reason key and language code.

    Parameters:
        reason: Fallback reason string or FallbackReason enum value.
        lang: Target language code ('en' or 'vi'). Defaults to 'en'.

    Returns:
        Localized message string. Falls back to English if target language is missing,
        or default not-found message if reason is unknown.
    """
    normalized_reason = reason.value if isinstance(reason, FallbackReason) else str(reason).upper()
    lang_key = (lang or "en").lower().strip()
    if lang_key.startswith("vi"):
        target_lang = "vi"
    else:
        target_lang = "en"

    messages_for_reason = FALLBACK_MESSAGES.get(normalized_reason)
    if not messages_for_reason:
        return FALLBACK_MESSAGES[FallbackReason.NOT_FOUND.value]["en"]

    return messages_for_reason.get(target_lang, messages_for_reason.get("en", ""))
