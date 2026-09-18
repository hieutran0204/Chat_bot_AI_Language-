# name: voice.py
# description: Abstract interfaces (ports) for Speech-To-Text (STT) and Text-To-Speech (TTS).

from abc import ABC, abstractmethod


class ISpeechToText(ABC):
    """
    Port interface for Speech-to-Text audio transcription.

    Concrete adapters can implement local Whisper, faster-whisper,
    or browser-level STT relay.
    """

    @abstractmethod
    async def transcribe(self, audio_bytes: bytes) -> tuple[str, float]:
        """
        Transcribe raw audio bytes into recognized text.

        Args:
            audio_bytes: Raw audio byte buffer (WAV, MP3, WebM).

        Returns:
            Tuple of (transcribed_text, confidence_score_between_0_and_1).
        """


class ITextToSpeech(ABC):
    """
    Port interface for Text-to-Speech audio synthesis.

    Concrete adapters can implement local Piper, Kokoro, edge-tts,
    or Coqui TTS.
    """

    @abstractmethod
    async def synthesize(self, text: str, voice: str | None = None) -> bytes:
        """
        Synthesize text into speech audio bytes.

        Args:
            text: Text to speak aloud.
            voice: Optional voice persona identifier.

        Returns:
            Raw audio bytes (typically MP3 or WAV).
        """
