"""Audio transcription for WhatsApp voice messages using OpenAI Whisper API.

Flow: WhatsApp audio → download from Meta → Whisper API → text
Text is then fed into the learning pipeline like any other message.

Audio files are NOT stored — only the transcribed text is kept.
"""

import io
import logging
import tempfile

import httpx

from core.config import settings

logger = logging.getLogger(__name__)


async def download_whatsapp_media(media_id: str) -> bytes | None:
    """Download media from WhatsApp Business API using media ID.

    Step 1: Get media URL from media ID
    Step 2: Download the actual file
    """
    if not settings.whatsapp_access_token:
        logger.error("No WhatsApp access token configured")
        return None

    headers = {"Authorization": f"Bearer {settings.whatsapp_access_token}"}

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            # Step 1: Get media URL
            url_resp = await client.get(
                f"https://graph.facebook.com/v22.0/{media_id}",
                headers=headers,
            )
            url_resp.raise_for_status()
            media_url = url_resp.json().get("url")

            if not media_url:
                logger.error(f"No URL returned for media {media_id}")
                return None

            # Step 2: Download file
            file_resp = await client.get(media_url, headers=headers)
            file_resp.raise_for_status()
            return file_resp.content

    except Exception as e:
        logger.error(f"Media download failed for {media_id}: {e}")
        return None


async def transcribe_audio(audio_bytes: bytes, language: str = "hi") -> str | None:
    """Transcribe audio bytes using OpenAI Whisper API.

    Args:
        audio_bytes: Raw audio file bytes (ogg/opus from WhatsApp)
        language: Language hint for Whisper (default: Hindi)

    Returns:
        Transcribed text or None on failure
    """
    if not settings.openai_api_key:
        logger.error("No OpenAI API key configured for Whisper")
        return None

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {settings.openai_api_key}"},
                data={"model": "whisper-1", "language": language},
                files={"file": ("audio.ogg", io.BytesIO(audio_bytes), "audio/ogg")},
            )
            response.raise_for_status()
            result = response.json()
            text = result.get("text", "").strip()

            if text:
                logger.info(f"Whisper transcription: {text[:80]}...")
                return text
            else:
                logger.warning("Whisper returned empty transcription")
                return None

    except Exception as e:
        logger.error(f"Whisper transcription failed: {e}")
        return None


async def process_whatsapp_audio(media_id: str, language: str = "hi") -> str | None:
    """Full pipeline: download WhatsApp audio → transcribe → return text.

    Audio bytes are NOT stored — only returned as text.
    """
    audio_bytes = await download_whatsapp_media(media_id)
    if not audio_bytes:
        return None

    text = await transcribe_audio(audio_bytes, language=language)
    return text
