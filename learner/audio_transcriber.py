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

# --- Audio Transcription Tracking ---
_audio_stats = {
    "total_transcribed": 0,
    "total_failed": 0,
    "today_transcribed": 0,
    "today_failed": 0,
    "today_date": "",
    "recent_transcriptions": [],  # Last 20 transcription previews
}
_audio_stats_loaded = False


def _load_audio_stats():
    """Load audio stats from DB on first access."""
    global _audio_stats, _audio_stats_loaded
    if _audio_stats_loaded:
        return
    _audio_stats_loaded = True
    try:
        from core.database import is_db_available, kv_get
        if is_db_available():
            saved = kv_get("audio_transcription_stats")
            if saved and isinstance(saved, dict):
                _audio_stats.update(saved)
    except Exception as e:
        logger.warning(f"[Audio Stats] DB load failed: {e}")


def _save_audio_stats():
    """Persist audio stats to DB."""
    try:
        from core.database import is_db_available, kv_set
        if is_db_available():
            kv_set("audio_transcription_stats", _audio_stats)
    except Exception:
        pass


def _reset_today_if_needed():
    """Reset today's counters if it's a new day."""
    from datetime import datetime, timezone, timedelta
    ist = timezone(timedelta(hours=5, minutes=30))
    today = datetime.now(ist).strftime("%d %b %Y")
    if _audio_stats["today_date"] != today:
        _audio_stats["today_date"] = today
        _audio_stats["today_transcribed"] = 0
        _audio_stats["today_failed"] = 0


def _track_transcription(text: str, customer_phone: str = "", source: str = "whatsapp"):
    """Track a successful audio transcription."""
    _load_audio_stats()
    _reset_today_if_needed()

    from datetime import datetime, timezone, timedelta
    ist = timezone(timedelta(hours=5, minutes=30))

    _audio_stats["total_transcribed"] += 1
    _audio_stats["today_transcribed"] += 1

    _audio_stats["recent_transcriptions"].append({
        "text_preview": text[:100],
        "phone_last4": customer_phone[-4:] if customer_phone else "",
        "source": source,
        "time": datetime.now(ist).strftime("%I:%M %p"),
        "date": _audio_stats["today_date"],
    })
    # Keep only last 20
    _audio_stats["recent_transcriptions"] = _audio_stats["recent_transcriptions"][-20:]

    _save_audio_stats()


def _track_transcription_failure():
    """Track a failed audio transcription."""
    _load_audio_stats()
    _reset_today_if_needed()
    _audio_stats["total_failed"] += 1
    _audio_stats["today_failed"] += 1
    _save_audio_stats()


def get_audio_stats() -> dict:
    """Get audio transcription stats for dashboard."""
    _load_audio_stats()
    _reset_today_if_needed()
    return {
        "total_transcribed": _audio_stats["total_transcribed"],
        "total_failed": _audio_stats["total_failed"],
        "today_transcribed": _audio_stats["today_transcribed"],
        "today_failed": _audio_stats["today_failed"],
        "recent_transcriptions": _audio_stats["recent_transcriptions"][-10:],
    }


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
                _track_transcription_failure()
                return None

    except Exception as e:
        logger.error(f"Whisper transcription failed: {e}")
        _track_transcription_failure()
        return None


async def process_whatsapp_audio(
    media_id: str,
    language: str = "hi",
    customer_phone: str = "",
) -> str | None:
    """Full pipeline: download WhatsApp audio → transcribe → return text.

    Audio bytes are NOT stored — only returned as text.
    Tracks transcription stats for dashboard visibility.
    """
    audio_bytes = await download_whatsapp_media(media_id)
    if not audio_bytes:
        _track_transcription_failure()
        return None

    text = await transcribe_audio(audio_bytes, language=language)
    if text:
        _track_transcription(text, customer_phone=customer_phone, source="whatsapp")

        # Track Whisper cost
        from core.cost_tracker import track_whisper_cost
        # Estimate audio duration from file size (OGG ~16kbps for voice)
        estimated_seconds = len(audio_bytes) / 2000  # rough estimate
        track_whisper_cost(duration_seconds=estimated_seconds, source="whatsapp-audio")

    return text
