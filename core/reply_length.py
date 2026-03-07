"""Reply Length Tracking with Auto-Constraint.

Tracks AI reply lengths vs Ketu's manual reply lengths.
If AI consistently writes 3x longer than Ketu, auto-adds a length constraint.

Data is persisted to DB and checked on every reply generation.
"""

import logging
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

# In-memory tracking (persisted to DB)
_length_data: dict = {
    "ai_reply_lengths": [],      # Last 50 AI reply word counts
    "ketu_reply_lengths": [],    # Last 50 Ketu manual reply word counts
    "auto_constraint_active": False,  # Whether auto-constraint rule was added
    "constraint_max_words": 0,   # Auto-calculated max words
    "total_ai_tracked": 0,
    "total_ketu_tracked": 0,
}
_length_loaded = False

# Thresholds
_MIN_KETU_SAMPLES = 15     # Need 15 Ketu replies before comparing
_RATIO_THRESHOLD = 2.5     # AI must be 2.5x longer to trigger constraint
_KEEP_SAMPLES = 50         # Rolling window of last 50 replies


def _load_data():
    """Load length tracking data from DB."""
    global _length_data, _length_loaded
    if _length_loaded:
        return
    _length_loaded = True
    try:
        from core.database import is_db_available, kv_get
        if is_db_available():
            data = kv_get("reply_length_tracking")
            if data and isinstance(data, dict):
                _length_data.update(data)
                logger.info(
                    f"[ReplyLength] Loaded: {_length_data['total_ai_tracked']} AI, "
                    f"{_length_data['total_ketu_tracked']} Ketu samples"
                )
    except Exception as e:
        logger.warning(f"[ReplyLength] DB load failed: {e}")


def _save_data():
    """Persist length data to DB."""
    try:
        from core.database import is_db_available, kv_set
        if is_db_available():
            kv_set("reply_length_tracking", _length_data)
    except Exception as e:
        logger.warning(f"[ReplyLength] DB save failed: {e}")


def track_ai_reply(reply: str):
    """Track an AI-generated reply's word count."""
    _load_data()
    words = len(reply.split())
    _length_data["ai_reply_lengths"].append(words)
    _length_data["ai_reply_lengths"] = _length_data["ai_reply_lengths"][-_KEEP_SAMPLES:]
    _length_data["total_ai_tracked"] += 1

    # Check if constraint should be activated
    _check_and_update_constraint()
    _save_data()


def track_ketu_reply(reply: str):
    """Track a manual Ketu reply's word count."""
    _load_data()
    words = len(reply.split())
    if words < 2:
        return  # Skip "Ok", "Ji" etc — they'd skew the average too low
    _length_data["ketu_reply_lengths"].append(words)
    _length_data["ketu_reply_lengths"] = _length_data["ketu_reply_lengths"][-_KEEP_SAMPLES:]
    _length_data["total_ketu_tracked"] += 1
    _check_and_update_constraint()
    _save_data()


def _check_and_update_constraint():
    """Check if AI replies are too long compared to Ketu's and update constraint."""
    ketu_samples = _length_data["ketu_reply_lengths"]
    ai_samples = _length_data["ai_reply_lengths"]

    if len(ketu_samples) < _MIN_KETU_SAMPLES or len(ai_samples) < _MIN_KETU_SAMPLES:
        return  # Not enough data

    ketu_avg = sum(ketu_samples) / len(ketu_samples)
    ai_avg = sum(ai_samples) / len(ai_samples)

    if ketu_avg == 0:
        return

    ratio = ai_avg / ketu_avg

    if ratio >= _RATIO_THRESHOLD:
        # AI is writing way too long — set constraint
        # Max = Ketu's average * 1.5 (allow some room above Ketu's average)
        max_words = int(ketu_avg * 1.5)
        max_words = max(max_words, 8)  # Never go below 8 words

        if not _length_data["auto_constraint_active"] or _length_data["constraint_max_words"] != max_words:
            logger.info(
                f"[ReplyLength] Auto-constraint ACTIVATED: AI avg={ai_avg:.0f} words, "
                f"Ketu avg={ketu_avg:.0f} words (ratio={ratio:.1f}x). "
                f"Setting max={max_words} words."
            )
            _length_data["auto_constraint_active"] = True
            _length_data["constraint_max_words"] = max_words
    elif ratio < 2.0 and _length_data["auto_constraint_active"]:
        # AI has improved — deactivate constraint
        logger.info(
            f"[ReplyLength] Auto-constraint DEACTIVATED: AI avg={ai_avg:.0f}, "
            f"Ketu avg={ketu_avg:.0f} (ratio={ratio:.1f}x, below 2.0)"
        )
        _length_data["auto_constraint_active"] = False
        _length_data["constraint_max_words"] = 0


def get_length_constraint() -> str | None:
    """Get the prompt constraint string if active, else None.

    Injected into the system prompt to enforce shorter replies.
    """
    _load_data()
    if _length_data["auto_constraint_active"] and _length_data["constraint_max_words"] > 0:
        max_w = _length_data["constraint_max_words"]
        return f">> LENGTH LIMIT: Max {max_w} words. Ketu replies in ~{max_w} words — match his brevity."
    return None


def get_length_stats() -> dict:
    """Get length tracking stats for dashboard."""
    _load_data()

    ketu_samples = _length_data["ketu_reply_lengths"]
    ai_samples = _length_data["ai_reply_lengths"]

    ketu_avg = round(sum(ketu_samples) / len(ketu_samples), 1) if ketu_samples else 0
    ai_avg = round(sum(ai_samples) / len(ai_samples), 1) if ai_samples else 0
    ratio = round(ai_avg / ketu_avg, 1) if ketu_avg > 0 else 0

    return {
        "ai_avg_words": ai_avg,
        "ketu_avg_words": ketu_avg,
        "ratio": ratio,
        "auto_constraint_active": _length_data["auto_constraint_active"],
        "constraint_max_words": _length_data["constraint_max_words"],
        "ai_samples": len(ai_samples),
        "ketu_samples": len(ketu_samples),
        "total_ai_tracked": _length_data["total_ai_tracked"],
        "total_ketu_tracked": _length_data["total_ketu_tracked"],
        "enough_data": len(ketu_samples) >= _MIN_KETU_SAMPLES,
    }
