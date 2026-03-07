"""Peak Hours Auto-Defer — tracks when Ketu is active on WhatsApp.

During Ketu's active hours: defer more borderline questions to him.
During off-hours: be more aggressive with AI replies so customers aren't waiting.

Learns Ketu's activity pattern from wwbun sync timestamps (when he manually replies).
"""

import logging
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

# In-memory activity tracking (persisted to DB)
_ketu_activity: dict = {
    "hourly_replies": {},   # hour (0-23) -> count of Ketu's manual replies
    "recent_active": [],    # Last 20 timestamps when Ketu was active
    "total_tracked": 0,
}
_activity_loaded = False

# Thresholds
_MIN_DATA_POINTS = 10       # Need at least 10 manual replies before using peak hours
_ACTIVE_WINDOW_MINUTES = 30  # Consider Ketu "currently active" if replied within 30 min
_PEAK_HOUR_MIN_REPLIES = 3   # Hour needs 3+ replies to be considered "peak"


def _load_activity():
    """Load Ketu's activity data from DB."""
    global _ketu_activity, _activity_loaded
    if _activity_loaded:
        return
    _activity_loaded = True
    try:
        from core.database import is_db_available, kv_get
        if is_db_available():
            data = kv_get("ketu_peak_hours")
            if data and isinstance(data, dict):
                _ketu_activity.update(data)
                logger.info(
                    f"[PeakHours] Loaded: {_ketu_activity['total_tracked']} replies tracked, "
                    f"active hours: {_get_peak_hours_list()}"
                )
    except Exception as e:
        logger.warning(f"[PeakHours] DB load failed: {e}")


def _save_activity():
    """Persist activity data to DB."""
    try:
        from core.database import is_db_available, kv_set
        if is_db_available():
            kv_set("ketu_peak_hours", _ketu_activity)
    except Exception as e:
        logger.warning(f"[PeakHours] DB save failed: {e}")


def track_ketu_reply(timestamp: datetime | None = None):
    """Track a manual reply from Ketu (called during wwbun sync or ketu-replied).

    Args:
        timestamp: When Ketu replied. Defaults to now (IST).
    """
    _load_activity()

    if timestamp is None:
        timestamp = datetime.now(IST)
    elif timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=IST)

    hour = timestamp.hour
    hour_str = str(hour)
    _ketu_activity["hourly_replies"][hour_str] = (
        _ketu_activity["hourly_replies"].get(hour_str, 0) + 1
    )

    # Track recent activity (keep last 20)
    _ketu_activity["recent_active"].append(timestamp.isoformat())
    _ketu_activity["recent_active"] = _ketu_activity["recent_active"][-20:]
    _ketu_activity["total_tracked"] += 1

    _save_activity()


def track_ketu_replies_batch(count: int = 1):
    """Track multiple Ketu replies at current time (simpler batch call)."""
    _load_activity()
    now = datetime.now(IST)
    hour_str = str(now.hour)
    _ketu_activity["hourly_replies"][hour_str] = (
        _ketu_activity["hourly_replies"].get(hour_str, 0) + count
    )
    _ketu_activity["recent_active"].append(now.isoformat())
    _ketu_activity["recent_active"] = _ketu_activity["recent_active"][-20:]
    _ketu_activity["total_tracked"] += count
    _save_activity()


def is_ketu_currently_active() -> bool:
    """Check if Ketu has replied within the last 30 minutes."""
    _load_activity()

    if not _ketu_activity["recent_active"]:
        return False

    now = datetime.now(IST)
    try:
        last_active_str = _ketu_activity["recent_active"][-1]
        last_active = datetime.fromisoformat(last_active_str)
        if last_active.tzinfo is None:
            last_active = last_active.replace(tzinfo=IST)
        minutes_ago = (now - last_active).total_seconds() / 60
        return minutes_ago <= _ACTIVE_WINDOW_MINUTES
    except (ValueError, IndexError):
        return False


def is_peak_hour() -> bool:
    """Check if current hour is one of Ketu's peak activity hours."""
    _load_activity()

    if _ketu_activity["total_tracked"] < _MIN_DATA_POINTS:
        return False  # Not enough data yet

    hour_str = str(datetime.now(IST).hour)
    count = _ketu_activity["hourly_replies"].get(hour_str, 0)
    return count >= _PEAK_HOUR_MIN_REPLIES


def should_defer_borderline(confidence_score: int) -> dict | None:
    """Decide if a borderline-confidence reply should be deferred to Ketu.

    During Ketu's active hours or when he's currently online: defer borderline cases.
    During off-hours: let AI handle it (customer shouldn't wait).

    Args:
        confidence_score: The confidence score from confidence.py (0-100)

    Returns:
        None if AI should reply normally.
        dict with {reason, defer_reply} if should defer to Ketu.
    """
    _load_activity()

    # Only applies to borderline scores (25-50 range)
    # Below 25 = always deferred by confidence.py
    # Above 50 = always handled by AI
    if confidence_score < 25 or confidence_score > 50:
        return None

    ketu_active = is_ketu_currently_active()
    peak = is_peak_hour()

    if ketu_active:
        # Ketu is online right now — defer borderline questions to him
        logger.info(
            f"[PeakHours] Deferring borderline (score={confidence_score}) — "
            f"Ketu is currently active"
        )
        return {
            "reason": "ketu_active_borderline",
            "defer_reply": "Bhai, Ketu sir abhi online hain — wo khud reply karenge.",
        }

    if peak and confidence_score < 40:
        # Peak hour + lower confidence — defer
        logger.info(
            f"[PeakHours] Deferring borderline (score={confidence_score}) — "
            f"peak hour, Ketu likely available"
        )
        return {
            "reason": "peak_hour_borderline",
            "defer_reply": "Bhai, ye Ketu sir khud confirm karenge — thodi der mein reply aayega.",
        }

    # Off-hours or higher borderline — AI should handle
    return None


def _get_peak_hours_list() -> list[int]:
    """Get list of hours that are peak hours."""
    peaks = []
    for h_str, count in _ketu_activity.get("hourly_replies", {}).items():
        if count >= _PEAK_HOUR_MIN_REPLIES:
            peaks.append(int(h_str))
    return sorted(peaks)


def get_peak_hours_stats() -> dict:
    """Get peak hours statistics for dashboard."""
    _load_activity()

    now = datetime.now(IST)
    current_hour = now.hour

    # Parse recent activity for "last active" display
    last_active_ago = None
    if _ketu_activity["recent_active"]:
        try:
            last = datetime.fromisoformat(_ketu_activity["recent_active"][-1])
            if last.tzinfo is None:
                last = last.replace(tzinfo=IST)
            minutes = int((now - last).total_seconds() / 60)
            if minutes < 60:
                last_active_ago = f"{minutes} min ago"
            elif minutes < 1440:
                last_active_ago = f"{minutes // 60}h ago"
            else:
                last_active_ago = f"{minutes // 1440}d ago"
        except (ValueError, IndexError):
            pass

    return {
        "total_replies_tracked": _ketu_activity["total_tracked"],
        "hourly_distribution": {
            int(h): c for h, c in _ketu_activity.get("hourly_replies", {}).items()
        },
        "peak_hours": _get_peak_hours_list(),
        "current_hour": current_hour,
        "is_peak_now": is_peak_hour(),
        "ketu_currently_active": is_ketu_currently_active(),
        "last_active_ago": last_active_ago,
        "enough_data": _ketu_activity["total_tracked"] >= _MIN_DATA_POINTS,
    }
