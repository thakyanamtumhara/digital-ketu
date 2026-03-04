"""Error tracker for Digital Ketu — tracks recent errors for dashboard visibility.

In-memory ring buffer of last 50 errors. No DB storage needed —
errors are transient and only useful for recent debugging.
"""

import logging
from datetime import datetime, timezone, timedelta
from collections import deque

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

# Ring buffer of recent errors (max 50)
_errors: deque[dict] = deque(maxlen=50)

# Error counts by source (for health check)
_error_counts: dict[str, int] = {}


def track_error(source: str, error_msg: str, context: dict | None = None):
    """Track an error event."""
    now = datetime.now(IST)
    _errors.append({
        "timestamp": now.isoformat(),
        "time": now.strftime("%I:%M %p IST"),
        "date": now.strftime("%d %b %Y"),
        "source": source,
        "error": error_msg[:200],  # Truncate long errors
        "context": context or {},
    })
    _error_counts[source] = _error_counts.get(source, 0) + 1
    logger.error(f"[ErrorTracker] {source}: {error_msg[:100]}")


def get_recent_errors(limit: int = 20) -> list[dict]:
    """Get recent errors, newest first."""
    return list(reversed(list(_errors)))[:limit]


def get_error_summary() -> dict:
    """Get error summary for health check."""
    now = datetime.now(IST)
    today = now.strftime("%d %b %Y")

    today_errors = [e for e in _errors if e.get("date") == today]
    last_hour = [
        e for e in _errors
        if (now - datetime.fromisoformat(e["timestamp"])).total_seconds() < 3600
    ]

    return {
        "total_errors": len(_errors),
        "today_errors": len(today_errors),
        "last_hour_errors": len(last_hour),
        "by_source": dict(_error_counts),
        "latest": list(_errors)[-1] if _errors else None,
    }


def clear_errors():
    """Clear all tracked errors."""
    _errors.clear()
    _error_counts.clear()
