"""Leave management for Digital Ketu.

Stores Ketu's leave/busy schedule:
- Full-day leaves (e.g., 4 Sep to 5 Sep)
- Partial busy hours (e.g., 2pm-4pm on a specific day)

When Ketu is on leave, AI auto-replies differently:
- Full day leave: "Ketu bhai leave pe hai, urgent ho toh godown pe call karo / website se order karo"
- Few hours busy: "Ketu bhai thodi der mein reply karenge"
"""

import logging
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))
DB_KEY = "ketu_leaves"


def _now_ist() -> datetime:
    return datetime.now(IST)


def _load_leaves() -> list[dict]:
    """Load all leaves from DB."""
    try:
        from core.database import is_db_available, kv_get
        if is_db_available():
            data = kv_get(DB_KEY, [])
            if isinstance(data, list):
                return data
    except Exception as e:
        logger.error(f"[Leave] Failed to load: {e}")
    return []


def _save_leaves(leaves: list[dict]):
    """Save leaves to DB."""
    try:
        from core.database import is_db_available, kv_set
        if is_db_available():
            kv_set(DB_KEY, leaves)
    except Exception as e:
        logger.error(f"[Leave] Failed to save: {e}")


def add_leave(
    leave_type: str,  # "full_day" or "busy_hours"
    start_date: str,  # "2026-09-04" format
    end_date: str | None = None,  # "2026-09-05" for multi-day, None for single day
    start_time: str | None = None,  # "14:00" for busy_hours
    end_time: str | None = None,  # "16:00" for busy_hours
    reason: str = "",
    contact_info: str = "",  # Godown number, website, etc.
) -> dict:
    """Add a new leave entry."""
    leaves = _load_leaves()

    leave = {
        "id": int(_now_ist().timestamp() * 1000),  # unique ID
        "type": leave_type,
        "start_date": start_date,
        "end_date": end_date or start_date,
        "start_time": start_time,
        "end_time": end_time,
        "reason": reason,
        "contact_info": contact_info,
        "created_at": _now_ist().isoformat(),
    }

    leaves.append(leave)
    _save_leaves(leaves)
    logger.info(f"[Leave] Added: {leave_type} from {start_date} to {end_date or start_date}")
    return leave


def remove_leave(leave_id: int) -> bool:
    """Remove a leave by ID."""
    leaves = _load_leaves()
    original_len = len(leaves)
    leaves = [l for l in leaves if l.get("id") != leave_id]
    if len(leaves) < original_len:
        _save_leaves(leaves)
        logger.info(f"[Leave] Removed leave ID {leave_id}")
        return True
    return False


def get_all_leaves() -> list[dict]:
    """Get all leaves (for dashboard display)."""
    return _load_leaves()


def get_active_leaves() -> list[dict]:
    """Get leaves that haven't ended yet (today or future)."""
    leaves = _load_leaves()
    today = _now_ist().strftime("%Y-%m-%d")
    return [l for l in leaves if l.get("end_date", l.get("start_date", "")) >= today]


def cleanup_past_leaves():
    """Remove leaves that ended more than 7 days ago."""
    leaves = _load_leaves()
    cutoff = (_now_ist() - timedelta(days=7)).strftime("%Y-%m-%d")
    active = [l for l in leaves if l.get("end_date", l.get("start_date", "")) >= cutoff]
    if len(active) < len(leaves):
        _save_leaves(active)
        logger.info(f"[Leave] Cleaned up {len(leaves) - len(active)} old leaves")


def check_leave_status() -> dict | None:
    """Check if Ketu is currently on leave or busy.

    Returns None if Ketu is available.
    Returns dict with leave info if Ketu is on leave/busy right now.
    """
    leaves = _load_leaves()
    now = _now_ist()
    today = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M")

    for leave in leaves:
        start_date = leave.get("start_date", "")
        end_date = leave.get("end_date", start_date)

        # Check if today falls within leave dates
        if start_date <= today <= end_date:
            if leave.get("type") == "full_day":
                # Full day leave — active all day
                days_total = _days_between(start_date, end_date)
                days_left = _days_between(today, end_date)
                return {
                    "active": True,
                    "type": "full_day",
                    "days_total": days_total,
                    "days_left": days_left,
                    "start_date": start_date,
                    "end_date": end_date,
                    "reason": leave.get("reason", ""),
                    "contact_info": leave.get("contact_info", ""),
                }
            elif leave.get("type") == "busy_hours":
                # Busy hours — only active during specified time
                start_time = leave.get("start_time", "00:00")
                end_time = leave.get("end_time", "23:59")
                if start_time <= current_time <= end_time:
                    return {
                        "active": True,
                        "type": "busy_hours",
                        "start_time": start_time,
                        "end_time": end_time,
                        "date": today,
                        "reason": leave.get("reason", ""),
                        "contact_info": leave.get("contact_info", ""),
                    }

    return None


def get_leave_auto_reply(leave_status: dict) -> str:
    """Generate appropriate auto-reply based on leave type."""
    if not leave_status or not leave_status.get("active"):
        return ""

    contact = leave_status.get("contact_info", "").strip()

    if leave_status["type"] == "full_day":
        days_left = leave_status.get("days_left", 1)
        if days_left > 1:
            reply = f"Bhai abhi Ketu sir {days_left} din ke liye leave pe hai."
        else:
            reply = "Bhai abhi Ketu sir aaj leave pe hai."

        if contact:
            reply += f" Urgent ho toh {contact}."
        else:
            reply += " Urgent ho toh website sale91.com pe order kar sakte ho, ya kal reply milega."

        return reply

    elif leave_status["type"] == "busy_hours":
        end_time = leave_status.get("end_time", "")
        if end_time:
            # Convert 24h to 12h for display
            try:
                h, m = end_time.split(":")
                h = int(h)
                period = "PM" if h >= 12 else "AM"
                h12 = h % 12 or 12
                time_display = f"{h12}:{m} {period}"
                reply = f"Bhai Ketu sir abhi busy hai, {time_display} tak reply karenge."
            except Exception:
                reply = "Bhai Ketu sir abhi busy hai, thodi der mein reply karenge."
        else:
            reply = "Bhai Ketu sir abhi busy hai, thodi der mein reply karenge."

        if contact:
            reply += f" Urgent ho toh {contact}."

        return reply

    return ""


def _days_between(date1: str, date2: str) -> int:
    """Calculate days between two date strings (YYYY-MM-DD)."""
    try:
        d1 = datetime.strptime(date1, "%Y-%m-%d")
        d2 = datetime.strptime(date2, "%Y-%m-%d")
        return max(1, (d2 - d1).days + 1)
    except Exception:
        return 1
