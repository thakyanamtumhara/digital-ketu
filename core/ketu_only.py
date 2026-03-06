"""Ketu-Only Question Detection & Queue.

Detects questions that ONLY the real Ketu can answer (stock timelines,
order status, custom pricing, specific delivery dates, payment issues).

When detected:
- AI sends a polite "Ketu sir will reply" message instead of fabricating answers
- Question is logged to a queue visible on the dashboard
- Ketu can review what's being deferred and verify the AI is categorizing correctly
"""

import json
import logging
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path

from core.config import KNOWLEDGE_DIR

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))
KETU_ONLY_FILE = KNOWLEDGE_DIR / "ketu_only.json"

# In-memory queue of deferred questions (persisted to DB)
_deferred_queue: list[dict] = []
_queue_loaded = False

# Cached detection config
_ketu_only_config: dict | None = None


def _load_config() -> dict:
    """Load ketu_only config from DB first, then file fallback."""
    global _ketu_only_config
    if _ketu_only_config is not None:
        return _ketu_only_config

    # Try DB first
    try:
        from core.database import is_db_available, load_knowledge_from_db
        if is_db_available():
            data = load_knowledge_from_db("ketu_only")
            if data:
                _ketu_only_config = data
                return _ketu_only_config
    except Exception:
        pass

    # File fallback
    try:
        with open(KETU_ONLY_FILE, "r", encoding="utf-8") as f:
            _ketu_only_config = json.load(f)
            return _ketu_only_config
    except Exception as e:
        logger.warning(f"Failed to load ketu_only.json: {e}")
        _ketu_only_config = {"categories": []}
        return _ketu_only_config


def invalidate_config_cache():
    """Clear cached config — call after learning new patterns."""
    global _ketu_only_config
    _ketu_only_config = None


def detect_ketu_only(message: str, customer_phone: str = "", customer_name: str = "") -> dict | None:
    """Check if a message is a ketu-only question.

    Returns:
        None if the AI can handle this question normally.
        dict with {category_id, category_name, defer_reply, reason} if Ketu must answer.
    """
    config = _load_config()
    msg_lower = message.strip().lower()

    for cat in config.get("categories", []):
        # Check keywords
        for kw in cat.get("keywords", []):
            if kw.lower() in msg_lower:
                return {
                    "category_id": cat["id"],
                    "category_name": cat["name"],
                    "defer_reply": cat.get("defer_reply", config.get("global_defer_reply", "")),
                    "reason": f"keyword match: '{kw}'",
                }

        # Check regex patterns
        for pattern in cat.get("patterns", []):
            try:
                if re.search(pattern, msg_lower):
                    return {
                        "category_id": cat["id"],
                        "category_name": cat["name"],
                        "defer_reply": cat.get("defer_reply", config.get("global_defer_reply", "")),
                        "reason": f"pattern match: '{pattern}'",
                    }
            except re.error:
                continue

    # Check learned patterns
    for lp in config.get("learned_patterns", []):
        pattern_text = lp.get("pattern", "") if isinstance(lp, dict) else str(lp)
        if pattern_text.lower() in msg_lower:
            return {
                "category_id": lp.get("category_id", "learned") if isinstance(lp, dict) else "learned",
                "category_name": lp.get("category_name", "Learned Pattern") if isinstance(lp, dict) else "Learned Pattern",
                "defer_reply": config.get("global_defer_reply", "Bhai, ye Ketu sir khud confirm karenge — thodi der mein reply aayega."),
                "reason": f"learned pattern: '{pattern_text}'",
            }

    return None


def _load_queue_from_db():
    """Load deferred queue from DB on first access."""
    global _deferred_queue, _queue_loaded
    if _queue_loaded:
        return
    _queue_loaded = True
    try:
        from core.database import is_db_available, kv_get
        if is_db_available():
            data = kv_get("ketu_only_queue")
            if data and isinstance(data, list):
                _deferred_queue.extend(data)
                logger.info(f"[KetuOnly] Loaded {len(data)} deferred questions from DB")
    except Exception as e:
        logger.warning(f"[KetuOnly] DB queue load failed: {e}")


def _save_queue_to_db():
    """Persist deferred queue to DB."""
    try:
        from core.database import is_db_available, kv_set
        if is_db_available():
            # Keep last 200 entries
            kv_set("ketu_only_queue", _deferred_queue[-200:])
    except Exception as e:
        logger.warning(f"[KetuOnly] DB queue save failed: {e}")


def log_deferred_question(
    customer_phone: str,
    customer_name: str,
    message: str,
    category_id: str,
    category_name: str,
    reason: str,
    defer_reply: str,
):
    """Log a question that was deferred to real Ketu."""
    _load_queue_from_db()

    now = datetime.now(IST)
    entry = {
        "phone_last4": customer_phone[-4:] if len(customer_phone) >= 4 else customer_phone,
        "customer_name": customer_name or "Unknown",
        "message": message[:300],
        "category_id": category_id,
        "category_name": category_name,
        "reason": reason,
        "defer_reply": defer_reply,
        "timestamp": now.isoformat(),
        "date": now.strftime("%d %b %Y"),
        "time": now.strftime("%I:%M %p"),
        "resolved": False,
    }
    _deferred_queue.append(entry)
    _save_queue_to_db()

    # Also log to activity feed
    from core.activity_log import log_activity
    log_activity(
        source="ketu-only",
        action="deferred",
        details={
            "customer_phone": entry["phone_last4"],
            "customer_name": customer_name,
            "category": category_name,
            "message_preview": message[:100],
        },
        items_count=1,
    )

    logger.info(f"[KetuOnly] Deferred to Ketu: {category_name} | {customer_phone[-4:]} | {message[:50]}")


def log_manual_takeover(customer_phone: str, customer_message: str, ketu_reply: str = ""):
    """Log when Ketu manually took over a conversation.

    Called when /api/ketu-replied fires. Adds the customer's last question
    to the ketu-only queue so the dashboard shows what Ketu handled.
    Deduplicates: skips if same phone logged within last 2 minutes.
    """
    _load_queue_from_db()

    phone_last4 = customer_phone[-4:] if len(customer_phone) >= 4 else customer_phone
    now = datetime.now(IST)

    # Dedup: skip if same phone was logged within last 2 minutes
    for entry in reversed(_deferred_queue[-10:]):
        if entry.get("phone_last4") == phone_last4 and entry.get("category_id") == "manual_takeover":
            try:
                entry_time = datetime.fromisoformat(entry["timestamp"])
                if (now - entry_time).total_seconds() < 120:
                    return  # Already logged recently
            except (ValueError, KeyError):
                pass
            break

    entry = {
        "phone_last4": phone_last4,
        "customer_name": "Unknown",
        "message": customer_message[:300],
        "category_id": "manual_takeover",
        "category_name": "Ketu Took Over",
        "reason": f"Ketu replied: '{ketu_reply[:50]}'",
        "defer_reply": ketu_reply,
        "timestamp": now.isoformat(),
        "date": now.strftime("%d %b %Y"),
        "time": now.strftime("%I:%M %p"),
        "resolved": True,  # Already resolved — Ketu already replied
    }
    _deferred_queue.append(entry)
    _save_queue_to_db()

    from core.activity_log import log_activity
    log_activity(
        source="ketu-only",
        action="manual-takeover",
        details={
            "customer_phone": phone_last4,
            "category": "Ketu Took Over",
            "message_preview": customer_message[:100],
            "ketu_reply_preview": ketu_reply[:80],
        },
        items_count=1,
    )
    logger.info(f"[KetuOnly] Manual takeover logged: {phone_last4} | {customer_message[:50]}")


def get_deferred_queue(limit: int = 50, pending_only: bool = False) -> list[dict]:
    """Get the deferred question queue for the dashboard."""
    _load_queue_from_db()
    items = _deferred_queue
    if pending_only:
        items = [q for q in items if not q.get("resolved")]
    return list(reversed(items[-limit:]))


def get_deferred_stats() -> dict:
    """Get summary stats for the dashboard."""
    _load_queue_from_db()
    now = datetime.now(IST)
    today_str = now.strftime("%d %b %Y")

    total = len(_deferred_queue)
    pending = sum(1 for q in _deferred_queue if not q.get("resolved"))
    today = sum(1 for q in _deferred_queue if q.get("date") == today_str)

    # Category breakdown
    by_category: dict[str, int] = {}
    for q in _deferred_queue:
        cat = q.get("category_name", "Unknown")
        by_category[cat] = by_category.get(cat, 0) + 1

    return {
        "total": total,
        "pending": pending,
        "today": today,
        "by_category": by_category,
    }


def mark_resolved(index: int) -> bool:
    """Mark a deferred question as resolved (Ketu replied manually)."""
    _load_queue_from_db()
    # Index from the reversed list
    actual_idx = len(_deferred_queue) - 1 - index
    if 0 <= actual_idx < len(_deferred_queue):
        _deferred_queue[actual_idx]["resolved"] = True
        _save_queue_to_db()
        return True
    return False


def add_learned_pattern(pattern: str, category_id: str = "learned", category_name: str = "Learned Pattern"):
    """Add a new learned pattern that should be deferred to Ketu."""
    config = _load_config()
    config.setdefault("learned_patterns", []).append({
        "pattern": pattern.lower(),
        "category_id": category_id,
        "category_name": category_name,
        "added_at": datetime.now(IST).isoformat(),
    })

    # Save to DB
    try:
        from core.database import is_db_available, save_knowledge_to_db
        if is_db_available():
            save_knowledge_to_db("ketu_only", config)
    except Exception:
        pass

    # Also save to file
    try:
        with open(KETU_ONLY_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

    invalidate_config_cache()
    logger.info(f"[KetuOnly] Learned new pattern: '{pattern}' -> {category_name}")
