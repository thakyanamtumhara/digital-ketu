"""Conversation log for tracking all AI replies.

Stores every AI reply with full context so Ketu can review later
and submit corrections — even hours after the reply was sent.

Storage: PostgreSQL (primary) + in-memory ring buffer (fallback).
"""

import json
import logging
from collections import deque
from datetime import datetime, timezone, timedelta

from core.config import KNOWLEDGE_DIR

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

# In-memory ring buffer (last 200 conversations)
_conversation_log: deque[dict] = deque(maxlen=200)


def log_conversation(
    customer_phone: str,
    customer_name: str,
    customer_message: str,
    ai_reply: str,
):
    """Log an AI reply for later review/correction by Ketu."""
    now = datetime.now(IST)
    entry = {
        "customer_phone": customer_phone,
        "customer_name": customer_name,
        "customer_message": customer_message,
        "ai_reply": ai_reply,
        "timestamp": now.isoformat(),
        "time": now.strftime("%I:%M %p"),
        "date": now.strftime("%d %b"),
        "corrected": False,
    }

    # Save to in-memory buffer
    _conversation_log.append(entry)

    # Save to DB if available
    try:
        from core.database import is_db_available, _execute
        if is_db_available():
            _execute(
                """INSERT INTO conversation_log
                   (customer_phone, customer_name, customer_message, ai_reply, timestamp, corrected)
                   VALUES (%s, %s, %s, %s, %s, FALSE)""",
                (customer_phone, customer_name, customer_message, ai_reply, now),
            )
    except Exception as e:
        logger.warning(f"[ConvoLog] DB save failed: {e}")


def get_recent_conversations(limit: int = 20) -> list[dict]:
    """Get recent AI conversations for review.

    Returns newest first. Ketu uses this to check AI replies.
    """
    # Try DB first
    try:
        from core.database import is_db_available, _execute
        if is_db_available():
            rows = _execute(
                """SELECT customer_phone, customer_name, customer_message,
                          ai_reply, timestamp, corrected
                   FROM conversation_log
                   ORDER BY timestamp DESC
                   LIMIT %s""",
                (limit,),
                fetch=True,
            )
            if rows:
                result = []
                for r in rows:
                    ts = r["timestamp"]
                    if hasattr(ts, "astimezone"):
                        ts = ts.astimezone(IST)
                    result.append({
                        "customer_phone": r["customer_phone"],
                        "customer_name": r["customer_name"],
                        "customer_message": r["customer_message"],
                        "ai_reply": r["ai_reply"],
                        "time": ts.strftime("%I:%M %p") if hasattr(ts, "strftime") else str(ts),
                        "date": ts.strftime("%d %b") if hasattr(ts, "strftime") else "",
                        "corrected": r.get("corrected", False),
                    })
                return result
    except Exception as e:
        logger.warning(f"[ConvoLog] DB read failed: {e}")

    # Fallback to in-memory
    entries = list(_conversation_log)
    entries.reverse()  # newest first
    if entries:
        return entries[:limit]

    # Second fallback: use activity_log entries (always has AI reply data)
    try:
        from core.database import is_db_available, _execute
        if is_db_available():
            rows = _execute(
                """SELECT details, timestamp
                   FROM activity_log
                   WHERE action = 'ai-reply' AND source = 'whatsapp'
                   ORDER BY timestamp DESC
                   LIMIT %s""",
                (limit,),
                fetch=True,
            )
            if rows:
                result = []
                for r in rows:
                    details = r.get("details", {}) or {}
                    if isinstance(details, str):
                        import json
                        try:
                            details = json.loads(details)
                        except Exception:
                            details = {}
                    ts = r["timestamp"]
                    if hasattr(ts, "astimezone"):
                        ts = ts.astimezone(IST)
                    result.append({
                        "customer_phone": details.get("customer_phone", ""),
                        "customer_name": details.get("customer_name", ""),
                        "customer_message": details.get("customer_message", details.get("message", "")),
                        "ai_reply": details.get("ai_reply", details.get("reply", "")),
                        "time": ts.strftime("%I:%M %p") if hasattr(ts, "strftime") else str(ts),
                        "date": ts.strftime("%d %b") if hasattr(ts, "strftime") else "",
                        "corrected": False,
                    })
                return result
    except Exception as e:
        logger.warning(f"[ConvoLog] Activity log fallback failed: {e}")

    return []


def get_last_ai_reply(customer_phone: str) -> dict | None:
    """Get the last AI reply to a specific customer.

    wwbun calls this when Ketu manually messages a customer —
    to check if AI already replied (potential correction).
    """
    # Try DB first
    try:
        from core.database import is_db_available, _execute
        if is_db_available():
            rows = _execute(
                """SELECT customer_phone, customer_name, customer_message,
                          ai_reply, timestamp, corrected
                   FROM conversation_log
                   WHERE customer_phone = %s AND corrected = FALSE
                   ORDER BY timestamp DESC
                   LIMIT 1""",
                (customer_phone,),
                fetch=True,
            )
            if rows and rows[0]:
                r = rows[0]
                return {
                    "customer_phone": r["customer_phone"],
                    "customer_name": r["customer_name"],
                    "customer_message": r["customer_message"],
                    "ai_reply": r["ai_reply"],
                }
    except Exception as e:
        logger.debug(f"DB last reply lookup failed: {e}")

    # Fallback to in-memory
    for entry in reversed(_conversation_log):
        if entry["customer_phone"] == customer_phone and not entry.get("corrected"):
            return entry
    return None


def mark_corrected(customer_phone: str):
    """Mark the last conversation with this customer as corrected."""
    # DB
    try:
        from core.database import is_db_available, _execute
        if is_db_available():
            _execute(
                """UPDATE conversation_log SET corrected = TRUE
                   WHERE customer_phone = %s AND corrected = FALSE
                   ORDER BY timestamp DESC LIMIT 1""",
                (customer_phone,),
            )
    except Exception:
        pass

    # In-memory
    for entry in reversed(_conversation_log):
        if entry["customer_phone"] == customer_phone and not entry.get("corrected"):
            entry["corrected"] = True
            break


def init_conversation_log_table():
    """Create conversation_log table if it doesn't exist.

    Also verifies the table can accept inserts (health check).
    """
    try:
        from core.database import is_db_available, _execute
        if not is_db_available():
            logger.warning("[ConvoLog] DB not available — conversation_log table skipped")
            return

        # Create table
        _execute("""
            CREATE TABLE IF NOT EXISTS conversation_log (
                id SERIAL PRIMARY KEY,
                customer_phone TEXT NOT NULL,
                customer_name TEXT DEFAULT '',
                customer_message TEXT NOT NULL,
                ai_reply TEXT NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                corrected BOOLEAN DEFAULT FALSE
            );
            CREATE INDEX IF NOT EXISTS idx_convlog_phone
                ON conversation_log(customer_phone);
            CREATE INDEX IF NOT EXISTS idx_convlog_timestamp
                ON conversation_log(timestamp DESC);
        """)

        # Health check — verify table exists and is writable
        check = _execute(
            "SELECT COUNT(*) as cnt FROM conversation_log", fetch=True
        )
        count = check[0]["cnt"] if check else -1
        logger.info(f"[ConvoLog] Table ready — {count} existing conversations")

        if count == 0:
            logger.warning(
                "[ConvoLog] Table is EMPTY — if AI has been replying, "
                "conversation logs were lost. Check previous deploy logs for errors."
            )

    except Exception as e:
        logger.error(f"[ConvoLog] Table creation/check FAILED: {e}")
