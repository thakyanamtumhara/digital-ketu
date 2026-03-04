"""PostgreSQL persistence layer for Digital Ketu.

Stores all knowledge, activity logs, and learned data in Railway PostgreSQL.
This survives deploys — no more data loss on redeploy.

Tables:
- knowledge_store: JSON blobs for each knowledge file (faq, products, style, prompt, company)
- activity_log: All learning/reply events with timestamps
- learned_files: YouTube knowledge extracts and other learned content
- kv_store: Key-value store for internal state (processed_videos, backfill_state)

Env var: DATABASE_URL (auto-set by Railway PostgreSQL plugin)
"""

import json
import logging
import os
from datetime import datetime, timezone, timedelta

import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

_conn = None


def get_connection():
    """Get or create a database connection."""
    global _conn
    database_url = os.environ.get("DATABASE_URL", "")
    if not database_url:
        return None

    try:
        if _conn is None or _conn.closed:
            _conn = psycopg2.connect(database_url)
            _conn.autocommit = True
            logger.info("[DB] Connected to PostgreSQL")
        return _conn
    except Exception as e:
        logger.error(f"[DB] Connection failed: {e}")
        _conn = None
        return None


def _reconnect():
    """Force reconnect on connection errors."""
    global _conn
    _conn = None
    return get_connection()


def _execute(query: str, params=None, fetch: bool = False):
    """Execute a query with auto-reconnect on failure."""
    conn = get_connection()
    if not conn:
        return None

    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, params)
            if fetch:
                return cur.fetchall()
            return True
    except psycopg2.OperationalError:
        # Connection lost — reconnect and retry once
        conn = _reconnect()
        if not conn:
            return None
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(query, params)
                if fetch:
                    return cur.fetchall()
                return True
        except Exception as e:
            logger.error(f"[DB] Query failed after reconnect: {e}")
            return None
    except Exception as e:
        logger.error(f"[DB] Query error: {e}")
        return None


def init_db():
    """Create tables if they don't exist. Call on startup."""
    conn = get_connection()
    if not conn:
        logger.info("[DB] No DATABASE_URL — running in JSON-only mode")
        return False

    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_store (
                    key TEXT PRIMARY KEY,
                    data JSONB NOT NULL,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );

                CREATE TABLE IF NOT EXISTS activity_log (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                    date TEXT,
                    time TEXT,
                    source TEXT NOT NULL,
                    action TEXT NOT NULL,
                    items_count INTEGER DEFAULT 0,
                    details JSONB DEFAULT '{}'::jsonb
                );

                CREATE TABLE IF NOT EXISTS learned_files (
                    filename TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata JSONB DEFAULT '{}'::jsonb,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );

                CREATE TABLE IF NOT EXISTS kv_store (
                    key TEXT PRIMARY KEY,
                    value JSONB NOT NULL,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );

                CREATE INDEX IF NOT EXISTS idx_activity_source ON activity_log(source);
                CREATE INDEX IF NOT EXISTS idx_activity_date ON activity_log(date);
                CREATE INDEX IF NOT EXISTS idx_activity_timestamp ON activity_log(timestamp DESC);
            """)
        logger.info("[DB] Tables ready")
        return True
    except Exception as e:
        logger.error(f"[DB] Init failed: {e}")
        return False


def is_db_available() -> bool:
    """Check if database is available."""
    return get_connection() is not None


# --- Knowledge Store ---

def save_knowledge(key: str, data: dict):
    """Save a knowledge JSON blob to DB."""
    _execute(
        """INSERT INTO knowledge_store (key, data, updated_at)
           VALUES (%s, %s, NOW())
           ON CONFLICT (key) DO UPDATE SET data = %s, updated_at = NOW()""",
        (key, json.dumps(data, ensure_ascii=False), json.dumps(data, ensure_ascii=False)),
    )


def load_knowledge_from_db(key: str) -> dict | None:
    """Load a knowledge JSON blob from DB."""
    rows = _execute(
        "SELECT data FROM knowledge_store WHERE key = %s",
        (key,),
        fetch=True,
    )
    if rows and rows[0]:
        return rows[0]["data"]
    return None


def load_all_knowledge_from_db() -> dict:
    """Load all knowledge blobs from DB."""
    rows = _execute("SELECT key, data FROM knowledge_store", fetch=True)
    if not rows:
        return {}
    return {row["key"]: row["data"] for row in rows}


# --- Activity Log ---

def save_activity(source: str, action: str, details: dict | None = None, items_count: int = 0):
    """Save an activity log entry to DB."""
    now = datetime.now(IST)
    _execute(
        """INSERT INTO activity_log (timestamp, date, time, source, action, items_count, details)
           VALUES (%s, %s, %s, %s, %s, %s, %s)""",
        (
            now,
            now.strftime("%d %b %Y"),
            now.strftime("%I:%M %p IST"),
            source,
            action,
            items_count,
            json.dumps(details or {}, ensure_ascii=False),
        ),
    )


def get_activity_from_db(
    limit: int = 50,
    source_filter: str | None = None,
    date_filter: str | None = None,
) -> list[dict]:
    """Get activity log entries from DB."""
    query = "SELECT * FROM activity_log WHERE 1=1"
    params = []

    if source_filter:
        query += " AND source = %s"
        params.append(source_filter)

    if date_filter:
        query += " AND date = %s"
        params.append(date_filter)

    query += " ORDER BY timestamp DESC LIMIT %s"
    params.append(limit)

    rows = _execute(query, params, fetch=True)
    if not rows:
        return []

    return [
        {
            "timestamp": row["timestamp"].isoformat() if row["timestamp"] else "",
            "date": row["date"],
            "time": row["time"],
            "source": row["source"],
            "action": row["action"],
            "items_count": row["items_count"],
            "details": row["details"] or {},
        }
        for row in rows
    ]


def get_today_summary_from_db() -> dict:
    """Get today's activity summary from DB."""
    today = datetime.now(IST).strftime("%d %b %Y")
    rows = _execute(
        "SELECT source, action, items_count, time FROM activity_log WHERE date = %s ORDER BY timestamp",
        (today,),
        fetch=True,
    )

    summary = {
        "date": today,
        "total_events": 0,
        "by_source": {},
        "total_items_learned": 0,
        "latest_activity": None,
    }

    if not rows:
        return summary

    summary["total_events"] = len(rows)

    for row in rows:
        src = row["source"]
        if src not in summary["by_source"]:
            summary["by_source"][src] = {"count": 0, "items_learned": 0, "last_time": ""}
        summary["by_source"][src]["count"] += 1
        summary["by_source"][src]["items_learned"] += row["items_count"] or 0
        summary["by_source"][src]["last_time"] = row["time"] or ""
        summary["total_items_learned"] += row["items_count"] or 0

    summary["latest_activity"] = {
        "source": rows[-1]["source"],
        "action": rows[-1]["action"],
        "time": rows[-1]["time"],
    }

    return summary


def get_activity_count() -> int:
    """Get total activity log entry count."""
    rows = _execute("SELECT COUNT(*) as cnt FROM activity_log", fetch=True)
    return rows[0]["cnt"] if rows else 0


# --- Learned Files ---

def save_learned_file(filename: str, content: str, metadata: dict | None = None):
    """Save a learned file to DB."""
    _execute(
        """INSERT INTO learned_files (filename, content, metadata, updated_at)
           VALUES (%s, %s, %s, NOW())
           ON CONFLICT (filename) DO UPDATE SET content = %s, metadata = %s, updated_at = NOW()""",
        (
            filename,
            content,
            json.dumps(metadata or {}, ensure_ascii=False),
            content,
            json.dumps(metadata or {}, ensure_ascii=False),
        ),
    )


def load_learned_file(filename: str) -> str | None:
    """Load a learned file from DB."""
    rows = _execute(
        "SELECT content FROM learned_files WHERE filename = %s",
        (filename,),
        fetch=True,
    )
    if rows and rows[0]:
        return rows[0]["content"]
    return None


def list_learned_files() -> list[dict]:
    """List all learned files in DB."""
    rows = _execute(
        "SELECT filename, metadata, updated_at FROM learned_files ORDER BY updated_at DESC",
        fetch=True,
    )
    if not rows:
        return []
    return [
        {
            "file": row["filename"],
            "metadata": row["metadata"] or {},
            "updated_at": row["updated_at"].isoformat() if row["updated_at"] else "",
        }
        for row in rows
    ]


def count_learned_files() -> int:
    """Count non-internal learned files."""
    rows = _execute(
        "SELECT COUNT(*) as cnt FROM learned_files WHERE filename NOT LIKE '\\_%'",
        fetch=True,
    )
    return rows[0]["cnt"] if rows else 0


# --- Key-Value Store ---

def kv_set(key: str, value):
    """Set a key-value pair."""
    _execute(
        """INSERT INTO kv_store (key, value, updated_at)
           VALUES (%s, %s, NOW())
           ON CONFLICT (key) DO UPDATE SET value = %s, updated_at = NOW()""",
        (key, json.dumps(value, ensure_ascii=False), json.dumps(value, ensure_ascii=False)),
    )


def kv_get(key: str, default=None):
    """Get a value by key."""
    rows = _execute("SELECT value FROM kv_store WHERE key = %s", (key,), fetch=True)
    if rows and rows[0]:
        return rows[0]["value"]
    return default


# --- Seed DB from JSON files ---

# --- Data Cleanup ---

def cleanup_old_activity_logs(days: int = 90) -> int:
    """Delete activity log entries older than N days. Returns count deleted."""
    rows = _execute(
        "DELETE FROM activity_log WHERE timestamp < NOW() - INTERVAL '%s days' RETURNING id",
        (days,),
        fetch=True,
    )
    count = len(rows) if rows else 0
    if count:
        logger.info(f"[DB] Cleaned up {count} activity logs older than {days} days")
    return count


def cleanup_old_learned_files(days: int = 90, keep_patterns: list[str] | None = None) -> int:
    """Delete learned files not updated in N days, except those matching keep_patterns.

    keep_patterns: list of LIKE patterns to keep (e.g., ['catalog_%'] for permanent files).
    Returns count deleted.
    """
    keep_patterns = keep_patterns or ["catalog_%"]

    query = "DELETE FROM learned_files WHERE updated_at < NOW() - INTERVAL '%s days'"
    params: list = [days]

    for pattern in keep_patterns:
        query += " AND filename NOT LIKE %s"
        params.append(pattern)

    query += " RETURNING filename"
    rows = _execute(query, params, fetch=True)
    count = len(rows) if rows else 0
    if count:
        deleted_files = [r["filename"] for r in rows]
        logger.info(f"[DB] Cleaned up {count} learned files older than {days} days: {deleted_files}")
    return count


def get_cleanup_stats() -> dict:
    """Get counts of records eligible for cleanup."""
    activity_rows = _execute(
        "SELECT COUNT(*) as cnt FROM activity_log WHERE timestamp < NOW() - INTERVAL '90 days'",
        fetch=True,
    )
    learned_rows = _execute(
        "SELECT COUNT(*) as cnt FROM learned_files WHERE updated_at < NOW() - INTERVAL '90 days' AND filename NOT LIKE 'catalog_%'",
        fetch=True,
    )
    return {
        "activity_logs_eligible": activity_rows[0]["cnt"] if activity_rows else 0,
        "learned_files_eligible": learned_rows[0]["cnt"] if learned_rows else 0,
    }


def seed_from_json_files(knowledge_dir):
    """One-time seed: load existing JSON files into DB.

    Only seeds if DB tables are empty (first run after migration).
    """
    if not is_db_available():
        return

    # Check if already seeded
    rows = _execute("SELECT COUNT(*) as cnt FROM knowledge_store", fetch=True)
    if rows and rows[0]["cnt"] > 0:
        logger.info("[DB] Already seeded — skipping")
        return

    logger.info("[DB] First run — seeding DB from JSON files...")
    seeded = 0

    # Seed knowledge files
    for f in knowledge_dir.glob("*.json"):
        if f.name == "activity_log.json":
            # Seed activity log separately
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    entries = json.load(fh)
                for entry in entries:
                    save_activity(
                        source=entry.get("source", ""),
                        action=entry.get("action", ""),
                        details=entry.get("details"),
                        items_count=entry.get("items_count", 0),
                    )
                seeded += 1
            except Exception as e:
                logger.error(f"[DB] Seed activity_log failed: {e}")
            continue

        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            save_knowledge(f.stem, data)
            seeded += 1
        except Exception as e:
            logger.error(f"[DB] Seed {f.name} failed: {e}")

    # Seed learned files
    learned_dir = knowledge_dir / "learned"
    if learned_dir.exists():
        for f in learned_dir.iterdir():
            if f.name.startswith("_"):
                # Internal files → kv_store
                try:
                    with open(f, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                    kv_set(f.stem, data)
                    seeded += 1
                except Exception:
                    pass
            else:
                try:
                    content = f.read_text(encoding="utf-8")
                    save_learned_file(f.name, content)
                    seeded += 1
                except Exception:
                    pass

    logger.info(f"[DB] Seeded {seeded} items from JSON files")
