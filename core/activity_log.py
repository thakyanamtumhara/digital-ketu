"""Activity logger for Digital Ketu — tracks all learning events.

Primary storage: PostgreSQL (survives deploys).
Fallback: JSON file (knowledge/activity_log.json).
"""

import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

from core.config import KNOWLEDGE_DIR

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))
LOG_FILE = KNOWLEDGE_DIR / "activity_log.json"
MAX_ENTRIES = 500


def _load_log() -> list[dict]:
    if LOG_FILE.exists():
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _save_log(entries: list[dict]):
    if len(entries) > MAX_ENTRIES:
        entries = entries[-MAX_ENTRIES:]
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


def log_activity(
    source: str,
    action: str,
    details: dict | None = None,
    items_count: int = 0,
):
    """Log a learning activity event to DB + JSON file."""
    # Save to DB (primary)
    from core.database import is_db_available, save_activity
    if is_db_available():
        save_activity(source, action, details, items_count)

    # Also save to JSON file (backup + backward compat)
    now = datetime.now(IST)
    entry = {
        "timestamp": now.isoformat(),
        "date": now.strftime("%d %b %Y"),
        "time": now.strftime("%I:%M %p IST"),
        "source": source,
        "action": action,
        "items_count": items_count,
        "details": details or {},
    }
    entries = _load_log()
    entries.append(entry)
    _save_log(entries)

    logger.info(f"[Activity] {source} → {action} ({items_count} items)")


def get_activity_log(
    limit: int = 50,
    source_filter: str | None = None,
    date_filter: str | None = None,
) -> list[dict]:
    """Get recent activity log entries. DB first, JSON fallback."""
    from core.database import is_db_available, get_activity_from_db
    if is_db_available():
        result = get_activity_from_db(limit, source_filter, date_filter)
        if result:
            return result

    # JSON fallback
    entries = _load_log()
    if source_filter:
        entries = [e for e in entries if e.get("source") == source_filter]
    if date_filter:
        entries = [e for e in entries if e.get("date") == date_filter]
    return list(reversed(entries[-limit:]))


def get_today_summary() -> dict:
    """Get summary of today's activity."""
    from core.database import is_db_available, get_today_summary_from_db
    if is_db_available():
        result = get_today_summary_from_db()
        if result and result.get("total_events", 0) > 0:
            return result

    # JSON fallback
    today = datetime.now(IST).strftime("%d %b %Y")
    entries = _load_log()
    today_entries = [e for e in entries if e.get("date") == today]

    summary = {
        "date": today,
        "total_events": len(today_entries),
        "by_source": {},
        "total_items_learned": 0,
        "latest_activity": today_entries[-1] if today_entries else None,
    }

    for entry in today_entries:
        src = entry.get("source", "unknown")
        if src not in summary["by_source"]:
            summary["by_source"][src] = {"count": 0, "items_learned": 0, "last_time": ""}
        summary["by_source"][src]["count"] += 1
        summary["by_source"][src]["items_learned"] += entry.get("items_count", 0)
        summary["by_source"][src]["last_time"] = entry.get("time", "")
        summary["total_items_learned"] += entry.get("items_count", 0)

    return summary


def get_storage_stats() -> dict:
    """Get knowledge base storage statistics."""
    from core.config import KNOWLEDGE_DIR, LEARNED_DIR
    from core.database import (
        is_db_available, load_knowledge_from_db, get_activity_count,
        count_learned_files, list_learned_files, kv_get,
    )

    knowledge_dir = KNOWLEDGE_DIR
    learned_dir = LEARNED_DIR
    use_db = is_db_available()

    stats = {
        "knowledge_files": {},
        "learned_files": [],
        "total_size_kb": 0,
        "storage_mode": "postgresql" if use_db else "json-files",
    }

    # Main knowledge files (from filesystem for file size info)
    for f in knowledge_dir.glob("*.json"):
        size = f.stat().st_size
        stats["knowledge_files"][f.stem] = {
            "file": f.name,
            "size_kb": round(size / 1024, 1),
            "modified": datetime.fromtimestamp(
                f.stat().st_mtime, tz=IST
            ).strftime("%d %b %Y %I:%M %p IST"),
        }
        stats["total_size_kb"] += size

    # Learned files
    if use_db:
        db_learned = list_learned_files()
        for f in db_learned:
            stats["learned_files"].append({
                "file": f["file"],
                "size_kb": 0,
                "modified": f.get("updated_at", ""),
            })
    elif learned_dir.exists():
        for f in learned_dir.iterdir():
            if f.name.startswith("_"):
                continue
            size = f.stat().st_size
            stats["learned_files"].append({
                "file": f.name,
                "size_kb": round(size / 1024, 1),
                "modified": datetime.fromtimestamp(
                    f.stat().st_mtime, tz=IST
                ).strftime("%d %b %Y %I:%M %p IST"),
            })
            stats["total_size_kb"] += size

    stats["total_size_kb"] = round(stats["total_size_kb"] / 1024, 1)

    # Activity log count
    if use_db:
        stats["activity_log_entries"] = get_activity_count()
    else:
        stats["activity_log_entries"] = len(_load_log())

    # Count items from DB or JSON
    if use_db:
        products_data = load_knowledge_from_db("products")
        faq_data = load_knowledge_from_db("faq")
        style_data = load_knowledge_from_db("style")
        prompt_data = load_knowledge_from_db("prompt")

        stats["products_count"] = len((products_data or {}).get("catalog", []))
        stats["faqs_count"] = len((faq_data or {}).get("faqs", []))
        stats["style_patterns_count"] = len((style_data or {}).get("learned_patterns", []))
        stats["example_conversations_count"] = len((style_data or {}).get("example_conversations", []))

        stats["evolved_traits_count"] = len((prompt_data or {}).get("evolved_traits", []))
        stats["evolved_phrases_count"] = len((prompt_data or {}).get("evolved_phrases", []))
        stats["evolved_rules_count"] = len((prompt_data or {}).get("evolved_rules", []))
        stats["prompt_version"] = (prompt_data or {}).get("version", 1)

        # YouTube processed videos from kv_store
        pv = kv_get("_processed_videos", {})
        stats["youtube_videos_processed"] = len(pv.get("video_ids", []) if isinstance(pv, dict) else [])

        stats["learned_files_count"] = count_learned_files()
    else:
        # JSON fallback (existing logic)
        try:
            with open(knowledge_dir / "products.json", "r") as f:
                stats["products_count"] = len(json.load(f).get("catalog", []))
        except Exception:
            stats["products_count"] = 0

        try:
            with open(knowledge_dir / "faq.json", "r") as f:
                stats["faqs_count"] = len(json.load(f).get("faqs", []))
        except Exception:
            stats["faqs_count"] = 0

        try:
            with open(knowledge_dir / "style.json", "r") as f:
                style = json.load(f)
                stats["style_patterns_count"] = len(style.get("learned_patterns", []))
                stats["example_conversations_count"] = len(style.get("example_conversations", []))
        except Exception:
            stats["style_patterns_count"] = 0
            stats["example_conversations_count"] = 0

        try:
            with open(knowledge_dir / "prompt.json", "r") as f:
                prompt_data = json.load(f)
                stats["evolved_traits_count"] = len(prompt_data.get("evolved_traits", []))
                stats["evolved_phrases_count"] = len(prompt_data.get("evolved_phrases", []))
                stats["evolved_rules_count"] = len(prompt_data.get("evolved_rules", []))
                stats["prompt_version"] = prompt_data.get("version", 1)
        except Exception:
            stats["evolved_traits_count"] = 0
            stats["evolved_phrases_count"] = 0
            stats["evolved_rules_count"] = 0
            stats["prompt_version"] = 1

        processed_file = learned_dir / "_processed_videos.json"
        if processed_file.exists():
            try:
                with open(processed_file, "r") as f:
                    stats["youtube_videos_processed"] = len(json.load(f).get("video_ids", []))
            except Exception:
                stats["youtube_videos_processed"] = 0
        else:
            stats["youtube_videos_processed"] = 0

    return stats
