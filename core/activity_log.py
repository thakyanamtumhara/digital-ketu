"""Activity logger for Digital Ketu — tracks all learning events.

Stores a rolling log of what Digital Ketu learned, when, and from where.
This powers the /api/dashboard endpoint so the owner can see live activity.

Storage: JSON file (knowledge/activity_log.json) — persists across restarts.
Max entries: 500 (oldest auto-pruned).
"""

import json
import logging
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))
LOG_FILE = Path(__file__).parent.parent / "knowledge" / "activity_log.json"
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
    # Prune to max entries
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
    """Log a learning activity event.

    Args:
        source: Where the data came from — "wwbun-sync", "whatsapp-export",
                "youtube", "catalog-sync", "api-reply"
        action: What happened — "learned", "synced", "replied", "error"
        details: Extra info (FAQs learned, patterns, products, etc.)
        items_count: Number of items processed/learned
    """
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
    """Get recent activity log entries.

    Args:
        limit: Max entries to return (default 50)
        source_filter: Filter by source (e.g., "youtube", "wwbun-sync")
        date_filter: Filter by date string (e.g., "04 Mar 2026")
    """
    entries = _load_log()

    if source_filter:
        entries = [e for e in entries if e.get("source") == source_filter]

    if date_filter:
        entries = [e for e in entries if e.get("date") == date_filter]

    # Return most recent first
    return list(reversed(entries[-limit:]))


def get_today_summary() -> dict:
    """Get summary of today's activity."""
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
            summary["by_source"][src] = {
                "count": 0,
                "items_learned": 0,
                "last_time": "",
            }
        summary["by_source"][src]["count"] += 1
        summary["by_source"][src]["items_learned"] += entry.get("items_count", 0)
        summary["by_source"][src]["last_time"] = entry.get("time", "")
        summary["total_items_learned"] += entry.get("items_count", 0)

    return summary


def get_storage_stats() -> dict:
    """Get knowledge base storage statistics."""
    knowledge_dir = Path(__file__).parent.parent / "knowledge"
    learned_dir = knowledge_dir / "learned"

    stats = {
        "knowledge_files": {},
        "learned_files": [],
        "total_size_kb": 0,
        "activity_log_entries": len(_load_log()),
    }

    # Main knowledge files
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
    if learned_dir.exists():
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

    # Count items in knowledge base
    try:
        with open(knowledge_dir / "products.json", "r") as f:
            products = json.load(f)
            stats["products_count"] = len(products.get("catalog", []))
    except Exception:
        stats["products_count"] = 0

    try:
        with open(knowledge_dir / "faq.json", "r") as f:
            faq = json.load(f)
            stats["faqs_count"] = len(faq.get("faqs", []))
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

    # Processed YouTube videos count
    processed_file = learned_dir / "_processed_videos.json"
    if processed_file.exists():
        try:
            with open(processed_file, "r") as f:
                data = json.load(f)
                stats["youtube_videos_processed"] = len(data.get("video_ids", []))
        except Exception:
            stats["youtube_videos_processed"] = 0
    else:
        stats["youtube_videos_processed"] = 0

    return stats
